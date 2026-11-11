"""Schedule bounded drawing jobs and select the supported completion backend."""

import asyncio
import time

from .generation import execute_image_request

LegacyCompletionEvent = None
run_legacy_completion = None
native_wakeup_import_error = None
legacy_wakeup_import_error = None
try:
    from astrbot.core.astr_agent_wakeup import wake_main_agent
    from astrbot.core.platform.agent_wakeup_event import AgentWakeupEvent
except ImportError as exc:
    native_wakeup_import_error = (type(exc).__name__, exc.name)
    wake_main_agent = None
    AgentWakeupEvent = None
    try:
        from .legacy_completion import LegacyCompletionEvent as LegacyCompletionEvent
        from .legacy_completion import run_legacy_completion
    except ImportError as legacy_exc:
        legacy_wakeup_import_error = (type(legacy_exc).__name__, legacy_exc.name)

BACKGROUND_TIMEOUT_SECONDS = 3600
RESULT_TIMEOUT_SECONDS = 600


async def run_background_job(plugin, job: dict, arguments: dict) -> None:
    """Run one bounded task and queue a separate main-agent completion turn.

    Args:
        plugin: Plugin owning this request's state and shared connections.
        job: Session-bound task state, cancellation and consumption signals.
        arguments: Validated drawing inputs frozen before returning the tool.
    """
    try:
        async with plugin._queue_changed:
            while True:
                if (
                    plugin._closing
                    or job["stop"].is_set()
                    or plugin._jobs.get(job["id"]) is not job
                ):
                    return
                busy_windows = {
                    item["umo"]
                    for item in plugin._jobs.values()
                    if item["state"] != "queued"
                }
                first = next(
                    (
                        item
                        for item in plugin._jobs.values()
                        if item["state"] == "queued" and item["umo"] not in busy_windows
                    ),
                    None,
                )
                running = sum(
                    item["state"] == "running" for item in plugin._jobs.values()
                )
                if first is job and running < job["concurrency"]:
                    job["state"] = "running"
                    job["started_at"] = time.time()
                    plugin._queue_changed.notify_all()
                    break
                await plugin._queue_changed.wait()
        current_cid = (
            await plugin.context.conversation_manager.get_curr_conversation_id(
                job["umo"]
            )
        )
        if plugin._closing or job["stop"].is_set() or current_cid != job["cid"]:
            return
        try:
            result = await asyncio.wait_for(
                execute_image_request(plugin, **arguments),
                timeout=BACKGROUND_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            result = (
                "后台绘图超时，已停止等待且不会自动重试。"
                "上游可能仍在处理，请确认后再发送新的绘图请求。"
            )
        if plugin._closing or job["stop"].is_set():
            return
        current_cid = (
            await plugin.context.conversation_manager.get_curr_conversation_id(
                job["umo"]
            )
        )
        if current_cid != job["cid"]:
            plugin.logger.debug(
                "Background image completion discarded: conversation_changed"
            )
            return
        job["result"] = result
        async with plugin._queue_changed:
            job["state"] = "awaiting_delivery"
            plugin._queue_changed.notify_all()
        prompt = (
            "A previously requested background image task has finished. "
            "Retrieve its saved result, inspect the pixels and deliver them "
            "to this session. Do not start another drawing request."
        )
        try:
            if wake_main_agent is not None:
                await wake_main_agent(
                    plugin.context,
                    umo=job["umo"],
                    source="plugin",
                    source_id=job["id"],
                    user_prompt=prompt,
                    role="member",
                    stop_signal=job["stop"],
                )
                await asyncio.wait_for(
                    job["claimed"].wait(), timeout=RESULT_TIMEOUT_SECONDS
                )
            else:
                await asyncio.wait_for(
                    run_legacy_completion(plugin, job, prompt),
                    timeout=RESULT_TIMEOUT_SECONDS,
                )
        except asyncio.TimeoutError:
            job["stop"].set()
            plugin.logger.warning("Background image completion timed out")
    except asyncio.CancelledError:
        job["stop"].set()
        plugin.logger.info("Background image task cancelled")
        raise
    except Exception as exc:
        job["stop"].set()
        plugin.logger.warning("Background image task failed: %s", type(exc).__name__)
    finally:
        job["result"] = None
        if plugin.quota is not None:
            plugin.quota.release(job["id"])
        async with plugin._queue_changed:
            if plugin._jobs.get(job["id"]) is job:
                plugin._jobs.pop(job["id"], None)
            plugin._queue_changed.notify_all()
