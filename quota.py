"""Persist rolling drawing counts; reserve queued requests in memory."""

import json
import math
import time
from pathlib import Path
from tempfile import NamedTemporaryFile


class QuotaExceeded(ValueError):
    """Signal that a chat window has no remaining drawing allowance."""


class DrawingQuota:
    """Keep admission and charging atomic on the plugin's event loop.

    Methods deliberately contain no await points. Small JSON writes use atomic
    replacement before a drawing call starts; pending jobs are never restored.
    """

    def __init__(self, path: Path, config: dict) -> None:
        """Load durable counts without resetting malformed state.

        Args:
            path: Plugin-owned ledger file.
            config: Live default limit settings.

        Raises:
            ValueError: The persisted ledger has an invalid structure.
        """
        self.path = path
        self.config = config
        self.reservations: dict[str, str] = {}
        self.used: dict[str, list[float]] = {}
        self.overrides: dict[str, list[int]] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("Invalid drawing quota ledger")
            used, overrides = data.get("used"), data.get("overrides")
            if not isinstance(used, dict) or not isinstance(overrides, dict):
                raise ValueError("Invalid drawing quota records")
            for stamps in used.values():
                if not isinstance(stamps, list) or any(
                    type(stamp) not in (float, int) or not math.isfinite(stamp)
                    for stamp in stamps
                ):
                    raise ValueError("Invalid drawing quota timestamps")
            for limits in overrides.values():
                if (
                    not isinstance(limits, list)
                    or len(limits) != 2
                    or any(type(limit) is not int or limit < 0 for limit in limits)
                ):
                    raise ValueError("Invalid drawing quota limits")
            self.used, self.overrides = used, overrides

    def snapshot(self, umo: str) -> dict:
        """Count rolling usage and pending reservations for one window.

        Args:
            umo: AstrBot's unified message origin.

        Returns:
            Limits, committed usage and outstanding reservations; zero means unlimited.
        """
        now = time.time()
        limits = self.overrides.get(umo)
        if limits is None:
            limits = [
                self.config.get("hourly_limit", 10),
                self.config.get("daily_limit", 30),
            ]
        if any(type(limit) is not int or limit < 0 for limit in limits):
            raise ValueError("Drawing limits must be nonnegative integers")
        stamps = self.used.get(umo, [])
        return {
            "umo": umo,
            "hourly_limit": limits[0],
            "daily_limit": limits[1],
            "used_hour": sum(stamp > now - 3600 for stamp in stamps),
            "used_day": sum(stamp > now - 86400 for stamp in stamps),
            "reserved": sum(window == umo for window in self.reservations.values()),
        }

    def reserve(self, token: str, umo: str, *, admin: bool) -> None:
        """Reserve both rolling allowances before accepting a request.

        Args:
            token: Unique submission identifier.
            umo: Window whose allowance is charged.
            admin: Trusted permission captured from the submitting event.

        Raises:
            QuotaExceeded: A nonzero hourly or daily limit has been reached.
        """
        if admin:
            return
        status = self.snapshot(umo)
        for period, label in (("hour", "小时"), ("day", "天")):
            limit = status["hourly_limit" if period == "hour" else "daily_limit"]
            if limit and status[f"used_{period}"] + status["reserved"] >= limit:
                raise QuotaExceeded(
                    f"本聊天窗口最近一{label}的绘图额度已满（{limit} 次，含排队预留）。"
                )
        self.reservations[token] = umo

    def commit(self, token: str) -> None:
        """Charge a reserved attempt immediately before calling the drawing API.

        Args:
            token: Reservation identifier; administrator requests have no reservation.
        """
        umo = self.reservations.get(token)
        if umo is None:
            return
        now = time.time()
        used = {
            window: [stamp for stamp in stamps if stamp > now - 86400]
            for window, stamps in self.used.items()
        }
        used = {window: stamps for window, stamps in used.items() if stamps}
        used.setdefault(umo, []).append(now)
        self._save(used, self.overrides)
        self.used = used
        self.reservations.pop(token)

    def release(self, token: str) -> None:
        """Release only unstarted work without refunding a drawing attempt.

        Args:
            token: Submission identifier.
        """
        self.reservations.pop(token, None)

    def update(self, umo: str, *, limits: list[int] | None = None) -> dict:
        """Set window limits, or reset usage while preserving pending work.

        Args:
            umo: Administrator-selected window.
            limits: Hourly and daily override; None resets usage only.

        Returns:
            Updated usage and limits.

        Raises:
            ValueError: Limits are not two nonnegative integers.
        """
        used, overrides = dict(self.used), dict(self.overrides)
        if limits is None:
            used.pop(umo, None)
        else:
            if len(limits) != 2 or any(type(n) is not int or n < 0 for n in limits):
                raise ValueError("Drawing limits must be nonnegative integers")
            overrides[umo] = limits
        self._save(used, overrides)
        self.used, self.overrides = used, overrides
        return self.snapshot(umo)

    def _save(self, used: dict, overrides: dict) -> None:
        """Atomically persist a charge, quota reset or limit update.

        Args:
            used: Proposed committed timestamps.
            overrides: Proposed per-window limits.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump({"version": 1, "used": used, "overrides": overrides}, handle)
                handle.flush()
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
