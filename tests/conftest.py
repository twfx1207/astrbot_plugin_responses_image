"""Isolate AstrBot's import-time runtime files from the user's configuration."""

import importlib.util
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

_runtime_root = TemporaryDirectory(
    prefix="astrbot-responses-image-tests-", ignore_cleanup_errors=True
)
_previous_root = os.environ.get("ASTRBOT_ROOT")
os.environ["ASTRBOT_ROOT"] = _runtime_root.name
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("ASTRBOT_TEST_MODE", "true")
_core_path = os.environ.get("RESPONSES_IMAGE_TEST_CORE")
if _core_path:
    _repository_root = Path(_core_path).expanduser().resolve()
    if not (_repository_root / "astrbot" / "__init__.py").is_file():
        raise pytest.UsageError(
            "RESPONSES_IMAGE_TEST_CORE must point to an AstrBot source checkout."
        )
    sys.path.insert(0, str(_repository_root))
if importlib.util.find_spec("astrbot") is None:
    raise pytest.UsageError(
        "Install AstrBot in the test environment or set RESPONSES_IMAGE_TEST_CORE "
        "to its source checkout; see CONTRIBUTING.md."
    )


def pytest_collection_modifyitems(items) -> None:
    """Skip native queue integration when the installed core has no wakeup API.

    Args:
        items: Collected tests, including dedicated released-core completion tests.
    """
    if importlib.util.find_spec("astrbot.core.astr_agent_wakeup") is None:
        unsupported = pytest.mark.skip(reason="This core has no native wakeup API")
        for item in items:
            if item.get_closest_marker("native_wakeup"):
                item.add_marker(unsupported)


def pytest_unconfigure(config) -> None:
    """Restore the runtime root and remove temporary test data.

    Args:
        config: The completed pytest session configuration.
    """
    if _previous_root is None:
        os.environ.pop("ASTRBOT_ROOT", None)
    else:
        os.environ["ASTRBOT_ROOT"] = _previous_root
    _runtime_root.cleanup()
