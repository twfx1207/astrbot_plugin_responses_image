# Development

This is a standalone AstrBot plugin repository. AstrBot is a host dependency, not vendored source. Runtime configuration and user data belong in AstrBot's data directory and must not be committed here.

## Modules

| File | Responsibility |
| --- | --- |
| `main.py` | Lifecycle, public tools and commands, queue admission and permissions. |
| `request_context.py` | Request-local tools, reference candidates and completion receipt validation. |
| `generation.py` | Reference collection, drawing workflow and returning pixels to the main agent. |
| `background.py` | Scheduling, cancellation and completion backend selection. |
| `legacy_completion.py` | Completion on released v4.29.4 without the native wakeup API. |
| `delivery.py` | Confirmed-send termination, captions and connector notifications. |
| `provider.py` | Isolated Responses and Images API clients using shared host connections. |
| `references.py` | Character library and validated event/history image selection. |
| `search_references.py`, `tavily_search.py` | Reference retrieval and Tavily integration. |
| `quota.py` | Rolling usage, reservations and administrator exemptions. |

Keep plugin/tool names, configuration keys and persistent data compatible. Do not register providers, mutate shared model configurations, retry generation automatically or modify AstrBot core. Retain the released-core completion path until the minimum supported AstrBot version provides its replacement.

Use English comments and logs, Google-style docstrings and conventional commits. Keep runtime syntax compatible with Python 3.10+, while testing AstrBot v4.29.4 on Python 3.12. Use feature branches for development and `release/*` only for stabilization.

## Tests

Use an isolated environment, not a running bot's environment:

```sh
python -m venv .venv
```

Activate it (`.venv/Scripts/Activate.ps1` on PowerShell, `source .venv/bin/activate` on Unix), then install the host and development dependencies:

```sh
git clone --depth 1 --branch v4.29.4 https://github.com/AstrBotDevs/AstrBot.git .test-core
python -m pip install ./.test-core -r requirements.txt -r requirements-dev.txt
python -m pytest -q
ruff format .
ruff check .
```

The commands above test the installed host package. To test another source checkout, install its dependencies in the same isolated environment and set `RESPONSES_IMAGE_TEST_CORE` to the directory containing its `astrbot/` package. No fixed plugin location inside that checkout is required.

PowerShell:

```powershell
$env:RESPONSES_IMAGE_TEST_CORE = 'C:/path/to/AstrBot'
python -m pytest -q
```

Unix:

```sh
RESPONSES_IMAGE_TEST_CORE=/path/to/AstrBot python -m pytest -q
```

Tests isolate runtime data in a temporary directory and replace model/network transports with offline fixtures. They cover generation, edits, reference isolation, quotas, cancellation races, delivery, connector payloads and history persistence; they do not contact QQ or paid model APIs.

Tests marked `native_wakeup` require that host API. Dedicated released-core tests exercise the real v4.29.4 builder, runner and history saver, and skip on hosts using the native implementation. Both variants must pass before release. CI tests the v4.29.4 tag and a pinned AstrBot revision with the native API; update the latter deliberately when changing the host contract.

## Release preparation

The plugin version lives in `metadata.yaml`; do not bump AstrBot's version. Create a short-lived `release/1.0.0` branch from the reviewed feature branch for final validation and version/documentation changes. The README describes current functionality and configuration without pre-1.0 changelog entries.

```sh
ruff format .
ruff check .
python -m pytest -q
python scripts/build_release.py
```

The standard-library build script validates metadata, configuration, Python 3.10 syntax and local imports. It writes `dist/astrbot_plugin_responses_image-v1.0.0.zip` and a `.zip.sha256` checksum. The installation ZIP contains runtime modules, dependency declarations, metadata, configuration, README, development instructions and license. Tests, Git history, caches and user data are excluded. File order, permissions, timestamps and line endings are normalized for reproducible builds with the same Python/zlib toolchain.

Complete source, tests and build scripts remain in Git. Create a source archive with:

```sh
git archive --format=zip --prefix=astrbot_plugin_responses_image/ --output=dist/astrbot_plugin_responses_image-source-v1.0.0.zip HEAD
```

After review and merging the release branch, tag the exact release commit `v1.0.0` and attach the verified installation ZIP and checksum. Keep `repo` in `metadata.yaml` aligned with this repository. Building archives and running CI do not publish or tag a release automatically.
