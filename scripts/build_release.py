"""Build a deterministic AstrBot installation archive using only the standard library."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def build_release(output_dir: Path) -> Path:
    """Validate release inputs and package runtime modules with their documentation.

    Args:
        output_dir: Destination for the ZIP archive and its SHA-256 checksum.

    Returns:
        Absolute path to the installation archive.

    Raises:
        ValueError: Metadata, configuration or local imports are invalid.
        SyntaxError: Runtime syntax is unsupported by Python 3.10.
        OSError: A required source cannot be read or the output cannot be written.
    """
    root = Path(__file__).resolve().parents[1]
    metadata = (root / "metadata.yaml").read_text(encoding="utf-8")
    name = re.search(r"^name: (astrbot_plugin_[a-z0-9_]+)$", metadata, re.MULTILINE)
    version = re.search(r"^version: (\d+\.\d+\.\d+)$", metadata, re.MULTILINE)
    if not name or not version:
        raise ValueError(
            "Release metadata must contain a plugin name and stable version."
        )
    schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
    if not isinstance(schema, dict) or not schema:
        raise ValueError("Plugin configuration schema must be a nonempty object.")
    sources = sorted(root.glob("*.py"))
    modules = {path.stem for path in sources}
    if "main" not in modules:
        raise ValueError("The plugin entry point is missing.")
    for source in sources:
        tree = ast.parse(
            source.read_text(encoding="utf-8"),
            filename=source.name,
            feature_version=(3, 10),
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            imported = (
                {node.module.split(".")[0]}
                if node.module
                else {alias.name for alias in node.names}
            )
            if node.level != 1 or not imported <= modules:
                raise ValueError(f"Unpackaged local import in {source.name}.")
    files = sorted(
        [
            *sources,
            *(
                root / filename
                for filename in (
                    "metadata.yaml",
                    "_conf_schema.json",
                    "requirements.txt",
                    "README.md",
                    "CONTRIBUTING.md",
                    "LICENSE",
                )
            ),
        ],
        key=lambda path: path.name,
    )
    # Normalize text so Windows and Unix checkouts produce the same archive.
    contents = {
        path.name: path.read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .encode("utf-8")
        for path in files
    }
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    package_name = name.group(1)
    archive = output_dir / f"{package_name}-v{version.group(1)}.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for filename, content in contents.items():
            entry = ZipInfo(
                f"{package_name}/{filename}", date_time=(1980, 1, 1, 0, 0, 0)
            )
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = ZIP_DEFLATED
            bundle.writestr(entry, content, compresslevel=9)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(
        f"{checksum}  {archive.name}\n", encoding="utf-8"
    )
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dist",
        help="Directory for the installation archive and checksum (default: dist).",
    )
    print(build_release(parser.parse_args().output_dir))
