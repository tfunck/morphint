import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "morphint"
IO_HELPERS_PATH = PACKAGE_DIR / "io_helpers.py"
MORPHINT_PATH = PACKAGE_DIR / "morphint.py"
EXPECTED_HELPERS = {
    "_atomic_ants_image_write",
    "_is_nonempty_file",
    "_is_readable_nifti",
}


def _module(name, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def _load_module(monkeypatch, module_name, file_path, stub_modules):
    for name, module in stub_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_io_helpers_handle_readability_checks_and_atomic_writes(tmp_path, monkeypatch):
    load_calls = []

    def fake_load(path):
        load_calls.append(path)
        if path.endswith("broken.nii.gz"):
            raise OSError("broken nifti")
        return object()

    def successful_image_write(image, path):
        Path(path).write_text(str(image), encoding="utf-8")

    io_helpers = _load_module(
        monkeypatch,
        "io_helpers_under_test",
        IO_HELPERS_PATH,
        {
            "ants": _module("ants", image_write=successful_image_write),
            "nibabel": _module("nibabel", load=fake_load),
        },
    )

    readable = tmp_path / "readable.nii.gz"
    readable.write_bytes(b"nifti")
    empty = tmp_path / "empty.nii.gz"
    empty.touch()
    broken = tmp_path / "broken.nii.gz"
    broken.write_bytes(b"bad")

    assert io_helpers._is_nonempty_file(str(readable))
    assert not io_helpers._is_nonempty_file(str(tmp_path / "missing.nii.gz"))

    assert io_helpers._is_readable_nifti(str(readable))
    assert not io_helpers._is_readable_nifti(str(empty))
    assert not io_helpers._is_readable_nifti(str(broken))
    assert load_calls == [str(readable), str(broken)]

    output_path = tmp_path / "written.nii.gz"
    io_helpers._atomic_ants_image_write("payload", str(output_path))
    assert output_path.read_text(encoding="utf-8") == "payload"
    assert not list(tmp_path.glob(f".{output_path.name}.tmp_*"))

    def failing_image_write(image, path):
        Path(path).write_text("partial", encoding="utf-8")
        raise RuntimeError("write failed")

    io_helpers = _load_module(
        monkeypatch,
        "io_helpers_under_test_failing",
        IO_HELPERS_PATH,
        {
            "ants": _module("ants", image_write=failing_image_write),
            "nibabel": _module("nibabel", load=fake_load),
        },
    )

    failed_output = tmp_path / "failed.nii.gz"
    with pytest.raises(RuntimeError, match="write failed"):
        io_helpers._atomic_ants_image_write("payload", str(failed_output))

    assert not failed_output.exists()
    assert not list(tmp_path.glob(f".{failed_output.name}.tmp_*"))


def test_morphint_imports_shared_io_helpers():
    tree = ast.parse(MORPHINT_PATH.read_text(encoding="utf-8"))
    imported_helpers = set()

    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue

        is_absolute_import = node.level == 0 and node.module == "morphint.io_helpers"
        is_relative_import = node.level == 1 and node.module == "io_helpers"
        if is_absolute_import or is_relative_import:
            imported_helpers.update(alias.name for alias in node.names)

    assert EXPECTED_HELPERS.issubset(imported_helpers)