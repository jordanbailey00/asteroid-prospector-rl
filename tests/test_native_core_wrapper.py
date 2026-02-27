import pytest
from asteroid_prospector.native_core import NativeProspectorCore, default_native_library_path


def test_default_native_library_path_points_to_dll_location() -> None:
    path = default_native_library_path()
    assert path.name == "abp_core.dll"
    assert path.parent.name == "build"


def test_native_core_raises_for_missing_library(tmp_path) -> None:
    missing = tmp_path / "missing_core.dll"
    with pytest.raises(FileNotFoundError):
        NativeProspectorCore(seed=0, library_path=missing)
