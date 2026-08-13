"""Compatibility re-export for FlyDSL's legacy buffer operations."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import flydsl


_SOURCE = (
    Path(flydsl.__file__).resolve().parents[2]
    / "kernels"
    / "common"
    / "buffer_ops.py"
)
if not _SOURCE.is_file():
    raise ImportError(
        "FlyDSL kernel buffer_ops.py was not found. Install FlyDSL in editable "
        f"mode from its source checkout (looked for {_SOURCE})."
    )

_SPEC = spec_from_file_location("_flydsl_legacy_buffer_ops", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load FlyDSL buffer operations from {_SOURCE}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

__all__ = list(_MODULE.__all__)
globals().update({name: getattr(_MODULE, name) for name in __all__})
