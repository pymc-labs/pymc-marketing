"""Injected code to the top of each notebook to mock long running code."""

from functools import partial

import numpy as np
import pymc as pm
import pymc.testing

# --- TEMPORARY DIAGNOSTIC (remove before merge) -------------------------------
# Capture oversized matplotlib Agg allocations behind the intermittent
# ``MemoryError: std::bad_alloc`` in notebook CI. We print the requested
# width/height/dpi (and a stack) *before* the C++ buffer is allocated, so the
# real figure dimensions are visible even when the allocation then fails.
try:  # pragma: no cover - diagnostic only
    import sys as _sys
    import traceback as _tb

    import matplotlib.backends.backend_agg as _bagg

    _orig_renderer_init = _bagg.RendererAgg.__init__

    def _diag_renderer_init(self, width, height, dpi):
        try:
            if float(width) * float(height) > 2e7:
                print(
                    f"[DIAG-OOM] RendererAgg width={width!r} height={height!r} "
                    f"dpi={dpi!r} px={float(width) * float(height):.3e}",
                    file=_sys.stderr,
                    flush=True,
                )
                _tb.print_stack(file=_sys.stderr)
                _sys.stderr.flush()
        except Exception as _diag_err:  # never let the probe break rendering
            print(f"[DIAG-OOM] probe error: {_diag_err!r}", file=_sys.stderr)
        return _orig_renderer_init(self, width, height, dpi)

    _bagg.RendererAgg.__init__ = _diag_renderer_init
    print("[DIAG-OOM] RendererAgg probe installed", file=_sys.stderr, flush=True)
except Exception as _diag_install_err:  # pragma: no cover
    print(f"[DIAG-OOM] probe install failed: {_diag_install_err!r}")
# --- END TEMPORARY DIAGNOSTIC -------------------------------------------------


def mock_diverging(size):
    return np.zeros(size, dtype=int)


pm.sample = partial(
    pymc.testing.mock_sample,
    sample_stats={"diverging": mock_diverging},
)
pm.HalfFlat = pm.HalfNormal
pm.Flat = pm.Normal
