"""Zero-shot quantum kernel selection on foundation-model embeddings.

A screening rule that predicts, before any quantum circuit is run, whether a
quantum kernel will beat a classical one on a given frozen embedding.

The rule in one line::

    from qks.diagnostics import classical_alignment
    skip = classical_alignment(X_train, y_train) < 0.05

See README.md for the pipeline and scripts/00_verify_install.py to confirm that
an installation reproduces the published results.
"""

__version__ = "1.0.0"

from . import analysis, config, diagnostics, evaluate, features, io, kernels

__all__ = ["analysis", "config", "diagnostics", "evaluate", "features", "io", "kernels"]
