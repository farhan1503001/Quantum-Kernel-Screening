"""Quantum feature maps and kernel matrices.

The paper's kernel is a single-layer IQP map on eight qubits with a linear chain
of ZZ interactions, evaluated as the fidelity overlap

    K_Q(x, x') = |<0|U(x')^dagger U(x)|0>|^2 .

Two equivalent routes are provided:

``pair_kernel``
    One circuit per pair of points, which is how the kernel must be obtained on
    hardware, where states cannot be read out and each overlap is estimated by
    measurement. This is what the main run used. Cost at N = 1000 was about
    83 minutes per kernel matrix on ``lightning.qubit``.

``fast_kernel``
    Simulate each point's state once and form every entry with one matrix
    product. On an exact statevector simulator this is *identical* to the
    pairwise route; it is not an approximation. ``self_check`` asserts the
    equality on random inputs and is called by every script that uses it.

The ablation feature maps are also defined here so that the robustness study and
the main pipeline share one definition of each circuit.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .config import N_QUBITS

# --------------------------------------------------------------------------
# Feature maps. Each takes a feature vector and a wire range.
# --------------------------------------------------------------------------


def iqp_1layer(x, wires):
    """Single-layer IQP map with a linear chain of ZZ interactions (the paper's map)."""
    n = len(wires)
    for w in wires:
        qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        qml.RZ(x[i], wires=w)
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.RZ(x[i] * x[i + 1], wires=wires[i + 1])
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def iqp_2layer(x, wires):
    """The same layer applied twice."""
    for _ in range(2):
        iqp_1layer(x, wires)


def iqp_1layer_all(x, wires):
    """IQP map with ZZ interactions on all pairs rather than adjacent pairs only."""
    n = len(wires)
    for w in wires:
        qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        qml.RZ(x[i], wires=w)
    for i in range(n):
        for j in range(i + 1, n):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(x[i] * x[j], wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])


def angle_map(x, wires):
    """Entanglement-free angle encoding, one RY rotation per qubit."""
    for i, w in enumerate(wires):
        qml.RY(x[i], wires=w)


#: The six kernels evaluated in the robustness study.
#: ``name -> (feature map, number of qubits)``.
#: With fewer than eight qubits only the first k of the eight scaled features
#: enter the quantum kernel; the classical alignment is always computed on all
#: eight, which is what makes the screen comparable across these settings.
FEATURE_MAPS = {
    "IQP-1L":     (iqp_1layer, 8),     # the paper's kernel
    "IQP-2L":     (iqp_2layer, 8),
    "Angle":      (angle_map, 8),
    "IQP-1L-all": (iqp_1layer_all, 8),
    "IQP-1L-6q":  (iqp_1layer, 6),
    "IQP-1L-4q":  (iqp_1layer, 4),
}

_DEVICES: dict[int, "qml.Device"] = {}


def _device(n_wires: int):
    """One cached ``lightning.qubit`` device per qubit count."""
    if n_wires not in _DEVICES:
        _DEVICES[n_wires] = qml.device("lightning.qubit", wires=n_wires)
    return _DEVICES[n_wires]


# --------------------------------------------------------------------------
# Kernel builders
# --------------------------------------------------------------------------


def make_pair_kernel(fmap=iqp_1layer, n_wires: int = N_QUBITS):
    """Return ``k(a, b)``, one circuit per pair (the hardware-equivalent route)."""
    @qml.qnode(_device(n_wires))
    def circuit(a, b):
        fmap(a, wires=range(n_wires))
        qml.adjoint(fmap)(b, wires=range(n_wires))
        return qml.probs(wires=range(n_wires))

    return lambda a, b: float(circuit(a[:n_wires], b[:n_wires])[0])


def make_fast_kernel(fmap=iqp_1layer, n_wires: int = N_QUBITS):
    """Return ``K(A, B=None)`` built from simulated states.

    Exactly equal to the pairwise route on a statevector simulator, because
    ``K(x, x') = |<phi(x')|phi(x)>|^2`` with ``|phi(x)> = U(x)|0>``.
    """
    @qml.qnode(_device(n_wires))
    def state(x):
        fmap(x, wires=range(n_wires))
        return qml.state()

    def K(A, B=None):
        S_A = np.array([state(a[:n_wires]) for a in A])
        S_B = S_A if B is None else np.array([state(b[:n_wires]) for b in B])
        return np.abs(S_A.conj() @ S_B.T) ** 2

    return K


def self_check(tol: float = 1e-10, n_points: int = 12, seed: int = 3, verbose: bool = True) -> bool:
    """Assert that the fast kernel equals the per-pair circuit for every feature map.

    Run this before any expensive computation. It takes a few seconds and is the
    guard that lets the rest of the code use ``fast_kernel`` safely.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.0, np.pi, (n_points, N_QUBITS))
    for name, (fmap, n_wires) in FEATURE_MAPS.items():
        pair = make_pair_kernel(fmap, n_wires)
        fast = make_fast_kernel(fmap, n_wires)
        slow = np.array([[pair(X[i], X[j]) for j in range(n_points)] for i in range(n_points)])
        diff = np.abs(fast(X) - slow).max()
        assert np.allclose(np.diag(slow), 1.0, atol=1e-9), f"{name}: diagonal is not 1"
        assert np.allclose(slow, slow.T, atol=1e-12), f"{name}: kernel is not symmetric"
        assert diff < tol, f"{name}: fast kernel differs from the circuit by {diff:.1e}"
        if verbose:
            print(f"  {name:<11} fast == per-pair circuit (max diff {diff:.1e})")
    if verbose:
        print("kernel self-check passed")
    return True


def quantum_kernel_matrices(X_train, X_test, fmap=iqp_1layer, n_wires: int = N_QUBITS):
    """Return ``(K_train, K_test)`` for the given feature map.

    ``K_train`` is ``N_train x N_train``; ``K_test`` is ``N_test x N_train``,
    which is the orientation scikit-learn expects for a precomputed kernel.
    """
    K = make_fast_kernel(fmap, n_wires)
    return K(X_train), K(X_test, X_train)


if __name__ == "__main__":
    self_check()
