from __future__ import annotations
import numpy as np

from .steering import DOA, doa_to_unit_vector, compute_delays_seconds, steering_vectors


def choose_delay_sign_by_ds_energy(
    X: np.ndarray,
    freqs: np.ndarray,
    mic_pos: np.ndarray,
    doa: DOA,
    c: float
) -> float:
    """
    Test both sign conventions using a delay-and-sum score and pick the one with higher output power.
    """
    u = doa_to_unit_vector(doa)

    def ds_score(sign: float) -> float:
        tau = compute_delays_seconds(mic_pos, u, c, sign)
        a = steering_vectors(freqs, tau)  # (F,M)
        M = a.shape[1]
        power = 0.0
        for fi in range(X.shape[0]):
            w = a[fi, :] / M
            y = X[fi, :, :] @ np.conj(w)
            power += float(np.mean(np.abs(y) ** 2))
        return power

    p_plus = ds_score(+1.0)
    p_minus = ds_score(-1.0)
    return +1.0 if p_plus >= p_minus else -1.0


def lcmv_weights_per_freq(R: np.ndarray, C: np.ndarray, f_vec: np.ndarray) -> np.ndarray:
    """
    w = R^{-1} C (C^H R^{-1} C)^{-1} f_vec
    """
    M = R.shape[0]
    trace = np.trace(R).real
    dl = 1e-3 * (trace / M + 1e-12)
    R_reg = R + dl * np.eye(M, dtype=R.dtype)

    try:
        RinvC = np.linalg.solve(R_reg, C)
    except np.linalg.LinAlgError:
        RinvC = np.linalg.pinv(R_reg) @ C

    G = C.conj().T @ RinvC

    try:
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        Ginv = np.linalg.pinv(G)

    w = RinvC @ (Ginv @ f_vec)
    return w


def beamform_lcmv(
    X: np.ndarray,         # (F,T,M)
    freqs: np.ndarray,     # (F,)
    mic_pos: np.ndarray,   # (M,3)
    doa_target: DOA,
    doa_interf: DOA,
    c: float,
    verbose: bool = False
) -> np.ndarray:
    """
    LCMV per-frequency:
      - pass target with gain 1
      - null interferer with gain 0
    """
    sign = choose_delay_sign_by_ds_energy(X, freqs, mic_pos, doa_target, c)
    if verbose:
        print(f"        [INFO] Using delay sign = {sign:+.0f} for target DOA")

    u_t = doa_to_unit_vector(doa_target)
    u_i = doa_to_unit_vector(doa_interf)

    tau_t = compute_delays_seconds(mic_pos, u_t, c, sign)
    tau_i = compute_delays_seconds(mic_pos, u_i, c, sign)

    a_t = steering_vectors(freqs, tau_t)  # (F,M)
    a_i = steering_vectors(freqs, tau_i)  # (F,M)

    F, T, M = X.shape
    Y = np.zeros((F, T), dtype=np.complex128)

    f_vec = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)

    for fi in range(F):
        Xf = X[fi, :, :]  # (T,M)
        R = (Xf.conj().T @ Xf) / max(T, 1)
        C = np.stack([a_t[fi, :], a_i[fi, :]], axis=1)  # (M,2)
        w = lcmv_weights_per_freq(R, C, f_vec)          # (M,)
        Y[fi, :] = Xf @ np.conj(w)

    return Y
