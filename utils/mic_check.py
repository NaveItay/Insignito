from __future__ import annotations
import numpy as np


def detect_bad_mics(
    x: np.ndarray,
    clip_threshold: float = 0.999,
    print_report: bool = True,
    print_all: bool = False,
) -> np.ndarray:
    """
    Simple robust mic selection + report.

    Rules:
      - dead:    rms < 0.05 * median_rms
      - noisy:   rms > 20.0 * median_rms
      - clipped: clip_ratio > 1e-3   (more than 0.1% samples saturated)

    Returns:
      good_mask: boolean shape (M,)
    """
    eps = 1e-12

    rms = np.sqrt(np.mean(x ** 2, axis=0) + eps)          # (M,)
    med = float(np.median(rms))

    clip_ratio = np.mean(np.abs(x) >= clip_threshold, axis=0)  # (M,)

    dead = rms < (0.05 * med)
    noisy = rms > (20.0 * med)
    clipped = clip_ratio > 1e-3
    not_finite = ~np.isfinite(rms)

    bad = dead | noisy | clipped | not_finite
    good = ~bad

    if print_report:
        M = x.shape[1]
        print("[MIC CHECK] thresholds:")
        print(f"  median_rms = {med:.6e}")
        print(f"  dead   if rms < {(0.05 * med):.6e}  (0.05 * median)")
        print(f"  noisy  if rms > {(20.0 * med):.6e}  (20 * median)")
        print(f"  clipped if clip_ratio > 1.0e-3  (0.1%) with |x| >= {clip_threshold}")
        print("")

        idxs = list(range(M)) if print_all else np.flatnonzero(bad).tolist()

        if len(idxs) == 0:
            print("[MIC CHECK] No bad microphones detected.")
        else:
            print("[MIC CHECK] Report:")
            for i in idxs:
                reasons = []
                if dead[i]:
                    reasons.append("dead (low RMS)")
                if noisy[i]:
                    reasons.append("noisy (high RMS)")
                if clipped[i]:
                    reasons.append("clipped")
                if not_finite[i]:
                    reasons.append("non-finite RMS")

                ratio = float(rms[i] / (med + eps))
                status = "BAD" if bad[i] else "OK "
                reason_str = ", ".join(reasons) if reasons else "—"

                print(
                    f"  ch {i:02d} | {status} | rms={rms[i]:.6e} (x{ratio:.2f} of med) "
                    f"| clip_ratio={clip_ratio[i]:.6e} | {reason_str}"
                )

        print("")
        bad_list = np.flatnonzero(bad).tolist()
        print(f"[MIC CHECK] bad mics: {bad_list}")
        print(f"[MIC CHECK] good mics: {int(np.sum(good))} / {M}")

    return good
