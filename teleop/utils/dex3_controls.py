"""Pure controller-trigger mapping helpers for the Unitree Dex3 hand."""

import numpy as np


def compose_dex3_targets(base_pose: np.ndarray, trigger: float, closed_pose: np.ndarray) -> np.ndarray:
    """Overlay a raw trigger toward the closed pose onto a retargeted pose.

    Televuer's raw trigger value is expected to be ``0.0`` when released and
    ``1.0`` when fully pressed. Invalid values fail open by preserving the
    retargeted pose; finite values are clamped before interpolation.
    """
    base_pose = np.asarray(base_pose, dtype=float)
    closed_pose = np.asarray(closed_pose, dtype=float)
    if base_pose.shape != (7,) or closed_pose.shape != (7,):
        raise ValueError("Dex3 poses must each contain seven joint targets")
    if not np.isfinite(trigger):
        amount = 0.0
    else:
        amount = float(np.clip(trigger, 0.0, 1.0))
    return base_pose + amount * (closed_pose - base_pose)


def trigger_to_dex3_targets(trigger: float, open_pose: np.ndarray, closed_pose: np.ndarray) -> np.ndarray:
    """Interpolate a seven-joint Dex3 pose from open to closed."""
    return compose_dex3_targets(open_pose, trigger, closed_pose)
