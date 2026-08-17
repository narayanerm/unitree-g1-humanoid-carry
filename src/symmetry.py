###############################################################################
# Project    : Humanoid Loco-Manipulation with the Unitree G1                 #
# Developer  : Narayane Ribeiro Medeiros                                      #
# Repository : https://github.com/narayanerm/Humanoid-Loco-Manipulation       #
###############################################################################
"""Left-right symmetry (sagittal-plane mirror) for the G1 carry task.

Extends the walk mirror with the box-carry observations: hand-to-box vectors,
trunk-to-box vector, per-wrist actuator torques, box position, box orientation
(quat), and box size. Under the sagittal mirror (y flips) with a uniform rigid
box, the physics remains consistent — mirror image of a symmetric box is the
same box.

Numerical operations on the concatenated obs vector:
- 3-D vector obs (world- or body-frame): flip y-component. Works uniformly
  because the sagittal mirror plane is the y=0 plane in both frames.
- box_orientation quaternion (w, x, y, z): (w, -x, y, -z).
- wrist_torque (6-D, [L_roll, L_pitch, L_yaw, R_roll, R_pitch, R_yaw]): swap
  L<->R and flip roll/yaw signs.
- box_size (3-D positive scalars): unchanged.
"""

from __future__ import annotations

import torch

# ==================== joint mapping (mirrors walk's) ====================

G1_29DOF_JOINT_ORDER = (
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",       "waist_roll_joint",     "waist_pitch_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
    "left_elbow_joint",           "left_wrist_roll_joint",     "left_wrist_pitch_joint",  "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",          "right_wrist_roll_joint",    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

_perm = list(range(29))
_perm[0:6] = list(range(6, 12))
_perm[6:12] = list(range(0, 6))
_perm[15:22] = list(range(22, 29))
_perm[22:29] = list(range(15, 22))
G1_MIRROR_PERM = tuple(_perm)

G1_MIRROR_SIGN_FLIP_IDX = (1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28)


def _assert_joint_order():
    """Verify runtime G1_JOINTS matches the hard-coded G1_29DOF_JOINT_ORDER."""
    from src.policy_carry import G1_JOINTS
    if tuple(G1_JOINTS) != G1_29DOF_JOINT_ORDER:
        mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(G1_JOINTS, G1_29DOF_JOINT_ORDER)) if a != b]
        raise AssertionError(
            "G1_JOINTS order in policy_carry differs from src.symmetry G1_29DOF_JOINT_ORDER. "
            f"Mismatches (idx, actual, expected): {mismatches[:5]} ..."
        )


_assert_joint_order()


def _switch_g1_joints_left_right(x: torch.Tensor) -> torch.Tensor:
    """Permute L<->R on the last dim of a (..., 29) tensor and flip roll/yaw signs.
    Args: x — tensor whose last dim is exactly 29 (joint_pos, joint_vel or actions).
    Returns: mirrored tensor, same shape.
    """
    perm = torch.as_tensor(G1_MIRROR_PERM, device=x.device, dtype=torch.long)
    out = torch.index_select(x, dim=-1, index=perm)
    signs = torch.ones(29, device=x.device, dtype=x.dtype)
    signs[list(G1_MIRROR_SIGN_FLIP_IDX)] = -1.0
    return out * signs


# ==================== wrist_torque mirror (6-D) ====================
# Actual entity order for actuator_names=(r".*wrist.*",) preserve_order=True is
# [L_roll, R_roll, L_pitch, L_yaw, R_pitch, R_yaw] — not the "L then R" grouping
# a naive reader might expect. Verified against ManagerBasedRlEnv.scene["robot"]
# .actuator_names at 2026-08-17. Swap L<->R AND flip roll/yaw sign (torques
# around x, z axes invert under sagittal mirror; pitch around y stays).
# Mapping:
#   new_L_roll  = -old_R_roll  → slot 0 <- slot 1, sign -1
#   new_R_roll  = -old_L_roll  → slot 1 <- slot 0, sign -1
#   new_L_pitch = +old_R_pitch → slot 2 <- slot 4, sign +1
#   new_L_yaw   = -old_R_yaw   → slot 3 <- slot 5, sign -1
#   new_R_pitch = +old_L_pitch → slot 4 <- slot 2, sign +1
#   new_R_yaw   = -old_L_yaw   → slot 5 <- slot 3, sign -1
_WRIST_PERM  = (1, 0, 4, 5, 2, 3)
_WRIST_SIGNS = (-1.0, -1.0, 1.0, -1.0, 1.0, -1.0)


def _mirror_wrist_torque(x: torch.Tensor) -> torch.Tensor:
    """Mirror 6-D wrist torque obs. Args: x — (..., 6). Returns: same shape."""
    perm = torch.as_tensor(_WRIST_PERM, device=x.device, dtype=torch.long)
    signs = torch.as_tensor(_WRIST_SIGNS, device=x.device, dtype=x.dtype)
    return torch.index_select(x, dim=-1, index=perm) * signs


# ==================== carry obs layout (124 dims) ====================

CARRY_OBS_DIM = 124

CARRY_OBS_LAYOUT = {
    "base_lin_vel":      (0, 3),
    "base_ang_vel":      (3, 6),
    "projected_gravity": (6, 9),
    "joint_pos":         (9, 38),
    "joint_vel":         (38, 67),
    "actions":           (67, 96),
    "command":           (96, 99),
    "left_hand_to_box":  (99, 102),
    "right_hand_to_box": (102, 105),
    "trunk_to_box":      (105, 108),
    "wrist_torque":      (108, 114),
    "box_pos":           (114, 117),
    "box_orientation":   (117, 121),
    "box_size":          (121, 124),
}
assert CARRY_OBS_LAYOUT["box_size"][1] == CARRY_OBS_DIM

# Sign vectors for terms that are element-wise (not permutation-based).
CARRY_OBS_SIGNS = {
    "base_lin_vel":      (1.0, -1.0,  1.0),
    "base_ang_vel":      (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0,  1.0),
    "command":           (1.0, -1.0, -1.0),
    "trunk_to_box":      (1.0, -1.0,  1.0),
    "box_pos":           (1.0, -1.0,  1.0),
    "box_orientation":   (1.0, -1.0,  1.0, -1.0),  # (w, -x, y, -z)
    "box_size":          (1.0,  1.0,  1.0),
}


def mirror_carry_obs(obs: torch.Tensor) -> torch.Tensor:
    """Mirror a concatenated carry obs tensor.
    Args: obs — (..., k*CARRY_OBS_DIM) tensor for k >= 1 stacked timesteps.
    Returns: mirrored tensor, same shape.
    """
    assert obs.shape[-1] % CARRY_OBS_DIM == 0, (
        f"Last dim {obs.shape[-1]} not multiple of {CARRY_OBS_DIM}"
    )
    n_stack = obs.shape[-1] // CARRY_OBS_DIM
    lead = obs.shape[:-1]
    x = obs.clone().view(*lead, n_stack, CARRY_OBS_DIM)
    for term, (a, b) in CARRY_OBS_LAYOUT.items():
        if term in {"joint_pos", "joint_vel", "actions"}:
            x[..., a:b] = _switch_g1_joints_left_right(x[..., a:b])
        elif term == "wrist_torque":
            x[..., a:b] = _mirror_wrist_torque(x[..., a:b])
        elif term == "left_hand_to_box":
            # L<->R swap of the two hand-to-box vectors, then flip y on each.
            lh_a, lh_b = CARRY_OBS_LAYOUT["left_hand_to_box"]
            rh_a, rh_b = CARRY_OBS_LAYOUT["right_hand_to_box"]
            lh = x[..., lh_a:lh_b].clone()
            rh = x[..., rh_a:rh_b].clone()
            signs = torch.tensor(CARRY_OBS_SIGNS.get("trunk_to_box"), device=obs.device, dtype=obs.dtype)
            x[..., lh_a:lh_b] = rh * signs
            x[..., rh_a:rh_b] = lh * signs
        elif term == "right_hand_to_box":
            pass  # handled with left_hand_to_box above
        else:
            signs = torch.tensor(CARRY_OBS_SIGNS[term], device=obs.device, dtype=obs.dtype)
            x[..., a:b] = x[..., a:b] * signs
    return x.view(*lead, obs.shape[-1])


def mirror_carry_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror carry actions (29-D joint targets).
    Args: actions — (..., 29) tensor.
    Returns: mirrored actions, same shape.
    """
    return _switch_g1_joints_left_right(actions)


# ==================== rsl_rl entrypoint ====================

@torch.no_grad()
def mirror_g1_carry(env=None, obs=None, actions=None):
    """rsl_rl symmetry_cfg entrypoint. Returns doubled batch [original; mirrored].
    Args: env — mjlab env (unused, kept for the rsl_rl API contract). obs — TensorDict with per-group keys, or plain tensor. actions — (N, 29) tensor, or None.
    Returns: (obs_out, actions_out); each is None if the corresponding input is None, else doubled along the batch dim.
    """
    obs_out = None
    actions_out = None

    if obs is not None:
        if hasattr(obs, "keys") and hasattr(obs, "batch_size"):
            batch_size = obs.batch_size[0]
            obs_out = obs.repeat(2)
            for key in obs.keys():
                obs_out[key][:batch_size] = obs[key]
                obs_out[key][batch_size:] = mirror_carry_obs(obs[key])
        else:
            obs_mirror = mirror_carry_obs(obs)
            obs_out = torch.cat([obs, obs_mirror], dim=0)

    if actions is not None:
        actions_mirror = mirror_carry_actions(actions)
        actions_out = torch.cat([actions, actions_mirror], dim=0)

    return obs_out, actions_out
