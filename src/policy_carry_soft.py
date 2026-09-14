##########################################################################################
# Project    : unitree-g1-humanoid-carry                                                 #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-carry                   #
##########################################################################################
"""G1 carry task with a deformable ball-lattice payload.

Same MDP as ``config/policy_carry`` (walk-warm-started, twist-driven, hugging-arm
reset), but the payload is a **soft box**: an N x N x N solid lattice of free-joint
spheres tied by tendon springs (axial edges + all internal face diagonals). Same
"composite hand-built" pattern proven in ``deformable_hack.py``.

The lattice replaces the single solid box:
* Each of ``N^3`` balls is a separate ``mjlab`` entity ``ball_<i>``. Each ball has a
  freejoint + sphere geom (radius ``BALL_R``, mass ``BALL_MASS``).
* Tendons connecting neighbours (axial + face diagonals) are attached at scene-compile
  time via ``SceneCfg.spec_fn``: this lets us reference sites across entities.
* Ball-ball self-collision is disabled via ``contype/conaffinity`` (only ball-ground
  contacts). Otherwise N^3 free spheres bumping each other would explode the solver.

**Deformability knobs**  (same semantics as ``deformable_hack.py``):
    SOFTNESS   > 1 = squishier (divides every tendon stiffness).
    BOUNCINESS > 1 = more elastic rebound (divides every tendon damping).

**Box observations** are replaced by lattice-aware analogues that treat the deformable
payload as a point cloud: position = centroid, lin_vel = mean velocity, orientation
falls back to identity (a deformable blob has no rigid pose). The orientation-based
rewards (``box_upright``, ``box_front_aligned``) are dropped; the ``hold_box`` reward
places the hand targets in robot-yaw frame instead of box-yaw frame.

Notebook usage::

    from src import policy_carry_soft
    policy_carry_soft.configure(n=3, softness=1.0, bounciness=1.0)
    policy_carry_soft.register()

Output: register() -> TASK_ID ("Mjlab-G1-Carry-Soft").
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import last_action, joint_pos_rel, joint_vel_rel, time_out, dr
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import base_ang_vel, base_lin_vel, projected_gravity, UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

# Reuse carry MDP pieces except reward/obs functions that need lattice awareness.
from src.policy_carry import (  # noqa: F401
    BOX_HALF,
    BOX_DROP_Z,
    BOX_HOLD_X_OFFSET,
    BOX_HOLD_Z,
    BOX_HOLD_Z_ABOVE_ROOT,
    CHEST_Z,
    G1_FOOT_GEOMS,
    G1_JOINTS,
    G1_HOME_ROOT_POS,
    G1_HOME_ROOT_ROT,
    G1_HOME_JOINT_POS,
    HOLDING_ARM_JOINTS,
    HOLDING_ARM_POSE,
    _LEFT_SYM_JOINTS,
    _RIGHT_SYM_JOINTS,
    action_rate_l2,
    body_vector,
    reset_arms_to_holding,
    single_stance_reward,
    symmetric_pose_penalty,
    upright_torso_reward,
    wrist_actuator_force_obs,
    _entity_pos,
)
from src.policy_carry import (
    g1_carry_ppo_runner_cfg as _g1_carry_ppo_runner_cfg,
)
from src.policy_walk import wz_sign_penalty
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)


# Lattice geometry defaults; override via configure().
SOFT_N              = 3
SOFT_SPACING        = 2.0 * BOX_HALF / (SOFT_N - 1) if SOFT_N > 1 else 0.0
SOFT_BALL_R         = 0.012      # ball radius in meters
SOFT_BALL_MASS      = 0.02       # per-ball mass in kg before DR override
SOFT_TOTAL_MASS_MIN = 0.30       # kg total payload lower bound
SOFT_TOTAL_MASS_MAX = 3.00       # kg total payload upper bound

# Deformability knobs divide the base stiffness and damping.
SOFTNESS   = 1.0
BOUNCINESS = 1.0

# Baseline tendon spring parameters before knobs (matches deformable_hack.py).
_AXIAL_K_BASE = 400.0
_AXIAL_D_BASE = 10.0
_DIAG_K_BASE  = 100.0
_DIAG_D_BASE  = 4.0

TASK_ID = "Mjlab-G1-Carry-Soft"

# Contract for src.collect_buffer.
SNAPSHOT_ENTITIES = ["robot", "box"]
SNAPSHOT_YAW_ALIGN = True


def _effective_springs() -> tuple[float, float, float, float]:
    """Apply SOFTNESS and BOUNCINESS knobs to the base spring parameters.
    Args: none.
    Returns: tuple (axial_k, axial_d, diag_k, diag_d).
    """
    return (
        _AXIAL_K_BASE / SOFTNESS,
        _AXIAL_D_BASE / BOUNCINESS,
        _DIAG_K_BASE  / SOFTNESS,
        _DIAG_D_BASE  / BOUNCINESS,
    )


def configure(n: int = 3, softness: float = 1.0, bounciness: float = 1.0,
              ball_radius_m: float | None = None) -> None:
    """Set lattice size and deformability knobs before register().
    Args: n: lattice edge count (>=2); softness: divides tendon stiffness; bounciness: divides tendon damping; ball_radius_m: override ball radius (must be <= 0.49 * spacing).
    Returns: none.
    """
    global SOFT_N, SOFT_SPACING, SOFT_BALL_R, SOFTNESS, BOUNCINESS
    assert n >= 2, f"SOFT_N must be >= 2, got {n}"
    assert softness   > 0.0, f"softness must be > 0, got {softness}"
    assert bounciness > 0.0, f"bounciness must be > 0, got {bounciness}"

    SOFT_N = int(n)
    SOFT_SPACING = 2.0 * BOX_HALF / (SOFT_N - 1)
    SOFTNESS   = float(softness)
    BOUNCINESS = float(bounciness)
    if ball_radius_m is not None:
        assert ball_radius_m > 0.0
        # Keep balls smaller than half-spacing to avoid overlap.
        max_r = 0.49 * SOFT_SPACING
        if ball_radius_m > max_r:
            raise ValueError(
                f"ball_radius_m={ball_radius_m} too large for SOFT_N={n} "
                f"(spacing={SOFT_SPACING*1000:.1f} mm, max radius {max_r*1000:.1f} mm)."
            )
        SOFT_BALL_R = float(ball_radius_m)


def _lattice_indices(n: int) -> list[tuple[int, int, int]]:
    """Enumerate (ix, iy, iz) tuples for a full n^3 solid lattice.
    Args: n: lattice edge count.
    Returns: list of index tuples in ix->iy->iz order.
    """
    return [(ix, iy, iz) for ix in range(n) for iy in range(n) for iz in range(n)]


def _index_name(ix: int, iy: int, iz: int) -> str:
    """Deterministic per-ball entity name from lattice indices.
    Args: ix, iy, iz: lattice indices.
    Returns: entity name string ball_ijk.
    """
    return f"ball_{ix}{iy}{iz}"


def _lattice_offsets(n: int) -> list[tuple[float, float, float]]:
    """Body-frame offset for each lattice node, centred at the origin.
    Args: n: lattice edge count.
    Returns: list of (dx, dy, dz) offsets in meters.
    """
    half = (n - 1) * SOFT_SPACING * 0.5
    return [
        (ix * SOFT_SPACING - half, iy * SOFT_SPACING - half, iz * SOFT_SPACING - half)
        for (ix, iy, iz) in _lattice_indices(n)
    ]


def _make_ball_spec_fn(name: str, radius: float, mass: float):
    """Build a spec_fn producing a single sphere with self-collision disabled.
    Args: name: body/geom base name; radius: sphere radius; mass: sphere mass.
    Returns: zero-arg callable returning a fresh mujoco.MjSpec.
    """
    def _spec() -> mujoco.MjSpec:
        """Instantiate the ball MjSpec captured by the enclosing closure.
        Args: none.
        Returns: mujoco.MjSpec for a single sphere body with a marker site.
        """
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name=name)
        fj = body.add_freejoint()
        fj.name = f"{name}_free"
        body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(radius, 0.0, 0.0),
            mass=mass,
            rgba=(0.85, 0.30, 0.20, 1.0),
            condim=3,
            friction=(1.0, 0.05, 0.02),
            solref=(0.005, 1.0),
            solimp=(0.9, 0.99, 0.001, 0.5, 2.0),
            # contype=2/conaffinity=1: matches ground but not other balls, disabling self-collision.
            contype=2, conaffinity=1,
        )
        body.add_site(name=f"{name}_s", pos=(0.0, 0.0, 0.0), size=(0.001, 0.001, 0.001))
        return spec
    return _spec


def _soft_box_tendons_spec_fn(spec: mujoco.MjSpec) -> None:
    """Scene-level spec_fn that wires all lattice tendons after ball entities are merged.
    Args: spec: merged scene MjSpec with per-ball sites already added.
    Returns: none (mutates spec in place).
    """
    n = SOFT_N
    axial_k, axial_d, diag_k, diag_d = _effective_springs()
    diag_len = SOFT_SPACING * math.sqrt(2.0)

    # Map lattice indices to namespaced site names in the merged spec.
    site_by_name: dict[str, mujoco.MjsSite] = {s.name: s for s in spec.sites}
    coord_to_site: dict[tuple[int, int, int], str] = {}
    for (ix, iy, iz) in _lattice_indices(n):
        want_suffix = f"{_index_name(ix, iy, iz)}_s"
        match = next((nm for nm in site_by_name if nm.endswith(want_suffix)), None)
        if match is None:
            raise RuntimeError(
                f"soft_box tendon builder: site for ({ix},{iy},{iz}) not found. "
                f"Have {len(site_by_name)} sites; wanted suffix '{want_suffix}'."
            )
        coord_to_site[(ix, iy, iz)] = match

    node_set = set(coord_to_site.keys())

    def add_spring(a: str, b: str, rest_len: float, k: float, d: float, tag: str):
        """Add a two-site spring tendon between sites a and b.
        Args: a: first site name; b: second site name; rest_len: rest length in meters; k: stiffness; d: damping; tag: short spring category label.
        Returns: none.
        """
        t = spec.add_tendon()
        t.name = f"soft_{tag}_{a}__{b}"
        t.stiffness[0] = k
        t.damping[0] = d
        t.springlength[:] = [rest_len, rest_len]
        t.wrap_site(a)
        t.wrap_site(b)

    # Axial edges along the three principal axes.
    for (ix, iy, iz) in _lattice_indices(n):
        for (dx, dy, dz) in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nb = (ix + dx, iy + dy, iz + dz)
            if nb in node_set:
                add_spring(coord_to_site[(ix, iy, iz)], coord_to_site[nb],
                           SOFT_SPACING, axial_k, axial_d, tag="ax")

    # Face diagonals across every internal face plane.
    face_axes = ((0, 1, 2), (1, 0, 2), (2, 0, 1))
    for const_ax, a, b in face_axes:
        for c_val in range(n):
            for u in range(n - 1):
                for v in range(n - 1):
                    p1 = [0, 0, 0]; p1[const_ax] = c_val; p1[a] = u;   p1[b] = v
                    p2 = [0, 0, 0]; p2[const_ax] = c_val; p2[a] = u+1; p2[b] = v+1
                    p3 = [0, 0, 0]; p3[const_ax] = c_val; p3[a] = u;   p3[b] = v+1
                    p4 = [0, 0, 0]; p4[const_ax] = c_val; p4[a] = u+1; p4[b] = v
                    add_spring(coord_to_site[tuple(p1)], coord_to_site[tuple(p2)],
                               diag_len, diag_k, diag_d, tag="dg")
                    add_spring(coord_to_site[tuple(p3)], coord_to_site[tuple(p4)],
                               diag_len, diag_k, diag_d, tag="dg")


def _all_ball_positions_w(env: ManagerBasedRlEnv, ball_names: tuple[str, ...]) -> torch.Tensor:
    """Stack world-frame ball positions into a single tensor.
    Args: env: RL env; ball_names: ordered tuple of ball entity names.
    Returns: (num_envs, K, 3) position tensor with K = len(ball_names).
    """
    return torch.stack(
        [env.scene[name].data.root_link_pos_w for name in ball_names], dim=1
    )


def _lattice_centroid(env: ManagerBasedRlEnv, ball_names: tuple[str, ...]) -> torch.Tensor:
    """Mean world-frame position of every ball in the lattice.
    Args: env: RL env; ball_names: ordered tuple of ball entity names.
    Returns: (num_envs, 3) centroid tensor.
    """
    return _all_ball_positions_w(env, ball_names).mean(dim=1)


def _lattice_mean_lin_vel(env: ManagerBasedRlEnv, ball_names: tuple[str, ...]) -> torch.Tensor:
    """Mean world-frame linear velocity across all balls.
    Args: env: RL env; ball_names: ordered tuple of ball entity names.
    Returns: (num_envs, 3) mean velocity tensor.
    """
    stacked = torch.stack(
        [env.scene[name].data.root_link_lin_vel_w for name in ball_names], dim=0
    )
    return stacked.mean(dim=0)


def _ball_positions_body_frame(
    env: ManagerBasedRlEnv, ball_names: tuple[str, ...],
) -> torch.Tensor:
    """Ball positions in the robot yaw-frame relative to root, yaw-invariant.
    Args: env: RL env; ball_names: ordered tuple of ball entity names.
    Returns: (num_envs, K, 3) body-frame position tensor.
    """
    positions_w = _all_ball_positions_w(env, ball_names)  # (num_envs, K, 3)
    root_pos = env.scene["robot"].data.root_link_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
    rel = positions_w - root_pos
    yaw = env.scene["robot"].data.heading_w
    cy = torch.cos(-yaw).unsqueeze(-1)
    sy = torch.sin(-yaw).unsqueeze(-1)
    rx = cy * rel[..., 0] - sy * rel[..., 1]
    ry = sy * rel[..., 0] + cy * rel[..., 1]
    rz = rel[..., 2]
    return torch.stack([rx, ry, rz], dim=-1)


# Nominal body-frame position of each ball at reset time; cached lazily.
_NOMINAL_KEY = "carry_soft_nominal_offsets"


def _nominal_body_positions(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Nominal body-frame position of each ball relative to the robot root.
    Args: env: RL env.
    Returns: (K, 3) nominal position tensor matching reset_robot_and_soft_box.
    """
    cached = env.extras.get(_NOMINAL_KEY)
    if cached is not None and cached.shape[0] == SOFT_N ** 3:
        return cached
    offsets = _lattice_offsets(SOFT_N)
    nominal = torch.tensor(
        [
            (BOX_HOLD_X_OFFSET + dx, 0.0 + dy, BOX_HOLD_Z_ABOVE_ROOT + dz)
            for (dx, dy, dz) in offsets
        ],
        device=env.device, dtype=torch.float32,
    )
    env.extras[_NOMINAL_KEY] = nominal
    return nominal


def _lattice_extents(env: ManagerBasedRlEnv, ball_names: tuple[str, ...]) -> torch.Tensor:
    """Body-frame AABB extent (max - min per axis) of the lattice.
    Args: env: RL env; ball_names: ordered tuple of ball entity names.
    Returns: (num_envs, 3) extent tensor with nominal 2*BOX_HALF per axis.
    """
    body = _ball_positions_body_frame(env, ball_names)  # (num_envs, K, 3)
    return body.amax(dim=1) - body.amin(dim=1)


def soft_box_position_obs(env, ball_names: tuple[str, ...]):
    """Observation: robot-relative lattice centroid, clipped for stability.
    Args: env: RL env; ball_names: ordered tuple of ball entity names.
    Returns: (num_envs, 3) tensor clamped to [-3, 3].
    """
    centroid = _lattice_centroid(env, ball_names)
    robot_pos = env.scene["robot"].data.root_link_pos_w
    rel = centroid - robot_pos
    return torch.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0).clamp(-3.0, 3.0)


def soft_box_lin_vel_obs(env, ball_names: tuple[str, ...]):
    """Observation: mean lattice linear velocity, clipped for stability.
    Args: env: RL env; ball_names: ordered tuple of ball entity names.
    Returns: (num_envs, 3) tensor clamped to [-5, 5].
    """
    v = _lattice_mean_lin_vel(env, ball_names)
    return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5.0, 5.0)


def soft_box_size_obs(env):
    """Observation: nominal lattice bounding-box size, matching the solid box.
    Args: env: RL env.
    Returns: (num_envs, 3) constant size tensor.
    """
    root_pos = env.scene["robot"].data.root_link_pos_w
    size = torch.tensor([2 * BOX_HALF, 2 * BOX_HALF, 2 * BOX_HALF], device=root_pos.device)
    return size.expand(root_pos.shape[0], -1)


def lattice_extents_obs(env, ball_names: tuple[str, ...]):
    """Observation: body-frame lattice AABB extents, clipped for stability.
    Args: env: RL env; ball_names: ordered tuple of ball entity names.
    Returns: (num_envs, 3) tensor clamped to [0, 1].
    """
    ext = _lattice_extents(env, ball_names)
    return torch.nan_to_num(ext, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)


def hold_soft_box_reward(env, left_hand_cfg, right_hand_cfg, ball_names: tuple[str, ...],
                         y_offset: float = 0.13, z_offset: float = 0.02) -> torch.Tensor:
    """Reward symmetric hand grip on the lattice centroid in the robot yaw frame.
    Args: env: RL env; left_hand_cfg/right_hand_cfg: hand body selectors; ball_names: tuple of ball entity names; y_offset: lateral hand spacing in meters; z_offset: vertical hand offset.
    Returns: (num_envs,) reward in (0, 1].
    """
    left  = _entity_pos(env, left_hand_cfg)
    right = _entity_pos(env, right_hand_cfg)
    centroid = _lattice_centroid(env, ball_names)
    yaw = env.scene["robot"].data.heading_w
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    # Robot +y in world xy is (-sin_yaw, cos_yaw); hands sit at ±y_offset along that.
    left_off  = torch.stack([-y_offset * sy,  y_offset * cy], dim=-1)
    right_off = -left_off
    desired_left  = centroid.clone(); desired_left[:, :2]  += left_off;  desired_left[:, 2]  += z_offset
    desired_right = centroid.clone(); desired_right[:, :2] += right_off; desired_right[:, 2] += z_offset
    err = (torch.linalg.norm(left  - desired_left,  dim=-1)
         + torch.linalg.norm(right - desired_right, dim=-1))
    return torch.exp(-8.0 * err)


def soft_box_at_chest_reward(env, ball_names: tuple[str, ...],
                             target_z: float = CHEST_Z) -> torch.Tensor:
    """Gaussian reward on lattice centroid height around chest height.
    Args: env: RL env; ball_names: tuple of ball entity names; target_z: desired centroid z in meters.
    Returns: (num_envs,) reward in (0, 1].
    """
    centroid_z = _lattice_centroid(env, ball_names)[:, 2]
    diff = centroid_z - target_z
    return torch.exp(-(diff ** 2) / 0.05)


def lattice_shape_preservation_reward(
    env, ball_names: tuple[str, ...], sigma: float = 0.06,
) -> torch.Tensor:
    """Gaussian reward on the RMS body-frame displacement from nominal lattice slots.
    Args: env: RL env; ball_names: tuple of ball entity names; sigma: Gaussian bandwidth in meters.
    Returns: (num_envs,) reward in (0, 1].
    """
    body = _ball_positions_body_frame(env, ball_names)          # (num_envs, K, 3)
    nominal = _nominal_body_positions(env)                      # (K, 3)
    disp = body - nominal.unsqueeze(0)                          # (num_envs, K, 3)
    rms2 = disp.pow(2).sum(dim=-1).mean(dim=-1)                 # (num_envs,)
    return torch.exp(-rms2 / (2.0 * sigma * sigma))


def lattice_bbox_penalty(env, ball_names: tuple[str, ...]) -> torch.Tensor:
    """L1 deviation of the current lattice extents from the nominal cube side.
    Args: env: RL env; ball_names: tuple of ball entity names.
    Returns: (num_envs,) non-negative penalty summed over xyz.
    """
    ext = _lattice_extents(env, ball_names)                     # (num_envs, 3)
    nominal = 2.0 * BOX_HALF
    return (ext - nominal).abs().sum(dim=-1)


def non_hand_soft_box_contact_penalty(
    env, trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg,
    ball_names: tuple[str, ...], threshold: float = 0.22,
) -> torch.Tensor:
    """Penalize the lattice centroid coming too close to trunk, hips, or shanks.
    Args: env: RL env; trunk_cfg/left_shank_cfg/right_shank_cfg/left_hip_cfg/right_hip_cfg: body selectors; ball_names: tuple of ball entity names; threshold: proximity distance in meters.
    Returns: (num_envs,) non-negative penalty summed across body parts.
    """
    centroid = _lattice_centroid(env, ball_names)
    penalty = torch.zeros(centroid.shape[0], device=centroid.device)
    for cfg in (trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg):
        body_pos = _entity_pos(env, cfg)
        dist = torch.linalg.norm(body_pos - centroid, dim=-1)
        penalty += torch.clamp(threshold - dist, min=0.0) / threshold
    return penalty


def soft_box_carry_failed(env, ball_names: tuple[str, ...], trunk_cfg,
                          limit_angle: float = math.radians(65.0),
                          box_min_z: float = BOX_DROP_Z) -> torch.Tensor:
    """Termination flag when the torso tips over or the lattice centroid drops.
    Args: env: RL env; ball_names: tuple of ball entity names; trunk_cfg: torso body; limit_angle: max torso tilt in radians; box_min_z: drop height threshold in meters.
    Returns: (num_envs,) boolean termination mask.
    """
    trunk_ent: Entity = env.scene[trunk_cfg.name]
    proj_grav = trunk_ent.data.projected_gravity_b
    tilted = -proj_grav[:, 2] < math.cos(limit_angle)
    centroid_z = _lattice_centroid(env, ball_names)[:, 2]
    dropped = centroid_z < box_min_z
    return tilted | dropped


def reset_robot_and_soft_box(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ball_cfgs: tuple[SceneEntityCfg, ...],
    xy_jitter_robot: float = 0.05,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    box_x_offset: float = BOX_HOLD_X_OFFSET,
    box_z_above_root: float = BOX_HOLD_Z_ABOVE_ROOT,
    box_xy_jitter: float = 0.02,
    lattice_z_jitter: float = 0.002,
) -> None:
    """Reset robot and every lattice ball in one pass with a shared sampled yaw.
    Args: env: RL env; env_ids: envs to reset (None = all); ball_cfgs: tuple of ball entities; xy_jitter_robot: root xy noise; yaw_range: base yaw range; box_x_offset: forward hand spacing; box_z_above_root: lattice height above root; box_xy_jitter: box lateral noise; lattice_z_jitter: per-ball vertical noise.
    Returns: none.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return

    yaw = torch.empty(n, device=env.device).uniform_(*yaw_range)
    x_jit = torch.empty(n, device=env.device).uniform_(-xy_jitter_robot, xy_jitter_robot)
    y_jit = torch.empty(n, device=env.device).uniform_(-xy_jitter_robot, xy_jitter_robot)
    origins = env.scene.env_origins[env_ids]
    robot_x = origins[:, 0] + x_jit
    robot_y = origins[:, 1] + y_jit
    robot_z = torch.full((n,), float(G1_HOME_ROOT_POS[2]), device=env.device)

    half = yaw * 0.5
    robot_quat = torch.stack(
        [torch.cos(half), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(half)], dim=-1
    )
    robot_pos = torch.stack([robot_x, robot_y, robot_z], dim=-1)
    robot: Entity = env.scene["robot"]
    robot.write_root_link_pose_to_sim(
        torch.cat([robot_pos, robot_quat], dim=-1), env_ids=env_ids,
    )
    robot.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids,
    )

    # Lattice centre: robot xy plus box_x_offset along yaw at chest height.
    cx = torch.cos(yaw); sy = torch.sin(yaw)
    box_jx = torch.empty(n, device=env.device).uniform_(-box_xy_jitter, box_xy_jitter)
    box_jy = torch.empty(n, device=env.device).uniform_(-box_xy_jitter, box_xy_jitter)
    box_x = robot_x + box_x_offset * cx + box_jx
    box_y = robot_y + box_x_offset * sy + box_jy
    box_z = robot_z + box_z_above_root + torch.empty(n, device=env.device).uniform_(-0.005, 0.005)

    identity_quat = torch.zeros(n, 4, device=env.device); identity_quat[:, 0] = 1.0

    # Each ball world pose is box centre plus yaw-rotated body-frame offset.
    offsets = _lattice_offsets(SOFT_N)
    for (dx, dy, dz), cfg in zip(offsets, ball_cfgs):
        wx = box_x + dx * cx - dy * sy
        wy = box_y + dx * sy + dy * cx
        wz_j = torch.empty(n, device=env.device).uniform_(-lattice_z_jitter, lattice_z_jitter)
        wz = box_z + dz + wz_j
        ball_pos = torch.stack([wx, wy, wz], dim=-1)
        ball: Entity = env.scene[cfg.name]
        ball.write_root_link_pose_to_sim(
            torch.cat([ball_pos, identity_quat], dim=-1), env_ids=env_ids,
        )
        ball.write_root_link_velocity_to_sim(
            torch.zeros(n, 6, device=env.device), env_ids=env_ids,
        )


# Startup DR: split one total payload equally across all N^3 balls.
_TOTAL_MASS_KEY = "carry_soft_total_mass"


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def uniform_split_ball_mass_dr(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ball_cfgs: tuple[SceneEntityCfg, ...],
    total_mass_range: tuple[float, float] = (SOFT_TOTAL_MASS_MIN, SOFT_TOTAL_MASS_MAX),
) -> None:
    """Sample per-env total payload and split equally across all lattice balls.
    Args: env: RL env; env_ids: envs to randomize (None = all); ball_cfgs: tuple of ball entities; total_mass_range: uniform range for total payload in kg.
    Returns: none.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return
    N = len(ball_cfgs)
    lo, hi = total_mass_range
    total = torch.empty(n, device=env.device).uniform_(lo, hi)
    per_ball = total / float(N)
    bm = env.sim.model.body_mass
    body_ids = torch.tensor(
        [int(env.scene[cfg.name].indexing.body_ids[0].item()) for cfg in ball_cfgs],
        device=env.device, dtype=torch.long,
    )
    env_grid, body_grid = torch.meshgrid(env_ids.long(), body_ids, indexing="ij")
    bm[env_grid, body_grid] = per_ball.unsqueeze(-1).expand(-1, N)

    buf = env.extras.get(_TOTAL_MASS_KEY)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.zeros(env.num_envs, 1, device=env.device)
        env.extras[_TOTAL_MASS_KEY] = buf
    buf[env_ids.long(), 0] = total


def build_g1_carry_soft_mdp():
    """Assemble the manager-cfg dict for the deformable-lattice carry task.
    Args: none.
    Returns: dict of manager configuration blocks consumed by g1_carry_soft_env_cfg.
    """
    robot_cfg = get_g1_robot_cfg()
    robot_cfg.init_state = EntityCfg.InitialStateCfg(
        pos=G1_HOME_ROOT_POS,
        rot=G1_HOME_ROOT_ROT,
        joint_pos=G1_HOME_JOINT_POS,
    )

    # Spawn each lattice ball at its nominal pose so tendon rest lengths match at compile.
    ball_indices = _lattice_indices(SOFT_N)
    ball_offsets = _lattice_offsets(SOFT_N)
    ball_entity_cfgs: list[tuple[str, EntityCfg]] = []
    ball_scene_cfgs: list[SceneEntityCfg] = []
    for (ix, iy, iz), (dx, dy, dz) in zip(ball_indices, ball_offsets):
        name = _index_name(ix, iy, iz)
        ball_entity_cfgs.append((name, EntityCfg(
            spec_fn=_make_ball_spec_fn(name, SOFT_BALL_R, SOFT_BALL_MASS),
            init_state=EntityCfg.InitialStateCfg(
                pos=(BOX_HOLD_X_OFFSET + dx, 0.0 + dy, BOX_HOLD_Z + dz),
                joint_pos={},
            ),
        )))
        ball_scene_cfgs.append(SceneEntityCfg(name, body_names=(name,)))
    ball_scene_cfgs_t = tuple(ball_scene_cfgs)
    ball_names_t = tuple(cfg.name for cfg in ball_scene_cfgs_t)

    g1_asset      = SceneEntityCfg("robot", joint_names=G1_JOINTS, preserve_order=True)
    holding_arms  = SceneEntityCfg("robot", joint_names=HOLDING_ARM_JOINTS, preserve_order=True)
    left_sym      = SceneEntityCfg("robot", joint_names=_LEFT_SYM_JOINTS,  preserve_order=True)
    right_sym     = SceneEntityCfg("robot", joint_names=_RIGHT_SYM_JOINTS, preserve_order=True)
    wrist_acts    = SceneEntityCfg("robot", actuator_names=(r".*wrist.*",), preserve_order=True)
    trunk       = SceneEntityCfg("robot", body_names=("torso_link",))
    left_hand   = SceneEntityCfg("robot", body_names=("left_wrist_yaw_link",))
    right_hand  = SceneEntityCfg("robot", body_names=("right_wrist_yaw_link",))
    left_shank  = SceneEntityCfg("robot", body_names=("left_knee_link",))
    right_shank = SceneEntityCfg("robot", body_names=("right_knee_link",))
    left_hip    = SceneEntityCfg("robot", body_names=("left_hip_pitch_link",))
    right_hip   = SceneEntityCfg("robot", body_names=("right_hip_pitch_link",))

    # Sensors: identical to walk.
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot")
                    for s in ("left_foot", "right_foot")),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )

    # Observations: walk-style + soft-box centroid.
    actor_terms = {
        "base_lin_vel":      ObservationTermCfg(func=base_lin_vel,      noise=Unoise(n_min=-0.2, n_max=0.2)),
        "base_ang_vel":      ObservationTermCfg(func=base_ang_vel,      noise=Unoise(n_min=-0.1, n_max=0.1)),
        "projected_gravity": ObservationTermCfg(func=projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
        "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-0.5, n_max=0.5)),
        "actions":   ObservationTermCfg(func=last_action),
        "command":       ObservationTermCfg(func=vel_mdp.generated_commands, params={"command_name": "twist"}),
        "wrist_torque":  ObservationTermCfg(func=wrist_actuator_force_obs, params={"asset_cfg": wrist_acts}, noise=Unoise(n_min=-0.2, n_max=0.2)),
        "box_pos":       ObservationTermCfg(func=soft_box_position_obs, params={"ball_names": ball_names_t}, noise=Unoise(n_min=-0.005, n_max=0.005)),
        "box_lin_vel":   ObservationTermCfg(func=soft_box_lin_vel_obs,  params={"ball_names": ball_names_t}, noise=Unoise(n_min=-0.05, n_max=0.05)),
        "box_size":      ObservationTermCfg(func=soft_box_size_obs),
        "lattice_extents": ObservationTermCfg(
            func=lattice_extents_obs,
            params={"ball_names": ball_names_t},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        ),
    }
    observations = {
        "actor":  ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=G1_JOINTS,
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
            preserve_order=True,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(4.0, 8.0),
            rel_standing_envs=0.20,
            rel_heading_envs=0.20,
            rel_forward_envs=0.10,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 0.8),
                lin_vel_y=(-0.4, 0.4),
                ang_vel_z=(-0.5, 0.5),
                heading=(-math.pi, math.pi),
            ),
        )
    }

    # Same reward budget as normal carry minus orientation-only terms (lattice has no rigid pose).
    rewards = {
        "track_linear_velocity":  RewardTermCfg(func=vel_mdp.track_linear_velocity,  weight=2.5,
                                                params={"command_name": "twist", "std": math.sqrt(0.25)}),
        "track_angular_velocity": RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=10.0,
                                                params={"command_name": "twist", "std": math.sqrt(0.10)}),
        "wz_sign_penalty": RewardTermCfg(func=wz_sign_penalty, weight=2.0, params={"command_name": "twist"}),
        "hold_box":       RewardTermCfg(func=hold_soft_box_reward, weight=3.0,
                                        params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                                                "ball_names": ball_names_t}),
        "box_at_chest":   RewardTermCfg(func=soft_box_at_chest_reward, weight=1.5,
                                        params={"ball_names": ball_names_t}),
        # Anti-deformation via shape-preservation reward plus explicit bbox penalty.
        "lattice_shape":  RewardTermCfg(func=lattice_shape_preservation_reward, weight=1.5,
                                        params={"ball_names": ball_names_t, "sigma": 0.06}),
        "lattice_bbox":   RewardTermCfg(func=lattice_bbox_penalty, weight=-1.0,
                                        params={"ball_names": ball_names_t}),
        "upright":        RewardTermCfg(func=upright_torso_reward, weight=1.0, params={"trunk_cfg": trunk}),
        "symmetric_pose": RewardTermCfg(
            func=symmetric_pose_penalty, weight=-0.3,
            params={"left_cfg": left_sym, "right_cfg": right_sym, "feet_moving_threshold": 0.1},
        ),
        "single_stance":  RewardTermCfg(func=single_stance_reward, weight=0.8,
                                        params={"sensor_name": "feet_ground_contact"}),
        "action_rate_l2": RewardTermCfg(func=action_rate_l2, weight=-0.05),
        "dof_pos_limits": RewardTermCfg(func=vel_mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": g1_asset}),
        "body_ang_vel":   RewardTermCfg(func=vel_mdp.body_angular_velocity_penalty, weight=-0.05, params={"asset_cfg": trunk}),
        "self_collisions": RewardTermCfg(
            func=vel_mdp.self_collision_cost, weight=-1.0,
            params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
        ),
        "non_hand_box":   RewardTermCfg(
            func=non_hand_soft_box_contact_penalty, weight=-2.0,
            params={"trunk_cfg": trunk, "left_shank_cfg": left_shank, "right_shank_cfg": right_shank,
                    "left_hip_cfg": left_hip, "right_hip_cfg": right_hip, "ball_names": ball_names_t},
        ),
    }

    events = {
        "reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset", params={}),
        "reset_robot_and_soft_box": EventTermCfg(
            func=reset_robot_and_soft_box,
            mode="reset",
            params={"ball_cfgs": ball_scene_cfgs_t,
                    "xy_jitter_robot": 0.05,
                    "yaw_range": (-math.pi, math.pi),
                    "box_x_offset": BOX_HOLD_X_OFFSET,
                    "box_z_above_root": BOX_HOLD_Z_ABOVE_ROOT,
                    "box_xy_jitter": 0.02},
        ),
        "reset_arms_holding": EventTermCfg(
            func=reset_arms_to_holding, mode="reset", params={"asset_cfg": holding_arms},
        ),
        "push_robot": EventTermCfg(
            func=vel_mdp.push_by_setting_velocity,
            mode="interval", interval_range_s=(1.0, 3.0),
            params={"velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
            }},
        ),
        "foot_friction": EventTermCfg(
            mode="startup", func=dr.geom_friction,
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=G1_FOOT_GEOMS),
                    "operation": "abs", "ranges": (0.3, 1.2), "shared_random": True},
        ),
        "encoder_bias": EventTermCfg(
            mode="startup", func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
        ),
        "base_com": EventTermCfg(
            mode="startup", func=dr.body_com_offset,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                    "operation": "add",
                    "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)}},
        ),
        "ball_mass_split": EventTermCfg(
            mode="startup", func=uniform_split_ball_mass_dr,
            params={"ball_cfgs": ball_scene_cfgs_t,
                    "total_mass_range": (SOFT_TOTAL_MASS_MIN, SOFT_TOTAL_MASS_MAX)},
        ),
    }

    terminations = {
        "time_out":  TerminationTermCfg(func=time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=soft_box_carry_failed,
            params={"ball_names": ball_names_t, "trunk_cfg": trunk,
                    "limit_angle": math.radians(65.0), "box_min_z": BOX_DROP_Z},
        ),
    }

    curriculum = {
        "command_vel": CurriculumTermCfg(
            func=vel_mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {"step": 0,           "lin_vel_x": (-0.5, 0.5), "ang_vel_z": (-0.3, 0.3)},
                    {"step":  2_000_000,  "lin_vel_x": (-0.8, 0.8), "ang_vel_z": (-0.45, 0.45)},
                    {"step":  6_000_000,  "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.6, 0.6)},
                ],
            },
        ),
    }

    return {
        "robot_cfg":         robot_cfg,
        "ball_entity_cfgs":  ball_entity_cfgs,
        "sensors":           (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg),
        "observations":      observations,
        "actions":           actions,
        "commands":          commands,
        "rewards":           rewards,
        "events":            events,
        "terminations":      terminations,
        "curriculum":        curriculum,
    }


def g1_carry_soft_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the ManagerBasedRlEnvCfg for the deformable-lattice carry task.
    Args: play: when True disables observation noise and push events.
    Returns: fully populated ManagerBasedRlEnvCfg.
    """
    parts = build_g1_carry_soft_mdp()
    entities: dict[str, EntityCfg] = {"robot": parts["robot_cfg"]}
    for name, cfg in parts["ball_entity_cfgs"]:
        entities[name] = cfg

    # Cross-entity tendons are added by scene.spec_fn after ball entities are merged.
    scene_cfg = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities=entities,
        sensors=parts["sensors"],
        num_envs=1,
        extent=2.0,
        spec_fn=_soft_box_tendons_spec_fn,
    )

    # Contact caps sized for a full lattice touching the ground.
    ntendons_est = SOFT_N * SOFT_N * (SOFT_N - 1) * 3 + 3 * SOFT_N * (SOFT_N - 1) ** 2 * 2

    cfg = ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        observations=parts["observations"],
        actions=parts["actions"],
        commands=parts["commands"],
        events=parts["events"],
        rewards=parts["rewards"],
        terminations=parts["terminations"],
        curriculum=parts["curriculum"],
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=2.5,
            elevation=-10.0,
            azimuth=125.0,
        ),
        sim=SimulationCfg(
            nconmax=200 + 20 * SOFT_N ** 3,
            njmax=1200 + 60 * SOFT_N ** 3,
            contact_sensor_maxmatch=500,
            # Implicit-in-tendon-damping integrator is required or soft lattices explode.
            mujoco=MujocoCfg(timestep=0.002, iterations=20, ls_iterations=30),
        ),
        decimation=12,   # 12 * dt=0.002 gives ~41.6 Hz control rate
        episode_length_s=8.0,
    )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
    return cfg


def g1_carry_soft_ppo_runner_cfg():
    """Build the PPO runner cfg for the soft task by cloning the carry runner.
    Args: none.
    Returns: RslRlOnPolicyRunnerCfg with experiment_name overridden.
    """
    rl = _g1_carry_ppo_runner_cfg()
    rl.experiment_name = "g1_carry_soft"
    # Carry's mirror_g1_carry hard-codes CARRY_OBS_DIM=124; soft has 117 (no box_orientation/box_size).
    rl.algorithm.symmetry_cfg = None
    return rl


def register():
    """Register the soft carry task with mjlab's task registry (idempotent).
    Args: none.
    Returns: task id string.
    """
    from mjlab.tasks import registry
    from mjlab.tasks.registry import register_mjlab_task

    registry._REGISTRY.pop(TASK_ID, None)
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=g1_carry_soft_env_cfg(),
        play_env_cfg=g1_carry_soft_env_cfg(play=True),
        rl_cfg=g1_carry_soft_ppo_runner_cfg(),
    )
    return TASK_ID
