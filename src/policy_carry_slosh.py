##########################################################################################
# Project    : unitree-g1-humanoid-carry                                                 #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-carry                   #
##########################################################################################
"""G1 carry task with a slosh payload: hollow semi-transparent box containing N free-floating balls.

Same MDP as ``config/policy_carry`` (walk-warm-started, twist-driven, hugging-arm reset),
but the payload is embodied physically as N loose spheres inside a hollow box:

* Box is a hollow cube built from 6 thin walls (outer half-extent ``BOX_HALF``,
  wall thickness ``BOX_WALL_T``). Walls are rendered with alpha ~0.30 so the balls
  are visible from outside. Box shell mass is fixed at ``BOX_WALL_MASS_TOTAL``.
* Balls are ``NUM_BALLS`` free-floating solid spheres of radius ``BALL_RADIUS`` each
  (default ``BOX_HALF / 4``), spawned in a k×k grid on the interior bottom so they
  do not overlap. Each ball is a *separate* mjlab entity ("ball_0", …, "ball_{N-1}")
  so per-ball mass and reset are trivial.
* **Domain randomisation is on the total payload mass**, not per-ball: at each startup
  we sample ``M ~ U(BALL_TOTAL_MASS_MIN, BALL_TOTAL_MASS_MAX)`` per env and assign
  ``M / NUM_BALLS`` to every ball body. So N=1 with M=1.5 kg gives one 1.5 kg ball;
  N=3 with M=1.5 kg gives three 0.5 kg balls (same total payload, different dynamics).

Notebook usage:
    from src import policy_carry_slosh
    policy_carry_slosh.configure(num_balls=3, ball_radius_m=0.02)  # before register()
    policy_carry_slosh.register()

Input:  play flag (bool) for g1_carry_slosh_env_cfg.
Output: register() -> TASK_ID ("Mjlab-G1-Carry-Slosh").
"""

from __future__ import annotations

import math

import mujoco
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.rl import RslRlOnPolicyRunnerCfg

# Reuse MDP-related pieces from the normal carry task.
from src.policy_carry import (  # noqa: F401  (re-exported for notebook use)
    BOX_DROP_Z,
    BOX_HOLD_Z,
    BOX_HOLD_Z_ABOVE_ROOT,
    CHEST_Z,
    G1_JOINTS,
    G1_HOME_ROOT_POS,
    G1_HOME_ROOT_ROT,
    G1_HOME_JOINT_POS,
    build_g1_carry_mdp,
    g1_carry_ppo_runner_cfg,
)

# Local box override: 15 cm side (half-extent 0.075 m) instead of the 21 cm rigid-carry box.
# Kept local so the rigid Mjlab-G1-Carry task keeps its own BOX_HALF=0.105.
BOX_HALF          = 0.075
BOX_HOLD_X_OFFSET = 0.29  # forward hand spacing; keeps the near face at the same root distance.

# Slosh geometry defaults; override via configure().
BOX_WALL_T          = 0.008        # wall half-thickness in meters
BOX_WALL_MASS_TOTAL = 0.20         # fixed box shell mass in kg
NUM_BALLS           = 1            # default ball count
BALL_RADIUS         = BOX_HALF / 4 # default ball radius
BALL_TOTAL_MASS_MIN = 0.30         # kg
BALL_TOTAL_MASS_MAX = 3.00         # kg

TASK_ID = "Mjlab-G1-Carry-Slosh"

# Contract for src.collect_buffer.
SNAPSHOT_ENTITIES = ["robot", "box"]
SNAPSHOT_YAW_ALIGN = True


# Tighter arm pose for the 15 cm box (shoulders closed and elbows more bent than the rigid-carry pose).
HOLDING_ARM_POSE_SMALL = {
    "left_shoulder_pitch_joint":  -0.65,
    "left_shoulder_roll_joint":    0.15,
    "left_shoulder_yaw_joint":     0.0,
    "left_elbow_joint":            0.75,
    "left_wrist_roll_joint":       0.0,
    "left_wrist_pitch_joint":      0.0,
    "left_wrist_yaw_joint":        0.0,
    "right_shoulder_pitch_joint": -0.65,
    "right_shoulder_roll_joint":  -0.15,
    "right_shoulder_yaw_joint":    0.0,
    "right_elbow_joint":           0.75,
    "right_wrist_roll_joint":      0.0,
    "right_wrist_pitch_joint":     0.0,
    "right_wrist_yaw_joint":       0.0,
}


def reset_arms_to_holding_small(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Override the 14 arm joints to HOLDING_ARM_POSE_SMALL after walk-buffer reset.
    Args: env: RL env; env_ids: envs to reset (None = all); asset_cfg: robot entity selecting arm joints.
    Returns: none.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return
    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids_t = torch.tensor(joint_ids, device=env.device)
    else:
        joint_ids_t = joint_ids
    pose = torch.tensor(
        [HOLDING_ARM_POSE_SMALL[j] for j in asset_cfg.joint_names],
        device=env.device,
    ).unsqueeze(0).expand(n, -1).clone()
    vel = torch.zeros_like(pose)
    asset.write_joint_state_to_sim(pose, vel, joint_ids=joint_ids_t, env_ids=env_ids)


def box_size_obs_small(env):
    """Observation: constant (2*BOX_HALF)^3 broadcast for the smaller slosh box.
    Args: env: RL env.
    Returns: (num_envs, 3) tensor of full box side lengths.
    """
    root_pos = env.scene["robot"].data.root_link_pos_w
    size = torch.tensor([2 * BOX_HALF, 2 * BOX_HALF, 2 * BOX_HALF], device=root_pos.device)
    return size.expand(root_pos.shape[0], -1)


def configure(num_balls: int = 1, ball_radius_m: float | None = None) -> None:
    """Set ball count and radius before register(); raises if the balls don't fit.
    Args: num_balls: number of balls to spawn; ball_radius_m: ball radius in meters (None keeps BOX_HALF/4).
    Returns: none.
    """
    global NUM_BALLS, BALL_RADIUS
    assert num_balls >= 1, f"num_balls must be >= 1, got {num_balls}"
    r = float(ball_radius_m) if ball_radius_m is not None else BOX_HALF / 4
    assert r > 0.0, f"ball_radius_m must be > 0, got {r}"
    check_fits(num_balls, r, verbose=False)
    NUM_BALLS = num_balls
    BALL_RADIUS = r


def radius_from_fill_pct(num_balls: int, interior_fill_pct: float) -> float:
    """Compute the ball radius achieving a target interior-volume fill percent.
    Args: num_balls: number of balls; interior_fill_pct: desired fill in (0, 100].
    Returns: ball radius in meters (does not verify geometric fit).
    """
    assert num_balls >= 1
    assert 0.0 < interior_fill_pct <= 100.0, "interior_fill_pct must be in (0, 100]"
    interior_side = 2.0 * (BOX_HALF - 2.0 * BOX_WALL_T)
    interior_vol  = interior_side ** 3
    per_ball_vol  = interior_vol * (interior_fill_pct / 100.0) / num_balls
    return (3.0 * per_ball_vol / (4.0 * math.pi)) ** (1.0 / 3.0)


def radius_from_vertical_fill_pct(
    num_balls: int,
    vertical_fill_pct: float,
    *,
    ball_gap_m: float = 0.001,
    wall_margin_m: float = 0.008,
) -> float:
    """Compute the ball radius so the cubic-grid pile reaches vertical_fill_pct of interior height.
    Args: num_balls: number of balls; vertical_fill_pct: pile top height as pct of interior height in (0, 100]; ball_gap_m: centre-to-centre clearance; wall_margin_m: clearance to interior walls.
    Returns: ball radius in meters that yields the closest achievable pile height.
    """
    assert num_balls >= 1
    assert 0.0 < vertical_fill_pct <= 100.0, "vertical_fill_pct must be in (0, 100]"
    interior_side = 2.0 * (BOX_HALF - 2.0 * BOX_WALL_T)
    usable = interior_side - 2.0 * wall_margin_m
    target = interior_side * vertical_fill_pct / 100.0
    best_r, best_diff = None, float("inf")
    r_lo, r_hi = 1e-4, usable / 2.0
    r_step = 1e-5
    r = r_lo
    while r <= r_hi:
        step = 2.0 * r + ball_gap_m
        if step > usable:
            r += r_step; continue
        n_per_axis = int(usable / step) + 1
        n_layers = math.ceil(num_balls / (n_per_axis * n_per_axis))
        pile = n_layers * step - ball_gap_m
        if pile > interior_side - wall_margin_m:
            r += r_step; continue
        diff = abs(pile - target)
        if diff < best_diff:
            best_diff, best_r = diff, r
        r += r_step
    if best_r is None:
        raise ValueError(
            f"cannot fit {num_balls} balls to reach {vertical_fill_pct}% vertical fill "
            f"in interior of {interior_side*1000:.1f} mm."
        )
    return best_r


def check_fits(
    num_balls: int,
    ball_radius_m: float,
    *,
    ball_gap_m: float = 0.001,
    wall_margin_m: float = 0.008,
    verbose: bool = True,
) -> dict:
    """Verify N balls fit inside the hollow box on an auto-sized 3D grid.
    Args: num_balls: number of balls; ball_radius_m: sphere radius; ball_gap_m: centre-to-centre clearance; wall_margin_m: reset-jitter margin from walls; verbose: print summary line.
    Returns: dict with grid layout, pile height, headroom, and fill fraction; raises ValueError on overflow.
    """
    assert num_balls >= 1
    assert ball_radius_m > 0.0
    inner_half   = BOX_HALF - 2.0 * BOX_WALL_T          # interior half-side (m)
    usable_half  = inner_half - wall_margin_m           # grid stays this far from walls
    step         = 2.0 * ball_radius_m + ball_gap_m     # centre-to-centre spacing
    if 2.0 * usable_half < 2.0 * ball_radius_m:
        raise ValueError(
            f"ball radius {ball_radius_m*1000:.2f} mm too large: a single ball "
            f"doesn't fit inside the interior ({inner_half*2000:.1f} mm side, "
            f"{wall_margin_m*1000:.0f} mm jitter margin)."
        )
    n_per_axis = int(math.floor((2.0 * usable_half) / step)) + 1
    per_layer  = n_per_axis * n_per_axis
    nz_needed  = math.ceil(num_balls / per_layer)
    pile_height    = nz_needed * step - ball_gap_m      # top of top layer's ball surfaces
    interior_side  = 2.0 * inner_half
    headroom       = interior_side - pile_height - wall_margin_m  # includes floor margin
    fill_volume    = num_balls * (4.0 / 3.0) * math.pi * ball_radius_m ** 3
    interior_vol   = interior_side ** 3
    fill_fraction  = fill_volume / interior_vol
    info = {
        "num_balls": num_balls,
        "ball_radius_mm": ball_radius_m * 1000,
        "ball_gap_mm": ball_gap_m * 1000,
        "wall_margin_mm": wall_margin_m * 1000,
        "interior_side_mm": interior_side * 1000,
        "grid_nx": n_per_axis,
        "grid_ny": n_per_axis,
        "grid_nz": nz_needed,
        "grid_capacity": n_per_axis * n_per_axis * nz_needed,
        "pile_height_mm": pile_height * 1000,
        "headroom_mm": headroom * 1000,
        "interior_fill_pct": fill_fraction * 100,
    }
    if headroom < 0.0:
        msg = (
            f"cannot fit {num_balls} balls of r={ball_radius_m*1000:.2f} mm: "
            f"pile height {pile_height*1000:.1f} mm exceeds interior "
            f"{interior_side*1000:.1f} mm minus the {wall_margin_m*1000:.0f} mm "
            f"jitter margin. Reduce num_balls or ball_radius_m."
        )
        if verbose:
            print("[check_fits] ✗ " + msg)
        raise ValueError(msg)
    if verbose:
        print(
            f"[check_fits] ✓ {num_balls} balls of r={ball_radius_m*1000:.2f} mm fit "
            f"in a {n_per_axis}×{n_per_axis}×{nz_needed} grid "
            f"({info['grid_capacity']} slots, {info['grid_capacity']-num_balls} unused).\n"
            f"             interior      = {interior_side*1000:.1f} mm cube\n"
            f"             pile height   = {pile_height*1000:.1f} mm\n"
            f"             free headroom = {headroom*1000:.1f} mm above the pile\n"
            f"             fill fraction = {fill_fraction*100:.1f}% of interior volume"
        )
    return info


def _get_hollow_box_spec() -> mujoco.MjSpec:
    """Build the hollow semi-transparent cube spec (6 wall geoms).
    Args: none.
    Returns: mujoco.MjSpec with a free-jointed box body of walls thickness 2*BOX_WALL_T.
    """
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box")
    fj = body.add_freejoint()
    fj.name = "box_free"

    inner = BOX_HALF - BOX_WALL_T
    rgba  = (0.80, 0.80, 0.82, 0.30)  # light gray semi-transparent, matches the soft aesthetic
    mass_each = BOX_WALL_MASS_TOTAL / 6.0
    # contype=1, conaffinity=3: walls collide with anything in group 1 (other walls) or 2 (balls).
    common = dict(
        rgba=rgba,
        condim=3,
        friction=(4.0, 0.25, 0.08),
        solref=(0.004, 1.0),
        solimp=(0.95, 0.995, 0.001, 0.5, 2.0),
        mass=mass_each,
        contype=1,
        conaffinity=3,
    )

    body.add_geom(name="box_bottom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=(BOX_HALF, BOX_HALF, BOX_WALL_T),
                  pos=(0.0, 0.0, -inner), **common)
    body.add_geom(name="box_top",    type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=(BOX_HALF, BOX_HALF, BOX_WALL_T),
                  pos=(0.0, 0.0, +inner), **common)
    side_yz = BOX_HALF - 2 * BOX_WALL_T
    body.add_geom(name="box_front", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=(BOX_WALL_T, side_yz, side_yz),
                  pos=(+inner, 0.0, 0.0), **common)
    body.add_geom(name="box_back",  type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=(BOX_WALL_T, side_yz, side_yz),
                  pos=(-inner, 0.0, 0.0), **common)
    body.add_geom(name="box_left",  type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=(side_yz, BOX_WALL_T, side_yz),
                  pos=(0.0, +inner, 0.0), **common)
    body.add_geom(name="box_right", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=(side_yz, BOX_WALL_T, side_yz),
                  pos=(0.0, -inner, 0.0), **common)
    return spec


def _make_ball_spec_fn(radius: float, name: str):
    """Build a spec_fn producing a single free-jointed sphere entity.
    Args: radius: sphere radius in meters; name: body name for the ball.
    Returns: zero-arg callable returning a fresh mujoco.MjSpec.
    """
    def _spec() -> mujoco.MjSpec:
        """Instantiate the ball MjSpec captured by the enclosing closure.
        Args: none.
        Returns: mujoco.MjSpec for a single sphere body.
        """
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name=name)
        fj = body.add_freejoint()
        fj.name = f"{name}_free"
        # contype=2, conaffinity=3: balls collide with walls (group 1) and with other balls (group 2).
        body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(radius, 0.0, 0.0),
            mass=BALL_TOTAL_MASS_MIN,  # per-env DR overrides this
            rgba=(0.85, 0.30, 0.20, 1.0),  # matches the soft-payload ball color
            condim=3,
            friction=(1.5, 0.05, 0.02),
            solref=(0.004, 1.0),
            solimp=(0.95, 0.995, 0.001, 0.5, 2.0),
            contype=2,
            conaffinity=3,
        )
        return spec
    return _spec


def _ball_grid_offsets(
    num_balls: int,
    radius: float,
    *,
    ball_gap_m: float = 0.001,
    wall_margin_m: float = 0.008,
) -> list[tuple[float, float, float]]:
    """Compute box-local grid offsets that pack N balls floor-up.
    Args: num_balls: ball count; radius: sphere radius; ball_gap_m: inter-ball clearance; wall_margin_m: clearance to interior walls.
    Returns: list of (dx, dy, dz) box-local offsets, length num_balls.
    """
    info = check_fits(num_balls, radius,
                      ball_gap_m=ball_gap_m, wall_margin_m=wall_margin_m,
                      verbose=False)
    nx, ny, nz = info["grid_nx"], info["grid_ny"], info["grid_nz"]
    step = 2.0 * radius + ball_gap_m
    # Centre the horizontal grid on the box centre.
    span_x = (nx - 1) * step
    span_y = (ny - 1) * step
    x_start = -span_x / 2.0
    y_start = -span_y / 2.0
    # First layer sits ball_radius + margin above the interior floor.
    z_floor = -(BOX_HALF - 2.0 * BOX_WALL_T) + radius + 0.001
    offsets: list[tuple[float, float, float]] = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                if len(offsets) == num_balls:
                    return offsets
                offsets.append((
                    x_start + ix * step,
                    y_start + iy * step,
                    z_floor + iz * step,
                ))
    return offsets


def reset_robot_box_and_balls(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    box_cfg: SceneEntityCfg,
    ball_cfgs: tuple[SceneEntityCfg, ...],
    xy_jitter_robot: float = 0.05,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    box_x_offset: float = BOX_HOLD_X_OFFSET,
    box_z_above_root: float = BOX_HOLD_Z_ABOVE_ROOT,
    box_xy_jitter: float = 0.02,
    ball_xy_jitter: float = 0.005,
) -> None:
    """Reset robot, box, and all balls in one write with a shared sampled yaw.
    Args: env: RL env; env_ids: envs to reset (None = all); box_cfg: box entity; ball_cfgs: tuple of ball entities; xy_jitter_robot: root xy noise; yaw_range: base yaw range; box_x_offset: forward hand spacing; box_z_above_root: box height above root; box_xy_jitter: box lateral noise; ball_xy_jitter: per-ball lateral noise.
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
    robot_quat = torch.stack([
        torch.cos(half),
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        torch.sin(half),
    ], dim=-1)
    robot_pos = torch.stack([robot_x, robot_y, robot_z], dim=-1)
    robot: Entity = env.scene["robot"]
    robot.write_root_link_pose_to_sim(
        torch.cat([robot_pos, robot_quat], dim=-1), env_ids=env_ids,
    )
    robot.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids,
    )

    # Box
    cx = torch.cos(yaw); sy = torch.sin(yaw)
    box_jx = torch.empty(n, device=env.device).uniform_(-box_xy_jitter, box_xy_jitter)
    box_jy = torch.empty(n, device=env.device).uniform_(-box_xy_jitter, box_xy_jitter)
    box_x = robot_x + box_x_offset * cx + box_jx
    box_y = robot_y + box_x_offset * sy + box_jy
    box_z = robot_z + box_z_above_root + torch.empty(n, device=env.device).uniform_(-0.005, 0.005)

    # Match the box yaw to the robot yaw so the front face aligns with the arm grip.
    box_pos = torch.stack([box_x, box_y, box_z], dim=-1)
    box: Entity = env.scene[box_cfg.name]
    box.write_root_link_pose_to_sim(
        torch.cat([box_pos, robot_quat], dim=-1), env_ids=env_ids,
    )
    box.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids,
    )

    # Clear per-env EMA buffer so previous episodes don't pollute the new one.
    buf = env.extras.get(_BALL_COM_EMA_KEY)
    if buf is not None and buf.shape[0] == env.num_envs:
        buf[env_ids.long()] = 0.0

    # Ball interior offsets are box-local; rotate them by the box yaw before writing world xyz.
    offsets = _ball_grid_offsets(len(ball_cfgs), BALL_RADIUS)
    for (dx, dy, dz), cfg in zip(offsets, ball_cfgs):
        jx = torch.empty(n, device=env.device).uniform_(-ball_xy_jitter, ball_xy_jitter)
        jy = torch.empty(n, device=env.device).uniform_(-ball_xy_jitter, ball_xy_jitter)
        wx = box_x + (dx * cx - dy * sy) + jx
        wy = box_y + (dx * sy + dy * cx) + jy
        wz = box_z + dz
        ball_pos = torch.stack([wx, wy, wz], dim=-1)
        ball: Entity = env.scene[cfg.name]
        ball.write_root_link_pose_to_sim(
            torch.cat([ball_pos, robot_quat], dim=-1), env_ids=env_ids,
        )
        ball.write_root_link_velocity_to_sim(
            torch.zeros(n, 6, device=env.device), env_ids=env_ids,
        )


# Startup DR: sample one total payload per env and split equally across N balls.
_TOTAL_MASS_KEY = "carry_slosh_total_mass"


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def uniform_split_ball_mass_dr(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ball_cfgs: tuple[SceneEntityCfg, ...],
    total_mass_range: tuple[float, float] = (BALL_TOTAL_MASS_MIN, BALL_TOTAL_MASS_MAX),
) -> None:
    """Sample per-env total payload and split it equally across the N balls.
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
    per_ball = total / float(N)   # (n,)

    bm = env.sim.model.body_mass  # (num_envs, nbody): real per-env storage
    body_ids = torch.tensor(
        [int(env.scene[cfg.name].indexing.body_ids[0].item()) for cfg in ball_cfgs],
        device=env.device, dtype=torch.long,
    )
    env_grid, body_grid = torch.meshgrid(env_ids.long(), body_ids, indexing="ij")
    bm[env_grid, body_grid] = per_ball.unsqueeze(-1).expand(-1, N)

    # Stash sampled total for privileged observations.
    buf = env.extras.get(_TOTAL_MASS_KEY)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.zeros(env.num_envs, 1, device=env.device)
        env.extras[_TOTAL_MASS_KEY] = buf
    buf[env_ids.long(), 0] = total


# Observations use the box's local frame so signals are pose-invariant.
_BALL_COM_EMA_KEY = "carry_slosh_ball_com_ema"


def _ball_com_local(env, box_cfg, ball_cfgs):
    """Mean ball position in the box local frame relative to the box centre.
    Args: env: RL env; box_cfg: box entity; ball_cfgs: tuple of ball entities.
    Returns: (num_envs, 3) box-local COM tensor.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse

    box     = env.scene[box_cfg.name]
    box_pos  = box.data.root_link_pos_w                       # (E, 3)
    box_quat = box.data.root_link_quat_w                      # (E, 4) wxyz

    ball_pos_w = torch.stack(
        [env.scene[cfg.name].data.root_link_pos_w for cfg in ball_cfgs],
        dim=1,
    )
    rel_w = ball_pos_w - box_pos.unsqueeze(1)                 # (E, N, 3)

    E, N, _ = rel_w.shape
    q_bcast = box_quat.unsqueeze(1).expand(-1, N, -1).reshape(-1, 4)
    rel_local = quat_apply_inverse(q_bcast, rel_w.reshape(-1, 3)).reshape(E, N, 3)
    return rel_local.mean(dim=1)                              # (E, 3)


def _ball_rel_vel_local(env, box_cfg, ball_cfgs):
    """Mean ball linear velocity relative to the box in the box local frame.
    Args: env: RL env; box_cfg: box entity; ball_cfgs: tuple of ball entities.
    Returns: (num_envs, 3) box-local relative velocity tensor.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse

    box = env.scene[box_cfg.name]
    box_vel_w  = box.data.root_link_lin_vel_w                 # (E, 3)
    box_quat   = box.data.root_link_quat_w                    # (E, 4)

    ball_vels_w = torch.stack(
        [env.scene[cfg.name].data.root_link_lin_vel_w for cfg in ball_cfgs],
        dim=1,
    )                                                         # (E, N, 3)
    rel_w = ball_vels_w - box_vel_w.unsqueeze(1)              # (E, N, 3)

    E, N, _ = rel_w.shape
    q_bcast = box_quat.unsqueeze(1).expand(-1, N, -1).reshape(-1, 4)
    rel_local = quat_apply_inverse(q_bcast, rel_w.reshape(-1, 3)).reshape(E, N, 3)
    return rel_local.mean(dim=1)                              # (E, 3)


def ball_com_offset_obs(env, box_cfg, ball_cfgs):
    """Observation: box-local mean ball position, clipped for stability.
    Args: env: RL env; box_cfg: box entity; ball_cfgs: tuple of ball entities.
    Returns: (num_envs, 3) tensor clamped to [-0.2, 0.2].
    """
    return torch.nan_to_num(_ball_com_local(env, box_cfg, ball_cfgs),
                            nan=0.0, posinf=0.0, neginf=0.0).clamp(-0.2, 0.2)


def ball_rel_lin_vel_obs(env, box_cfg, ball_cfgs):
    """Observation: box-local mean ball velocity relative to the box.
    Args: env: RL env; box_cfg: box entity; ball_cfgs: tuple of ball entities.
    Returns: (num_envs, 3) tensor clamped to [-5, 5].
    """
    return torch.nan_to_num(_ball_rel_vel_local(env, box_cfg, ball_cfgs),
                            nan=0.0, posinf=0.0, neginf=0.0).clamp(-5.0, 5.0)


def ball_slosh_penalty(env, box_cfg, ball_cfgs) -> torch.Tensor:
    """Quadratic penalty on mean squared ball speed relative to the box.
    Args: env: RL env; box_cfg: box entity; ball_cfgs: tuple of ball entities.
    Returns: (num_envs,) non-negative penalty (pair with a negative reward weight).
    """
    box_vel_w = env.scene[box_cfg.name].data.root_link_lin_vel_w             # (E, 3)
    ball_vels = torch.stack(
        [env.scene[cfg.name].data.root_link_lin_vel_w for cfg in ball_cfgs],
        dim=1,
    )                                                                        # (E, N, 3)
    rel = ball_vels - box_vel_w.unsqueeze(1)
    speeds_sq = (rel ** 2).sum(dim=-1)                                       # (E, N)
    return torch.nan_to_num(speeds_sq.mean(dim=1), nan=0.0, posinf=0.0, neginf=0.0)


def ball_com_stability_reward(
    env, box_cfg, ball_cfgs,
    sigma: float = 0.03,
    ema_alpha: float = 0.05,
) -> torch.Tensor:
    """Gaussian reward on the EMA-averaged box-local ball COM.
    Args: env: RL env; box_cfg: box entity; ball_cfgs: tuple of ball entities; sigma: Gaussian bandwidth in meters; ema_alpha: EMA per-step weight.
    Returns: (num_envs,) reward in (0, 1].
    """
    com_now = _ball_com_local(env, box_cfg, ball_cfgs)                       # (E, 3)

    buf = env.extras.get(_BALL_COM_EMA_KEY)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = com_now.clone()
    else:
        buf = (1.0 - ema_alpha) * buf + ema_alpha * com_now
    env.extras[_BALL_COM_EMA_KEY] = buf

    dist_sq = (buf ** 2).sum(dim=-1)                                          # (E,)
    return torch.nan_to_num(torch.exp(-dist_sq / (sigma ** 2)),
                            nan=0.0, posinf=0.0, neginf=0.0)


def build_g1_carry_slosh_mdp():
    """Adapt the normal-carry MDP dict for the hollow box plus N ball entities.
    Args: none.
    Returns: manager-cfg dict extended with ball entities, reset, DR, obs, and rewards.
    """
    parts = build_g1_carry_mdp()

    parts["box_cfg"] = EntityCfg(
        spec_fn=_get_hollow_box_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(BOX_HOLD_X_OFFSET, 0.0, BOX_HOLD_Z),
            joint_pos={},
        ),
    )

    # One entity per ball, positioned on the interior floor grid.
    offsets = _ball_grid_offsets(NUM_BALLS, BALL_RADIUS)
    ball_entity_cfgs = []
    ball_names = []
    for i, (dx, dy, dz) in enumerate(offsets):
        name = f"ball_{i}"
        ball_names.append(name)
        ball_entity_cfgs.append((name, EntityCfg(
            spec_fn=_make_ball_spec_fn(BALL_RADIUS, name),
            init_state=EntityCfg.InitialStateCfg(
                pos=(BOX_HOLD_X_OFFSET + dx, 0.0 + dy, BOX_HOLD_Z + dz),
                joint_pos={},
            ),
        )))
    parts["ball_entity_cfgs"] = ball_entity_cfgs

    box = SceneEntityCfg("box")
    ball_scene_cfgs = tuple(SceneEntityCfg(name, body_names=(name,)) for name in ball_names)

    events = dict(parts["events"])
    events.pop("reset_robot_and_box_held", None)
    events["reset_robot_box_balls"] = EventTermCfg(
        func=reset_robot_box_and_balls,
        mode="reset",
        params={"box_cfg": box, "ball_cfgs": ball_scene_cfgs},
    )
    events.pop("box_mass", None)
    events["ball_mass_split"] = EventTermCfg(
        mode="startup",
        func=uniform_split_ball_mass_dr,
        params={
            "ball_cfgs": ball_scene_cfgs,
            "total_mass_range": (BALL_TOTAL_MASS_MIN, BALL_TOTAL_MASS_MAX),
        },
    )
    # Swap the shared reset_arms_holding for the tighter small-box arm pose,
    # preserving the joint-selecting asset_cfg built by build_g1_carry_mdp.
    old_arms = events["reset_arms_holding"]
    events["reset_arms_holding"] = EventTermCfg(
        func=reset_arms_to_holding_small,
        mode="reset",
        params=old_arms.params,
    )
    parts["events"] = events

    # Feed ball-cloud observations to both actor and critic groups.
    from mjlab.managers import ObservationTermCfg
    from mjlab.utils.noise import UniformNoiseCfg as Unoise
    ball_obs = {
        "ball_com_offset": ObservationTermCfg(
            func=ball_com_offset_obs,
            params={"box_cfg": box, "ball_cfgs": ball_scene_cfgs},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        ),
        "ball_rel_lin_vel": ObservationTermCfg(
            func=ball_rel_lin_vel_obs,
            params={"box_cfg": box, "ball_cfgs": ball_scene_cfgs},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
    }
    for group_name, group in parts["observations"].items():
        group.terms.update(ball_obs)
        # Report the shadowed 10 cm side, not the rigid-carry 21 cm.
        if "box_size" in group.terms:
            group.terms["box_size"] = ObservationTermCfg(func=box_size_obs_small)

    # Quadratic slosh penalty combined with EMA-averaged COM-stability reward.
    from mjlab.managers import RewardTermCfg
    parts["rewards"]["ball_slosh"] = RewardTermCfg(
        func=ball_slosh_penalty, weight=-0.5,
        params={"box_cfg": box, "ball_cfgs": ball_scene_cfgs},
    )
    parts["rewards"]["ball_com_stability"] = RewardTermCfg(
        func=ball_com_stability_reward, weight=0.5,
        params={"box_cfg": box, "ball_cfgs": ball_scene_cfgs,
                "sigma": 0.03, "ema_alpha": 0.05},
    )

    # Gait shaping ported from policy_walk: variable_posture + swing_height_error, plus
    # bumped single/double stance weights. Leg+waist joints only so the arm-holding pose is not fought.
    from mjlab.tasks.velocity import mdp as vel_mdp
    from src.policy_walk import swing_height_error as walk_swing_height_error
    leg_asset = SceneEntityCfg(
        "robot",
        joint_names=(
            r".*hip_pitch.*", r".*hip_roll.*", r".*hip_yaw.*",
            r".*knee.*",
            r".*ankle_pitch.*", r".*ankle_roll.*",
            r".*waist.*",
        ),
        preserve_order=True,
    )
    parts["rewards"]["pose"] = RewardTermCfg(
        func=vel_mdp.variable_posture,
        weight=0.6,
        params={
            "asset_cfg": leg_asset,
            "command_name": "twist",
            "std_standing": {".*": 0.05},
            "std_walking": {
                r".*hip_pitch.*":   0.45,
                r".*hip_roll.*":    0.15,
                r".*hip_yaw.*":     0.15,
                r".*knee.*":        0.50,
                r".*ankle_pitch.*": 0.25,
                r".*ankle_roll.*":  0.10,
                r".*waist_yaw.*":   0.20,
                r".*waist_roll.*":  0.08,
                r".*waist_pitch.*": 0.10,
            },
            "std_running": {
                r".*hip_pitch.*":   0.50,
                r".*hip_roll.*":    0.20,
                r".*hip_yaw.*":     0.20,
                r".*knee.*":        0.60,
                r".*ankle_pitch.*": 0.35,
                r".*ankle_roll.*":  0.15,
                r".*waist_yaw.*":   0.30,
                r".*waist_roll.*":  0.08,
                r".*waist_pitch.*": 0.20,
            },
            "walking_threshold": 0.05,
            "running_threshold": 1.2,
        },
    )
    parts["rewards"]["swing_height_error"] = RewardTermCfg(
        func=walk_swing_height_error,
        weight=-0.4,
        params={
            "sensor_name": "feet_ground_contact",
            "height_sensor_name": "foot_height_scan",
            "target_height": 0.08,
            "sigma": 0.08,
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    )
    if "single_stance" in parts["rewards"]:
        parts["rewards"]["single_stance"].weight = 1.2
    if "double_stance" in parts["rewards"]:
        parts["rewards"]["double_stance"].weight = 0.8
    if "upright" in parts["rewards"]:
        parts["rewards"]["upright"].weight = 1.5

    return parts


def g1_carry_slosh_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the ManagerBasedRlEnvCfg for the slosh carry task.
    Args: play: when True disables observation noise and push events.
    Returns: fully populated ManagerBasedRlEnvCfg.
    """
    parts = build_g1_carry_slosh_mdp()
    entities = {
        "robot": parts["robot_cfg"],
        "box":   parts["box_cfg"],
    }
    for name, cfg in parts["ball_entity_cfgs"]:
        entities[name] = cfg

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=entities,
            sensors=parts["sensors"],
            num_envs=1,
            extent=2.0,
        ),
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
            # Ball-ball collision ON (balls have conaffinity=3). Budget accounts for the C(N,2)
            # potential ball-ball pairs plus 12 wall/floor contacts per ball; njmax uses ~4
            # Jacobian rows per contact for condim=3 (cone friction, isotropic).
            nconmax=400 + 12 * NUM_BALLS + NUM_BALLS * (NUM_BALLS - 1) // 2,
            njmax=2000 + 4 * (400 + 12 * NUM_BALLS + NUM_BALLS * (NUM_BALLS - 1) // 2),
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=10),
        ),
        decimation=5,
        episode_length_s=8.0,
    )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
    return cfg


def g1_carry_slosh_ppo_runner_cfg():
    """Build the PPO runner cfg for the slosh task by cloning the carry runner.
    Args: none.
    Returns: RslRlOnPolicyRunnerCfg with experiment_name overridden.
    """
    rl = g1_carry_ppo_runner_cfg()
    rl.experiment_name = "g1_carry_slosh"
    # Carry's mirror_g1_carry hard-codes CARRY_OBS_DIM=124; slosh adds 6 extra dims.
    rl.algorithm.symmetry_cfg = None
    return rl


def register():
    """Register the slosh carry task with mjlab's task registry (idempotent).
    Args: none.
    Returns: task id string.
    """
    from mjlab.tasks import registry
    from mjlab.tasks.registry import register_mjlab_task

    registry._REGISTRY.pop(TASK_ID, None)
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=g1_carry_slosh_env_cfg(),
        play_env_cfg=g1_carry_slosh_env_cfg(play=True),
        rl_cfg=g1_carry_slosh_ppo_runner_cfg(),
    )
    return TASK_ID
