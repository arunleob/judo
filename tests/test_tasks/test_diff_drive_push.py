# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Unit tests for the diff_drive_push task rewards and dynamics.

Each reward term is tested in isolation by zeroing the other weights and constructing
`states`/`sensors` arrays with known values at the task's real indices, then comparing the
reward against a hand-computed value.
"""

import mujoco
import numpy as np

from judo.tasks.diff_drive_push import DiffDrivePush


def _empty_arrays(task: DiffDrivePush, batch: int = 1, horizon: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build zeroed (states, sensors, controls) arrays of the correct shapes for the task."""
    nqv = task.model.nq + task.model.nv
    states = np.zeros((batch, horizon, nqv))
    sensors = np.zeros((batch, horizon, task.model.nsensordata))
    controls = np.zeros((batch, horizon, task.model.nu))
    return states, sensors, controls


def _zero_weights(task: DiffDrivePush) -> None:
    """Zero out all reward weights so a single term can be isolated per test."""
    task.config.w_pusher_proximity = 0.0
    task.config.w_pusher_velocity = 0.0
    task.config.w_cart_position = 0.0
    task.config.w_pusher_heading = 0.0


def test_model_builds_with_two_controls() -> None:
    """The diff-drive pusher exposes exactly two controls: forward speed and heading."""
    task = DiffDrivePush()
    assert task.nu == 2
    # State is (x, y, theta) for the pusher plus (x, y) for the cart.
    assert task.model.nq == 5
    assert task.model.nv == 5


def test_pusher_proximity_reward() -> None:
    """Pusher-proximity penalizes distance to the pusher goal (offset behind the cart)."""
    task = DiffDrivePush()
    _zero_weights(task)
    task.config.w_pusher_proximity = 1.0
    task.config.goal_pos = np.array([0.0, 0.0])
    offset = task.config.pusher_goal_offset

    states, sensors, controls = _empty_arrays(task)
    # Cart at (1, 0); cart->goal direction is (-1, 0), so pusher_goal = cart - offset*dir = (1+offset, 0).
    states[..., task.cart_pos_idx : task.cart_pos_idx + 2] = [1.0, 0.0]
    pusher_goal = np.array([1.0 + offset, 0.0])
    # Place the pusher 0.25 m short of the pusher goal along x.
    pusher_pos = pusher_goal - np.array([0.25, 0.0])
    sensors[..., task.pusher_pos_idx : task.pusher_pos_idx + 2] = pusher_pos

    reward = task.reward(states, sensors, controls)
    expected = -0.5 * (0.25**2)
    assert np.allclose(reward, expected)


def test_cart_position_reward() -> None:
    """Cart-position penalizes the squared distance from the cart to the goal."""
    task = DiffDrivePush()
    _zero_weights(task)
    task.config.w_cart_position = 1.0
    task.config.goal_pos = np.array([0.0, 0.0])

    states, sensors, controls = _empty_arrays(task)
    states[..., task.cart_pos_idx : task.cart_pos_idx + 2] = [0.5, 0.0]

    reward = task.reward(states, sensors, controls)
    expected = -0.5 * (0.5**2)
    assert np.allclose(reward, expected)


def test_velocity_reward() -> None:
    """Velocity penalizes squared forward (linear) and angular pusher velocity."""
    task = DiffDrivePush()
    _zero_weights(task)
    task.config.w_pusher_velocity = 1.0
    task.config.goal_pos = np.array([0.0, 0.0])

    states, sensors, controls = _empty_arrays(task)
    # Keep the cart away from the goal to avoid a degenerate direction (0/0).
    states[..., task.cart_pos_idx : task.cart_pos_idx + 2] = [1.0, 0.0]
    # pusher (vx, vy, omega).
    states[..., task.pusher_vel_idx : task.pusher_vel_idx + 3] = [0.3, 0.4, 0.5]

    reward = task.reward(states, sensors, controls)
    expected = -0.5 * (0.3**2 + 0.4**2 + 0.5**2)
    assert np.allclose(reward, expected)


def test_heading_reward_aligned_is_zero() -> None:
    """When the pusher points exactly at the pusher goal, the heading penalty is zero."""
    task = DiffDrivePush()
    _zero_weights(task)
    task.config.w_pusher_heading = 1.0
    task.config.goal_pos = np.array([0.0, 0.0])

    states, sensors, controls = _empty_arrays(task)
    states[..., task.cart_pos_idx : task.cart_pos_idx + 2] = [1.0, 0.0]
    sensors[..., task.pusher_pos_idx : task.pusher_pos_idx + 2] = [0.0, 0.0]
    # Pusher goal is at (1 + offset, 0), directly along +x from the pusher; heading +x is aligned.
    sensors[..., task.pusher_heading_idx : task.pusher_heading_idx + 2] = [1.0, 0.0]

    reward = task.reward(states, sensors, controls)
    assert np.allclose(reward, 0.0)


def test_heading_reward_anti_aligned_is_max() -> None:
    """When the pusher points away from the pusher goal, the heading penalty is maximal (2)."""
    task = DiffDrivePush()
    _zero_weights(task)
    task.config.w_pusher_heading = 1.0
    task.config.goal_pos = np.array([0.0, 0.0])

    states, sensors, controls = _empty_arrays(task)
    states[..., task.cart_pos_idx : task.cart_pos_idx + 2] = [1.0, 0.0]
    sensors[..., task.pusher_pos_idx : task.pusher_pos_idx + 2] = [0.0, 0.0]
    sensors[..., task.pusher_heading_idx : task.pusher_heading_idx + 2] = [-1.0, 0.0]

    reward = task.reward(states, sensors, controls)
    assert np.allclose(reward, -2.0)


def test_heading_reward_gated_off_when_close() -> None:
    """The heading term is gated off when the pusher is within the gate distance of its goal."""
    task = DiffDrivePush()
    _zero_weights(task)
    task.config.w_pusher_heading = 1.0
    task.config.goal_pos = np.array([0.0, 0.0])
    offset = task.config.pusher_goal_offset

    states, sensors, controls = _empty_arrays(task)
    states[..., task.cart_pos_idx : task.cart_pos_idx + 2] = [1.0, 0.0]
    pusher_goal = np.array([1.0 + offset, 0.0])
    # Place the pusher well within the gate distance of the pusher goal.
    close = pusher_goal - np.array([task.config.pointing_gate_distance * 0.5, 0.0])
    sensors[..., task.pusher_pos_idx : task.pusher_pos_idx + 2] = close
    # Even with a maximally wrong heading, the gate zeroes the penalty.
    sensors[..., task.pusher_heading_idx : task.pusher_heading_idx + 2] = [-1.0, 0.0]

    reward = task.reward(states, sensors, controls)
    assert np.allclose(reward, 0.0)


def test_heading_reward_sums_over_horizon() -> None:
    """Reward terms are summed over the horizon (checked via the heading term)."""
    task = DiffDrivePush()
    _zero_weights(task)
    task.config.w_pusher_heading = 1.0
    task.config.goal_pos = np.array([0.0, 0.0])

    horizon = 4
    states, sensors, controls = _empty_arrays(task, horizon=horizon)
    states[..., task.cart_pos_idx : task.cart_pos_idx + 2] = [1.0, 0.0]
    sensors[..., task.pusher_pos_idx : task.pusher_pos_idx + 2] = [0.0, 0.0]
    sensors[..., task.pusher_heading_idx : task.pusher_heading_idx + 2] = [-1.0, 0.0]

    reward = task.reward(states, sensors, controls)
    assert np.allclose(reward, -2.0 * horizon)


def test_forward_drive_pushes_cart_without_sideways_drift() -> None:
    """A forward velocity command drives the pusher along its heading and pushes the cart."""
    task = DiffDrivePush()
    model, data = task.model, task.data
    px = task.get_joint_position_start_index("pusher_x")
    pth = task.get_joint_position_start_index("pusher_theta")

    data.qpos[:] = 0.0
    data.qpos[px] = -0.6  # pusher just behind the cart
    data.qpos[pth] = 0.0  # facing +x
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    cart_y0 = data.qpos[task.cart_pos_idx + 1]
    for _ in range(200):
        data.ctrl[:] = [1.0, 0.0]  # forward 1 m/s, hold heading 0
        mujoco.mj_step(model, data)

    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    # Cart pushed forward along +x with negligible sideways (y) motion.
    assert data.qpos[task.cart_pos_idx] > 0.5
    assert abs(data.qpos[task.cart_pos_idx + 1] - cart_y0) < 0.05


def test_turn_in_place_holds_position() -> None:
    """Commanding a heading change with zero forward speed turns in place anywhere."""
    task = DiffDrivePush()
    model, data = task.model, task.data
    px = task.get_joint_position_start_index("pusher_x")
    py = task.get_joint_position_start_index("pusher_y")
    pth = task.get_joint_position_start_index("pusher_theta")

    data.qpos[:] = 0.0
    data.qpos[px] = 2.0
    data.qpos[py] = 1.0
    data.qpos[pth] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    for _ in range(200):
        data.ctrl[:] = [0.0, np.pi / 2]  # no forward, command 90 deg heading
        mujoco.mj_step(model, data)

    assert np.isclose(data.qpos[px], 2.0, atol=0.05)
    assert np.isclose(data.qpos[py], 1.0, atol=0.05)
    assert np.isclose(data.qpos[pth], np.pi / 2, atol=0.1)
