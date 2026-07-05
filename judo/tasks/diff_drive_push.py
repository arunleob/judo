# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.tasks.cost_functions import quadratic_norm
from judo.utils.fields import np_1d_field

XML_PATH = str(MODEL_PATH / "xml/diff_drive_push.xml")
GOAL_DISTANCE_THRESHOLD = 0.3


@slider("w_pusher_proximity", 0.0, 5.0, 0.1)
@slider("w_pusher_heading", 0.0, 5.0, 0.1)
@dataclass
class DiffDrivePushConfig(TaskConfig):
    """Reward configuration for the differential-drive push task."""

    w_pusher_proximity: float = 0.5
    w_pusher_velocity: float = 0.0
    w_cart_position: float = 0.1
    w_pusher_heading: float = 0.1
    pusher_goal_offset: float = 0.25
    pointing_gate_distance: float = 0.1
    # Pushing-phase gate: when the pusher is within `pusher_goal_gate_distance` of the pusher goal
    # (i.e. it is behind the cart and in contact), the proximity/cart-position weights switch to
    # their "_pushing" values. By default the pusher-goal proximity pull is turned off while pushing
    # so the cart-to-goal term becomes the sole signal; this lets the cart settle tighter on the goal
    # (final cart-goal distance ~0.06 m vs ~0.11 m when proximity stays on) without hurting time or
    # success. `w_cart_position_pushing` is kept at the base value: boosting it instead causes the
    # pusher to shove through and overshoot the goal.
    pusher_goal_gate_distance: float = 0.3
    w_pusher_proximity_pushing: float = 0.0
    w_cart_position_pushing: float = 0.1
    goal_pos: np.ndarray = np_1d_field(
        np.array([0.0, 0.0]),
        names=["x", "y"],
        mins=[-1.0, -1.0],
        maxs=[1.0, 1.0],
        steps=[0.01, 0.01],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
        xyz_vis_defaults=[0.0, 0.0, 0.0],
    )


class DiffDrivePush(Task[DiffDrivePushConfig]):
    """Planar push task where the pusher has differential-drive dynamics.

    The pusher can turn in place and drive forward/backward along its heading, but cannot drift
    sideways. It pushes a free planar cart toward a goal.
    """

    name: str = "diff_drive_push"
    config_t: type[DiffDrivePushConfig] = DiffDrivePushConfig

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the differential-drive push task."""
        super().__init__(model_path=model_path, sim_model_path=sim_model_path)
        self.cart_pos_idx = self.get_joint_position_start_index("slider_cart_x")
        self.pusher_vel_idx = self.get_joint_velocity_start_index("pusher_x")
        self.pusher_pos_idx = self.get_sensor_start_index("trace_pusher")
        self.pusher_heading_idx = self.get_sensor_start_index("pusher_heading")
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the differential-drive push reward.

        Maps a batch of states, sensors, and controls to a batch of rewards (summed over time).

        The reward has four terms:
            * `pusher_reward`, penalizing the distance between the pusher and a target point offset
              "behind" the cart along the cart-to-goal direction (the pusher goal).
            * `velocity_reward`, penalizing the squared forward and angular velocity of the pusher.
            * `goal_reward`, penalizing the distance from the cart to the goal.
            * `heading_reward`, rewarding the pusher for pointing toward the pusher goal. This term
              is gated off when the pusher is very close to the pusher goal so the target pointing
              direction does not change rapidly.

        Since we return rewards, each penalty term is returned as negative. The max reward is zero.
        """
        batch_size = states.shape[0]

        pusher_pos = sensors[..., self.pusher_pos_idx : self.pusher_pos_idx + 2]
        pusher_heading = sensors[..., self.pusher_heading_idx : self.pusher_heading_idx + 2]
        cart_pos = states[..., self.cart_pos_idx : self.cart_pos_idx + 2]
        pusher_vel = states[..., self.pusher_vel_idx : self.pusher_vel_idx + 3]
        cart_goal = self.config.goal_pos[0:2]

        cart_to_goal = cart_goal - cart_pos
        cart_to_goal_norm = np.linalg.norm(cart_to_goal, axis=-1, keepdims=True)
        cart_to_goal_direction = cart_to_goal / cart_to_goal_norm

        pusher_goal = cart_pos - self.config.pusher_goal_offset * cart_to_goal_direction

        pusher_to_goal = pusher_goal - pusher_pos
        pusher_to_goal_dist = np.linalg.norm(pusher_to_goal, axis=-1, keepdims=True)
        pusher_to_goal_direction = pusher_to_goal / np.clip(pusher_to_goal_dist, 1e-6, None)

        # Pushing-phase gate (per timestep): 1 once the pusher is behind the cart and in contact
        # (within gate distance of the pusher goal). While pushing, blend to the "_pushing" weights
        # so the cart-to-goal term can dominate; otherwise use the approach-phase weights.
        pushing = (pusher_to_goal_dist[..., 0] <= self.config.pusher_goal_gate_distance).astype(pusher_pos.dtype)
        w_proximity = self.config.w_pusher_proximity * (1.0 - pushing) + self.config.w_pusher_proximity_pushing * pushing
        w_cart = self.config.w_cart_position * (1.0 - pushing) + self.config.w_cart_position_pushing * pushing

        pusher_proximity = quadratic_norm(pusher_pos - pusher_goal)
        pusher_reward = -(w_proximity * pusher_proximity).sum(-1)

        velocity_reward = -self.config.w_pusher_velocity * quadratic_norm(pusher_vel).sum(-1)

        goal_proximity = quadratic_norm(cart_pos - cart_goal)
        goal_reward = -(w_cart * goal_proximity).sum(-1)

        alignment = (pusher_heading * pusher_to_goal_direction).sum(-1)
        pointing_gate = (pusher_to_goal_dist[..., 0] > self.config.pointing_gate_distance).astype(alignment.dtype)
        heading_penalty = pointing_gate * (1.0 - alignment)
        heading_reward = -self.config.w_pusher_heading * heading_penalty.sum(-1)

        assert pusher_reward.shape == (batch_size,)
        assert velocity_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert heading_reward.shape == (batch_size,)

        return pusher_reward + velocity_reward + goal_reward + heading_reward

    def success(self, model: mujoco.MjModel, data: mujoco.MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the cart is close to the goal position."""
        cart_pos = data.qpos[self.cart_pos_idx : self.cart_pos_idx + 2]
        goal_pos = self.config.goal_pos[0:2]
        return bool(np.linalg.norm(cart_pos - goal_pos) < GOAL_DISTANCE_THRESHOLD)

    def reset(self) -> None:
        """Resets the model to a default (random) state."""
        pusher_x_idx = self.get_joint_position_start_index("pusher_x")
        pusher_y_idx = self.get_joint_position_start_index("pusher_y")
        pusher_theta_idx = self.get_joint_position_start_index("pusher_theta")

        pusher_angle = 2 * np.pi * np.random.rand()
        cart_angle = 2 * np.pi * np.random.rand()

        self.data.qpos[:] = 0.0
        self.data.qpos[pusher_x_idx] = np.cos(pusher_angle)
        self.data.qpos[pusher_y_idx] = np.sin(pusher_angle)
        self.data.qpos[pusher_theta_idx] = 2 * np.pi * np.random.rand()
        self.data.qpos[self.cart_pos_idx] = 2 * np.cos(cart_angle)
        self.data.qpos[self.cart_pos_idx + 1] = 2 * np.sin(cart_angle)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
