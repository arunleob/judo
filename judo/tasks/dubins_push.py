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

XML_PATH = str(MODEL_PATH / "xml/dubins_push.xml")
GOAL_DISTANCE_THRESHOLD = 0.3


@slider("w_pusher_proximity", 0.0, 5.0, 0.1)
@dataclass
class DubinsPushConfig(TaskConfig):
    """Reward configuration for the Dubins push task."""

    w_pusher_proximity: float = 0.5
    w_pusher_velocity: float = 0.0
    w_cart_position: float = 0.1
    pusher_goal_offset: float = 0.25
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


class DubinsPush(Task[DubinsPushConfig]):
    """Defines the Dubins push task with position-controlled pusher forward offset and heading."""

    name: str = "dubins_push"
    config_t: type[DubinsPushConfig] = DubinsPushConfig

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the Dubins push task."""
        super().__init__(model_path=model_path, sim_model_path=sim_model_path)
        self.cart_pos_idx = self.get_joint_position_start_index("slider_cart_x")
        self.cart_vel_idx = self.get_joint_velocity_start_index("slider_cart_x")
        self.pusher_pos_idx = self.get_sensor_start_index("trace_pusher")
        self.pusher_vel_idx = self.get_sensor_start_index("pusher_linvel")
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the Dubins push reward (same structure as cylinder_push).

        Maps a list of states, list of controls, to a batch of rewards (summed over time) for each rollout.

        The Dubins push reward has three terms:
            * `pusher_reward`, penalizing the distance between the pusher and the cart.
            * `velocity_reward` penalizing squared linear velocity of the pusher.
            * `goal_reward`, penalizing the distance from the cart to the goal.

        Since we return rewards, each penalty term is returned as negative. The max reward is zero.
        """
        batch_size = states.shape[0]

        pusher_pos = sensors[..., self.pusher_pos_idx : self.pusher_pos_idx + 2]
        cart_pos = states[..., self.cart_pos_idx : self.cart_pos_idx + 2]
        pusher_vel = sensors[..., self.pusher_vel_idx : self.pusher_vel_idx + 2]
        cart_goal = self.config.goal_pos[0:2]

        cart_to_goal = cart_goal - cart_pos
        cart_to_goal_norm = np.linalg.norm(cart_to_goal, axis=-1, keepdims=True)
        cart_to_goal_direction = cart_to_goal / cart_to_goal_norm

        pusher_goal = cart_pos - self.config.pusher_goal_offset * cart_to_goal_direction

        pusher_proximity = quadratic_norm(pusher_pos - pusher_goal)
        pusher_reward = -self.config.w_pusher_proximity * pusher_proximity.sum(-1)

        velocity_reward = -self.config.w_pusher_velocity * quadratic_norm(pusher_vel).sum(-1)

        goal_proximity = quadratic_norm(cart_pos - cart_goal)
        goal_reward = -self.config.w_cart_position * goal_proximity.sum(-1)

        assert pusher_reward.shape == (batch_size,)
        assert velocity_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)

        return pusher_reward + velocity_reward + goal_reward

    def success(self, model: mujoco.MjModel, data: mujoco.MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the cart is close to the goal position."""
        cart_pos = data.qpos[self.cart_pos_idx : self.cart_pos_idx + 2]
        goal_pos = self.config.goal_pos[0:2]
        return bool(np.linalg.norm(cart_pos - goal_pos) < GOAL_DISTANCE_THRESHOLD)

    def reset(self) -> None:
        """Resets the model to a default (random) state."""
        pusher_theta_idx = self.get_joint_position_start_index("pusher_theta")
        pusher_x_idx = self.get_joint_position_start_index("pusher_x")

        pusher_angle = 2 * np.pi * np.random.rand()
        cart_angle = 2 * np.pi * np.random.rand()

        self.data.qpos[:] = 0.0
        self.data.qpos[pusher_theta_idx] = pusher_angle
        self.data.qpos[pusher_x_idx] = 1.0
        self.data.qpos[self.cart_pos_idx] = 2 * np.cos(cart_angle)
        self.data.qpos[self.cart_pos_idx + 1] = 2 * np.sin(cart_angle)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
