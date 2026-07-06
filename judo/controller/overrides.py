# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from judo.config import set_config_overrides
from judo.controller.controller import ControllerConfig


def set_default_cylinder_push_overrides() -> None:
    """Sets the default task-specific controller config overrides for the cylinder push task."""
    set_config_overrides(
        "cylinder_push",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "zero",
        },
    )


def set_default_diff_drive_push_overrides() -> None:
    """Sets the default task-specific controller config overrides for the diff drive push task.

    Mirrors run_mpc/configs/diff_drive_push.json so the interactive app matches the tuned config.
    The longer horizon (vs the 1.0 s default) lets the planner see the arc-around-the-cart
    maneuver, which is what prevents the pusher from lining up behind the cart and stalling.
    `min_max` action normalization samples in a per-actuator normalized space, so the heading
    actuator (ctrlrange +-10) gets exploration proportional to its range instead of ~5x less than
    the forward actuator (+-2). This keeps the sampler turning and is the main lever for escaping the
    collinear (pusher/cart/goal in a line) saddle.
    """
    set_config_overrides(
        "diff_drive_push",
        ControllerConfig,
        {
            "horizon": 1.5,
            "spline_order": "zero",
            "action_normalizer": "min_max",
        },
    )


def set_default_cartpole_overrides() -> None:
    """Sets the default task-specific controller config overrides for the cartpole task."""
    set_config_overrides(
        "cartpole",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "zero",
        },
    )


def set_default_leap_cube_overrides() -> None:
    """Sets the default task-specific controller config overrides for the leap cube task."""
    set_config_overrides(
        "leap_cube",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "max_num_traces": 1,
        },
    )


def set_default_leap_cube_down_overrides() -> None:
    """Sets the default task-specific controller config overrides for the leap cube down task."""
    set_config_overrides(
        "leap_cube_down",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "max_num_traces": 1,
        },
    )


def set_default_caltech_leap_cube_overrides() -> None:
    """Sets the default task-specific controller config overrides for the caltech leap cube task."""
    set_config_overrides(
        "caltech_leap_cube",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "max_num_traces": 1,
        },
    )


_SPOT_TASK_NAMES = [
    "spot_base",
    "spot_box_push",
    "spot_navigate",
    "spot_tire_roll",
    "spot_tire_upright",
]


def set_default_spot_overrides() -> None:
    """Sets the default task-specific controller config overrides for all Spot tasks."""
    for task_name in _SPOT_TASK_NAMES:
        set_config_overrides(
            task_name,
            ControllerConfig,
            {
                "horizon": 2.0,
            },
        )


def set_default_fr3_pick_overrides() -> None:
    """Sets the default task-specific controller config overrides for the fr3 pick task."""
    set_config_overrides(
        "fr3_pick",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "linear",
            "max_num_traces": 3,
            "control_freq": 20.0,
        },
    )
