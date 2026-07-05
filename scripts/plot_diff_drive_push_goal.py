# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Visualize the diff_drive_push pusher goal for a random cart, goal, and pusher pose.

The pusher goal is the point offset "behind" the cart along the cart-to-goal direction:
    pusher_goal = cart_pos - pusher_goal_offset * unit(goal - cart)

Run:
    python scripts/plot_diff_drive_push_goal.py [--seed N] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from judo.tasks.diff_drive_push import DiffDrivePushConfig

CYLINDER_RADIUS = 0.25


def main() -> None:
    """Sample a random scene and save an annotated plot of the pusher goal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "out" / "diff_drive_push_pusher_goal.png"),
        help="Output image path.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = DiffDrivePushConfig()
    offset = cfg.pusher_goal_offset

    # Random scene.
    cart_pos = rng.uniform(-2.0, 2.0, size=2)
    cart_goal = rng.uniform(-2.0, 2.0, size=2)
    pusher_pos = rng.uniform(-2.0, 2.0, size=2)
    pusher_theta = rng.uniform(-np.pi, np.pi)

    # Pusher goal: offset behind the cart along the cart-to-goal direction.
    cart_to_goal = cart_goal - cart_pos
    cart_to_goal_dir = cart_to_goal / np.linalg.norm(cart_to_goal)
    pusher_goal = cart_pos - offset * cart_to_goal_dir

    fig, ax = plt.subplots(figsize=(7, 7))

    # Cart-to-goal direction line.
    ax.plot(
        [cart_pos[0], cart_goal[0]],
        [cart_pos[1], cart_goal[1]],
        color="gray",
        ls="--",
        lw=1.0,
        zorder=1,
        label="cart -> goal",
    )

    # Cart (filled circle) and its goal (target marker).
    ax.add_patch(Circle(cart_pos, CYLINDER_RADIUS, color="teal", alpha=0.6, zorder=2))
    ax.plot(*cart_pos, "o", color="teal", ms=5, zorder=3, label="cart")
    ax.plot(*cart_goal, "*", color="green", ms=18, zorder=3, label="cart goal")

    # Pusher (filled circle) with heading arrow.
    ax.add_patch(Circle(pusher_pos, CYLINDER_RADIUS, color="salmon", alpha=0.6, zorder=2))
    ax.plot(*pusher_pos, "o", color="firebrick", ms=5, zorder=3, label="pusher")
    heading = np.array([np.cos(pusher_theta), np.sin(pusher_theta)])
    ax.annotate(
        "",
        xy=pusher_pos + 0.5 * heading,
        xytext=pusher_pos,
        arrowprops=dict(arrowstyle="-|>", color="firebrick", lw=2),
        zorder=4,
    )

    # Pusher goal (what the pusher-proximity reward pulls the pusher toward).
    ax.plot(*pusher_goal, "X", color="black", ms=13, zorder=5, label="pusher goal")
    ax.annotate(
        f"offset = {offset}",
        xy=pusher_goal,
        xytext=(pusher_goal[0] + 0.15, pusher_goal[1] + 0.15),
        fontsize=9,
    )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("diff_drive_push: pusher goal (offset behind cart toward cart goal)")
    ax.legend(loc="best", fontsize=9)

    all_pts = np.array([cart_pos, cart_goal, pusher_pos, pusher_goal])
    lo = all_pts.min(0) - 0.8
    hi = all_pts.max(0) + 0.8
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    print(f"cart={cart_pos}, cart_goal={cart_goal}, pusher={pusher_pos}, pusher_goal={pusher_goal}")


if __name__ == "__main__":
    main()
