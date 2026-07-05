# diff_drive_push implementation log

## Step 0 — Refactor dubins_push → diff_drive_push
- Renamed `judo/tasks/dubins_push.py`, `judo/models/xml/dubins_push.xml`,
  `run_mpc/configs/dubins_push.json` to `diff_drive_push.*` via `git mv`, updated identifiers
  (`DiffDrivePush`, `DiffDrivePushConfig`, `name="diff_drive_push"`, model/task names) and the
  task registry in `judo/tasks/__init__.py`.
- Verified import/instantiation in the `locomanip` conda env (mujoco 3.6.0): registered,
  `nq=5, nv=5, nu=2`, reset OK.

## Step 1 — Design de-risk (dynamics)
- Prototyped world-`x`/`y` slides + heading hinge with a site-transmission `<velocity>` forward
  actuator. Confirmed: straight motion along heading, turn-in-place holds position anywhere,
  curved paths keep lateral speed ≈ 0, no NaNs. Wrote PLAN.md; approved (velocity servo, hard
  pointing gate).

## Step 2 — Implement diff-drive XML + reward + config
- XML (`diff_drive_push.xml`): pusher now has `pusher_x`/`pusher_y` (world slides) + `pusher_theta`
  (hinge); cart unchanged. Actuators: `actuator_forward` = `<velocity>` with site transmission on
  `pusher_site` (`gear="1 0 0 0 0 0"`, `kv=20`, `ctrlrange="-2 2"`) drives forward speed along the
  heading; `actuator_heading` = `<position>` on `pusher_theta`. Added `pusher_heading`
  (`framexaxis`) and `pusher_angvel` (`frameangvel`) sensors; added a small non-colliding "nose"
  geom so heading is visible. Damping: x/y = 8, theta = 2.
- Reward (`diff_drive_push.py`): kept the three cylinder_push-style terms (pusher proximity to the
  offset pusher goal, cart-to-goal, pusher velocity) reading pusher pos from the `trace_pusher`
  sensor, cart pos from state, and pusher velocity (vx, vy, omega) from the 3 qvel entries. Added
  the pointing term `w_pusher_heading * (1 - heading . unit(pusher_goal - pusher_pos))`, hard-gated
  to 0 when within `pointing_gate_distance` of the pusher goal. New config fields
  `w_pusher_heading` (GUI slider) and `pointing_gate_distance`. `reset()` randomizes pusher x/y,
  heading, and cart position.
- Config JSON: added `w_pusher_heading` and `pointing_gate_distance`.
- Verified indices (cart_pos=3, pusher_vel=5, sensors: trace_pusher=0, pusher_heading=3) and reward
  shapes. Physics check: forward command pushes the cart straight (dy≈0), turn-in-place holds
  position at (2,1); all finite.
- What worked: site-transmission velocity actuator gives clean diff-drive motion and stays stable.
  Note (unchanged from cylinder_push): if the cart sits exactly on the goal the cart->goal
  direction is 0/0 and the pusher-proximity term is NaN; real states avoid this, tests use
  non-degenerate cart positions.

## Step 3 — Tests (`tests/test_tasks/test_diff_drive_push.py`)
- Per-term reward tests (proximity, cart position, velocity, heading aligned/anti-aligned/gated,
  horizon sum) plus dynamics tests (forward push without sideways drift, turn in place). 10 passed.
- Full `tests/test_tasks` + `tests/test_config.py`: 36 passed.

## Step 4 — Plot (`scripts/plot_diff_drive_push_goal.py`)
- Samples a random cart, cart goal, pusher pose; computes `pusher_goal`; saves annotated figure to
  `out/diff_drive_push_pusher_goal.png`. Verified the pusher goal sits "behind" the cart relative
  to the cart goal.

