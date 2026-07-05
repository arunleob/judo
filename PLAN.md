# Plan: `diff_drive_push` task

## Objective
Turn the refactored `diff_drive_push` task into a proper planar pushing task where the
pusher behaves like a differential-drive robot: it can turn in place and drive
forward/backward along its heading, but cannot drift sideways.

## 1. Dynamics / model (`judo/models/xml/diff_drive_push.xml`)

**Generalized coordinates (state = `qpos + qvel`)**
- Pusher: `pusher_x` (world slide), `pusher_y` (world slide), `pusher_theta` (hinge).
  → gives a clean `(x, y, theta)` state (requirement satisfied).
- Cart: `slider_cart_x`, `slider_cart_y` (identical planar dynamics to cylinder_push, no heading).

**Actuators (task control dim `nu = 2`)**
- `heading`: `<position>` actuator on `pusher_theta` → commanded target heading (position control on theta).
- `forward`: `<velocity>` actuator with **site transmission** on `pusher_site`,
  `gear = "1 0 0 0 0 0"`, so the actuator applies force along the pusher's *local x-axis*
  (its heading) and servos the *forward* speed to the commanded value.

**Why this design (key decision)**
A true nonholonomic (no-sideways-velocity) constraint cannot be expressed with holonomic
MuJoCo joints inside a batched, actuator-only rollout. Two candidate approximations were considered:
1. *Local forward-slide nested under the heading hinge.* Rejected: position becomes
   `q_f · (cosθ, sinθ)` measured from the hinge origin, so the robot only "turns in place" at
   the origin and sweeps large arcs when turning elsewhere — it drifts sideways.
2. *World `x, y` slides + heading hinge, forward driven by a site-transmission velocity
   actuator (chosen).* The forward actuator only ever applies force along the current heading;
   sideways motion can arise solely from contact and is suppressed by joint damping on `x, y`.
   This keeps `(x, y, theta)` as clean coordinates and lets the robot turn in place anywhere.

This approach was prototyped and verified: forward motion tracks heading, turning-in-place holds
position anywhere, curved paths keep lateral speed ≈ 0, and the sim is stable (no NaNs) at the
default timestep. A velocity servo (rather than a raw force/motor) is used for the forward drive
so the reference stays clean and bounded under sampling.

**Damping**: keep moderate damping on `pusher_x`, `pusher_y` (suppresses lateral drift) and on
`pusher_theta` (stability). Cart keeps its existing damping.

## 2. Sensors
- `trace_pusher` — `framepos` of `pusher_site` (pusher world XY).
- `pusher_heading` — `framexaxis` of `pusher_site` (unit heading direction in world; gives
  `(cosθ, sinθ)` directly, avoiding trig/angle-wrap in the reward).
- `pusher_linvel` — `framelinvel` of `pusher_site`.
- `pusher_angvel` — `frameangvel` of `pusher_site`.
- `trace_cart` — `framepos` of `cart_site` (for visualization traces).

## 3. Reward (`judo/tasks/diff_drive_push.py`)
Same structure/sign convention as cylinder_push (each term is a negative penalty; max reward 0),
summed over the horizon.

Let `cart_to_goal = goal_pos - cart_pos`, `dir = cart_to_goal / |cart_to_goal|`,
and `pusher_goal = cart_pos - pusher_goal_offset · dir` (exactly as in cylinder_push).

1. **`w_pusher_proximity`** — quadratic penalty on `pusher_pos - pusher_goal`.
2. **`w_cart_position`** — quadratic penalty on `cart_pos - goal_pos`.
3. **`w_pusher_velocity`** — quadratic penalty on the pusher's forward (linear) speed and angular
   speed, so it does not move too fast.
4. **`w_pusher_heading`** (new) — rewards the pusher pointing toward the pusher goal so it can
   drive that way. `alignment = heading_dir · unit(pusher_goal - pusher_pos)`, penalty
   `= (1 - alignment)` (0 when aligned, up to 2 when opposite). **Gated off** when
   `|pusher_pos - pusher_goal| < pointing_gate_distance`, so the target direction (and thus the
   reward) does not change rapidly once the pusher is essentially on top of the pusher goal.

**Config fields** (`DiffDrivePushConfig`)
- `w_pusher_proximity = 0.5`
- `w_pusher_velocity = 0.0` (tunable)
- `w_cart_position = 0.1`
- `w_pusher_heading = 0.1` (new)
- `pusher_goal_offset = 0.25`
- `pointing_gate_distance = 0.1` (new)
- `goal_pos = [0, 0]`
- `w_pusher_heading` exposed as a GUI slider (like `w_pusher_proximity`).

`reset()` places the cart and pusher at randomized positions with a random pusher heading and
zero velocity; `success()` unchanged (cart within threshold of goal).

## 4. Config JSON (`run_mpc/configs/diff_drive_push.json`)
Add `w_pusher_heading` and `pointing_gate_distance` to `task_config`; keep optimizer/controller
settings, tuning only if needed for stability.

## 5. Tests (`tests/test_tasks/test_diff_drive_push.py`)
One focused test per reward term. Each builds `states`/`sensors` arrays with known values at the
task's real indices, isolates a single weight (others set to 0), and asserts the reward equals a
hand-computed value:
- pusher-proximity term (incl. correct `pusher_goal` offset geometry),
- cart-position term,
- velocity term (forward + angular),
- heading/pointing term: aligned → 0 penalty, anti-aligned → max penalty, and **gated to 0** when
  within `pointing_gate_distance` of the pusher goal.
Plus a smoke test that the model builds with `nu == 2` and a short rollout stays finite.

## 6. Plot
`scripts/plot_diff_drive_push_goal.py`: samples a random cart position, cart goal, and pusher
pose, computes `pusher_goal`, and saves an annotated figure (cart, goal, pusher + heading arrow,
pusher goal, and the cart→goal direction line) to `out/diff_drive_push_pusher_goal.png`.

## 7. Constraints respected
Only the task's own files are touched: `diff_drive_push.py`, `diff_drive_push.xml`,
`diff_drive_push.json`, the task registry (`judo/tasks/__init__.py`), a new test file, and a new
plot script. No core Judo code is modified. Progress recorded in `LOG.md`.
