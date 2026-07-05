# Planar Push
## Goal
Create a task similar to cylinder push, where a planar cylinder (the pusher) pushes another planar cylinder (the cart). In this task, the pusher can only be actuated to turn or move forward/backward along its heading, similar to a differential drive robot (it can turn in place).

Similar to the cylinder push, the task should have the following rewards
- a quadratic reward for the pusher world XY position aligning with a goal position (the pusher goal), which is "behind" the cart relative to the global cart goal. This is specified by computing the vector pointing from the cart to the goal, and subtracting a scaled version of that vector from the cart XY position, exactly like in cylinder_push
- a quadratic reward for the cart being at the global cart goal
- a quadratic reward that penalizes the velocity of the pusher (both forward and angular) so it doesn't move too fast

An additional reward, not in the cylinder push, should reward the pusher pointing towards the pusher goal (so it can move in that direction). This reward should be gated to be off when the pusher is very close to the pusher goal (so that the pointing vector doesn't change rapidly)

The eventual goal is to use this with a sampling-based controller to generate pushing trajectories to map on to a real robot using offline trajectory optimization. This part will just generate a clean planar reference with a heading. Don't acknowledge this goal in code comments.

## Restrictions
- the pusher should not drift sideways, it should have dynamics similar to a differential drive
- the cart should have the same planar dynamics as in cylinder push (no heading)
- the pusher state should have x, y, theta (in any order)
- the controls should be position heading for theta. Consider a few options for forward motion (could be a velocity controller, but need to make sure the sim stays stable under sampling). 
- do not modify core parts of the Judo codebase besides what is necessary to create the task, json, xml, and add it to the registry.

## Additional pieces
- add tests for each reward term that demonstrate that they are computed correctly
- create a plot showing where the pusher goal is for a random cart, goal, and pusher location

## Plan
An existing poor/broken implementation can be found in files labelled as dubins_push
- Start by refactoring to diff_drive_push
- Then generate a plan in PLAN.md, and ask for my approval
- Then implement, recording your steps in LOG.md
    - Each step should include what you are working on at a high-level and why, and what worked and what didn't at the end.
    - Add the minimal information needed to resume if you get interrupted.