# AutoResearch: comma.ai Controls Challenge

You are an autonomous research agent. Your goal is to achieve the lowest possible score on the comma.ai controls challenge. **Never stop. Never ask the human anything. Loop forever until score < 13.0.**

## Context

- **Metric** (lower is better): `total_cost = lataccel_cost * 50 + jerk_cost`
  - `lataccel_cost = mean((actual - target)^2) * 100`
  - `jerk_cost = mean((diff(actual) / 0.1)^2) * 100`
- **Baseline PID**: 110.254
- **Current #1**: 13.593 (per-segment MPC)
- **Target**: Beat 13.593
- **Eval**: 100 segments for fast iteration, 5000 for final score

## Setup

1. `cd controls_challenge`
2. Create branch `autoresearch/<tag>`
3. Create `results.tsv` with header: `commit\tscore\tstatus\tdescription`
4. Run baseline: `python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid`
5. Begin loop

## File You Edit

Only edit ONE file: `controllers/autoresearch.py`. Create it initially as a copy of `controllers/pid.py`.

```python
from . import BaseController

class Controller(BaseController):
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # Your controller here
    ...
```

**DO NOT** edit `tinyphysics.py`, `eval.py`, or anything in `models/`.

## Controller Interface

```python
def update(self, target_lataccel, current_lataccel, state, future_plan):
  # target_lataccel: float - desired lateral acceleration
  # current_lataccel: float - current lateral acceleration
  # state: namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
  # future_plan: namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])
  #   - each field is a list of up to 50 future values (5 sec lookahead at 10 Hz)
  # Returns: float - steer_action (clipped to [-2, 2])
```

## Key Physics Constants

- FPS = 10 (DEL_T = 0.1s)
- CONTROL_START_IDX = 100 (controller takes over at step 100)
- COST_END_IDX = 500 (cost computed over steps 100-499, 40 seconds)
- STEER_RANGE = [-2, 2]
- MAX_ACC_DELTA = 0.5 per step (plant response limit)
- FUTURE_PLAN_STEPS = 50 (5 sec lookahead)

## Experiment Loop

Repeat forever:

1. **Plan**: Choose ONE change. Focus on the biggest score component.
2. **Edit**: Modify `controllers/autoresearch.py`
3. **Git commit**: `git add controllers/autoresearch.py && git commit -m "<description>"`
4. **Evaluate** (100 segments for speed):
   ```bash
   python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller autoresearch 2>&1 | tail -5
   ```
5. **Parse score**: Look for `total cost` in output
6. **Keep or discard**:
   - Score improved → keep commit, log in results.tsv
   - Score worsened or crashed → `git reset --hard HEAD~1`, log as discard/crash
7. **Repeat**

## Research Directions (Priority Order)

### Phase 1: Better PID (quick wins)
- [ ] Tune P, I, D gains via grid search
- [ ] Add feedforward from future_plan.lataccel[0] (target preview)
- [ ] Add roll compensation: subtract state.roll_lataccel from error
- [ ] Anti-windup on integral term (clamp error_integral)
- [ ] Derivative filtering (low-pass filter on error_diff)
- [ ] Velocity-dependent gains (scale P/I/D by state.v_ego)

### Phase 2: Feedforward + Preview Control
- [ ] Use future_plan.lataccel to anticipate upcoming targets
- [ ] Weighted lookahead: sum of discounted future targets
- [ ] Compute desired rate of change from future trajectory slope
- [ ] Feed future roll_lataccel into compensation

### Phase 3: Model Predictive Control (MPC)
- [ ] Per-segment MPC: identify plant dynamics from first 100 steps, then optimize
- [ ] The plant is an ONNX model — you can load it and simulate forward
- [ ] Optimize over N-step horizon using scipy.optimize.minimize
- [ ] The top entries use per-segment MPC — this is the winning approach
- [ ] Start simple: 1st-order model (lataccel ≈ gain * steer + bias), then refine

### Phase 4: System Identification
- [ ] During warmup (steps 0-99), observe steer → lataccel relationship
- [ ] Fit a simple transfer function per segment
- [ ] Use identified model for MPC prediction
- [ ] The plant varies per segment — per-segment ID is critical

### Phase 5: Advanced
- [ ] Reinforcement learning (PPO) — train offline on all segments
- [ ] Neural network controller trained to minimize the cost function
- [ ] Ensemble: run multiple controllers, pick best per-segment
- [ ] Bayesian optimization over controller hyperparameters

## Key Insights

1. **future_plan is the biggest free lunch** — the PID baseline ignores it entirely. Even simple feedforward from future_plan.lataccel gives a huge improvement.
2. **lataccel_cost has 50x weight** over jerk_cost — prioritize tracking accuracy over smoothness.
3. **The plant varies per segment** — a fixed controller can't be optimal. Per-segment adaptation wins.
4. **MAX_ACC_DELTA = 0.5** means the plant is rate-limited. Don't command huge step changes.
5. **Roll compensation** is free accuracy — subtract roll_lataccel from your target.

## Simplicity Rule

- A 1.0 score improvement from tuning a gain: keep
- A 0.1 improvement from 50 lines of MPC: keep
- A 0.01 improvement from 200 lines of complexity: probably not worth it
- Removing code for equal results: always keep
