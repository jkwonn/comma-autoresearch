from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  Enhanced PID: weighted lookahead target, rate-aware error, adaptive integral.
  """
  def __init__(self):
    self.p = 0.2
    self.i = 0.1
    self.d = -0.1
    self.error_integral = 0
    self.prev_error = 0
    self.prev_target = 0
    self.prev_steer = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # Weighted lookahead: blend current target with future targets
    # This anticipates upcoming changes
    lookahead_target = target_lataccel
    if future_plan and len(future_plan.lataccel) > 5:
      weights = [0.6, 0.15, 0.1, 0.1, 0.05]
      targets = [target_lataccel] + [future_plan.lataccel[i] for i in [1, 3, 5, 8] if i < len(future_plan.lataccel)]
      targets = targets[:len(weights)]
      w = weights[:len(targets)]
      w_sum = sum(w)
      lookahead_target = sum(t * wi for t, wi in zip(targets, w)) / w_sum

    error = lookahead_target - current_lataccel
    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -5, 5)
    error_diff = error - self.prev_error
    self.prev_error = error

    pid = self.p * error + self.i * self.error_integral + self.d * error_diff

    # Feedforward on lookahead target, compensating for roll
    ff_target = lookahead_target - state.roll_lataccel
    ff = 0.3 * ff_target

    # Future roll compensation
    future_roll_ff = 0.0
    if future_plan and len(future_plan.roll_lataccel) > 3:
      future_roll = future_plan.roll_lataccel[3]
      future_roll_ff = 0.05 * (state.roll_lataccel - future_roll)

    # Target rate feedforward: if target is changing, anticipate the rate
    target_rate_ff = 0.0
    if future_plan and len(future_plan.lataccel) > 1:
      target_rate = future_plan.lataccel[1] - target_lataccel
      target_rate_ff = 0.15 * target_rate

    steer = pid + ff + future_roll_ff + target_rate_ff
    self.prev_target = target_lataccel
    self.prev_steer = steer
    return steer
