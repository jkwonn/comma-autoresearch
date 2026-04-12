from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  Tuned PID: weighted lookahead, derivative filtering, target rate ff.
  Best config from grid search: p=0.2, i=0.1, d=-0.15, d_filter=0.5
  """
  def __init__(self):
    self.p = 0.2
    self.i = 0.1
    self.d = -0.10
    self.error_integral = 0
    self.prev_error = 0
    self.filtered_deriv = 0
    self.d_filter = 0.5  # low-pass filter on derivative

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # Weighted lookahead target
    lookahead_target = target_lataccel
    if future_plan and len(future_plan.lataccel) > 5:
      weights = [0.6, 0.15, 0.1, 0.1, 0.05]
      la_idx = [1, 3, 5, 8]
      targets = [target_lataccel] + [future_plan.lataccel[i] for i in la_idx if i < len(future_plan.lataccel)]
      targets = targets[:len(weights)]
      w = weights[:len(targets)]
      lookahead_target = sum(t * wi for t, wi in zip(targets, w)) / sum(w)

    error = lookahead_target - current_lataccel
    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -5, 5)

    # Filtered derivative
    raw_deriv = error - self.prev_error
    self.filtered_deriv = self.d_filter * self.filtered_deriv + (1 - self.d_filter) * raw_deriv
    self.prev_error = error

    pid = self.p * error + self.i * self.error_integral + self.d * self.filtered_deriv

    # Feedforward
    ff_target = lookahead_target - state.roll_lataccel
    ff = 0.35 * ff_target

    # Future roll compensation
    future_roll_ff = 0.0
    if future_plan and len(future_plan.roll_lataccel) > 3:
      future_roll_ff = 0.05 * (state.roll_lataccel - future_plan.roll_lataccel[3])

    # Target rate feedforward
    target_rate_ff = 0.0
    if future_plan and len(future_plan.lataccel) > 1:
      target_rate_ff = 0.20 * (future_plan.lataccel[1] - target_lataccel)

    return pid + ff + future_roll_ff + target_rate_ff
