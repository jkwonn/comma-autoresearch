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
    # Use future target (1 step ahead) as primary - compensates for plant delay
    lookahead_target = target_lataccel
    if future_plan and len(future_plan.lataccel) > 9:
      lookahead_target = (0.4 * target_lataccel +
                          0.3 * future_plan.lataccel[0] +
                          0.15 * future_plan.lataccel[2] +
                          0.1 * future_plan.lataccel[4] +
                          0.05 * future_plan.lataccel[8])

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
