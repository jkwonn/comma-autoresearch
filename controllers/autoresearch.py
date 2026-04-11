from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  PID with proper feedforward: steer ≈ gain * (target - roll) + PID correction
  """
  def __init__(self):
    self.p = 0.3
    self.i = 0.05
    self.d = -0.1
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = target_lataccel - current_lataccel
    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -10, 10)
    error_diff = error - self.prev_error
    self.prev_error = error

    # PID correction
    pid_output = self.p * error + self.i * self.error_integral + self.d * error_diff

    # Feedforward: estimate steer needed to produce target lataccel
    # Subtract roll component since that comes for free from road geometry
    ff_target = target_lataccel - state.roll_lataccel
    ff_gain = 0.3
    ff = ff_gain * ff_target

    # Preview feedforward: anticipate future target changes
    preview_ff = 0.0
    if future_plan and len(future_plan.lataccel) > 2:
      future_target = future_plan.lataccel[2]  # look 0.3s ahead
      future_roll = future_plan.roll_lataccel[2] if len(future_plan.roll_lataccel) > 2 else state.roll_lataccel
      preview_ff = 0.1 * ((future_target - future_roll) - ff_target)

    return pid_output + ff + preview_ff
