from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  PID with roll compensation and feedforward from future_plan
  """
  def __init__(self):
    self.p = 0.3
    self.i = 0.05
    self.d = -0.1
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # Roll compensation: subtract roll contribution from target
    adjusted_target = target_lataccel - state.roll_lataccel
    adjusted_current = current_lataccel - state.roll_lataccel

    error = adjusted_target - adjusted_current
    self.error_integral += error
    # Anti-windup
    self.error_integral = np.clip(self.error_integral, -10, 10)
    error_diff = error - self.prev_error
    self.prev_error = error

    # PID output
    pid_output = self.p * error + self.i * self.error_integral + self.d * error_diff

    # Feedforward from future plan
    ff = 0.0
    if future_plan and len(future_plan.lataccel) > 0:
      ff = future_plan.lataccel[0] * 0.3

    return pid_output + ff
