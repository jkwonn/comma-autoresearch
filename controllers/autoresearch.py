from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  Tuned PID with feedforward and preview.
  Best from grid search: p=0.25, i=0.1, d=-0.1, ff=0.3, preview=0.1
  """
  def __init__(self):
    self.p = 0.25
    self.i = 0.1
    self.d = -0.1
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = target_lataccel - current_lataccel
    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -10, 10)
    error_diff = error - self.prev_error
    self.prev_error = error

    pid = self.p * error + self.i * self.error_integral + self.d * error_diff

    ff_target = target_lataccel - state.roll_lataccel
    ff = 0.3 * ff_target

    preview_ff = 0.0
    if future_plan and len(future_plan.lataccel) > 2:
      ft = future_plan.lataccel[2]
      fr = future_plan.roll_lataccel[2] if len(future_plan.roll_lataccel) > 2 else state.roll_lataccel
      preview_ff = 0.1 * ((ft - fr) - ff_target)

    return pid + ff + preview_ff
