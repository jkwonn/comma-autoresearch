from . import BaseController
import numpy as np
import onnxruntime as ort
from pathlib import Path

CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
MAX_ACC_DELTA = 0.5

class Controller(BaseController):
  """PID with per-segment offset search via ONNX simulation of actual PID."""
  def __init__(self):
    model_path = str(Path(__file__).resolve().parent.parent / "models" / "tinyphysics.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    self.ort = ort.InferenceSession(model_path, options, ['CPUExecutionProvider'])
    self.bins = np.linspace(-5, 5, VOCAB_SIZE).astype(np.float32)
    self.states = []; self.actions = []; self.lats = []
    self.step = 0; self.offset = 0.0; self.searched = False
    self.ei = 0.0; self.ep = 0.0; self.fd = 0.0

  def _pid_compute(self, targets, h, current, roll, ei, ep, fd):
    """Compute PID matching pid_output exactly. targets is full target array."""
    t = targets[h]
    # Match the lookahead from pid_output
    la = t
    if h + 9 < len(targets):
      la = 0.4*t + 0.3*targets[h+1] + 0.15*targets[min(h+3,len(targets)-1)] + 0.1*targets[min(h+5,len(targets)-1)] + 0.05*targets[min(h+9,len(targets)-1)]
    e = la - current
    ei = np.clip(ei + e, -5, 5)
    rd = e - ep; fd = 0.5*fd + 0.5*rd; ep = e
    out = 0.2*e + 0.1*ei - 0.1*fd + 0.35*(la - roll)
    if h + 1 < len(targets):
      out += 0.2*(targets[h+1] - t)
    return float(np.clip(out, -2, 2)), ei, ep, fd

  def _predict(self, st_ctx, act_ctx, lat_ctx, draw):
    tok = np.digitize(np.clip(lat_ctx, -5, 5), self.bins, right=True).astype(np.int64)
    st = np.array(st_ctx, dtype=np.float32)
    acts = np.array(act_ctx, dtype=np.float32).reshape(-1, 1)
    si = np.concatenate([acts, st], axis=1)
    res = self.ort.run(None, {'states': si[np.newaxis].astype(np.float32), 'tokens': tok[np.newaxis]})[0]
    logits = res[0, -1] / 0.8; logits -= logits.max()
    probs = np.exp(logits); probs /= probs.sum()
    token = min(int(np.searchsorted(np.cumsum(probs), draw)), VOCAB_SIZE - 1)
    return float(self.bins[token])

  def _simulate_pid_offset(self, offset, targets, fp, draws):
    H = len(draws)  # use draws length as horizon
    st_ctx = list(self.states[-CONTEXT_LENGTH:])
    act_ctx = list(self.actions[-(CONTEXT_LENGTH-1):])
    lat_ctx = list(self.lats[-CONTEXT_LENGTH:])
    cur = self.lats[-1]
    ei, ep, fd = self.ei, self.ep, self.fd
    cost = 0.0
    for h in range(H):
      if h > 0 and h-1 < len(fp.roll_lataccel):
        st_ctx.append([fp.roll_lataccel[h-1],
                       fp.v_ego[h-1] if h-1<len(fp.v_ego) else st_ctx[-1][1],
                       fp.a_ego[h-1] if h-1<len(fp.a_ego) else st_ctx[-1][2]])
      elif h > 0:
        st_ctx.append(st_ctx[-1])
      roll = st_ctx[-1][0]
      steer, ei, ep, fd = self._pid_compute(targets, h, cur, roll, ei, ep, fd)
      steer = float(np.clip(steer + offset, -2, 2))
      act_ctx.append(steer)
      pred = self._predict(st_ctx[-CONTEXT_LENGTH:], act_ctx[-CONTEXT_LENGTH:],
                           lat_ctx[-CONTEXT_LENGTH:], draws[h])
      pred = np.clip(pred, cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)
      cost += (targets[h] - pred)**2 * 12.5 + (pred - cur)**2 * 25.0
      lat_ctx.append(pred); cur = pred
    return cost

  def pid_output(self, target, current, state, fp):
    la = target
    if fp and len(fp.lataccel) > 9:
      la = 0.4*target + 0.3*fp.lataccel[0] + 0.15*fp.lataccel[2] + 0.1*fp.lataccel[4] + 0.05*fp.lataccel[8]
    e = la - current
    self.ei = np.clip(self.ei + e, -5, 5)
    rd = e - self.ep; self.fd = 0.5*self.fd + 0.5*rd; self.ep = e
    out = 0.2*e + 0.1*self.ei - 0.1*self.fd + 0.35*(la - state.roll_lataccel)
    if fp and len(fp.lataccel) > 1: out += 0.2*(fp.lataccel[1] - target)
    return float(np.clip(out, -2, 2))

  def update(self, target_lataccel, current_lataccel, state, future_plan=None):
    self.states.append([state.roll_lataccel, state.v_ego, state.a_ego])
    self.lats.append(current_lataccel)
    self.step += 1
    steer = self.pid_output(target_lataccel, current_lataccel, state, future_plan)

    # One-time offset search after enough context
    if (not self.searched and self.step == 110 and
        future_plan and len(future_plan.lataccel) >= 20 and
        len(self.actions) >= CONTEXT_LENGTH - 1):
      self.searched = True
      n_avail = len(future_plan.lataccel)
      sim_H = 20  # simulation horizon
      targets = [target_lataccel] + list(future_plan.lataccel[:min(sim_H + 9, n_avail)])
      rng = np.random.get_state()
      draws = [np.random.random() for _ in range(sim_H)]
      np.random.set_state(rng)
      best_cost, best_off = float('inf'), 0.0
      for off in [-0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3]:
        c = self._simulate_pid_offset(off, targets, future_plan, draws)
        if c < best_cost: best_cost = c; best_off = off
      self.offset = best_off

    self.actions.append(float(np.clip(steer + self.offset, -2, 2)))
    return self.actions[-1]
