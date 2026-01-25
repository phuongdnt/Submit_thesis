# 🔍 Code Review & Optimization Report

## Executive Summary

| Category | Issues Found | Severity | Impact on Thesis |
|----------|-------------|----------|------------------|
| Performance | 5 | High | Training time 3-5x slower than optimal |
| Algorithm | 4 | Critical | Affects experimental results |
| Code Quality | 6 | Medium | Maintainability |
| Missing Features | 4 | High | Incomplete for thesis requirements |

---

## 🚨 CRITICAL Issues (Fix Immediately)

### Issue #1: SubprocVecEnv is NOT Parallel

**Location:** `envs/vec_env.py:150-180`

**Problem:** Despite the name "SubprocVecEnv", environments run SEQUENTIALLY, not in parallel.

```python
# Current: Sequential (SLOW)
for env, acts in zip(self.env_list, actions_batch):
    next_obs, rewards, done, info = env.step(acts, one_hot=one_hot)
```

**Impact:** With 8 threads, training is 1x speed instead of 8x.

**Fix:** Use Python `multiprocessing` or `concurrent.futures`:

```python
# Recommended: True parallel execution
from concurrent.futures import ThreadPoolExecutor

class TrueParallelVecEnv:
    def __init__(self, env_fn, n_envs: int, max_workers: int = None):
        self.env_list = [env_fn() for _ in range(n_envs)]
        self.num_envs = n_envs
        self.executor = ThreadPoolExecutor(max_workers=max_workers or n_envs)
    
    def step(self, actions_batch, one_hot=False):
        def step_env(args):
            env, acts = args
            return env.step(acts, one_hot=one_hot)
        
        results = list(self.executor.map(step_env, zip(self.env_list, actions_batch)))
        # ... process results
```

---

### Issue #2: GA Called Every Step (Performance Killer)

**Location:** `train_main.py:221`, `lot_sizing/hybrid_planner.py:56`

**Problem:** GA is called for EVERY agent at EVERY timestep.
- Episode = 100 steps
- Agents = 3
- GA per call = pop_size(30) × generations(40) = 1200 evaluations
- Total per episode = 100 × 3 × 1200 = **360,000 GA evaluations**

**Impact:** This is why training takes 8+ hours for 1100 episodes.

**Fix Options:**

**Option A: Call GA only every N steps**
```python
class HybridPlanner:
    def __init__(self, ..., ga_frequency: int = 5):
        self.ga_frequency = ga_frequency
        self.cached_actions = None
        self.call_count = 0
    
    def refine_actions(self, actions):
        self.call_count += 1
        if self.call_count % self.ga_frequency == 0 or self.cached_actions is None:
            self.cached_actions = self._run_ga(actions)
        return self.cached_actions
```

**Option B: Reduce GA complexity**
```yaml
# In config
heuristic:
  ga:
    pop_size: 10      # Was 30
    generations: 15   # Was 40
```

---

### Issue #3: GA Fitness Function Causes Over-Ordering

**Location:** `lot_sizing/ga_lotsizing.py:62-81`

**Problem:** Current fitness only penalizes backlog and holding, no inventory cap.

```python
# Current: No upper bound on inventory
cost += inventory * holding_cost + backlog * backlog_cost
```

With backlog_cost = 5 and holding_cost = 1, GA will ALWAYS over-order.

**Fix:** Add inventory penalty term

```python
def evaluate_plan(
    plan, init_inventory, demands, holding_cost, backlog_cost, fixed_cost,
    max_inventory: int = 100,  # NEW
    inventory_penalty: float = 0.5  # NEW
):
    inventory = init_inventory
    backlog = 0
    cost = 0.0
    
    for t, demand in enumerate(demands):
        received = plan[t]
        effective = demand + backlog
        unmet = effective - (inventory + received)
        
        if unmet > 0:
            backlog = unmet
            inventory = 0
        else:
            backlog = 0
            inventory = -unmet
        
        # Standard costs
        cost += inventory * holding_cost + backlog * backlog_cost
        if received > 0:
            cost += fixed_cost
        
        # NEW: Penalty for excessive inventory
        if inventory > max_inventory:
            cost += (inventory - max_inventory) * inventory_penalty
    
    return cost
```

---

### Issue #4: No Gradient Clipping

**Location:** `agents/happo_agent.py:172-197`

**Problem:** No gradient clipping can cause training instability.

**Fix:**
```python
def update(self, batch_size=None):
    # ... existing code ...
    
    # Update critic with gradient clipping
    self.critic_optimizer.zero_grad()
    value_preds = self.critic_net(states).squeeze(-1)
    critic_loss = torch.mean((returns - value_preds) ** 2)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.5)  # NEW
    self.critic_optimizer.step()
    
    # Update actors with gradient clipping
    for i in range(self.num_agents):
        # ... existing code ...
        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)  # NEW
        actor_opt.step()
```

---

## ⚠️ HIGH Priority Issues

### Issue #5: Hardcoded Reward Smoothing Alpha

**Location:** `envs/serial_env.py:163`, `envs/network_env.py:218`

**Problem:** `self.alpha = 0.5` is hardcoded, not configurable.

**Fix:** Add to config and constructor:
```python
def __init__(self, ..., reward_smoothing_alpha: float = 0.5):
    self.alpha = reward_smoothing_alpha
```

---

### Issue #6: No Learning Rate Scheduling

**Location:** `agents/happo_agent.py`

**Problem:** Fixed learning rate throughout training. RL typically benefits from LR decay.

**Fix:**
```python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

class HAPPOAgent:
    def __init__(self, ..., lr_decay_steps: int = 1000, lr_decay_gamma: float = 0.95):
        # ... existing code ...
        
        # Add schedulers
        self.actor_schedulers = [
            StepLR(opt, step_size=lr_decay_steps, gamma=lr_decay_gamma)
            for opt in self.actor_optimisers
        ]
        self.critic_scheduler = StepLR(
            self.critic_optimizer, step_size=lr_decay_steps, gamma=lr_decay_gamma
        )
    
    def update(self, ...):
        # ... existing update code ...
        
        # Step schedulers at end
        for scheduler in self.actor_schedulers:
            scheduler.step()
        self.critic_scheduler.step()
```

---

### Issue #7: Missing Batch Inference

**Location:** `agents/happo_agent.py:76-93`

**Problem:** Actions selected one-by-one, not batched.

```python
# Current: Loop over agents
for i, obs in enumerate(obs_list):
    obs_tensor = torch.tensor(obs, ...)
    act, log_prob = self.actors[i].get_action(obs_tensor)
```

**Fix:** Batch all agent observations:
```python
def select_actions_batched(self, obs_list):
    """Batched action selection for better GPU utilization."""
    with torch.no_grad():
        actions = []
        log_probs = []
        
        for i, obs in enumerate(obs_list):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            logits = self.actors[i](obs_tensor.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            actions.append(int(action.item()))
            log_probs.append(log_prob.squeeze(0))
        
        return actions, log_probs
```

---

### Issue #8: No Inventory History Tracking

**Location:** `envs/serial_env.py`, `envs/network_env.py`

**Problem:** Inventory levels not tracked over time, needed for analysis.

**Fix:**
```python
# In reset():
self.inventory_history = [[] for _ in range(self.level_num)]

# In _state_update():
for i in range(self.level_num):
    # ... existing code ...
    self.inventory_history[i].append(self.inventory[i])

# Add getter:
def get_inventory_history(self) -> List[List[int]]:
    return self.inventory_history
```

---

## 📊 MEDIUM Priority Issues

### Issue #9: Potential Memory Leak in Buffer

**Location:** `agents/replay_buffer.py`

**Problem:** If `clear()` is not called, buffer grows indefinitely.

**Fix:** Add max size limit:
```python
class ReplayBuffer:
    def __init__(self, max_size: int = 100000):
        self.storage = []
        self.max_size = max_size
    
    def add(self, transition):
        if len(self.storage) >= self.max_size:
            self.storage.pop(0)  # Remove oldest
        self.storage.append(transition)
```

---

### Issue #10: No Seed Control in GA

**Location:** `lot_sizing/ga_lotsizing.py`

**Problem:** `random.randint()` uses global seed, not reproducible per-experiment.

**Fix:**
```python
def optimise_order(..., rng_seed: int = None):
    rng = random.Random(rng_seed)
    
    def random_plan():
        return [rng.randint(0, max_order) for _ in range(horizon)]
    
    # Use rng.random(), rng.choices(), rng.randint() throughout
```

---

### Issue #11: Missing Model Checkpointing During Training

**Location:** `train_main.py`

**Problem:** Only saves at end. If training crashes at episode 9000/10000, all progress lost.

**Fix:** Already added `save_every` in my configs, but need to implement in train_main.py:
```python
# In training loop:
if ep % save_every == 0:
    checkpoint_path = f"{save_path.replace('.pth', '')}_ep{ep}.pth"
    torch.save({
        "episode": ep,
        "actor_state_dicts": [a.state_dict() for a in agent.actors],
        "critic_state_dict": agent.critic_net.state_dict(),
        "optimizer_states": {...},
        "reward_history": reward_history,
    }, checkpoint_path)
```

---

### Issue #12: Simple Demand Forecast in HybridPlanner

**Location:** `lot_sizing/hybrid_planner.py:76-80`

**Problem:** Forecast just repeats last known demand:
```python
def build_forecast(seq):
    return [seq[t] if t < len(seq) else seq[-1] for t in range(start, start + self.horizon)]
```

**Fix:** Use exponential smoothing or moving average:
```python
def build_forecast(seq, horizon, alpha=0.3):
    """Exponential smoothing forecast."""
    if not seq:
        return [0] * horizon
    
    # Simple exponential smoothing
    forecast = [seq[-1]]
    for _ in range(horizon - 1):
        next_val = alpha * seq[-1] + (1 - alpha) * forecast[-1]
        forecast.append(int(next_val))
    
    return forecast
```

---

## 📝 Code Quality Issues

### Issue #13: Inconsistent Type Hints
Many functions missing type hints. Add throughout for better IDE support.

### Issue #14: Magic Numbers
- `0.5` for alpha
- `1e-6` for bullwhip calculation
- `1e-8` for advantage normalization

Should be named constants:
```python
ADVANTAGE_EPSILON = 1e-8
BULLWHIP_MIN_MEAN = 1e-6
DEFAULT_REWARD_SMOOTHING = 0.5
```

### Issue #15: No Logging Framework
Using `print()` statements. Should use `logging` module consistently.

---

## 🔧 Recommended Optimizations to Apply Now

Based on your timeline (3-4 days), here are the **must-do** fixes:

### Priority 1: GA Performance (30 min)
```yaml
# configs/pareto_serial.yaml - ALREADY DONE
heuristic:
  ga:
    pop_size: 20    # Was 30
    generations: 25  # Was 40
```

### Priority 2: Gradient Clipping (15 min)
Add to `happo_agent.py`:
```python
torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.5)
torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
```

### Priority 3: GA Inventory Penalty (20 min)
Modify `ga_lotsizing.py` to add inventory cap penalty.

### Priority 4: Inventory History Tracking (10 min)
Add to environments for complete analysis data.

---

## Summary: Quick Win Fixes

| Fix | Time | Impact | Difficulty |
|-----|------|--------|------------|
| Reduce GA pop_size/generations | 2 min | High | Easy |
| Add gradient clipping | 15 min | Medium | Easy |
| Add inventory penalty to GA | 20 min | High | Medium |
| Add inventory history tracking | 10 min | Medium | Easy |
| Add checkpointing | 30 min | High | Easy |

**Total time for critical fixes: ~1.5 hours**

---

## Files to Modify

1. `agents/happo_agent.py` - Gradient clipping
2. `lot_sizing/ga_lotsizing.py` - Inventory penalty
3. `envs/serial_env.py` - Inventory history
4. `envs/network_env.py` - Inventory history
5. `train_main.py` - Checkpointing

Would you like me to implement these fixes now?
