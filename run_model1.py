"""
run_model1.py - Model 1: Hierarchical Rule Selection Baseline
==============================================================

Script chạy Model 1 baseline để so sánh với:
- Model 2: Pure HAPPO
- Model 3: Hybrid HAPPO + GA

Usage:
    python run_model1.py --config configs/pareto_serial.yaml
    python run_model1.py --topology serial --episodes 30
    python run_model1.py --topology network --episodes 30

Output:
    - results/model1_serial/model1_results.json
    - results/model1_serial/model1_summary.txt
    - Console output với format giống các model khác
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


# ============================================================================
# CLASSICAL LOT-SIZING HEURISTICS
# ============================================================================

class LotSizingRule(Enum):
    """Available lot-sizing heuristics."""
    FOQ = "Fixed Order Quantity"
    POQ = "Periodic Order Quantity"
    SILVER_MEAL = "Silver-Meal"
    LUC = "Least Unit Cost"
    PPB = "Part Period Balancing"
    EOQ = "Economic Order Quantity"


@dataclass
class HeuristicParams:
    """Parameters for lot-sizing heuristics."""
    holding_cost: float = 1.0
    fixed_cost: float = 1.0
    backlog_cost: float = 5.0
    lead_time: int = 2
    review_period: int = 3
    service_level: float = 0.95


class ClassicalLotSizingHeuristics:
    """Collection of classical lot-sizing heuristics."""
    
    @staticmethod
    def compute_eoq(mean_demand: float, holding_cost: float, fixed_cost: float,
                    periods_per_year: int = 100) -> float:
        """Compute Economic Order Quantity (EOQ)."""
        if holding_cost <= 0 or fixed_cost <= 0:
            return mean_demand
        annual_demand = mean_demand * periods_per_year
        eoq = np.sqrt(2 * fixed_cost * annual_demand / holding_cost)
        return max(1, eoq)
    
    @staticmethod
    def fixed_order_quantity(current_inventory: float, backlog: float,
                            order_quantity: float, reorder_point: float) -> float:
        """FOQ: Order fixed quantity when inventory falls below reorder point."""
        inventory_position = current_inventory - backlog
        if inventory_position <= reorder_point:
            return order_quantity
        return 0.0
    
    @staticmethod
    def periodic_order_quantity(current_inventory: float, backlog: float,
                               demand_forecast: List[float], target_periods: int = 3) -> float:
        """POQ: Order enough to cover demand for target_periods periods."""
        inventory_position = current_inventory - backlog
        forecast_sum = sum(demand_forecast[:target_periods])
        order = max(0, forecast_sum - inventory_position)
        return order
    
    @staticmethod
    def silver_meal(current_inventory: float, backlog: float,
                   demand_forecast: List[float], holding_cost: float, fixed_cost: float) -> float:
        """Silver-Meal Heuristic: Minimize average cost per period."""
        if not demand_forecast or fixed_cost <= 0:
            return sum(demand_forecast[:1]) if demand_forecast else 0
        
        inventory_position = current_inventory - backlog
        best_periods = 1
        min_avg_cost = float('inf')
        cumulative_demand = 0
        cumulative_holding = 0
        
        for k, demand in enumerate(demand_forecast):
            cumulative_demand += demand
            cumulative_holding += demand * holding_cost * k
            total_cost = fixed_cost + cumulative_holding
            avg_cost = total_cost / (k + 1)
            
            if avg_cost < min_avg_cost:
                min_avg_cost = avg_cost
                best_periods = k + 1
            else:
                break
        
        order_qty = sum(demand_forecast[:best_periods])
        order = max(0, order_qty - inventory_position)
        return order
    
    @staticmethod
    def least_unit_cost(current_inventory: float, backlog: float,
                       demand_forecast: List[float], holding_cost: float, fixed_cost: float) -> float:
        """Least Unit Cost (LUC): Minimize cost per unit ordered."""
        if not demand_forecast or fixed_cost <= 0:
            return sum(demand_forecast[:1]) if demand_forecast else 0
        
        inventory_position = current_inventory - backlog
        best_periods = 1
        min_unit_cost = float('inf')
        cumulative_demand = 0
        cumulative_holding = 0
        
        for k, demand in enumerate(demand_forecast):
            cumulative_demand += demand
            cumulative_holding += demand * holding_cost * k
            
            if cumulative_demand > 0:
                total_cost = fixed_cost + cumulative_holding
                unit_cost = total_cost / cumulative_demand
                
                if unit_cost < min_unit_cost:
                    min_unit_cost = unit_cost
                    best_periods = k + 1
                else:
                    break
        
        order_qty = sum(demand_forecast[:best_periods])
        order = max(0, order_qty - inventory_position)
        return order
    
    @staticmethod
    def part_period_balancing(current_inventory: float, backlog: float,
                             demand_forecast: List[float], holding_cost: float, fixed_cost: float) -> float:
        """Part Period Balancing (PPB): Balance holding cost with ordering cost."""
        if not demand_forecast or fixed_cost <= 0:
            return sum(demand_forecast[:1]) if demand_forecast else 0
        
        inventory_position = current_inventory - backlog
        cumulative_demand = 0
        cumulative_holding = 0
        periods_to_cover = 0
        
        for k, demand in enumerate(demand_forecast):
            cumulative_demand += demand
            cumulative_holding += demand * holding_cost * k
            periods_to_cover = k + 1
            
            if cumulative_holding >= fixed_cost:
                break
        
        order_qty = sum(demand_forecast[:periods_to_cover])
        order = max(0, order_qty - inventory_position)
        return order


# ============================================================================
# META-LEARNER RULE SELECTOR
# ============================================================================

class MetaLearnerRuleSelector:
    """Meta-learner that selects the best lot-sizing rule based on current state."""
    
    def __init__(self, params: HeuristicParams, safety_factor: float = 1.5,
                 volatility_threshold: float = 0.3):
        self.params = params
        self.safety_factor = safety_factor
        self.volatility_threshold = volatility_threshold
        self.heuristics = ClassicalLotSizingHeuristics()
        self.rule_usage_count = {rule: 0 for rule in LotSizingRule}
        
    def _compute_demand_stats(self, demand_forecast: List[float]) -> tuple:
        if not demand_forecast:
            return 0.0, 0.0, 0.0
        mean_demand = np.mean(demand_forecast)
        std_demand = np.std(demand_forecast)
        cv = std_demand / mean_demand if mean_demand > 0 else 0.0
        return mean_demand, std_demand, cv
    
    def _assess_urgency(self, inventory: float, backlog: float,
                       mean_demand: float, lead_time: int) -> float:
        safety_stock = mean_demand * lead_time * self.safety_factor
        inventory_position = inventory - backlog
        
        if inventory_position <= 0:
            return 1.0
        elif inventory_position < safety_stock * 0.5:
            return 0.8
        elif inventory_position < safety_stock:
            return 0.5
        else:
            return 0.2
    
    def select_rule(self, inventory: float, backlog: float,
                   demand_forecast: List[float]) -> LotSizingRule:
        mean_demand, std_demand, cv = self._compute_demand_stats(demand_forecast)
        urgency = self._assess_urgency(inventory, backlog, mean_demand, self.params.lead_time)
        
        if urgency >= 0.8:
            selected_rule = LotSizingRule.SILVER_MEAL
        elif cv > self.volatility_threshold:
            selected_rule = LotSizingRule.LUC
        elif cv < 0.1:
            selected_rule = LotSizingRule.FOQ
        elif urgency >= 0.5:
            selected_rule = LotSizingRule.PPB
        else:
            selected_rule = LotSizingRule.POQ
        
        self.rule_usage_count[selected_rule] += 1
        return selected_rule
    
    def compute_order(self, inventory: float, backlog: float,
                     demand_forecast: List[float], max_order: int = 40) -> int:
        rule = self.select_rule(inventory, backlog, demand_forecast)
        mean_demand, _, _ = self._compute_demand_stats(demand_forecast)
        
        if rule == LotSizingRule.FOQ:
            safety_stock = mean_demand * self.params.lead_time * self.safety_factor
            eoq = self.heuristics.compute_eoq(mean_demand, self.params.holding_cost, self.params.fixed_cost)
            order = self.heuristics.fixed_order_quantity(inventory, backlog, eoq, safety_stock)
        elif rule == LotSizingRule.POQ:
            order = self.heuristics.periodic_order_quantity(inventory, backlog, demand_forecast, self.params.review_period)
        elif rule == LotSizingRule.SILVER_MEAL:
            order = self.heuristics.silver_meal(inventory, backlog, demand_forecast, self.params.holding_cost, self.params.fixed_cost)
        elif rule == LotSizingRule.LUC:
            order = self.heuristics.least_unit_cost(inventory, backlog, demand_forecast, self.params.holding_cost, self.params.fixed_cost)
        elif rule == LotSizingRule.PPB:
            order = self.heuristics.part_period_balancing(inventory, backlog, demand_forecast, self.params.holding_cost, self.params.fixed_cost)
        else:
            eoq = self.heuristics.compute_eoq(mean_demand, self.params.holding_cost, self.params.fixed_cost)
            safety_stock = mean_demand * self.params.lead_time
            order = self.heuristics.fixed_order_quantity(inventory, backlog, eoq, safety_stock)
        
        return int(max(0, min(order, max_order)))
    
    def get_rule_statistics(self) -> Dict[str, int]:
        return {rule.value: count for rule, count in self.rule_usage_count.items()}


# ============================================================================
# MODEL 1 POLICY
# ============================================================================

class HierarchicalRulePolicy:
    """Complete Model 1 Policy: Hierarchical Rule Selection."""
    
    def __init__(self, n_agents: int = 3, holding_cost: Optional[List[float]] = None,
                 backlog_cost: Optional[List[float]] = None, fixed_cost: float = 1.0,
                 lead_time: int = 2, forecast_horizon: int = 5, action_dim: int = 41):
        self.n_agents = n_agents
        self.holding_cost = holding_cost or [1.0] * n_agents
        self.backlog_cost = backlog_cost or [5.0, 3.0, 2.0][:n_agents]
        self.fixed_cost = fixed_cost
        self.lead_time = lead_time
        self.forecast_horizon = forecast_horizon
        self.action_dim = action_dim
        
        self.meta_learners = []
        for i in range(n_agents):
            params = HeuristicParams(
                holding_cost=self.holding_cost[i],
                fixed_cost=self.fixed_cost,
                backlog_cost=self.backlog_cost[i],
                lead_time=lead_time,
            )
            self.meta_learners.append(MetaLearnerRuleSelector(params))
        
        self.demand_history = [[] for _ in range(n_agents)]
    
    def update_demand_history(self, demands: List[float]):
        for i, d in enumerate(demands):
            self.demand_history[i].append(d)
            if len(self.demand_history[i]) > 100:
                self.demand_history[i] = self.demand_history[i][-100:]
    
    def get_demand_forecast(self, agent_idx: int, horizon: int) -> List[float]:
        history = self.demand_history[agent_idx]
        if not history:
            return [10.0] * horizon
        
        window = min(10, len(history))
        avg_demand = np.mean(history[-window:])
        
        if len(history) >= 2:
            trend = history[-1] - history[-2]
            forecast = [max(0, avg_demand + trend * (i * 0.1)) for i in range(horizon)]
        else:
            forecast = [avg_demand] * horizon
        
        return forecast
    
    def get_action(self, inventory: float, backlog: float, agent_idx: int) -> int:
        forecast = self.get_demand_forecast(agent_idx, self.forecast_horizon)
        return self.meta_learners[agent_idx].compute_order(
            inventory, backlog, forecast, self.action_dim - 1
        )
    
    def get_actions(self, inventories: List[float], backlogs: List[float],
                   outstandings: Optional[List[float]] = None) -> List[int]:
        actions = []
        for i in range(self.n_agents):
            action = self.get_action(inventories[i], backlogs[i], i)
            actions.append(action)
        return actions
    
    def get_rule_statistics(self) -> Dict[str, Dict[str, int]]:
        return {f"agent_{i}": ml.get_rule_statistics() for i, ml in enumerate(self.meta_learners)}


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_bullwhip(order_history: List[List[int]]) -> List[float]:
    """Compute bullwhip coefficients from order history."""
    bullwhip = []
    for orders in order_history:
        if not orders or np.mean(orders) < 1e-6:
            bullwhip.append(0.0)
        else:
            cv = float(np.std(orders) / np.mean(orders))
            bullwhip.append(cv)
    return bullwhip


def compute_service_levels(demand_history: List[List[int]], fulfilled_history: List[List[int]]) -> List[float]:
    """Compute service level (fill rate) for each agent."""
    service_levels = []
    for i in range(len(demand_history)):
        total_demand = sum(demand_history[i])
        total_fulfilled = sum(fulfilled_history[i])
        if total_demand > 0:
            sl = max(0.0, min(1.0, total_fulfilled / total_demand))
        else:
            sl = 1.0
        service_levels.append(sl)
    return service_levels


def compute_cycle_service_level(backlog_history: List[List[int]]) -> List[float]:
    """Compute Type 2 Service Level (Cycle Service Level)."""
    service_levels = []
    for backlogs in backlog_history:
        if not backlogs:
            service_levels.append(1.0)
            continue
        total_periods = len(backlogs)
        periods_without_stockout = sum(1 for b in backlogs if b == 0)
        sl = periods_without_stockout / total_periods
        service_levels.append(sl)
    return service_levels


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model1(env, policy: HierarchicalRulePolicy, n_episodes: int = 30,
                   verbose: bool = True) -> Dict[str, Any]:
    """Evaluate Model 1 on the environment (supports both serial and network)."""
    
    all_costs = []
    all_fill_rates = []
    all_bullwhips = []
    all_csl = []
    
    num_agents = env.agent_num
    is_network = hasattr(env, 'leaf_indices')  # Network env has leaf_indices
    
    for ep in range(n_episodes):
        obs = env.reset(train=False)
        orders = [[] for _ in range(num_agents)]
        ep_costs = [0.0] * num_agents
        
        policy.demand_history = [[] for _ in range(num_agents)]
        
        step = 0
        while True:
            inventories = env.inventory
            backlogs = env.backlog
            
            # Update demand history - different logic for serial vs network
            if is_network:
                # Network env: use external_demand_list for retailers
                if step == 0:
                    demands = [0] * num_agents
                    for i, leaf_idx in enumerate(env.leaf_indices):
                        if hasattr(env, 'external_demand_list') and env.external_demand_list:
                            demands[leaf_idx] = env.external_demand_list[i][0] if env.external_demand_list[i] else 10
                        else:
                            demands[leaf_idx] = 10
                else:
                    demands = [0] * num_agents
                    for i, leaf_idx in enumerate(env.leaf_indices):
                        if hasattr(env, 'external_demand_list') and env.external_demand_list:
                            dem_list = env.external_demand_list[i]
                            demands[leaf_idx] = dem_list[step-1] if step-1 < len(dem_list) else dem_list[-1] if dem_list else 10
                        else:
                            demands[leaf_idx] = 10
                    # For non-leaf nodes, use downstream orders
                    for i in range(num_agents):
                        if i not in env.leaf_indices:
                            child_orders = [orders[c][-1] if orders[c] else 0 for c in env.child_ids[i]]
                            demands[i] = sum(child_orders)
            else:
                # Serial env: use demand_list
                if step == 0:
                    demands = [env.demand_list[0] if hasattr(env, 'demand_list') and env.demand_list else 10]
                    demands += [0] * (num_agents - 1)
                else:
                    dem_idx = step - 1
                    if hasattr(env, 'demand_list') and env.demand_list:
                        demands = [env.demand_list[dem_idx] if dem_idx < len(env.demand_list) else env.demand_list[-1]]
                    else:
                        demands = [10]
                    for i in range(1, num_agents):
                        demands.append(orders[i-1][-1] if orders[i-1] else 0)
            
            policy.update_demand_history(demands)
            
            actions = policy.get_actions(inventories, backlogs)
            
            for i, a in enumerate(actions):
                orders[i].append(a)
            
            obs, rewards, done, infos = env.step(actions, one_hot=False)
            step += 1
            
            for i, r in enumerate(rewards):
                cost = -r[0] if isinstance(r, list) else -r
                ep_costs[i] += cost
            
            if all(done):
                break
        
        all_costs.append(ep_costs)
        all_bullwhips.append(compute_bullwhip(orders))
        all_fill_rates.append(compute_service_levels(
            env.get_demand_history(), env.get_fulfilled_history()
        ))
        
        csl = []
        for agent_backlogs in env.backlog_history:
            if agent_backlogs:
                zero_periods = sum(1 for b in agent_backlogs if b == 0)
                csl.append(zero_periods / len(agent_backlogs))
            else:
                csl.append(1.0)
        all_csl.append(csl)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: Cost={sum(ep_costs):.1f}, "
                  f"FR={np.mean(all_fill_rates[-1])*100:.1f}%")
    
    # Aggregate
    per_agent_cost = [np.mean([c[i] for c in all_costs]) for i in range(num_agents)]
    per_agent_fr = [np.mean([f[i] for f in all_fill_rates]) for i in range(num_agents)]
    per_agent_bw = [np.mean([b[i] for b in all_bullwhips]) for i in range(num_agents)]
    per_agent_csl = [np.mean([s[i] for s in all_csl]) for i in range(num_agents)]
    
    return {
        "model_name": "Model 1: Hierarchical Rule Selection",
        "total_cost": float(sum(per_agent_cost)),
        "cost_std": float(np.std([sum(c) for c in all_costs])),
        "fill_rate_mean": float(np.mean(per_agent_fr)),
        "service_level_mean": float(np.mean(per_agent_csl)),
        "bullwhip_mean": float(np.mean(per_agent_bw)),
        "cost_per_agent": [float(c) for c in per_agent_cost],
        "fill_rate_per_agent": [float(f) for f in per_agent_fr],
        "service_level_per_agent": [float(s) for s in per_agent_csl],
        "bullwhip_per_agent": [float(b) for b in per_agent_bw],
        "rule_statistics": policy.get_rule_statistics(),
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run Model 1: Hierarchical Rule Selection Baseline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--topology", type=str, default="serial", choices=["serial", "network"])
    parser.add_argument("--episodes", type=int, default=30, help="Number of evaluation episodes")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-set if None)")
    parser.add_argument("--verbose", action="store_true", default=True)
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        # Try relative paths
        for candidate in [
            config_path,
            os.path.join(os.path.dirname(__file__), config_path),
            os.path.join(os.getcwd(), config_path)
        ]:
            if os.path.exists(candidate):
                config_path = candidate
                break
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(path: str, base_dirs: List[str]) -> str:
    """Resolve relative path against multiple base directories."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for base in base_dirs:
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return path


def create_serial_env(env_cfg: Dict):
    """Create serial supply chain environment."""
    from envs.serial_env import SerialInventoryEnv
    
    eval_data_dir = env_cfg.get("eval_data_dir", "test_data/test_demand_merton")
    eval_data_dir = resolve_path(eval_data_dir, [os.getcwd(), os.path.dirname(__file__)])
    
    return SerialInventoryEnv(
        level_num=env_cfg.get("level_num", env_cfg.get("agent_num", 3)),
        lead_time=env_cfg.get("lead_time", 2),
        episode_len=env_cfg.get("episode_len", 100),
        action_dim=env_cfg.get("action_dim", 41),
        init_inventory=env_cfg.get("init_inventory", 20),
        init_outstanding=env_cfg.get("init_outstanding", 10),
        holding_cost=env_cfg.get("holding_cost", [1.0, 1.0, 1.0]),
        backlog_cost=env_cfg.get("backlog_cost", [5.0, 3.0, 2.0]),
        fixed_cost=env_cfg.get("fixed_cost", 1.0),
        eval_data_dir=eval_data_dir
    )


def create_network_env(env_cfg: Dict):
    """Create network supply chain environment."""
    from envs.network_env import NetworkInventoryEnv
    
    # Parse topology from config
    children_raw = env_cfg.get("children", {0: [], 1: [], 2: [0], 3: [1], 4: [2], 5: [3]})
    parents_raw = env_cfg.get("parents", {0: 2, 1: 3, 2: 4, 3: 5, 4: None, 5: None})
    
    # Convert string keys to int if necessary
    children = {int(k): [int(x) for x in v] for k, v in children_raw.items()}
    parents = {int(k): (int(v) if v is not None else None) for k, v in parents_raw.items()}
    
    # Resolve eval_data_dirs
    eval_data_dirs_raw = env_cfg.get("eval_data_dirs", ["test_data/test_demand_merton", "test_data/test_demand_merton"])
    eval_data_dirs = [resolve_path(d, [os.getcwd(), os.path.dirname(__file__)]) for d in eval_data_dirs_raw]
    
    n_agents = len(children)
    
    return NetworkInventoryEnv(
        children=children,
        parents=parents,
        lead_time=env_cfg.get("lead_time", 2),
        episode_len=env_cfg.get("episode_len", 100),
        action_dim=env_cfg.get("action_dim", 41),
        init_inventory=env_cfg.get("init_inventory", 20),
        init_outstanding=env_cfg.get("init_outstanding", 10),
        holding_cost=env_cfg.get("holding_cost", [1.0] * n_agents),
        backlog_cost=env_cfg.get("backlog_cost", [5.0, 5.0, 3.0, 3.0, 2.0, 2.0][:n_agents]),
        fixed_cost=env_cfg.get("fixed_cost", 1.0),
        eval_data_dirs=eval_data_dirs
    )


def main():
    args = parse_args()
    
    print("=" * 70)
    print("MODEL 1: HIERARCHICAL RULE SELECTION BASELINE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load config if provided
    if args.config:
        print(f"\nLoading config from: {args.config}")
        config = load_config(args.config)
        env_cfg = config.get("env", {})
        # Auto-detect topology from config
        topology = env_cfg.get("env_type", args.topology)
    else:
        topology = args.topology
        if topology == "serial":
            env_cfg = {
                "level_num": 3,
                "lead_time": 2,
                "episode_len": 100,
                "action_dim": 41,
                "init_inventory": 20,
                "init_outstanding": 10,
                "holding_cost": [1.0, 1.0, 1.0],
                "backlog_cost": [5.0, 3.0, 2.0],
                "fixed_cost": 1.0,
                "eval_data_dir": "test_data/test_demand_merton"
            }
        else:  # network
            env_cfg = {
                "children": {0: [], 1: [], 2: [0], 3: [1], 4: [2], 5: [3]},
                "parents": {0: 2, 1: 3, 2: 4, 3: 5, 4: None, 5: None},
                "lead_time": 2,
                "episode_len": 100,
                "action_dim": 41,
                "init_inventory": 20,
                "init_outstanding": 10,
                "holding_cost": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "backlog_cost": [5.0, 5.0, 3.0, 3.0, 2.0, 2.0],
                "fixed_cost": 1.0,
                "eval_data_dirs": ["test_data/test_demand_merton", "test_data/test_demand_merton"]
            }
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"results/model1_{topology}"
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Topology: {topology}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Output: {output_dir}")
    print(f"  Holding cost: {env_cfg.get('holding_cost')}")
    print(f"  Backlog cost: {env_cfg.get('backlog_cost')}")
    
    # Create environment based on topology
    print("\nCreating environment...")
    try:
        if topology == "network":
            env = create_network_env(env_cfg)
            print(f"  Type: Network (2x3 parallel chains)")
        else:
            env = create_serial_env(env_cfg)
            print(f"  Type: Serial (3-echelon chain)")
    except ImportError as e:
        print(f"Error: Could not import environment: {e}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)
    
    print(f"  Agents: {env.agent_num}")
    print(f"  Episode length: {env.episode_len}")
    print(f"  Action space: {env.action_dim}")
    print(f"  Eval scenarios: {env.n_eval}")
    
    # Create policy
    print("\nCreating Model 1 policy...")
    policy = HierarchicalRulePolicy(
        n_agents=env.agent_num,
        holding_cost=env.holding_cost,
        backlog_cost=env.backlog_cost,
        fixed_cost=env.fixed_cost,
        lead_time=env.lead_time,
        action_dim=env.action_dim,
    )
    
    # Run evaluation
    print(f"\nRunning evaluation ({args.episodes} episodes)...")
    print("-" * 50)
    results = evaluate_model1(env, policy, n_episodes=args.episodes, verbose=args.verbose)
    results["topology"] = topology
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTotal Cost: {results['total_cost']:.1f} ± {results['cost_std']:.1f}")
    print(f"Fill Rate (Type 1): {results['fill_rate_mean']*100:.2f}%")
    print(f"Cycle Service Level (Type 2): {results['service_level_mean']*100:.2f}%")
    print(f"Bullwhip Effect: {results['bullwhip_mean']:.3f}")
    
    print("\nPer-Agent Breakdown:")
    print(f"  {'Agent':<15} {'Cost':>10} {'Fill Rate':>12} {'CSL':>10} {'Bullwhip':>10}")
    print("  " + "-" * 59)
    
    # Agent names based on topology
    if topology == "network":
        agent_names = ["Retailer 1", "Retailer 2", "Distributor 1", "Distributor 2", "Factory 1", "Factory 2"]
    else:
        agent_names = ["Retailer", "Distributor", "Manufacturer"]
    
    for i in range(env.agent_num):
        name = agent_names[i] if i < len(agent_names) else f"Agent {i}"
        cost = results['cost_per_agent'][i]
        fr = results['fill_rate_per_agent'][i] * 100
        csl = results['service_level_per_agent'][i] * 100
        bw = results['bullwhip_per_agent'][i]
        print(f"  {name:<15} {cost:>10.1f} {fr:>11.2f}% {csl:>9.2f}% {bw:>10.3f}")
    
    print("\nRule Usage Statistics:")
    for agent, stats in results['rule_statistics'].items():
        print(f"  {agent}:")
        total = sum(stats.values())
        for rule, count in stats.items():
            if count > 0:
                pct = count / total * 100 if total > 0 else 0
                print(f"    {rule}: {count} ({pct:.1f}%)")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    results_file = output_path / "model1_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    # Save summary text
    summary_file = output_path / "model1_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL 1: HIERARCHICAL RULE SELECTION - RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Topology: {topology}\n")
        f.write(f"Episodes: {args.episodes}\n\n")
        f.write(f"Total Cost: {results['total_cost']:.1f} ± {results['cost_std']:.1f}\n")
        f.write(f"Fill Rate: {results['fill_rate_mean']*100:.2f}%\n")
        f.write(f"Cycle Service Level: {results['service_level_mean']*100:.2f}%\n")
        f.write(f"Bullwhip Effect: {results['bullwhip_mean']:.3f}\n")
    print(f"Summary saved to: {summary_file}")
    
    print("\n" + "=" * 70)
    print("✅ Model 1 evaluation complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()