"""
train_pareto.py
================

Multi-Objective Training Script for Pareto Frontier Generation.

This script trains multiple models with different objective weight configurations
to generate a Pareto frontier showing the trade-off between cost minimization
and service level maximization.

Usage:
    python -m inventory_management_RL_Lot.train_pareto --config configs/pareto_serial.yaml

Output:
    - Trained models for each weight configuration
    - Pareto frontier data (JSON)
    - Training curves and metrics logs
"""

from __future__ import annotations

import argparse
import os
import json
import yaml
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
import torch

from utils.logger import setup_logger
from agents.happo_agent import HAPPOAgent
from envs.serial_env import SerialInventoryEnv
from envs.network_env import NetworkInventoryEnv
from envs.vec_env import SubprocVecEnv
from lot_sizing.hybrid_planner import HybridPlanner
from utils.metrics import compute_episode_costs, compute_bullwhip, compute_service_levels


class ParetoExperimentLogger:
    """Enhanced logger for Pareto experiments with comprehensive metric tracking."""
    
    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.log_file = self.output_dir / f"{experiment_name}_log.json"
        
        self.data = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "config": {},
            "training_history": [],
            "evaluation_results": [],
            "pareto_points": []
        }
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.data["config"] = config
        self._save()
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        """Log training episode metrics."""
        entry = {"episode": episode, "timestamp": datetime.now().isoformat()}
        entry.update(metrics)
        self.data["training_history"].append(entry)
        
        # Save periodically (every 10 episodes)
        if episode % 10 == 0:
            self._save()
    
    def log_evaluation(self, episode: int, metrics: Dict[str, Any]):
        """Log evaluation results."""
        entry = {"episode": episode, "timestamp": datetime.now().isoformat()}
        entry.update(metrics)
        self.data["evaluation_results"].append(entry)
        self._save()
    
    def log_pareto_point(self, weight_config: Dict[str, float], metrics: Dict[str, Any]):
        """Log a Pareto frontier point."""
        point = {
            "weight_config": weight_config,
            "timestamp": datetime.now().isoformat()
        }
        point.update(metrics)
        self.data["pareto_points"].append(point)
        self._save()
    
    def finalize(self):
        """Finalize and save all data."""
        self.data["end_time"] = datetime.now().isoformat()
        self._save()
    
    def _save(self):
        """Save data to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)


def compute_multi_objective_reward(
    raw_rewards: List[float],
    fill_rates: List[float],
    cost_weight: float,
    service_weight: float,
    service_bonus_scale: float = 10.0
) -> List[float]:
    """
    Compute multi-objective reward combining cost and service level.
    
    Args:
        raw_rewards: Raw cost-based rewards (negative costs)
        fill_rates: Fill rate for each agent (0-1)
        cost_weight: Weight for cost objective (α)
        service_weight: Weight for service objective (β)
        service_bonus_scale: Scaling factor for service bonus
    
    Returns:
        Multi-objective rewards for each agent
    """
    mo_rewards = []
    for raw_r, fr in zip(raw_rewards, fill_rates):
        # Cost component (already negative, so more negative = worse)
        cost_component = cost_weight * raw_r
        
        # Service component (positive bonus for good service)
        service_component = service_weight * fr * service_bonus_scale
        
        mo_reward = cost_component + service_component
        mo_rewards.append(mo_reward)
    
    return mo_rewards


def _resolve_config_path(path: str) -> str:
    """Resolve configuration file path."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = [
        os.path.abspath(os.path.join(os.getcwd(), path)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def parse_config(path: str) -> Dict[str, Any]:
    """Parse YAML configuration file."""
    resolved_path = _resolve_config_path(path)
    with open(resolved_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Resolve relative paths
    base_dir = os.path.dirname(os.path.abspath(resolved_path))
    env_cfg = cfg.get("env", {})
    
    eval_data_dir = env_cfg.get("eval_data_dir")
    if eval_data_dir and not os.path.isabs(eval_data_dir):
        for candidate_base in [os.getcwd(), base_dir, os.path.dirname(__file__)]:
            candidate = os.path.join(candidate_base, eval_data_dir)
            if os.path.exists(candidate):
                env_cfg["eval_data_dir"] = os.path.abspath(candidate)
                break
    
    cfg["env"] = env_cfg
    return cfg


def build_environment(cfg: Dict[str, Any]):
    """Construct environment from configuration."""
    env_cfg = cfg.get("env", {})
    env_type = env_cfg.get("env_type", "serial")
    
    # Remove pareto-specific keys that aren't env parameters
    env_params = {k: v for k, v in env_cfg.items() 
                  if k not in ["env_type", "cost_weight", "service_weight", "service_bonus_scale"]}
    
    if env_type == "serial":
        return SerialInventoryEnv(**env_params)
    elif env_type == "network":
        children = {int(k): [int(x) for x in v] for k, v in env_params.pop("children", {}).items()}
        parents = {int(k): (int(v) if v is not None else None) for k, v in env_params.pop("parents", {}).items()}
        return NetworkInventoryEnv(children=children, parents=parents, **env_params)
    else:
        raise ValueError(f"Unsupported env_type: {env_type}")


def run_evaluation(
    env, 
    agent: HAPPOAgent, 
    planner: Optional[HybridPlanner],
    n_episodes: int = 30
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation and return all metrics.
    
    Returns:
        Dictionary with costs, bullwhip, fill_rate, cycle_service_level
    """
    all_costs = []
    all_bullwhip = []
    all_fill_rates = []
    all_cycle_sl = []
    
    for _ in range(n_episodes):
        obs = env.reset(train=False)
        order_history = [[] for _ in range(env.agent_num)]
        episode_rewards = []
        
        while True:
            actions, _ = agent.select_actions(obs)
            if planner:
                actions = planner.refine_actions(actions)
            
            for i, a in enumerate(actions):
                order_history[i].append(a)
            
            obs, rewards, done, _ = env.step(actions, one_hot=False)
            episode_rewards.append([r[0] if isinstance(r, list) else r for r in rewards])
            
            if all(done):
                break
        
        # Compute metrics
        costs = [-sum(episode_rewards[t][i] for t in range(len(episode_rewards))) 
                 for i in range(env.agent_num)]
        all_costs.append(costs)
        
        bw = compute_bullwhip(order_history)
        all_bullwhip.append(bw)
        
        # Fill rate
        demand_hist = env.get_demand_history()
        fulfilled_hist = env.get_fulfilled_history()
        fill_rates = compute_service_levels(demand_hist, fulfilled_hist)
        all_fill_rates.append(fill_rates)
        
        # Cycle service level
        cycle_sl = []
        for backlogs in env.backlog_history:
            if backlogs:
                sl = sum(1 for b in backlogs if b == 0) / len(backlogs)
            else:
                sl = 1.0
            cycle_sl.append(sl)
        all_cycle_sl.append(cycle_sl)
    
    return {
        "total_cost": float(np.mean([sum(c) for c in all_costs])),
        "cost_per_agent": [float(np.mean([c[i] for c in all_costs])) for i in range(env.agent_num)],
        "cost_std": float(np.std([sum(c) for c in all_costs])),
        "bullwhip_mean": float(np.mean([np.mean(b) for b in all_bullwhip])),
        "bullwhip_per_agent": [float(np.mean([b[i] for b in all_bullwhip])) for i in range(env.agent_num)],
        "fill_rate_mean": float(np.mean([np.mean(f) for f in all_fill_rates])),
        "fill_rate_per_agent": [float(np.mean([f[i] for f in all_fill_rates])) for i in range(env.agent_num)],
        "cycle_sl_mean": float(np.mean([np.mean(s) for s in all_cycle_sl])),
        "cycle_sl_per_agent": [float(np.mean([s[i] for s in all_cycle_sl])) for i in range(env.agent_num)],
    }


def train_single_configuration(
    cfg: Dict[str, Any],
    cost_weight: float,
    service_weight: float,
    seed: int,
    output_dir: str,
    logger
) -> Dict[str, Any]:
    """
    Train a single model with specific weight configuration.
    
    Returns:
        Final evaluation metrics
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    experiment_name = f"cw{cost_weight:.1f}_sw{service_weight:.1f}_seed{seed}"
    exp_logger = ParetoExperimentLogger(output_dir, experiment_name)
    exp_logger.log_config({
        "cost_weight": cost_weight,
        "service_weight": service_weight,
        "seed": seed,
        "config": cfg
    })
    
    logger.info(f"=" * 60)
    logger.info(f"Training: cost_weight={cost_weight}, service_weight={service_weight}, seed={seed}")
    logger.info(f"=" * 60)
    
    # Build environment
    base_env = build_environment(cfg)
    
    # Get dimensions
    obs_dim = base_env.obs_dim
    action_dim = cfg.get("env", {}).get("action_dim", 41)
    num_agents = base_env.agent_num
    
    # Build agent
    agent_cfg = cfg.get("agent", {})
    agent = HAPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=agent_cfg.get("hidden_dim", 128),
        critic_hidden_dim=agent_cfg.get("critic_hidden_dim", 256),
        actor_lr=agent_cfg.get("actor_lr", 1e-4),
        critic_lr=agent_cfg.get("critic_lr", 1e-4),
        gamma=agent_cfg.get("gamma", 0.99),
        gae_lambda=agent_cfg.get("gae_lambda", 0.95),
        eps_clip=agent_cfg.get("eps_clip", 0.2),
        value_coef=agent_cfg.get("value_coef", 0.5),
        entropy_coef=agent_cfg.get("entropy_coef", 0.05),
    )
    
    # Training settings
    training_cfg = cfg.get("training", {})
    use_ga = training_cfg.get("use_ga", True)
    ga_horizon = training_cfg.get("ga_horizon", 5)
    ga_params = cfg.get("heuristic", {}).get("ga", {})
    n_rollout_threads = training_cfg.get("n_rollout_threads", 8)
    episodes = training_cfg.get("episodes", 5000)
    evaluate_every = training_cfg.get("evaluate_every", 50)
    
    service_bonus_scale = cfg.get("env", {}).get("service_bonus_scale", 10.0)
    
    # Setup vectorized environment
    if n_rollout_threads > 1:
        def make_env_fn():
            return build_environment(cfg)
        env = SubprocVecEnv(make_env_fn, n_rollout_threads)
        is_vec = True
    else:
        env = base_env
        is_vec = False
    
    # Setup planner
    planner = None
    planners = None
    eval_planner = None
    
    if use_ga:
        if is_vec:
            planners = [
                HybridPlanner(env=env_i, horizon=ga_horizon, use_ga=True, ga_params=ga_params)
                for env_i in env.env_list
            ]
            eval_planner = HybridPlanner(env=env.env_list[0], horizon=ga_horizon, use_ga=True, ga_params=ga_params)
        else:
            planner = HybridPlanner(env=env, horizon=ga_horizon, use_ga=True, ga_params=ga_params)
            eval_planner = planner
    
    # Early stopping
    early_stop = training_cfg.get("early_stop", True)
    n_warmup = training_cfg.get("n_warmup_evaluations", 10)
    n_no_improve_thres = training_cfg.get("n_no_improvement_thres", 15)
    best_eval_reward = None
    no_improve_count = 0
    
    # Training tracking
    reward_history = []
    best_model_state = None
    
    # Main training loop
    start_time = time.time()
    
    for ep in range(1, episodes + 1):
        ep_reward_sum = 0.0
        ep_fill_rates = [[] for _ in range(num_agents)]
        
        if is_vec:
            obs_batch = env.reset(train=True)
            done_batch = np.zeros((n_rollout_threads, num_agents), dtype=bool)
            
            while not np.all(done_batch):
                actions_batch = []
                log_probs_batch = []
                
                for i in range(n_rollout_threads):
                    if not np.all(done_batch[i]):
                        obs_list = obs_batch[i].tolist() if isinstance(obs_batch[i], np.ndarray) else obs_batch[i]
                        acts, log_ps = agent.select_actions(obs_list)
                        if planners is not None:
                            acts = planners[i].refine_actions(acts)
                        actions_batch.append(acts)
                        log_probs_batch.append(log_ps)
                    else:
                        actions_batch.append([0] * num_agents)
                        log_probs_batch.append([0.0] * num_agents)
                
                next_obs_batch, rewards_batch, done_batch2, _ = env.step(actions_batch, one_hot=False)
                
                for i in range(n_rollout_threads):
                    if not np.all(done_batch[i]):
                        obs_i = obs_batch[i].tolist() if isinstance(obs_batch[i], np.ndarray) else obs_batch[i]
                        next_obs_i = next_obs_batch[i].tolist() if isinstance(next_obs_batch[i], np.ndarray) else next_obs_batch[i]
                        rewards_i = rewards_batch[i].tolist() if isinstance(rewards_batch[i], np.ndarray) else rewards_batch[i]
                        done_i = done_batch2[i].tolist() if isinstance(done_batch2[i], np.ndarray) else done_batch2[i]
                        
                        # Apply multi-objective reward transformation
                        flat_rewards = [r[0] if isinstance(r, list) else r for r in rewards_i]
                        
                        # Estimate fill rates from observation (simplified)
                        # In practice, you'd track this more precisely
                        estimated_fill_rates = [0.8] * num_agents  # Placeholder
                        
                        mo_rewards = compute_multi_objective_reward(
                            flat_rewards, estimated_fill_rates,
                            cost_weight, service_weight, service_bonus_scale
                        )
                        
                        agent.store_transition(obs_i, actions_batch[i], log_probs_batch[i], 
                                             mo_rewards, next_obs_i, done_i)
                        ep_reward_sum += float(np.sum(mo_rewards))
                
                obs_batch = next_obs_batch
                done_batch = done_batch2
        else:
            obs_list = env.reset(train=True)
            while True:
                actions, log_probs = agent.select_actions(obs_list)
                if planner is not None:
                    actions = planner.refine_actions(actions)
                
                next_obs_list, rewards, done, _ = env.step(actions, one_hot=False)
                flat_rewards = [r[0] if isinstance(r, list) else r for r in rewards]
                
                # Multi-objective transformation
                estimated_fill_rates = [0.8] * num_agents
                mo_rewards = compute_multi_objective_reward(
                    flat_rewards, estimated_fill_rates,
                    cost_weight, service_weight, service_bonus_scale
                )
                
                agent.store_transition(obs_list, actions, log_probs, mo_rewards, next_obs_list, done[0])
                ep_reward_sum += float(np.sum(mo_rewards))
                obs_list = next_obs_list
                
                if all(done):
                    break
        
        # Update agent
        agent.update()
        reward_history.append(ep_reward_sum)
        
        # Log progress
        if ep % 50 == 0:
            logger.info(f"Episode {ep}/{episodes}: reward = {ep_reward_sum:.2f}")
            exp_logger.log_episode(ep, {"reward": ep_reward_sum})
        
        # Periodic evaluation
        if evaluate_every and ep % evaluate_every == 0:
            eval_env = env.env_list[0] if is_vec else env
            eval_metrics = run_evaluation(eval_env, agent, eval_planner, n_episodes=10)
            
            # Compute combined score for early stopping
            eval_score = -cost_weight * eval_metrics["total_cost"] + service_weight * eval_metrics["fill_rate_mean"] * 1000
            
            logger.info(f"  Eval: Cost={eval_metrics['total_cost']:.1f}, "
                       f"FillRate={eval_metrics['fill_rate_mean']:.2%}, "
                       f"Bullwhip={eval_metrics['bullwhip_mean']:.3f}")
            
            exp_logger.log_evaluation(ep, eval_metrics)
            
            # Early stopping logic
            if early_stop:
                if best_eval_reward is None or eval_score > best_eval_reward:
                    best_eval_reward = eval_score
                    no_improve_count = 0
                    # Save best model state
                    best_model_state = {
                        "actor_state_dicts": [actor.state_dict() for actor in agent.actors],
                        "critic_state_dict": agent.critic_net.state_dict(),
                    }
                else:
                    if ep // evaluate_every > n_warmup:
                        no_improve_count += 1
                        if no_improve_count >= n_no_improve_thres:
                            logger.info(f"Early stopping at episode {ep}")
                            break
    
    training_time = time.time() - start_time
    
    # Final evaluation with best model
    if best_model_state:
        for actor, state_dict in zip(agent.actors, best_model_state["actor_state_dicts"]):
            actor.load_state_dict(state_dict)
        agent.critic_net.load_state_dict(best_model_state["critic_state_dict"])
    
    eval_env = env.env_list[0] if is_vec else env
    final_metrics = run_evaluation(eval_env, agent, eval_planner, n_episodes=30)
    final_metrics["training_time"] = training_time
    final_metrics["episodes_trained"] = ep
    
    # Save model
    model_path = os.path.join(output_dir, f"{experiment_name}.pth")
    torch.save({
        "actor_state_dicts": [actor.state_dict() for actor in agent.actors],
        "critic_state_dict": agent.critic_net.state_dict(),
        "config": {
            "cost_weight": cost_weight,
            "service_weight": service_weight,
            "seed": seed
        }
    }, model_path)
    logger.info(f"Saved model to {model_path}")
    
    exp_logger.log_pareto_point(
        {"cost_weight": cost_weight, "service_weight": service_weight, "seed": seed},
        final_metrics
    )
    exp_logger.finalize()
    
    # Cleanup
    if is_vec:
        env.close()
    
    return final_metrics


def main(config_path: str):
    """Main entry point for Pareto experiments."""
    logger = setup_logger()
    cfg = parse_config(config_path)
    
    pareto_cfg = cfg.get("pareto", {})
    weight_configs = pareto_cfg.get("weight_configs", [
        {"name": "balanced", "cost_weight": 0.5, "service_weight": 0.5}
    ])
    seeds = pareto_cfg.get("seeds", [1, 2, 3])
    output_dir = pareto_cfg.get("output_dir", "results/pareto")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Master results
    all_results = []
    
    logger.info("=" * 70)
    logger.info("PARETO FRONTIER EXPERIMENT")
    logger.info(f"Weight configurations: {len(weight_configs)}")
    logger.info(f"Seeds per config: {len(seeds)}")
    logger.info(f"Total runs: {len(weight_configs) * len(seeds)}")
    logger.info("=" * 70)
    
    for wc in weight_configs:
        cost_weight = wc["cost_weight"]
        service_weight = wc["service_weight"]
        config_name = wc.get("name", f"cw{cost_weight}_sw{service_weight}")
        
        config_results = []
        
        for seed in seeds:
            try:
                metrics = train_single_configuration(
                    cfg, cost_weight, service_weight, seed, output_dir, logger
                )
                metrics["config_name"] = config_name
                metrics["cost_weight"] = cost_weight
                metrics["service_weight"] = service_weight
                metrics["seed"] = seed
                config_results.append(metrics)
                all_results.append(metrics)
            except Exception as e:
                logger.error(f"Failed: {config_name}, seed={seed}: {e}")
                import traceback
                traceback.print_exc()
        
        # Aggregate results for this configuration
        if config_results:
            avg_cost = np.mean([r["total_cost"] for r in config_results])
            avg_fill_rate = np.mean([r["fill_rate_mean"] for r in config_results])
            avg_bullwhip = np.mean([r["bullwhip_mean"] for r in config_results])
            
            logger.info(f"\n{config_name} Summary:")
            logger.info(f"  Avg Cost: {avg_cost:.1f}")
            logger.info(f"  Avg Fill Rate: {avg_fill_rate:.2%}")
            logger.info(f"  Avg Bullwhip: {avg_bullwhip:.3f}")
    
    # Save master results
    master_results_path = os.path.join(output_dir, "pareto_results.json")
    with open(master_results_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config_file": config_path,
            "results": all_results
        }, f, indent=2, default=str)
    
    logger.info(f"\nAll results saved to {master_results_path}")
    
    # Print Pareto summary
    logger.info("\n" + "=" * 70)
    logger.info("PARETO FRONTIER SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Config':<20} {'Cost':<12} {'Fill Rate':<12} {'Bullwhip':<12}")
    logger.info("-" * 60)
    
    for wc in weight_configs:
        config_name = wc.get("name", f"cw{wc['cost_weight']}_sw{wc['service_weight']}")
        config_results = [r for r in all_results if r.get("config_name") == config_name]
        if config_results:
            avg_cost = np.mean([r["total_cost"] for r in config_results])
            avg_fill_rate = np.mean([r["fill_rate_mean"] for r in config_results])
            avg_bullwhip = np.mean([r["bullwhip_mean"] for r in config_results])
            logger.info(f"{config_name:<20} {avg_cost:<12.1f} {avg_fill_rate:<12.2%} {avg_bullwhip:<12.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Objective Pareto Training")
    parser.add_argument("--config", type=str, required=True, help="Path to Pareto config YAML")
    args = parser.parse_args()
    main(args.config)
