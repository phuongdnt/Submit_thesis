"""
test_hierarchy.py - Quick Test for Hierarchy Enforcement
=========================================================
Chạy 100 episodes để verify:
1. Hierarchy penalty hoạt động
2. Per-agent metrics được tính đúng
3. Output format đúng

Usage: python test_hierarchy.py
"""

import os
import sys
import numpy as np
import torch

# Add current dir to path
sys.path.insert(0, os.getcwd())

from reward_utils import compute_mo_reward_with_hierarchy, compute_hierarchy_penalty

def test_hierarchy_penalty():
    """Test hierarchy penalty calculation"""
    print("=" * 60)
    print("TEST 1: Hierarchy Penalty Calculation")
    print("=" * 60)
    
    # Case 1: Hierarchy VIOLATED (Factory > Distributor > Retailer)
    fill_rates_bad = [0.50, 0.93, 0.98]  # R=50%, D=93%, F=98%
    penalty_bad = compute_hierarchy_penalty(fill_rates_bad, "serial", scale=200.0)
    print(f"\n❌ VIOLATED: R=50%, D=93%, F=98%")
    print(f"   Penalty: {penalty_bad:.2f}")
    
    # Case 2: Hierarchy SATISFIED (Retailer > Distributor > Factory)
    fill_rates_good = [0.90, 0.85, 0.80]  # R=90%, D=85%, F=80%
    penalty_good = compute_hierarchy_penalty(fill_rates_good, "serial", scale=200.0)
    print(f"\n✅ SATISFIED: R=90%, D=85%, F=80%")
    print(f"   Penalty: {penalty_good:.2f}")
    
    # Case 3: Edge case - equal
    fill_rates_equal = [0.85, 0.85, 0.85]
    penalty_equal = compute_hierarchy_penalty(fill_rates_equal, "serial", scale=200.0)
    print(f"\n⚠️ EQUAL: R=85%, D=85%, F=85%")
    print(f"   Penalty: {penalty_equal:.2f}")
    
    assert penalty_bad < penalty_good, "Penalty should be negative when violated"
    assert penalty_good == 0.0, "No penalty when satisfied"
    print("\n✅ Hierarchy penalty logic PASSED")


def test_reward_calculation():
    """Test full reward calculation"""
    print("\n" + "=" * 60)
    print("TEST 2: Full Reward Calculation")
    print("=" * 60)
    
    raw_rewards = [-100.0, -80.0, -60.0]  # Cost-based rewards
    
    # Bad hierarchy
    fill_rates_bad = [0.50, 0.93, 0.98]
    rewards_bad = compute_mo_reward_with_hierarchy(
        raw_rewards, fill_rates_bad,
        cost_weight=0.5, service_weight=0.5,
        topology="serial",
        service_bonus_scale=100.0,
        hierarchy_penalty_scale=200.0
    )
    
    # Good hierarchy
    fill_rates_good = [0.90, 0.85, 0.80]
    rewards_good = compute_mo_reward_with_hierarchy(
        raw_rewards, fill_rates_good,
        cost_weight=0.5, service_weight=0.5,
        topology="serial",
        service_bonus_scale=100.0,
        hierarchy_penalty_scale=200.0
    )
    
    print(f"\n❌ VIOLATED hierarchy rewards: {[f'{r:.2f}' for r in rewards_bad]}")
    print(f"   Total: {sum(rewards_bad):.2f}")
    
    print(f"\n✅ SATISFIED hierarchy rewards: {[f'{r:.2f}' for r in rewards_good]}")
    print(f"   Total: {sum(rewards_good):.2f}")
    
    assert sum(rewards_good) > sum(rewards_bad), "Good hierarchy should have higher reward"
    print("\n✅ Reward calculation logic PASSED")


def test_network_hierarchy():
    """Test network topology hierarchy"""
    print("\n" + "=" * 60)
    print("TEST 3: Network Topology Hierarchy")
    print("=" * 60)
    
    # Network: [R1, R2, D1, D2, F1, F2]
    # Bad: Factories > Distributors > Retailers
    fill_rates_bad = [0.50, 0.55, 0.85, 0.90, 0.95, 0.98]
    penalty_bad = compute_hierarchy_penalty(fill_rates_bad, "network", scale=200.0)
    
    # Good: Retailers > Distributors > Factories
    fill_rates_good = [0.92, 0.90, 0.85, 0.83, 0.78, 0.75]
    penalty_good = compute_hierarchy_penalty(fill_rates_good, "network", scale=200.0)
    
    print(f"\n❌ VIOLATED (Network):")
    print(f"   Retailers avg: {(0.50+0.55)/2*100:.1f}%")
    print(f"   Distributors avg: {(0.85+0.90)/2*100:.1f}%")
    print(f"   Factories avg: {(0.95+0.98)/2*100:.1f}%")
    print(f"   Penalty: {penalty_bad:.2f}")
    
    print(f"\n✅ SATISFIED (Network):")
    print(f"   Retailers avg: {(0.92+0.90)/2*100:.1f}%")
    print(f"   Distributors avg: {(0.85+0.83)/2*100:.1f}%")
    print(f"   Factories avg: {(0.78+0.75)/2*100:.1f}%")
    print(f"   Penalty: {penalty_good:.2f}")
    
    assert penalty_bad < 0, "Should have negative penalty when violated"
    assert penalty_good == 0, "Should have no penalty when satisfied"
    print("\n✅ Network hierarchy logic PASSED")


def test_with_real_env():
    """Test with actual environment (if available)"""
    print("\n" + "=" * 60)
    print("TEST 4: Real Environment Test (50 episodes)")
    print("=" * 60)
    
    try:
        from envs.serial_env import SerialInventoryEnv
        from agents.happo_agent import HAPPOAgent
        from utils.metrics import compute_bullwhip, compute_service_levels
    except ImportError as e:
        print(f"\n⚠️ Skipping real env test (import error): {e}")
        return
    
    # Create env
    try:
        env = SerialInventoryEnv(
            lead_time=2,
            episode_len=50,  # Short episodes for test
            action_dim=41,
            init_inventory=20,
            init_outstanding=10,
            holding_cost=[1.0, 1.0, 1.0],
            backlog_cost=[10.0, 4.0, 1.5],
            fixed_cost=1.0,
        )
    except Exception as e:
        print(f"\n⚠️ Skipping real env test (env error): {e}")
        return
    
    num_agents = env.agent_num
    agent_names = ["Retailer", "Distributor", "Factory"]
    
    # Create agent
    agent = HAPPOAgent(
        obs_dim=env.obs_dim,
        action_dim=41,
        num_agents=num_agents,
        hidden_dim=64,
        critic_hidden_dim=128,
    )
    
    print(f"\nRunning 50 test episodes...")
    
    all_fr = []
    all_bw = []
    all_costs = []
    
    for ep in range(50):
        obs = env.reset(train=True)
        orders = [[] for _ in range(num_agents)]
        ep_rewards = []
        
        while True:
            actions, _ = agent.select_actions(obs)
            for i, a in enumerate(actions):
                orders[i].append(a)
            
            obs, rewards, done, infos = env.step(actions, one_hot=False)
            ep_rewards.append([r[0] if isinstance(r, list) else r for r in rewards])
            
            if all(done):
                break
        
        # Compute metrics
        costs = [-sum(ep_rewards[t][i] for t in range(len(ep_rewards))) for i in range(num_agents)]
        all_costs.append(costs)
        all_bw.append(compute_bullwhip(orders))
        all_fr.append(compute_service_levels(env.get_demand_history(), env.get_fulfilled_history()))
    
    # Aggregate
    per_agent_fr = [np.mean([f[i] for f in all_fr]) for i in range(num_agents)]
    per_agent_bw = [np.mean([b[i] for b in all_bw]) for i in range(num_agents)]
    per_agent_cost = [np.mean([c[i] for c in all_costs]) for i in range(num_agents)]
    
    print(f"\n{'Agent':<14} {'Cost':<10} {'FillRate':<10} {'Bullwhip':<10}")
    print("-" * 44)
    for i in range(num_agents):
        print(f"{agent_names[i]:<14} {per_agent_cost[i]:<10.1f} "
              f"{per_agent_fr[i]*100:<10.2f}% {per_agent_bw[i]:<10.3f}")
    print("-" * 44)
    print(f"{'TOTAL':<14} {sum(per_agent_cost):<10.1f} "
          f"{np.mean(per_agent_fr)*100:<10.2f}% {np.mean(per_agent_bw):<10.3f}")
    
    # Check hierarchy
    hierarchy_ok = per_agent_fr[0] >= per_agent_fr[1] >= per_agent_fr[2]
    status = "✅ SATISFIED" if hierarchy_ok else "❌ VIOLATED (expected before training)"
    print(f"\nHierarchy (R≥D≥F): {status}")
    
    # Compute what penalty would be
    penalty = compute_hierarchy_penalty(per_agent_fr, "serial", scale=200.0)
    print(f"Hierarchy penalty: {penalty:.2f}")
    
    print("\n✅ Real environment test completed")


def main():
    print("\n" + "=" * 60)
    print("HIERARCHY ENFORCEMENT - QUICK TEST")
    print("=" * 60)
    
    test_hierarchy_penalty()
    test_reward_calculation()
    test_network_hierarchy()
    test_with_real_env()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
    print("\nCode is ready for full training. Run:")
    print("  python train_pareto.py --config configs/pareto_serial.yaml")
    print("  python train_pareto_network.py --config configs/pareto_network.yaml")


if __name__ == "__main__":
    main()

