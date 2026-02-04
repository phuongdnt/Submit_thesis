"""
test_quick.py - Quick Validation Test
======================================
Test reward function and baselines before running full training.

Usage: python test_quick.py
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from reward_utils import compute_mo_reward, compute_cycle_service_level


def test_mo_reward():
    """Test multi-objective reward calculation."""
    print("=" * 50)
    print("TEST: Multi-Objective Reward")
    print("=" * 50)
    
    raw_rewards = [-10.0, -8.0, -5.0]  # Negative costs
    fill_rates = [0.75, 0.85, 0.95]
    
    # Test cost-focused
    rewards_cost = compute_mo_reward(raw_rewards, fill_rates, 
                                     cost_weight=1.0, service_weight=0.0)
    print(f"\nCost-focused (cw=1.0, sw=0.0):")
    print(f"  Rewards: {[f'{r:.2f}' for r in rewards_cost]}")
    
    # Test balanced
    rewards_balanced = compute_mo_reward(raw_rewards, fill_rates,
                                         cost_weight=0.5, service_weight=0.5)
    print(f"\nBalanced (cw=0.5, sw=0.5):")
    print(f"  Rewards: {[f'{r:.2f}' for r in rewards_balanced]}")
    
    # Test service-focused
    rewards_service = compute_mo_reward(raw_rewards, fill_rates,
                                        cost_weight=0.0, service_weight=1.0)
    print(f"\nService-focused (cw=0.0, sw=1.0):")
    print(f"  Rewards: {[f'{r:.2f}' for r in rewards_service]}")
    
    # Verify logic
    assert rewards_cost[0] < rewards_cost[2], "Higher cost should mean lower reward"
    assert rewards_service[0] < rewards_service[2], "Higher FR should mean higher reward"
    print("\n✅ MO Reward test PASSED")


def test_cycle_service_level():
    """Test cycle service level calculation."""
    print("\n" + "=" * 50)
    print("TEST: Cycle Service Level")
    print("=" * 50)
    
    # Agent 0: 80% CSL, Agent 1: 100% CSL, Agent 2: 60% CSL
    backlog_history = [
        [0, 0, 5, 0, 0, 0, 0, 0, 3, 0],  # Agent 0: 8/10 = 80%
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 1: 10/10 = 100%
        [2, 0, 0, 3, 0, 1, 0, 0, 4, 2],  # Agent 2: 6/10 = 60%
    ]
    
    csl = compute_cycle_service_level(backlog_history)
    print(f"\nBacklog history (3 agents, 10 periods):")
    print(f"  Agent 0 CSL: {csl[0]*100:.0f}% (expected: 80%)")
    print(f"  Agent 1 CSL: {csl[1]*100:.0f}% (expected: 100%)")
    print(f"  Agent 2 CSL: {csl[2]*100:.0f}% (expected: 60%)")
    
    assert abs(csl[0] - 0.80) < 0.01, f"Agent 0 CSL should be 80%, got {csl[0]*100}%"
    assert abs(csl[1] - 1.00) < 0.01, f"Agent 1 CSL should be 100%, got {csl[1]*100}%"
    assert abs(csl[2] - 0.60) < 0.01, f"Agent 2 CSL should be 60%, got {csl[2]*100}%"
    print("\n✅ Cycle Service Level test PASSED")


def test_baselines():
    """Test baseline policies."""
    print("\n" + "=" * 50)
    print("TEST: Baseline Policies")
    print("=" * 50)
    
    from baselines import BaseStockPolicy, sSPolicy
    
    # Test BaseStockPolicy
    policy = BaseStockPolicy(target_levels=[30.0, 25.0, 20.0])
    actions = policy.get_actions([10, 15, 20], [5, 5, 0])
    print(f"\nBase-stock policy:")
    print(f"  Target levels: [30, 25, 20]")
    print(f"  Inventory: [10, 15, 20], Outstanding: [5, 5, 0]")
    print(f"  Actions: {actions}")
    assert actions == [15, 5, 0], f"Expected [15, 5, 0], got {actions}"
    
    # Test sSPolicy
    policy = sSPolicy(reorder_points=[15, 12, 10], target_levels=[35, 30, 25])
    actions = policy.get_actions([10, 15, 20], [5, 5, 0])
    print(f"\n(s,S) policy:")
    print(f"  Reorder points: [15, 12, 10]")
    print(f"  Target levels: [35, 30, 25]")
    print(f"  Inventory position: [15, 20, 20]")
    print(f"  Actions: {actions}")
    # Agent 0: pos=15 <= s=15, order to S=35, action=20
    # Agent 1: pos=20 > s=12, no order
    # Agent 2: pos=20 > s=10, no order
    assert actions == [20, 0, 0], f"Expected [20, 0, 0], got {actions}"
    
    print("\n✅ Baseline Policies test PASSED")


def test_pareto_configs():
    """Test different Pareto weight configurations."""
    print("\n" + "=" * 50)
    print("TEST: Pareto Configurations")
    print("=" * 50)
    
    configs = [
        ("cost_focused", 1.0, 0.0),
        ("cost_leaning", 0.7, 0.3),
        ("balanced", 0.5, 0.5),
        ("service_leaning", 0.3, 0.7),
        ("service_focused", 0.0, 1.0),
    ]
    
    raw_rewards = [-10.0, -8.0, -5.0]
    fill_rates = [0.75, 0.85, 0.95]
    
    print(f"\n{'Config':<18} {'Total Reward':<15} {'Interpretation'}")
    print("-" * 55)
    
    for name, cw, sw in configs:
        rewards = compute_mo_reward(raw_rewards, fill_rates, cw, sw)
        total = sum(rewards)
        
        if cw > sw:
            interp = "Favors low cost"
        elif sw > cw:
            interp = "Favors high service"
        else:
            interp = "Balanced"
        
        print(f"{name:<18} {total:<15.2f} {interp}")
    
    print("\n✅ Pareto Configurations test PASSED")


def main():
    print("\n" + "=" * 50)
    print("LIU + PARETO CLEAN - VALIDATION TESTS")
    print("=" * 50)
    
    test_mo_reward()
    test_cycle_service_level()
    test_baselines()
    test_pareto_configs()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✅")
    print("=" * 50)
    print("\nReady for training! Run:")
    print("  Terminal 1: python train_pareto.py --config configs/pareto_serial.yaml")
    print("  Terminal 2: python train_pareto_network.py --config configs/pareto_network.yaml")
    print("\nAfter training, run evaluation:")
    print("  python evaluate_all.py --serial results/pareto_serial")


if __name__ == "__main__":
    main()
