"""
analyze_pareto.py
==================

Analyze Pareto experiment results and generate visualizations.

This script reads the results from Pareto experiments and produces:
1. Pareto frontier plot (Cost vs Fill Rate)
2. Trade-off analysis tables
3. Statistical significance tests
4. LaTeX-ready tables for thesis

Usage:
    python -m inventory_management_RL_Lot.analyze_pareto --results results/pareto_serial/pareto_results.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_results(results_path: str) -> Dict[str, Any]:
    """Load Pareto experiment results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def aggregate_by_config(results: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate results by weight configuration.
    
    Returns:
        Dictionary mapping config_name to aggregated metrics with mean and std.
    """
    configs = {}
    
    for r in results:
        name = r.get("config_name", "unknown")
        if name not in configs:
            configs[name] = {
                "cost_weight": r.get("cost_weight", 0),
                "service_weight": r.get("service_weight", 0),
                "costs": [],
                "fill_rates": [],
                "bullwhips": [],
                "cycle_sls": [],
                "training_times": []
            }
        
        configs[name]["costs"].append(r["total_cost"])
        configs[name]["fill_rates"].append(r["fill_rate_mean"])
        configs[name]["bullwhips"].append(r["bullwhip_mean"])
        if "cycle_sl_mean" in r:
            configs[name]["cycle_sls"].append(r["cycle_sl_mean"])
        if "training_time" in r:
            configs[name]["training_times"].append(r["training_time"])
    
    # Compute statistics
    for name, data in configs.items():
        data["cost_mean"] = float(np.mean(data["costs"]))
        data["cost_std"] = float(np.std(data["costs"]))
        data["fill_rate_mean"] = float(np.mean(data["fill_rates"]))
        data["fill_rate_std"] = float(np.std(data["fill_rates"]))
        data["bullwhip_mean"] = float(np.mean(data["bullwhips"]))
        data["bullwhip_std"] = float(np.std(data["bullwhips"]))
        
        if data["cycle_sls"]:
            data["cycle_sl_mean"] = float(np.mean(data["cycle_sls"]))
            data["cycle_sl_std"] = float(np.std(data["cycle_sls"]))
        
        if data["training_times"]:
            data["training_time_mean"] = float(np.mean(data["training_times"]))
        
        data["n_seeds"] = len(data["costs"])
    
    return configs


def identify_pareto_frontier(configs: Dict[str, Dict]) -> List[str]:
    """
    Identify configurations on the Pareto frontier.
    
    A point is Pareto-optimal if no other point dominates it
    (i.e., better in cost AND better in fill rate).
    
    Returns:
        List of config names on the Pareto frontier.
    """
    points = [(name, data["cost_mean"], data["fill_rate_mean"]) 
              for name, data in configs.items()]
    
    pareto_frontier = []
    
    for name, cost, fill_rate in points:
        is_dominated = False
        for other_name, other_cost, other_fill_rate in points:
            if other_name == name:
                continue
            # Check if other dominates this point
            # (lower cost AND higher fill rate)
            if other_cost <= cost and other_fill_rate >= fill_rate:
                if other_cost < cost or other_fill_rate > fill_rate:
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_frontier.append(name)
    
    return pareto_frontier


def compute_statistical_tests(configs: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Perform statistical tests between configurations.
    
    Returns:
        Dictionary with test results.
    """
    if not HAS_SCIPY:
        return {"error": "scipy not available for statistical tests"}
    
    results = {}
    config_names = list(configs.keys())
    
    # Pairwise t-tests for cost
    cost_tests = {}
    for i, name1 in enumerate(config_names):
        for name2 in config_names[i+1:]:
            t_stat, p_value = stats.ttest_ind(
                configs[name1]["costs"],
                configs[name2]["costs"]
            )
            cost_tests[f"{name1}_vs_{name2}"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            }
    results["cost_tests"] = cost_tests
    
    # Pairwise t-tests for fill rate
    fill_rate_tests = {}
    for i, name1 in enumerate(config_names):
        for name2 in config_names[i+1:]:
            t_stat, p_value = stats.ttest_ind(
                configs[name1]["fill_rates"],
                configs[name2]["fill_rates"]
            )
            fill_rate_tests[f"{name1}_vs_{name2}"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            }
    results["fill_rate_tests"] = fill_rate_tests
    
    return results


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def generate_latex_table(configs: Dict[str, Dict], pareto_points: List[str]) -> str:
    """Generate LaTeX table for thesis."""
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Pareto Frontier Results: Cost vs Service Level Trade-off}",
        r"\label{tab:pareto_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Configuration & $\alpha$ & $\beta$ & Cost & Fill Rate & Bullwhip & Pareto \\",
        r"\midrule"
    ]
    
    # Sort by cost weight descending
    sorted_configs = sorted(configs.items(), key=lambda x: -x[1]["cost_weight"])
    
    for name, data in sorted_configs:
        is_pareto = r"\checkmark" if name in pareto_points else ""
        line = (f"{name} & {data['cost_weight']:.1f} & {data['service_weight']:.1f} & "
                f"{data['cost_mean']:.1f} $\\pm$ {data['cost_std']:.1f} & "
                f"{data['fill_rate_mean']*100:.1f}\\% $\\pm$ {data['fill_rate_std']*100:.1f}\\% & "
                f"{data['bullwhip_mean']:.3f} $\\pm$ {data['bullwhip_std']:.3f} & "
                f"{is_pareto} \\\\")
        lines.append(line)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    return "\n".join(lines)


def plot_pareto_frontier(
    configs: Dict[str, Dict],
    pareto_points: List[str],
    output_path: str
):
    """Generate Pareto frontier plot."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot generation")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color map based on service weight
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(configs)))
    
    # Sort by service weight for consistent coloring
    sorted_configs = sorted(configs.items(), key=lambda x: x[1]["service_weight"])
    
    for i, (name, data) in enumerate(sorted_configs):
        cost = data["cost_mean"]
        fill_rate = data["fill_rate_mean"] * 100  # Convert to percentage
        cost_std = data["cost_std"]
        fill_rate_std = data["fill_rate_std"] * 100
        
        # Marker style
        marker = 's' if name in pareto_points else 'o'
        size = 150 if name in pareto_points else 100
        
        ax.errorbar(cost, fill_rate, 
                   xerr=cost_std, yerr=fill_rate_std,
                   fmt=marker, markersize=np.sqrt(size),
                   color=colors[i], capsize=5, capthick=2,
                   label=f"{name} (α={data['cost_weight']:.1f})")
    
    # Draw Pareto frontier line
    pareto_data = [(configs[name]["cost_mean"], configs[name]["fill_rate_mean"]*100) 
                   for name in pareto_points]
    pareto_data.sort(key=lambda x: x[0])
    
    if len(pareto_data) > 1:
        pareto_costs, pareto_fill_rates = zip(*pareto_data)
        ax.plot(pareto_costs, pareto_fill_rates, 'k--', linewidth=2, alpha=0.5,
               label='Pareto Frontier')
    
    ax.set_xlabel('Total Cost', fontsize=12)
    ax.set_ylabel('Fill Rate (%)', fontsize=12)
    ax.set_title('Multi-Objective Trade-off: Cost vs Service Level', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for trade-off direction
    ax.annotate('Better Service →', xy=(0.95, 0.95), xycoords='axes fraction',
               fontsize=10, ha='right', va='top', color='green')
    ax.annotate('← Lower Cost', xy=(0.05, 0.05), xycoords='axes fraction',
               fontsize=10, ha='left', va='bottom', color='blue')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pareto frontier plot saved to {output_path}")


def plot_multi_metric_comparison(
    configs: Dict[str, Dict],
    output_path: str
):
    """Generate multi-metric bar chart comparison."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sorted_configs = sorted(configs.items(), key=lambda x: x[1]["cost_weight"], reverse=True)
    names = [name for name, _ in sorted_configs]
    
    # Cost comparison
    costs = [configs[name]["cost_mean"] for name in names]
    cost_stds = [configs[name]["cost_std"] for name in names]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))
    
    axes[0].bar(names, costs, yerr=cost_stds, color=colors, capsize=5)
    axes[0].set_ylabel('Total Cost')
    axes[0].set_title('Cost by Configuration')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Fill Rate comparison
    fill_rates = [configs[name]["fill_rate_mean"]*100 for name in names]
    fill_rate_stds = [configs[name]["fill_rate_std"]*100 for name in names]
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(names)))
    
    axes[1].bar(names, fill_rates, yerr=fill_rate_stds, color=colors, capsize=5)
    axes[1].set_ylabel('Fill Rate (%)')
    axes[1].set_title('Service Level by Configuration')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim([0, 100])
    
    # Bullwhip comparison
    bullwhips = [configs[name]["bullwhip_mean"] for name in names]
    bullwhip_stds = [configs[name]["bullwhip_std"] for name in names]
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(names)))
    
    axes[2].bar(names, bullwhips, yerr=bullwhip_stds, color=colors, capsize=5)
    axes[2].set_ylabel('Bullwhip Effect (CV)')
    axes[2].set_title('Order Variability by Configuration')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Neutral (CV=1)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-metric comparison saved to {output_path}")


def generate_summary_report(
    configs: Dict[str, Dict],
    pareto_points: List[str],
    stat_tests: Dict[str, Any],
    output_path: str
):
    """Generate comprehensive text summary report."""
    
    lines = [
        "=" * 70,
        "PARETO FRONTIER ANALYSIS REPORT",
        "=" * 70,
        "",
        "1. CONFIGURATION SUMMARY",
        "-" * 40,
        ""
    ]
    
    # Summary table
    header = f"{'Config':<20} {'α':<6} {'β':<6} {'Cost':<15} {'Fill Rate':<15} {'Bullwhip':<12}"
    lines.append(header)
    lines.append("-" * len(header))
    
    sorted_configs = sorted(configs.items(), key=lambda x: -x[1]["cost_weight"])
    
    for name, data in sorted_configs:
        pareto_marker = "*" if name in pareto_points else " "
        line = (f"{name:<20} {data['cost_weight']:<6.1f} {data['service_weight']:<6.1f} "
                f"{data['cost_mean']:.1f}±{data['cost_std']:.1f}  "
                f"{data['fill_rate_mean']*100:.1f}%±{data['fill_rate_std']*100:.1f}%  "
                f"{data['bullwhip_mean']:.3f}±{data['bullwhip_std']:.3f} {pareto_marker}")
        lines.append(line)
    
    lines.extend([
        "",
        "* = Pareto optimal point",
        "",
        "2. PARETO FRONTIER",
        "-" * 40,
        f"Pareto optimal configurations: {pareto_points}",
        "",
    ])
    
    # Trade-off analysis
    if len(pareto_points) >= 2:
        lines.append("Trade-off Analysis:")
        pareto_data = [(name, configs[name]) for name in pareto_points]
        pareto_data.sort(key=lambda x: x[1]["cost_mean"])
        
        for i in range(len(pareto_data) - 1):
            name1, data1 = pareto_data[i]
            name2, data2 = pareto_data[i + 1]
            
            cost_diff = data2["cost_mean"] - data1["cost_mean"]
            fill_rate_diff = (data2["fill_rate_mean"] - data1["fill_rate_mean"]) * 100
            
            lines.append(f"  {name1} → {name2}:")
            lines.append(f"    Cost increase: +{cost_diff:.1f}")
            lines.append(f"    Fill Rate gain: +{fill_rate_diff:.1f}%")
            if cost_diff > 0:
                lines.append(f"    Cost per 1% Fill Rate: {cost_diff/fill_rate_diff:.1f}")
    
    # Statistical tests
    if "cost_tests" in stat_tests:
        lines.extend([
            "",
            "3. STATISTICAL SIGNIFICANCE",
            "-" * 40,
            "Cost Comparisons (p < 0.05 is significant):",
        ])
        
        for comparison, result in stat_tests["cost_tests"].items():
            sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
            lines.append(f"  {comparison}: p={result['p_value']:.4f} {sig}")
    
    lines.extend([
        "",
        "4. KEY FINDINGS",
        "-" * 40,
    ])
    
    # Find extremes
    cost_focused = min(sorted_configs, key=lambda x: x[1]["cost_mean"])
    service_focused = max(sorted_configs, key=lambda x: x[1]["fill_rate_mean"])
    
    lines.append(f"Lowest cost: {cost_focused[0]} (Cost={cost_focused[1]['cost_mean']:.1f})")
    lines.append(f"Highest fill rate: {service_focused[0]} (Fill Rate={service_focused[1]['fill_rate_mean']*100:.1f}%)")
    
    # Cost-service trade-off
    cost_range = max(d["cost_mean"] for _, d in sorted_configs) - min(d["cost_mean"] for _, d in sorted_configs)
    fill_rate_range = (max(d["fill_rate_mean"] for _, d in sorted_configs) - 
                       min(d["fill_rate_mean"] for _, d in sorted_configs)) * 100
    
    lines.extend([
        "",
        f"Total cost range: {cost_range:.1f}",
        f"Total fill rate range: {fill_rate_range:.1f}%",
        f"Average cost per 1% fill rate improvement: {cost_range/fill_rate_range:.1f}" if fill_rate_range > 0 else "",
    ])
    
    report = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReport saved to {output_path}")


def main(results_path: str, output_dir: str = None):
    """Main analysis function."""
    
    # Load results
    data = load_results(results_path)
    results = data.get("results", [])
    
    if not results:
        print("No results found in file")
        return
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate by configuration
    configs = aggregate_by_config(results)
    print(f"Found {len(configs)} configurations with {sum(d['n_seeds'] for d in configs.values())} total runs")
    
    # Identify Pareto frontier
    pareto_points = identify_pareto_frontier(configs)
    print(f"Pareto optimal points: {pareto_points}")
    
    # Statistical tests
    stat_tests = compute_statistical_tests(configs)
    
    # Generate outputs
    # 1. Pareto frontier plot
    plot_path = os.path.join(output_dir, "pareto_frontier.png")
    plot_pareto_frontier(configs, pareto_points, plot_path)
    
    # 2. Multi-metric comparison
    comparison_path = os.path.join(output_dir, "metric_comparison.png")
    plot_multi_metric_comparison(configs, comparison_path)
    
    # 3. LaTeX table
    latex_table = generate_latex_table(configs, pareto_points)
    latex_path = os.path.join(output_dir, "pareto_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {latex_path}")
    
    # 4. Summary report
    report_path = os.path.join(output_dir, "analysis_report.txt")
    generate_summary_report(configs, pareto_points, stat_tests, report_path)
    
    # 5. Save aggregated data
    aggregated_path = os.path.join(output_dir, "aggregated_results.json")
    with open(aggregated_path, 'w') as f:
        json.dump({
            "configs": {k: {key: val for key, val in v.items() 
                          if key not in ["costs", "fill_rates", "bullwhips", "cycle_sls", "training_times"]}
                       for k, v in configs.items()},
            "pareto_points": pareto_points,
            "statistical_tests": stat_tests
        }, f, indent=2)
    print(f"Aggregated results saved to {aggregated_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Pareto experiment results")
    parser.add_argument("--results", type=str, required=True, 
                       help="Path to pareto_results.json")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: same as results)")
    args = parser.parse_args()
    
    main(args.results, args.output)
