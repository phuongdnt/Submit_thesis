# Multi-Objective Pareto Frontier Experiments

## Quick Start

### 1. Test Setup First (5-10 minutes)
```bash
cd /path/to/hierarchicallotsizing_inventorymanagement_phuong-main
python -m inventory_management_RL_Lot.train_pareto --config configs/pareto_test.yaml
```

### 2. Run Full Serial Experiments (~12-16 hours)
```bash
python -m inventory_management_RL_Lot.train_pareto --config configs/pareto_serial.yaml
```

### 3. Run Full Network Experiments (~20-24 hours)
```bash
python -m inventory_management_RL_Lot.train_pareto --config configs/pareto_network.yaml
```

### 4. Run Pure HAPPO Baseline (~8-12 hours)
```bash
python -m inventory_management_RL_Lot.train_pareto --config configs/baseline_pure_happo.yaml
```

### 5. Analyze Results
```bash
# After Serial experiments complete:
python -m inventory_management_RL_Lot.analyze_pareto --results results/pareto_serial/pareto_results.json

# After Network experiments complete:
python -m inventory_management_RL_Lot.analyze_pareto --results results/pareto_network/pareto_results.json

# After Pure HAPPO baseline complete:
python -m inventory_management_RL_Lot.analyze_pareto --results results/baseline_pure_happo/pareto_results.json
```

---

## Experiment Configurations

### Weight Configurations (5 Pareto Points)
| Config Name | α (Cost) | β (Service) | Expected Behavior |
|-------------|----------|-------------|-------------------|
| cost_focused | 1.0 | 0.0 | Minimize cost, low service |
| cost_leaning | 0.8 | 0.2 | Mostly cost, some service |
| balanced | 0.5 | 0.5 | Equal trade-off |
| service_leaning | 0.2 | 0.8 | Mostly service, some cost |
| service_focused | 0.0 | 1.0 | Maximize service, high cost |

### Training Parameters
| Parameter | Serial | Network |
|-----------|--------|---------|
| Episodes | 5,000 | 8,000 |
| Seeds | 3 | 3 |
| Total runs | 15 | 15 |
| GA pop_size | 20 | 20 |
| GA generations | 25 | 25 |

---

## Output Files

After experiments complete, find these in `results/pareto_serial/` or `results/pareto_network/`:

| File | Description |
|------|-------------|
| `pareto_results.json` | Raw experiment data (all seeds, all configs) |
| `pareto_frontier.png` | Pareto frontier visualization |
| `metric_comparison.png` | Bar chart comparison of all metrics |
| `pareto_table.tex` | LaTeX table ready for thesis |
| `analysis_report.txt` | Text summary with statistics |
| `aggregated_results.json` | Mean ± std for each configuration |
| `*.pth` | Trained model checkpoints |
| `*_log.json` | Detailed training logs per experiment |

---

## Expected Timeline

### Option A: Sequential (Safe, ~2-3 days)
```
Day 1 Morning:  Run pareto_test.yaml (verify setup)
Day 1 Afternoon: Start pareto_serial.yaml (let run overnight)
Day 2 Morning:  Check results, start pareto_network.yaml
Day 2-3:        Network experiments complete
Day 3:          Run analysis, generate plots
```

### Option B: Parallel (Faster, requires 2 GPUs or machines)
```
Machine 1: pareto_serial.yaml + baseline_pure_happo.yaml
Machine 2: pareto_network.yaml
Total time: ~24 hours
```

---

## How to Use Results in Thesis

### Chapter 5: Results Analysis

1. **Replace Table 5-4** with `pareto_table.tex` content
2. **Add new Figure**: `pareto_frontier.png` as Figure 5-2
3. **Add new Section 5.5**: "Multi-Objective Analysis"
   - Use data from `analysis_report.txt`
   - Discuss trade-offs between configurations
   - Show Pareto-optimal points

### Key Metrics to Report
From `aggregated_results.json`:
- Total Cost (mean ± std)
- Fill Rate % (mean ± std)  
- Bullwhip Effect (mean ± std)
- Statistical significance (p-values)

---

## Troubleshooting

### Out of Memory
Reduce `n_rollout_threads` from 8 to 4:
```yaml
training:
  n_rollout_threads: 4
```

### Training Too Slow
Reduce GA complexity:
```yaml
heuristic:
  ga:
    pop_size: 15
    generations: 20
```

### Experiments Interrupted
Results are saved periodically. To resume:
1. Check which configs completed in `pareto_results.json`
2. Modify `weight_configs` in yaml to only include remaining configs
3. Re-run

---

## File Structure After Experiments

```
results/
├── pareto_serial/
│   ├── pareto_results.json
│   ├── pareto_frontier.png
│   ├── metric_comparison.png
│   ├── pareto_table.tex
│   ├── analysis_report.txt
│   ├── aggregated_results.json
│   ├── cw1.0_sw0.0_seed1.pth
│   ├── cw1.0_sw0.0_seed1_log.json
│   ├── ... (more model files)
│   
├── pareto_network/
│   └── ... (same structure)
│
└── baseline_pure_happo/
    └── ... (same structure)
```

---

## Contact
For issues with this experiment setup, check the training logs in `logs/` directory.
