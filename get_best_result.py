import json
import numpy as np

# Đường dẫn file log
file_path = 'results/pareto_network/net_cw1.0_sw0.0_s1_log.json'

def generate_thesis_tables(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        evals = data.get('evals', [])
        if not evals:
            print("Lỗi: Không tìm thấy dữ liệu 'evals'.")
            return

        # 1. Tìm Epoch tốt nhất (dựa trên Total Cost thấp nhất)
        best_entry = min(evals, key=lambda x: x['total_cost'])
        epoch = best_entry['ep']
        
        print(f"=== DỮ LIỆU CHO BÁO CÁO (LẤY TỪ EPOCH {epoch}) ===\n")

        # --- XỬ LÝ DỮ LIỆU ---
        # Network 2x3 có 6 agents: 0,1 (Retailers), 2,3 (Distributors), 4,5 (Manufacturers)
        costs = best_entry['cost_per_agent']
        fill_rates = best_entry['fill_rate_per_agent']
        bullwhips = best_entry['bullwhip_per_agent']
        # Service level mean trong log thường là Cycle Service Level (Type 2)
        sl_type2 = best_entry.get('service_level_per_agent', [0]*6) 

        # Gom nhóm
        retailers_cost = sum(costs[0:2])
        distributors_cost = sum(costs[2:4])
        manufacturers_cost = sum(costs[4:6])
        total_real_cost = best_entry['total_cost']

        avg_bw_retailer = np.mean(bullwhips[0:2])
        avg_bw_distributor = np.mean(bullwhips[2:4])
        avg_bw_factory = np.mean(bullwhips[4:6])
        
        avg_fr_retailer = np.mean(fill_rates[0:2])
        avg_sl_retailer = np.mean(sl_type2[0:2])

        # --- IN RA DẠNG BẢNG KHỚP VỚI THESIS ---

        print(">>> DATA FOR TABLE: COST PERFORMANCE (Cost Breakdown)")
        print(f"{'Echelon':<20} | {'Total Cost':<15} | {'% of Total':<10}")
        print("-" * 50)
        print(f"{'Retailers (Ag 0,1)':<20} | {retailers_cost:,.1f}        | {retailers_cost/total_real_cost*100:.1f}%")
        print(f"{'Distributors (Ag 2,3)':<20} | {distributors_cost:,.1f}        | {distributors_cost/total_real_cost*100:.1f}%")
        print(f"{'Factories (Ag 4,5)':<20} | {manufacturers_cost:,.1f}        | {manufacturers_cost/total_real_cost*100:.1f}%")
        print(f"{'TOTAL':<20} | {total_real_cost:,.1f}        | 100%")
        print("\n")

        print(">>> DATA FOR TABLE: BULLWHIP EFFECT ANALYSIS (Network)")
        print(f"{'Metric':<30} | {'Value (Hybrid)':<15}")
        print("-" * 50)
        print(f"{'Global Average BW':<30} | {best_entry['bullwhip_mean']:.4f}")
        print(f"{'  - Retailer BW':<30} | {avg_bw_retailer:.4f}")
        print(f"{'  - Distributor BW':<30} | {avg_bw_distributor:.4f}")
        print(f"{'  - Factory BW':<30} | {avg_bw_factory:.4f}")
        print("\n")

        print(">>> DATA FOR TABLE: SERVICE LEVEL COMPARISON")
        print(f"{'Metric':<30} | {'Value (Hybrid)':<15}")
        print("-" * 50)
        print(f"{'Fill Rate (Global Mean)':<30} | {best_entry['fill_rate_mean']*100:.2f}%")
        print(f"{'Fill Rate (Retailers Only)':<30} | {avg_fr_retailer*100:.2f}%")
        print(f"{'Cycle Service Level (CSL)':<30} | {best_entry['service_level_mean']*100:.2f}%")
        print(f"{'CSL (Retailers Only)':<30} | {avg_sl_retailer*100:.2f}%")

    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    generate_thesis_tables(file_path)