import numpy as np
import os
import time

# --- CẤU HÌNH TÊN FILE MỚI ---
DATASETS_CONFIG = [
    {"num_nodes": 10000, "name": "VRP4_10k_Dataset"},
    {"num_nodes": 20000, "name": "VRP4_20k_Dataset"},
    {"num_nodes": 30000, "name": "VRP4_30k_Dataset"},
    {"num_nodes": 40000, "name": "VRP4_40k_Dataset"},
    {"num_nodes": 50000, "name": "VRP4_50k_Dataset"},
]

PROHIBITED_RANGE = (1000, 5000)
COMMODITIES = ["food", "drink", "electronics", "household"]

VEHICLE_TYPES = [
    {"type": "Type_A", "cost_factor": 1.0, "capacity_scale": 1.0},
    {"type": "Type_B", "cost_factor": 1.3, "capacity_scale": 1.3},
    {"type": "Type_C", "cost_factor": 1.8, "capacity_scale": 1.8}
]

def generate_nodes_balanced(num_nodes, x_range, y_range):
    """ Sinh node phân phối cụm (Clustered) + Nhiễu nền """
    num_clustered = int(num_nodes * 0.9)
    num_uniform = num_nodes - num_clustered
    
    num_clusters = int(np.sqrt(num_nodes)) * 2 
    nodes_per_cluster = num_clustered // num_clusters
    
    centers_x = np.random.randint(x_range[0], x_range[1], num_clusters)
    centers_y = np.random.randint(y_range[0], y_range[1], num_clusters)
    
    std_dev_x = (x_range[1] - x_range[0]) / (num_clusters * 0.5) 
    std_dev_y = (y_range[1] - y_range[0]) / (num_clusters * 0.5)
    
    all_x = []
    all_y = []
    
    for i in range(num_clusters):
        count = nodes_per_cluster + (1 if i < num_clustered % num_clusters else 0)
        cx = np.random.normal(centers_x[i], std_dev_x, count)
        cy = np.random.normal(centers_y[i], std_dev_y, count)
        all_x.append(cx)
        all_y.append(cy)
        
    uniform_x = np.random.uniform(x_range[0], x_range[1], num_uniform)
    uniform_y = np.random.uniform(y_range[0], y_range[1], num_uniform)
    
    all_x.append(uniform_x)
    all_y.append(uniform_y)
    
    final_x = np.concatenate(all_x).astype(int)
    final_y = np.concatenate(all_y).astype(int)
    final_x = np.clip(final_x, x_range[0], x_range[1])
    final_y = np.clip(final_y, y_range[0], y_range[1])
    
    ids = np.arange(1, num_nodes + 1)
    coords_matrix = np.column_stack((final_x, final_y))
    cluster_centers = np.column_stack((centers_x, centers_y))
    
    return ids, coords_matrix, cluster_centers

def select_depots_mixed(ids, coords_matrix, cluster_centers, num_depots):
    """ 70% Inner - 30% Outer """
    print(f"  - Đang chọn {num_depots} depots...")
    num_inner = int(num_depots * 0.7)
    num_outer = num_depots - num_inner
    selected_indices = []
    selected_set = set()
    
    # 70% Inner
    chosen_centers_indices = np.random.choice(len(cluster_centers), size=min(num_inner, len(cluster_centers)), replace=False)
    chosen_centers = cluster_centers[chosen_centers_indices]
    
    for center in chosen_centers:
        dists = np.sum((coords_matrix - center)**2, axis=1)
        sorted_indices = np.argsort(dists)
        for idx in sorted_indices:
            if idx not in selected_set:
                selected_indices.append(idx)
                selected_set.add(idx)
                break
                
    while len(selected_indices) < num_inner:
        idx = np.random.randint(0, len(coords_matrix))
        if idx not in selected_set:
            selected_indices.append(idx)
            selected_set.add(idx)

    # 30% Outer
    map_center = np.mean(coords_matrix, axis=0)
    dist_from_map_center = np.sum((coords_matrix - map_center)**2, axis=1)
    furthest_indices = np.argsort(dist_from_map_center)[::-1]
    
    count = 0
    for idx in furthest_indices:
        if count >= num_outer: break
        if idx not in selected_set:
            selected_indices.append(idx)
            selected_set.add(idx)
            count += 1
            
    return np.array(selected_indices)

def generate_prohibited_edges(ids, min_edges, max_edges):
    num_edges = np.random.randint(min_edges, max_edges + 1)
    buffer = int(num_edges * 1.5)
    sources = np.random.choice(ids, buffer)
    targets = np.random.choice(ids, buffer)
    mask = sources != targets
    return sources[mask][:num_edges], targets[mask][:num_edges]

def write_vrp_file(config, ids, coords, depot_indices, vehicle_types, commodities):
    output_dir = "dataset"
    filename = f"{output_dir}/{config['name']}.vrp"
    os.makedirs(output_dir, exist_ok=True)
    
    num_nodes = len(ids)
    all_indices = np.arange(num_nodes)
    depot_set = set(depot_indices)
    market_vendor_indices = np.array([i for i in all_indices if i not in depot_set])
    
    np.random.shuffle(market_vendor_indices)
    split_idx = int(len(market_vendor_indices) * 0.4)
    vendor_indices = market_vendor_indices[:split_idx]
    market_indices = market_vendor_indices[split_idx:]
    
    # =========================================================================
    # 1. SINH SUPPLY (VENDOR)
    # =========================================================================
    vendor_supply = np.round(np.random.uniform(20, 60, (len(vendor_indices), len(commodities))), 1)
    
    # Tính tổng supply của toàn bộ Vendor
    total_supply = np.sum(vendor_supply, axis=0)
    
    # =========================================================================
    # 2. SINH DEMAND (MARKET) = CHÍNH XÁC 30% VENDOR SUPPLY
    # =========================================================================
    # Bước 2.1: Sinh ngẫu nhiên trọng số nhu cầu cho từng market
    raw_market_demand = np.random.uniform(10, 40, (len(market_indices), len(commodities)))
    
    # Bước 2.2: Tính tổng demand thô hiện tại
    current_market_sum = np.sum(raw_market_demand, axis=0)
    
    # Bước 2.3: Tính mục tiêu demand (30% Supply)
    target_market_total = total_supply * 0.30
    
    # Bước 2.4: Scale demand thô để khớp với mục tiêu
    scale_factors = target_market_total / current_market_sum
    market_demand = raw_market_demand * scale_factors
    market_demand = np.round(market_demand, 1) # Làm tròn 1 chữ số thập phân
    
    # Gán vào mảng tổng
    all_demands = np.zeros((num_nodes, len(commodities)))
    all_demands[vendor_indices] = vendor_supply
    all_demands[market_indices] = -market_demand 

    # =========================================================================
    # 3. DEPOT CAPACITY = GẤP 4 LẦN VENDOR SUPPLY
    # =========================================================================
    num_depots = len(depot_indices)
    
    # Tổng sức chứa toàn hệ thống = 4 * Total Supply
    total_system_capacity_needed = total_supply * 4.0
    
    # Chia đều sức chứa này cho các Depot (có thể thay đổi nếu muốn random)
    cap_per_depot = total_system_capacity_needed / num_depots
    cap_per_depot = np.round(cap_per_depot, 1)
    
    # Tạo mảng capacity cho tất cả depot
    depot_caps = np.tile(cap_per_depot, (num_depots, 1))

    # Gán node type
    node_types = np.zeros(num_nodes, dtype=int)
    node_types[vendor_indices] = 1
    node_types[market_indices] = 2
    node_types[depot_indices] = 0

    # =========================================================================
    # 4. VEHICLE CONFIG
    # =========================================================================
    # Giữ nguyên logic cũ hoặc chỉnh lại nếu cần. 
    # Hiện tại target capacity xe = 30% Supply (tức là đủ sức chở hết demand 1 lượt)
    vehicles_lines = []
    target_cap = total_supply * 0.3
    base_cap = target_cap / (len(depot_indices) * 3 * 1.36)
    
    v_id = 1
    real_depot_ids = ids[depot_indices]
    
    for i, d_id in enumerate(real_depot_ids):
        for vt in vehicle_types:
            cap_vals = np.round(base_cap * vt['capacity_scale'], 1)
            vehicles_lines.append(f"{v_id} {d_id} {vt['type']} {vt['cost_factor']} {' '.join(map(str, cap_vals))}")
            v_id += 1

    p_sources, p_targets = generate_prohibited_edges(ids, PROHIBITED_RANGE[0], PROHIBITED_RANGE[1])

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"NAME : {config['name']}\n")
        f.write("COMMENT : VRP4 Dataset (Manhattan, Demand=30%Supply, DepotCap=4xSupply)\n")
        f.write("TYPE : CVRP_EXTENDED\n")
        f.write(f"DIMENSION : {num_nodes}\n")
        f.write(f"VEHICLES : {len(vehicles_lines)}\n")
        f.write(f"COMMODITIES : {len(commodities)}\n")
        f.write(f"PROHIBITED_EDGES_COUNT : {len(p_sources)}\n")
        f.write("EDGE_WEIGHT_TYPE : MAN_2D\n") 
        
        f.write("COMMODITY_SECTION\n")
        for i, c in enumerate(commodities): f.write(f"{i+1} {c}\n")
            
        f.write("NODE_COORD_SECTION\n")
        for i in range(num_nodes): f.write(f"{ids[i]} {coords[i][0]} {coords[i][1]}\n")
            
        f.write("NODE_TYPE_SECTION\n")
        for i in range(num_nodes): f.write(f"{ids[i]} {node_types[i]}\n")
            
        f.write("DEMAND_MultiCommodity_SECTION\n")
        for i in range(num_nodes): f.write(f"{ids[i]} {' '.join(map(str, all_demands[i]))}\n")
        
        f.write("DEPOT_CAPACITY_SECTION\n")
        for i, d_id in enumerate(real_depot_ids):
            cap_str = ' '.join(map(str, depot_caps[i]))
            f.write(f"{d_id} {cap_str}\n")
            
        f.write("DEPOT_SECTION\n")
        for d in real_depot_ids: f.write(f"{d}\n")
        f.write("-1\n")
        
        f.write("VEHICLE_SECTION\n")
        for line in vehicles_lines: f.write(line + "\n")

        f.write("PROHIBITED_SECTION\n")
        f.writelines([f"{s} {t}\n" for s, t in zip(p_sources, p_targets)])
        f.write("EOF\n")
    
    print(f"✅ Đã tạo file: {filename}")

# --- RUN ---
if __name__ == "__main__":
    start_total = time.time()
    # Tạo thư mục nếu chưa có
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    for conf in DATASETS_CONFIG:
        st = time.time()
        print(f"--- Đang xử lý: {conf['name']} ---")
        ids, coords, centers = generate_nodes_balanced(conf['num_nodes'], (496000, 504500), (4786000, 4798000))
        depots = select_depots_mixed(ids, coords, centers, int(conf['num_nodes']*0.1))
        write_vrp_file(conf, ids, coords, depots, VEHICLE_TYPES, COMMODITIES)
    print(f"\n🎉 HOÀN TẤT! Dữ liệu nằm trong thư mục 'dataset'.")