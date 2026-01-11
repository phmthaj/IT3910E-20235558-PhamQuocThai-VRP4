import argparse
import math
import random
import time
import numpy as np
import copy
import concurrent.futures
import multiprocessing
import os  # Thêm thư viện os để xử lý đường dẫn
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict

# ==========================================
# 0. TĂNG TỐC ĐỘ TÍNH TOÁN (NUMBA)
# ==========================================
try:
    from numba import njit
    HAS_NUMBA = True
    print(">> [System] Numba JIT detected. Acceleration ON.")
except ImportError:
    HAS_NUMBA = False
    print(">> [System] Numba not found. Running in pure Python (Slower).")
    def njit(func=None, **kwargs):
        def wrapper(f): return f
        if func is not None: return wrapper(func)
        return wrapper

@njit(fastmath=True)
def calc_manhattan_matrix(coords):
    """Tính ma trận khoảng cách Manhattan (N x N) cực nhanh."""
    n = len(coords)
    dist_matrix = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = abs(coords[i, 0] - coords[j, 0]) + \
                                abs(coords[i, 1] - coords[j, 1])
    return dist_matrix

@njit(fastmath=True)
def check_capacity_static(current_load, new_demand, capacity):
    """Kiểm tra sức chứa (Vectorized) cho Multi-commodity."""
    n_dim = len(capacity)
    for i in range(n_dim):
        if current_load[i] + new_demand[i] > capacity[i] + 1e-4:
            return False
    return True

# ==========================================
# 1. CẤU TRÚC DỮ LIỆU (DATA STRUCTURES)
# ==========================================

@dataclass
class Node:
    id: int
    x: float
    y: float
    type: int  # 0: Depot, 1: Vendor, 2: Market
    demand: np.ndarray 
    original_id: int 

@dataclass
class VehicleType:
    id: int         # ID định danh loại xe
    name: str       # Tên hiển thị
    depot_id: int
    capacity: np.ndarray
    cost_factor: float
    original_count: int = 1
    cloned_count: int = 1

@dataclass
class Route:
    depot_id: int
    vehicle: VehicleType
    path: List[int] = field(default_factory=list)
    load: np.ndarray = field(default_factory=lambda: np.array([]))
    base_cost: float = 0.0 # Chi phí gốc (Dist * Factor) chưa nhân phạt
    dist: float = 0.0
    phase: int = 1     # 1: Pickup, 2: Delivery

    @property
    def cost(self):
        """Trả về base_cost. Penalty được tính ở cấp độ Solver."""
        return self.base_cost

    def refresh_stats(self, dist_matrix, map_id_to_idx, demands):
        """Cập nhật lại tải trọng, khoảng cách và chi phí cơ bản."""
        if not self.path:
            self.base_cost = 0.0
            self.dist = 0.0
            self.load = np.zeros_like(self.vehicle.capacity)
            return

        depot_idx = map_id_to_idx[self.depot_id]
        d = 0.0
        
        # Depot -> First Node
        first_idx = map_id_to_idx[self.path[0]]
        d += dist_matrix[depot_idx, first_idx]
        
        # Node -> Node
        self.load = np.zeros_like(self.vehicle.capacity)
        self.load += demands[self.path[0]]
        
        for i in range(len(self.path) - 1):
            u, v = self.path[i], self.path[i+1]
            u_idx, v_idx = map_id_to_idx[u], map_id_to_idx[v]
            d += dist_matrix[u_idx, v_idx]
            self.load += demands[v]
            
        # Last Node -> Depot
        last_idx = map_id_to_idx[self.path[-1]]
        d += dist_matrix[last_idx, depot_idx]
        
        self.dist = d
        self.base_cost = d * self.vehicle.cost_factor

    def clone(self):
        r = Route(self.depot_id, self.vehicle, phase=self.phase)
        r.path = self.path[:]
        r.load = self.load.copy()
        r.base_cost = self.base_cost
        r.dist = self.dist
        return r

# ==========================================
# 2. LOGIC XỬ LÝ (PRE-PROCESSING & CLUSTERING)
# ==========================================

def split_large_nodes(nodes, vehicles):
    """Chia nhỏ node nếu demand lớn hơn sức chứa xe."""
    if not vehicles: return nodes
    all_caps = np.array([v.capacity for v in vehicles])
    if len(all_caps) == 0: return nodes
    max_cap = np.max(all_caps, axis=0) * 0.95 

    new_nodes = {}
    next_id = max(nodes.keys()) + 1
    
    for nid, node in nodes.items():
        if node.type == 0:
            new_nodes[nid] = node
            continue
        demand = node.demand.copy()
        if np.any(demand > max_cap):
            while np.any(demand > 0.001):
                chunk = np.minimum(demand, max_cap)
                new_nodes[next_id] = Node(next_id, node.x, node.y, node.type, chunk.copy(), node.original_id)
                next_id += 1
                demand -= chunk
        else:
            new_nodes[nid] = node
    return new_nodes

def clone_vehicles_logic(depots, vehicles, nodes_allocations):
    """Tạo pool xe đủ lớn để thuật toán chạy."""
    cloned_vehicles = []
    for d_id in depots:
        assigned_nodes = nodes_allocations.get(d_id, [])
        if not assigned_nodes: continue
        
        total_demand = np.sum([n.demand for n in assigned_nodes], axis=0)
        depot_vehs = [v for v in vehicles if v.depot_id == d_id]
        if not depot_vehs: continue
        
        total_cap = np.sum([v.capacity for v in depot_vehs], axis=0)
        
        ratio = 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(total_cap > 0, total_demand / total_cap, 0)
            ratio = np.max(ratios)
        
        # Tạo dư xe
        multiplier = math.ceil(ratio) + 5 
        
        for v in depot_vehs:
            new_v = copy.deepcopy(v)
            new_v.cloned_count = int(v.original_count * multiplier)
            cloned_vehicles.append(new_v)
            
    return cloned_vehicles

def clustering_report_logic(nodes, depots, vehicles):
    """Gán node về Depot gần nhất."""
    allocations = {d_id: [] for d_id in depots}
    depot_objs = {n.id: n for n in nodes.values() if n.type == 0}
    targets = [n for n in nodes.values() if n.type != 0]
    for node in targets:
        best_d = -1
        min_dist = float('inf')
        for d_id in depots:
            if d_id not in depot_objs: continue
            dist = abs(node.x - depot_objs[d_id].x) + abs(node.y - depot_objs[d_id].y)
            if dist < min_dist:
                min_dist = dist
                best_d = d_id
        if best_d != -1:
            allocations[best_d].append(node)
    return allocations

# ==========================================
# 3. BỘ GIẢI ALNS (ALNS SOLVER)
# ==========================================

class ALNS_Solver:
    def __init__(self, cluster_nodes, depot_id, vehicles, dist_matrix, map_id_to_idx, params):
        self.nodes = cluster_nodes
        self.depot_id = depot_id
        self.vehicles = vehicles 
        self.dist_matrix = dist_matrix
        self.map_id_to_idx = map_id_to_idx
        self.params = params
        self.node_ids = [n.id for n in cluster_nodes]
        self.demands = {n.id: n.demand for n in cluster_nodes}
        self.depot_idx = map_id_to_idx[depot_id]
        
        # Lấy giới hạn chuyến từ params
        self.limit_trips = params.get('limit_trips', 30)
        
        # --- TIME LIMIT FIX: Lấy Deadline (Timestamp) thay vì duration ---
        # Mặc định là hiện tại + 60s nếu không có tham số
        self.deadline = params.get('deadline', time.time() + 60)
        
        self.phase_type = 1
        if cluster_nodes and cluster_nodes[0].type == 2:
            self.phase_type = 2

    def calculate_solution_cost(self, routes: List[Route]) -> float:
        """
        Tính tổng cost toàn cục:
        - Gom theo Vehicle ID.
        - Sort base_cost giảm dần (ưu tiên chuyến đắt được dùng quota giá gốc).
        - Quá quota -> nhân 3.
        """
        usage_map = defaultdict(list)
        for r in routes:
            usage_map[r.vehicle.id].append(r.base_cost)
            
        total_cost = 0.0
        
        for vid, costs in usage_map.items():
            costs.sort(reverse=True)
            for i, c in enumerate(costs):
                if i < self.limit_trips:
                    total_cost += c         # Giá gốc
                else:
                    total_cost += c * 3.0   # Phạt x3
        return total_cost

    def _get_best_fit_vehicle(self, load_needed):
        sorted_vehs = sorted(self.vehicles, key=lambda v: np.sum(v.capacity))
        for v in sorted_vehs:
            if check_capacity_static(np.zeros_like(v.capacity), load_needed, v.capacity):
                return v
        return sorted_vehs[-1]

    def generate_initial_solution(self) -> List[Route]:
        routes = []
        unassigned = list(self.node_ids)
        random.shuffle(unassigned)
        while unassigned:
            # --- TIME LIMIT FIX: Check deadline ngay trong bước khởi tạo ---
            if time.time() > self.deadline:
                # Nếu hết giờ, buộc phải break để trả về những gì đang có
                break

            best_veh = max(self.vehicles, key=lambda v: np.sum(v.capacity))
            r = Route(self.depot_id, best_veh, phase=self.phase_type)
            r.load = np.zeros_like(best_veh.capacity)
            curr_idx = self.depot_idx
            while True:
                best_next = None
                min_dist = 1e9
                best_remove_idx = -1
                for i, uid in enumerate(unassigned):
                    u_demand = self.demands[uid]
                    if check_capacity_static(r.load, u_demand, r.vehicle.capacity):
                        u_idx = self.map_id_to_idx[uid]
                        d = self.dist_matrix[curr_idx, u_idx]
                        if d < min_dist:
                            min_dist = d
                            best_next = uid
                            best_remove_idx = i
                if best_next is not None:
                    r.path.append(best_next)
                    r.load += self.demands[best_next]
                    curr_idx = self.map_id_to_idx[best_next]
                    unassigned.pop(best_remove_idx)
                else: break
            r.vehicle = self._get_best_fit_vehicle(r.load)
            r.refresh_stats(self.dist_matrix, self.map_id_to_idx, self.demands)
            routes.append(r)
        return routes

    def destroy(self, routes, n_remove):
        new_routes = [r.clone() for r in routes]
        removed = []
        all_nodes = []
        for r_idx, r in enumerate(new_routes):
            for u in r.path: all_nodes.append((r_idx, u))
        if not all_nodes: return new_routes, []
        k = min(n_remove, len(all_nodes))
        indices = random.sample(range(len(all_nodes)), k)
        mod_map = {i: set() for i in range(len(new_routes))}
        for idx in indices:
            r_idx, u = all_nodes[idx]
            mod_map[r_idx].add(u)
            removed.append(u)
        final = []
        for i, r in enumerate(new_routes):
            if mod_map[i]:
                r.path = [u for u in r.path if u not in mod_map[i]]
                if r.path:
                    r.load = sum(self.demands[u] for u in r.path)
                    r.vehicle = self._get_best_fit_vehicle(r.load)
                    r.refresh_stats(self.dist_matrix, self.map_id_to_idx, self.demands)
                    final.append(r)
            else: final.append(r)
        return final, removed

    def repair(self, routes, unassigned):
        """
        LOGIC REPAIR SỬ DỤNG MARGINAL COST.
        Kiểm tra chi phí biên khi tạo route mới để tránh bị phạt oan.
        """
        routes = [r.clone() for r in routes]
        random.shuffle(unassigned)
        
        # Cache để tính toán nhanh
        veh_costs = defaultdict(list)
        veh_min_cost = {} 

        for r in routes:
            veh_costs[r.vehicle.id].append(r.base_cost)
        
        for vid, costs in veh_costs.items():
            if costs:
                veh_min_cost[vid] = min(costs)
            else:
                veh_min_cost[vid] = float('inf')

        for u in unassigned:
            # --- TIME LIMIT FIX: Check deadline trong vòng lặp repair ---
            if time.time() > self.deadline:
                 # Nếu hết giờ, tạo nhanh 1 route đơn giản cho node này rồi next
                 # để đảm bảo node không bị bỏ rơi hoàn toàn (hoặc có thể skip)
                 min_veh = self._get_best_fit_vehicle(self.demands[u])
                 nr = Route(self.depot_id, min_veh, phase=self.phase_type)
                 nr.path = [u]
                 nr.load = self.demands[u].copy()
                 nr.refresh_stats(self.dist_matrix, self.map_id_to_idx, self.demands)
                 routes.append(nr)
                 continue

            u_dem = self.demands[u]
            u_idx = self.map_id_to_idx[u]
            best_diff = float('inf')
            best_r = -1
            best_pos = -1
            
            # 1. Chèn vào Route cũ
            for r_idx, r in enumerate(routes):
                if not check_capacity_static(r.load, u_dem, r.vehicle.capacity): continue
                path = [self.depot_idx] + [self.map_id_to_idx[x] for x in r.path] + [self.depot_idx]
                for i in range(len(path)-1):
                    p, n = path[i], path[i+1]
                    add = (self.dist_matrix[p, u_idx] + self.dist_matrix[u_idx, n] - self.dist_matrix[p, n])
                    cost_inc = add * r.vehicle.cost_factor
                    if cost_inc < best_diff:
                        best_diff = cost_inc
                        best_r = r_idx
                        best_pos = i
            
            # 2. Tạo Route mới (Tính Marginal Cost)
            min_veh = self._get_best_fit_vehicle(u_dem)
            new_dist = self.dist_matrix[self.depot_idx, u_idx] * 2
            base_new_cost = new_dist * min_veh.cost_factor
            
            current_count = len(veh_costs[min_veh.id])
            
            if current_count < self.limit_trips:
                estimated_new_cost = base_new_cost
            else:
                # Logic Marginal Cost
                current_min = veh_min_cost.get(min_veh.id, float('inf'))
                penalty_component = 2.0 * min(base_new_cost, current_min)
                estimated_new_cost = base_new_cost + penalty_component
            
            if estimated_new_cost < best_diff:
                nr = Route(self.depot_id, min_veh, phase=self.phase_type)
                nr.path = [u]
                nr.load = u_dem.copy()
                nr.dist = new_dist
                nr.base_cost = base_new_cost
                routes.append(nr)
                
                # Update Cache
                veh_costs[min_veh.id].append(base_new_cost)
                current_min = veh_min_cost.get(min_veh.id, float('inf'))
                veh_min_cost[min_veh.id] = min(current_min, base_new_cost)
                
            elif best_r != -1:
                r = routes[best_r]
                r.path.insert(best_pos, u)
                r.load += u_dem
                r.refresh_stats(self.dist_matrix, self.map_id_to_idx, self.demands)
                
        return routes

    def solve(self):
        curr = self.generate_initial_solution()
        best = copy.deepcopy(curr)
        best_cost = self.calculate_solution_cost(best)
        
        it_val = self.params.get('iters', 500)
        
        T = max(10, best_cost * 0.05) if best_cost > 0 else 10.0
        
        for _ in range(it_val):
            # --- TIME LIMIT FIX: Sử dụng Deadline tuyệt đối ---
            # So sánh thời gian hiện tại với mốc Deadline được truyền vào từ Main
            if time.time() > self.deadline:
                break
                
            n_rem = random.randint(1, max(2, int(len(self.nodes)*0.3)))
            tmp, rem = self.destroy(curr, n_rem)
            tmp = self.repair(tmp, rem)
            
            tc = self.calculate_solution_cost(tmp)
            curr_cost = self.calculate_solution_cost(curr)
            
            delta = tc - curr_cost
            if delta < 0 or random.random() < math.exp(-delta/T):
                curr = tmp
                if tc < best_cost:
                    best_cost = tc
                    best = copy.deepcopy(tmp)
            T *= 0.99
        return best

# ==========================================
# 4. CÁC HÀM HỖ TRỢ CHẠY & XUẤT FILE
# ==========================================

def worker_solve_cluster(args):
    d_id, cluster_nodes, vehicles, full_coords, params, prohibited, base_seed = args
    if not cluster_nodes: return []
    
    random.seed(base_seed + d_id)
    np.random.seed(base_seed + d_id)
    
    local_ids = [d_id] + [n.id for n in cluster_nodes]
    map_id_to_idx = {uid: i for i, uid in enumerate(local_ids)}
    local_coords = np.array([[full_coords[uid][0], full_coords[uid][1]] for uid in local_ids], dtype=np.float32)
    dist_matrix = calc_manhattan_matrix(local_coords)
    
    for u, v in prohibited:
        if u in map_id_to_idx and v in map_id_to_idx:
            i, j = map_id_to_idx[u], map_id_to_idx[v]
            dist_matrix[i, j] = 1e9
            dist_matrix[j, i] = 1e9
            
    cluster_vehs = [v for v in vehicles if v.depot_id == d_id]
    if not cluster_vehs: return []
    
    solver = ALNS_Solver(cluster_nodes, d_id, cluster_vehs, dist_matrix, map_id_to_idx, params)
    return solver.solve()

def parse_vrp_real(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    nodes_map, vehicles, depot_ids, prohibited = {}, [], [], set()
    commodities, section = 0, None
    node_coords, node_types, node_demands = {}, {}, {}
    
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("DIMENSION"): pass
        elif line.startswith("COMMODITIES"): commodities = int(line.split(":")[1])
        elif line.startswith("NODE_COORD_SECTION"): section = "COORD"
        elif line.startswith("NODE_TYPE_SECTION"): section = "TYPE"
        elif line.startswith("DEMAND_MultiCommodity_SECTION"): section = "DEMAND"
        elif line.startswith("DEPOT_SECTION"): section = "DEPOT"
        elif line.startswith("VEHICLE_SECTION"): section = "VEHICLE"
        elif line.startswith("PROHIBITED_SECTION"): section = "PROHIBITED"
        elif line.startswith("EOF"): break
        elif "SECTION" in line or ":" in line: continue
        else:
            parts = line.split()
            if section == "COORD": node_coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif section == "TYPE": node_types[int(parts[0])] = int(parts[1])
            elif section == "DEMAND": node_demands[int(parts[0])] = np.array([float(x) for x in parts[1:]])
            elif section == "DEPOT":
                if parts[0] != "-1": depot_ids.append(int(parts[0]))
            elif section == "VEHICLE":
                v = VehicleType(int(parts[0]), parts[2], int(parts[1]), np.array([float(x) for x in parts[4:]]), float(parts[3]))
                vehicles.append(v)
            elif section == "PROHIBITED":
                u, v = int(parts[0]), int(parts[1])
                prohibited.add((u, v)); prohibited.add((v, u))
                
    for nid, coords in node_coords.items():
        ntype = node_types.get(nid, 1)
        demand = node_demands.get(nid, np.zeros(commodities) if commodities > 0 else np.array([0.]))
        if nid in depot_ids: ntype = 0
        nodes_map[nid] = Node(nid, coords[0], coords[1], ntype, demand, nid)
    return nodes_map, vehicles, depot_ids, prohibited

def recalculate_global_cost_with_penalty(routes, limit_trips):
    """Tính lại cost tổng thể để hiển thị báo cáo."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in routes:
        grouped[r.depot_id][r.vehicle.id].append(r)
        
    total_final_cost = 0.0
    penalized_route_ids = set()
    
    for d_id, veh_map in grouped.items():
        for vid, r_list in veh_map.items():
            r_list.sort(key=lambda x: x.base_cost, reverse=True)
            for i, r in enumerate(r_list):
                if i < limit_trips:
                    total_final_cost += r.base_cost
                else:
                    total_final_cost += r.base_cost * 3.0
                    penalized_route_ids.add(id(r))
                    
    return total_final_cost, penalized_route_ids

def export_formatted_solution(filepath, routes, total_cost, execution_time, limit_trips, penalized_ids):
    """Lưu file solution với Header đầy đủ thông tin số xe và số chuyến phạt."""
    phase1_routes = [r for r in routes if r.phase == 1]
    phase2_routes = [r for r in routes if r.phase == 2]

    def group_by_depot(route_list):
        grouped = {}
        for r in route_list:
            if r.depot_id not in grouped: grouped[r.depot_id] = []
            grouped[r.depot_id].append(r)
        return dict(sorted(grouped.items()))

    p1_grouped, p2_grouped = group_by_depot(phase1_routes), group_by_depot(phase2_routes)

    def format_load(arr):
        items = [f"{x:.1f}" if x % 1 != 0 else f"{x}." for x in arr]
        return f"[{','.join(items)}]"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Total Cost: {total_cost:.2f}\n")
        f.write(f"Execution Time: {execution_time:.2f}s\n")
        f.write(f"Total Routes (Vehicles): {len(routes)}\n")
        f.write(f"Penalized Routes: {len(penalized_ids)} (Limit {limit_trips})\n")
        f.write("-" * 40 + "\n\n")

        def write_phase(group, phase_name):
            if group:
                f.write(f"{phase_name}\n")
                for d_id, d_routes in group.items():
                    f.write(f"  Depot {d_id}:\n")
                    d_routes.sort(key=lambda x: x.vehicle.name)
                    
                    for i, r in enumerate(d_routes, 1):
                        is_penalized = id(r) in penalized_ids
                        note = " [PENALIZED x3]" if is_penalized else ""
                        final_c = r.base_cost * 3 if is_penalized else r.base_cost
                        
                        f.write(f"    Trip {i} [{r.vehicle.name}]{note}: Cost {final_c:.2f} | Dist {r.dist:.2f}\n")
                        f.write(f"      Path: {' -> '.join(map(str, [r.depot_id] + r.path + [r.depot_id]))}\n")
                        f.write(f"      Load: {format_load(r.load)}\n")

        if p1_grouped:
            write_phase(p1_grouped, "PHASE 1: VENDOR -> DEPOT")
        if p2_grouped:
            if p1_grouped: f.write("\n")
            write_phase(p2_grouped, "PHASE 2: DEPOT -> MARKET")

    print(f">> [Output] Solution saved to: {filepath}")

# ==========================================
# 5. HÀM MAIN
# ==========================================

def main():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, required=True, help="Input .vrp file path")
    parser.add_argument("--iters", type=int, default=300, help="Number of ALNS iterations")
    parser.add_argument("--time", type=int, default=1000, help="Time limit in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit_trips", type=int, default=30, help="Max trips per vehicle type per depot before penalty")
    args = parser.parse_args()
    
    # --- BẮT ĐẦU TÍNH GIỜ & TẠO DEADLINE ---
    start_time = time.time()
    deadline = start_time + args.time
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f">>> Loading {args.instance}...")
    try:
        nodes, vehicles, depot_ids, prohibited = parse_vrp_real(args.instance)
    except Exception as e: print(f"Error reading file: {e}"); return
    
    print(f">> Loaded {len(nodes)} nodes, {len(vehicles)} vehicle types, {len(depot_ids)} depots.")
    print(f">> Trip Limit: {args.limit_trips} (Excess cost x3)")
    
    nodes = split_large_nodes(nodes, vehicles)
    allocations = clustering_report_logic(nodes, depot_ids, vehicles)
    cloned_vehicles = clone_vehicles_logic(depot_ids, vehicles, allocations)
    full_coords = {n.id: (n.x, n.y) for n in nodes.values()}
    
    print(f"[Run] Solving {len(depot_ids)} clusters in parallel...")
    print(f"[Timer] Max Duration: {args.time}s | Deadline Timestamp: {deadline:.2f}")

    tasks = []
    # --- TRUYỀN DEADLINE VÀO PARAMS ---
    alns_params = {
        'iters': args.iters, 
        'deadline': deadline, # Dùng deadline thay vì duration
        'limit_trips': args.limit_trips
    }
    
    for d_id in depot_ids:
        cluster_nodes = allocations[d_id]
        if cluster_nodes:
            tasks.append((d_id, cluster_nodes, cloned_vehicles, full_coords, alns_params, prohibited, args.seed))
            
    total_routes = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(worker_solve_cluster, tasks)
        for res in results: total_routes.extend(res)
    
    final_cost, penalized_ids = recalculate_global_cost_with_penalty(total_routes, args.limit_trips)
    execution_time = time.time() - start_time
    
    print("\n" + "="*40 + f"\nFINAL SOLUTION SUMMARY\nTotal Routes      : {len(total_routes)}")
    print(f"Penalized Routes  : {len(penalized_ids)} (Limit {args.limit_trips})")
    print(f"Total Cost        : {final_cost:.2f}")
    print(f"Time              : {execution_time:.2f}s (Limit: {args.time}s)\n" + "="*40)
    
    # --- XỬ LÝ OUTPUT VÀO THƯ MỤC 'Log' ---
    log_dir = "Log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    filename = os.path.basename(args.instance).rsplit('.', 1)[0] + ".sol"
    output_path = os.path.join(log_dir, filename)
    
    export_formatted_solution(
        output_path,
        total_routes, 
        final_cost, 
        execution_time, 
        args.limit_trips,
        penalized_ids
    )

if __name__ == "__main__":
    main()