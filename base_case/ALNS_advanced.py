import argparse
import math
import random
import time
from typing import List, Tuple, Dict, Any
import numpy as np
import os

def parse_vrp_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    n = None
    capacity = None
    edge_weight_type = None
    edge_weight_format = None
    demands = None
    depot_idx = None
    ew_values = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"):
            parts = line.split(":")
            n = int(parts[1].strip())
        elif line.startswith("CAPACITY"):
            parts = line.split(":")
            capacity = int(parts[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("EDGE_WEIGHT_FORMAT"):
            edge_weight_format = line.split(":")[1].strip()
        elif line == "EDGE_WEIGHT_SECTION":
            i += 1
            while i < len(lines):
                ln = lines[i]
                if ln in ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF"):
                    i -= 1
                    break
                parts = ln.split()
                try:
                    ew_values.extend(map(float, parts))
                except ValueError:
                    i -= 1
                    break
                i += 1
        elif line == "DEMAND_SECTION":
            if demands is None:
                demands = np.zeros(n, dtype=int)
            i += 1
            while i < len(lines):
                ln = lines[i]
                if ln in ("DEPOT_SECTION", "EOF", "NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION"):
                    i -= 1
                    break
                parts = ln.split()
                if len(parts) >= 2:
                    idx = int(parts[0]) - 1
                    dem = int(parts[1])
                    if idx < n: demands[idx] = dem
                i += 1
        elif line == "DEPOT_SECTION":
            i += 1
            depots = []
            while i < len(lines):
                ln = lines[i]
                if ln.startswith("-1") or ln == "EOF":
                    break
                try:
                    val = int(ln)
                    if val != -1: depots.append(val)
                except ValueError:
                    pass
                i += 1
            depot_idx = (depots[0] - 1) if depots else 0
        i += 1

    if n is None or capacity is None:
        raise ValueError("Error: Missing DIMENSION or CAPACITY.")

    if edge_weight_type is None or edge_weight_type.upper() != "EXPLICIT":
        edge_weight_format = edge_weight_format or "FULL_MATRIX"

    dist = np.zeros((n, n), dtype=float)

    if edge_weight_format.upper() in ("FULL_MATRIX", "FULL_MATRIX_DIAG", "FUNCTION"):
        expected = n * n
        if len(ew_values) >= expected:
            dist = np.array(ew_values[:expected], dtype=float).reshape(n, n)
    elif edge_weight_format.upper() in ("LOWER_ROW", "LOWER_ROW_DIAG", "LOWER_TRIANGLE"):
        k = 0
        for r in range(1, n):
            for c in range(r):
                if k < len(ew_values):
                    w = float(ew_values[k])
                    dist[r, c] = w
                    dist[c, r] = w
                    k += 1
    
    np.fill_diagonal(dist, 0.0)
    if demands is None: demands = np.zeros(n, dtype=int)
    demands[depot_idx] = 0

    return {"n": n, "capacity": capacity, "depot": depot_idx, "demands": demands, "dist": dist}

def total_cost(dist: np.ndarray, depot: int, routes: List[List[int]]) -> float:
    cost = 0.0
    for r in routes:
        if not r: continue
        path = np.array([depot] + r + [depot], dtype=int)
        cost += np.sum(dist[path[:-1], path[1:]])
    return cost

def flatten(routes: List[List[int]]) -> List[int]:
    return [c for r in routes for c in r]

def calculate_insertion_vectorized(dist: np.ndarray, route: List[int], nodes: np.ndarray, depot: int):
    if not route:
        costs = dist[depot, nodes] + dist[nodes, depot]
        return costs, np.zeros(len(nodes), dtype=int)

    prev_pts = np.array([depot] + route, dtype=int)
    next_pts = np.array(route + [depot], dtype=int)
    
    current_edges = dist[prev_pts, next_pts] 
    
    d_prev_node = dist[prev_pts][:, nodes] 
    d_node_next = dist[nodes][:, next_pts].T 
    
    deltas = (d_prev_node + d_node_next) - current_edges[:, None]
    
    best_pos = np.argmin(deltas, axis=0)
    min_deltas = deltas[best_pos, np.arange(len(nodes))]
    
    return min_deltas, best_pos


def repair_greedy(dist: np.ndarray, depot: int, demands: np.ndarray, cap: int, 
                  routes: List[List[int]], removed: List[int]) -> List[List[int]]:
    routes = [list(r) for r in routes]
    route_loads = [sum(demands[n] for n in r) for r in routes]
    
    random.shuffle(removed)
    
    for cust in removed:
        dem = demands[cust]
        best_val = float('inf')
        best_rid = -1
        best_pos = -1
        
        for rid, r in enumerate(routes):
            if route_loads[rid] + dem > cap:
                continue
            
            prev_node = depot
            for i in range(len(r) + 1):
                next_node = r[i] if i < len(r) else depot
                delta = dist[prev_node, cust] + dist[cust, next_node] - dist[prev_node, next_node]
                
                if delta < best_val:
                    best_val = delta
                    best_rid = rid
                    best_pos = i
                prev_node = next_node
        
        cost_new = dist[depot, cust] + dist[cust, depot]
        if cost_new < best_val:
            best_val = cost_new
            best_rid = -1
            best_pos = 0
            
        if best_rid == -1:
            routes.append([cust])
            route_loads.append(dem)
        else:
            routes[best_rid].insert(best_pos, cust)
            route_loads[best_rid] += dem
            
    return routes

def repair_regret_2_matrix(dist: np.ndarray, depot: int, demands: np.ndarray, cap: int, 
                           routes: List[List[int]], removed: List[int]) -> List[List[int]]:
    routes = [list(r) for r in routes]
    route_loads = np.array([sum(demands[n] for n in r) for r in routes], dtype=int)
    
    unassigned = np.array(removed, dtype=int)
    if len(unassigned) == 0: return routes
    
    num_u = len(unassigned)
    num_r = len(routes)
    
    cost_matrix = np.full((num_u, num_r), np.inf)
    pos_matrix = np.zeros((num_u, num_r), dtype=int)
    new_route_costs = dist[depot, unassigned] + dist[unassigned, depot]
    
    for rid in range(num_r):
        fits = (route_loads[rid] + demands[unassigned]) <= cap
        if np.any(fits):
            valid_custs = unassigned[fits]
            c, p = calculate_insertion_vectorized(dist, routes[rid], valid_custs, depot)
            cost_matrix[fits, rid] = c
            pos_matrix[fits, rid] = p

    while len(unassigned) > 0:
        if num_r > 0:
            if num_r >= 2:
                sorted_costs = np.partition(cost_matrix, 1, axis=1)
                min1_ex = sorted_costs[:, 0]
                min2_ex = sorted_costs[:, 1]
            else:
                min1_ex = cost_matrix[:, 0]
                min2_ex = np.full(len(unassigned), np.inf)
        else:
            min1_ex = np.full(len(unassigned), np.inf)
            min2_ex = np.full(len(unassigned), np.inf)
            
        comps = np.column_stack((min1_ex, min2_ex, new_route_costs))
        comps.sort(axis=1) 
        
        final_min1 = comps[:, 0]
        final_min2 = comps[:, 1]
        regrets = final_min2 - final_min1
        
        best_idx_in_arr = np.argmax(regrets)
        cust_id = unassigned[best_idx_in_arr]
        
        val_best = final_min1[best_idx_in_arr]
        val_new = new_route_costs[best_idx_in_arr]
        
        target_rid = -1
        target_pos = 0
        
        if abs(val_best - val_new) < 1e-9:
            target_rid = -1
        else:
            row_costs = cost_matrix[best_idx_in_arr, :]
            found_cols = np.where(np.abs(row_costs - val_best) < 1e-9)[0]
            if len(found_cols) > 0:
                target_rid = found_cols[0]
                target_pos = pos_matrix[best_idx_in_arr, target_rid]
            else:
                target_rid = -1

        if target_rid == -1:
            routes.append([cust_id])
            route_loads = np.append(route_loads, demands[cust_id])
            new_rid_idx = num_r
            num_r += 1
            cost_matrix = np.hstack((cost_matrix, np.full((len(unassigned), 1), np.inf)))
            pos_matrix = np.hstack((pos_matrix, np.zeros((len(unassigned), 1), dtype=int)))
            target_rid = new_rid_idx
        else:
            routes[target_rid].insert(target_pos, cust_id)
            route_loads[target_rid] += demands[cust_id]
        
        keep_mask = np.ones(len(unassigned), dtype=bool)
        keep_mask[best_idx_in_arr] = False
        unassigned = unassigned[keep_mask]
        cost_matrix = cost_matrix[keep_mask, :]
        pos_matrix = pos_matrix[keep_mask, :]
        new_route_costs = new_route_costs[keep_mask]
        
        if len(unassigned) == 0: break
        
        r_mod = routes[target_rid]
        load_mod = route_loads[target_rid]
        fits = (load_mod + demands[unassigned]) <= cap
        cost_matrix[:, target_rid] = np.inf
        if np.any(fits):
            valid_custs = unassigned[fits]
            c, p = calculate_insertion_vectorized(dist, r_mod, valid_custs, depot)
            cost_matrix[fits, target_rid] = c
            pos_matrix[fits, target_rid] = p

    return routes


def destroy_random(routes: List[List[int]], d_frac: float) -> Tuple[List[List[int]], List[int]]:
    flat = flatten(routes)
    if not flat: return routes, []
    k = max(1, min(len(flat), int(len(flat) * d_frac)))
    removed = set(random.sample(flat, k))
    new_routes = []
    for r in routes:
        nr = [c for c in r if c not in removed]
        if nr: new_routes.append(nr)
    return new_routes, list(removed)

def destroy_worst(dist: np.ndarray, depot: int, routes: List[List[int]], d_frac: float, p_exp: float = 3.0):
    flat = flatten(routes)
    if not flat: return routes, []
    k = max(1, min(len(flat), int(len(flat) * d_frac)))
    
    candidates = []
    for rid, r in enumerate(routes):
        if not r: continue
        r_arr = np.array(r)
        prevs = np.array([depot] + r[:-1])
        nexts = np.array(r[1:] + [depot])
        savings = dist[prevs, r_arr] + dist[r_arr, nexts] - dist[prevs, nexts]
        for idx, sav in enumerate(savings):
            candidates.append((sav, r_arr[idx]))
            
    candidates.sort(key=lambda x: x[0], reverse=True)
    removed = set()
    while len(removed) < k and candidates:
        idx = int(len(candidates) * (random.random() ** p_exp))
        if idx >= len(candidates): idx = len(candidates) - 1
        removed.add(candidates.pop(idx)[1])
        
    new_routes = []
    for r in routes:
        nr = [c for c in r if c not in removed]
        if nr: new_routes.append(nr)
    return new_routes, list(removed)

def initial_solution(n, depot, demands, cap, dist):
    custs = [i for i in range(n) if i != depot]
    return repair_greedy(dist, depot, demands, cap, [], custs)


def alns(dist, depot, demands, cap, iters=1000, seed=42, time_limit=60,
         reward_improve=10.0, reward_accept=2.0, 
         weight_decay=0.8, weight_learning=0.2, 
         cooling=0.9995, T0_factor=0.05):
    
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    
    routes = initial_solution(len(demands), depot, demands, cap, dist)
    best_routes = [list(r) for r in routes]
    best_cost = total_cost(dist, depot, routes)
    cur_cost = best_cost
    
    print(f"Start ALNS: Init Cost = {best_cost:.2f} | Vehicles = {len(routes)}")
    
    T = T0_factor * best_cost 
    d_fracs = [0.06,0.09,0.12,0.15]
    
    ops_d = ["random", "worst"]
    ops_r = ["greedy", "regret2"]
    w_d = {op: 1.0 for op in ops_d}
    w_r = {op: 1.0 for op in ops_r}
    score_d = {op: 0.0 for op in ops_d}
    score_r = {op: 0.0 for op in ops_r}
    count_d = {op: 0 for op in ops_d}
    count_r = {op: 0 for op in ops_r}
    
    def select(w):
        s = sum(w.values())
        if s == 0: return list(w.keys())[0]
        r = random.uniform(0, s)
        c = 0
        for k, v in w.items():
            c += v
            if r <= c: return k
        return list(w.keys())[-1]
    
    for it in range(iters):
        if time.time() - start_time > time_limit:
            print("Time limit reached.")
            break
            
        d_name = select(w_d)
        r_name = select(w_r)
        frac = random.choice(d_fracs)
        
        if d_name == "random":
            p_routes, removed = destroy_random(routes, frac)
        else:
            p_routes, removed = destroy_worst(dist, depot, routes, frac)
            
        if r_name == "greedy":
            c_routes = repair_greedy(dist, depot, demands, cap, p_routes, removed)
        else:
            c_routes = repair_regret_2_matrix(dist, depot, demands, cap, p_routes, removed)
            
        c_cost = total_cost(dist, depot, c_routes)
        delta = c_cost - cur_cost
        
        accept = False
        reward = 0
        
        if delta < 0:
            accept = True
            if c_cost < best_cost - 1e-5:
                best_cost = c_cost
                best_routes = [list(r) for r in c_routes]
                reward = reward_improve 
                print(f"Iter {it+1}: Best Cost = {best_cost:.2f} | Vehicles = {len(best_routes)} | Op: {d_name}+{r_name}")
            else:
                reward = reward_accept
        else:
            if T > 1e-9:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept = True
                    reward = reward_accept

        if accept:
            routes = c_routes
            cur_cost = c_cost
            
        score_d[d_name] += reward
        score_r[r_name] += reward
        count_d[d_name] += 1
        count_r[r_name] += 1
        
        if it % 50 == 0:
            for op in ops_d:
                if count_d[op] > 0:
                    w_d[op] = weight_decay * w_d[op] + weight_learning * (score_d[op] / count_d[op])
                score_d[op] = 0; count_d[op] = 0
            for op in ops_r:
                if count_r[op] > 0:
                    w_r[op] = weight_decay * w_r[op] + weight_learning * (score_r[op] / count_r[op])
                score_r[op] = 0; count_r[op] = 0
        
        T *= cooling
        
    return best_routes, best_cost, time.time() - start_time

def print_detailed_routes(routes, demands, dist, depot):
    print("\n" + "="*50)
    print(" CHI TIẾT LỘ TRÌNH CÁC XE (DETAILED ROUTES)")
    print("="*50)
    
    total_load_all = 0
    total_dist_all = 0
    
    for i, r in enumerate(routes):
        path = [depot] + r + [depot]
        
        load = sum(demands[n] for n in r)
        total_load_all += load
        
        d_val = 0
        path_str = f"{depot}"
        
        for k in range(len(path) - 1):
            d_val += dist[path[k], path[k+1]]
            path_str += f" -> {path[k+1]}"
            
        total_dist_all += d_val
        
        print(f"Vehicle {i+1:02d}: {path_str}")
        print(f"    - Load    : {load}")
        print(f"    - Distance: {d_val:.2f}")
        print("-" * 30)
        
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=".")
    parser.add_argument("--instances", nargs="+", required=True)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--time", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--reward_improve", type=float, default=10.0)
    parser.add_argument("--reward_accept", type=float, default=2.0)
    parser.add_argument("--weight_decay", type=float, default=0.8)
    parser.add_argument("--weight_learning", type=float, default=0.2)
    parser.add_argument("--cooling", type=float, default=0.9995)
    parser.add_argument("--T0_factor", type=float, default=0.05)
    
    args = parser.parse_args()
    
    for ins in args.instances:
        path = os.path.join(args.folder, ins)
        if not os.path.exists(path): 
            print(f"File not found: {path}")
            continue
        print(f"\nLoading {ins}...")
        try:
            d = parse_vrp_file(path)
            
            r, c, t = alns(
                d['dist'], d['depot'], d['demands'], d['capacity'], 
                iters=args.iters, 
                seed=args.seed, 
                time_limit=args.time,
                reward_improve=args.reward_improve,
                reward_accept=args.reward_accept,
                weight_decay=args.weight_decay,
                weight_learning=args.weight_learning,
                cooling=args.cooling,
                T0_factor=args.T0_factor
            )
            
            print(f"\n✔ Final Summary for {ins}")
            print(f"  Best Cost: {c:.2f}")
            print(f"  Time     : {t:.2f}s")
            print(f"  Vehicles : {len(r)}")
            
            print_detailed_routes(r, d['demands'], d['dist'], d['depot'])
            
        except Exception as e:
            print(f"Error: {e}")