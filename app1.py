import random
import numpy as np
import math
import heapq
from datetime import timedelta, datetime
from flask import Flask, render_template, request, jsonify
from functools import lru_cache
import os
import sys
import requests 
from shared_utils import load_stop_data

app = Flask(__name__)
FIXED_DESTINATION_NAME = "면목역 1번출구"
NUM_ROUTES_TO_FIND = 5
SEARCH_TIME_HORIZON_MINS = 150 
#
KAKAO_API_KEY = "YOUR_API_KEY"
if not KAKAO_API_KEY:
    raise ValueError("FATAL ERROR: KAKAO_REST_API_KEY environment variable not set. Please provide your Kakao REST API key to run the application.")


STOPS_BASE_DATA = load_stop_data()
ID_TO_STOP = {stop['id']: stop for stop in STOPS_BASE_DATA}
NAME_TO_ID = {stop['name']: stop['id'] for stop in STOPS_BASE_DATA}
print("\n" + "="*40)
print("--- AVAILABLE START LOCATIONS ---")
# We print a sorted list of all valid names the API will accept
for name in sorted(NAME_TO_ID.keys()):
    # We exclude the fixed destination from the list of valid *start* points
    if name != FIXED_DESTINATION_NAME:
        print(f"- {name}")
print("="*40 + "\n")
if FIXED_DESTINATION_NAME not in NAME_TO_ID:
    raise KeyError(f"The fixed destination '{FIXED_DESTINATION_NAME}' was not found in location.csv.")

def load_precomputed_matrices(dist_path='distance_matrix.npy', dura_path='duration_matrix.npy'):
    if not os.path.exists(dist_path) or not os.path.exists(dura_path):
        print("\n" + "="*60); print(" ! ERROR: Pre-computed matrix files not found."); print("   Please run 'python build_matrices.py' first to generate them."); print(f"   Expected to find: '{dist_path}' and '{dura_path}'"); print("="*60 + "\n"); sys.exit(1)
    return np.load(dist_path), np.load(dura_path)

DISTANCE_MATRIX_M, DURATION_MATRIX_S = load_precomputed_matrices()

def simulate_demand(stops, sim_time_str="09:00"):
    """
    Simulates passenger demand based on the 'avg_passenger' data from the stops file.
    Applies a multiplier during defined peak hours and ensures the result is an integer.
    """
    # Peak hours are defined here.
    is_peak = (("07:00" <= sim_time_str <= "09:30") or 
               ("11:30" <= sim_time_str <= "13:00") or 
               ("17:30" <= sim_time_str <= "20:00"))
    
    peak_multiplier = 1.5
    simulated = []

    for s in stops:
        try:
            base_demand = float(s.get('avg_passenger', 1))
        except (ValueError, TypeError):
            base_demand = 1
            print(f"Warning: Could not parse 'avg_passenger' for stop {s.get('id', 'N/A')}. Defaulting to 1.", file=sys.stderr)

        if is_peak:
            final_demand = base_demand * peak_multiplier
        else:
            final_demand = base_demand
        
        demand = int(final_demand)

        # Append the stop data with the newly calculated, data-driven demand.
        simulated.append({
            'id': s['id'], 
            'name': s['name'], 
            'lat': s['lat'], 
            'lng': s['lng'], 
            'demand': demand
        })
        
    return simulated

def compute_route_fitness(route_stop_ids, all_stops_map, duration_matrix_s, max_duration_mins, capacity, maximize_coverage=False):
    if maximize_coverage:
        demand_weight, efficiency_weight, coverage_weight = 0.5, 0.2, 0.3
    else:
        demand_weight, efficiency_weight, coverage_weight = 0.7, 0.3, 0.0

    current_time_s = 0; current_passengers = 0; total_demand = 0
    for i in range(len(route_stop_ids) - 1):
        from_stop_id = route_stop_ids[i]; to_stop_id = route_stop_ids[i+1]
        travel_seconds = duration_matrix_s[from_stop_id][to_stop_id]; current_time_s += travel_seconds
        
        if i < len(route_stop_ids) - 2:
            to_stop_info = all_stops_map.get(to_stop_id)
            if to_stop_info: 
                demand_at_stop = to_stop_info.get('demand', 0)
                current_passengers += demand_at_stop
                total_demand += demand_at_stop
        
        if current_passengers > capacity: return -1, 0, 0

    total_time_mins = current_time_s / 60
    if total_time_mins > max_duration_mins: return -1, 0, 0
    
    total_demand_score = total_demand * demand_weight
    efficiency_score = (total_demand / total_time_mins) * efficiency_weight if total_time_mins > 0 else 0
    
    num_intermediate_stops = max(0, len(route_stop_ids) - 2)
    coverage_score = (num_intermediate_stops * coverage_weight) * 10

    fitness = total_demand_score + efficiency_score + coverage_score
    return fitness, total_demand, total_time_mins

def generate_greedy_route(source_id, dest_id, intermediate_stops, all_stops_map, duration_matrix_s, max_duration_mins, capacity):
    route = [source_id]; remaining_stops = list(intermediate_stops)
    while remaining_stops:
        current_stop_id = route[-1]; best_next_stop = None; best_score = -1
        for next_stop in remaining_stops:
            travel_time = duration_matrix_s[current_stop_id][next_stop['id']]
            if travel_time == 0: continue
            score = next_stop['demand'] / travel_time
            _, _, temp_time = compute_route_fitness(route + [next_stop['id'], dest_id], all_stops_map, duration_matrix_s, max_duration_mins, capacity)
            if temp_time > 0 and score > best_score: best_score = score; best_next_stop = next_stop
        if best_next_stop: route.append(best_next_stop['id']); remaining_stops.remove(best_next_stop)
        else: break
    return tuple(route[1:])

def two_opt_refine(route_ids, duration_matrix_s):
    if len(route_ids) < 4: return route_ids
    best_route = list(route_ids); improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route)):
                if j == i + 1: continue
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                original_dist = sum(duration_matrix_s[best_route[k-1]][best_route[k]] for k in range(1, len(best_route)))
                new_dist = sum(duration_matrix_s[new_route[k-1]][new_route[k]] for k in range(1, len(new_route)))
                if new_dist < original_dist: best_route = new_route; improved = True; break
            if improved: break
    return best_route

def find_optimized_routes(source_id, dest_id, intermediate_stops, all_stops_map, duration_matrix_s, max_duration_mins, capacity, maximize_coverage=False, mandatory_stop_ids=None):
    if mandatory_stop_ids is None: mandatory_stop_ids = []
    mandatory_set = set(mandatory_stop_ids)
    
    num_optional_stops = len(intermediate_stops)
    num_bats = 40; max_iter = 150; temp_initial = 100.0; temp_cooling_rate = 0.995

    @lru_cache(maxsize=None)
    def _compute_fitness_cached(route_tuple):
        full_route = (source_id,) + route_tuple + (dest_id,)
        return compute_route_fitness(full_route, all_stops_map, duration_matrix_s, max_duration_mins, capacity, maximize_coverage)

    def eval_fitness(route_tuple):
        if mandatory_set and not mandatory_set.issubset(set(route_tuple)):
            return -1.0, 0, 0
        return _compute_fitness_cached(route_tuple)

    bats = []; optional_intermediate_ids = [s['id'] for s in intermediate_stops]
    
    # Only run greedy algorithm if no mandatory stops are selected
    if not mandatory_set:
        greedy_start_route = generate_greedy_route(source_id, dest_id, intermediate_stops, all_stops_map, duration_matrix_s, max_duration_mins, capacity)
        if greedy_start_route: bats.append(greedy_start_route)

    # Populate bats, ensuring mandatory stops are included if specified
    while len(bats) < num_bats:
        if mandatory_set:
            num_to_add = random.randint(0, len(optional_intermediate_ids))
            stops_to_add = random.sample(optional_intermediate_ids, num_to_add)
            new_bat_list = list(mandatory_set) + stops_to_add
            random.shuffle(new_bat_list)
            bats.append(tuple(new_bat_list))
        else:
            bats.append(tuple(random.sample(optional_intermediate_ids, random.randint(0, num_optional_stops))))

    # Get fitness of a route tuple, accessing only the fitness score [0]
    def get_fitness_score(route_tuple):
        return eval_fitness(route_tuple)[0]

    best_route = max(bats, key=get_fitness_score, default=tuple()); best_fitness = get_fitness_score(best_route); temperature = temp_initial
    for _ in range(max_iter):
        for i in range(len(bats)):
            current_route_tuple = bats[i]; mutated_route_list = list(current_route_tuple)
            
            if maximize_coverage:
                op = random.choices(['add', 'remove', 'swap', 'reverse'], weights=[0.5, 0.1, 0.2, 0.2], k=1)[0]
            else:
                op = random.choice(['add', 'remove', 'swap', 'reverse'])

            if op == 'add' and len(optional_intermediate_ids) > 0 and len(mutated_route_list) < (len(optional_intermediate_ids) + len(mandatory_set)):
                possible_adds = list(set(optional_intermediate_ids) - set(mutated_route_list))
                if possible_adds: mutated_route_list.insert(random.randint(0, len(mutated_route_list)), random.choice(possible_adds))
            elif op == 'remove' and len(mutated_route_list) > len(mandatory_set):
                removable_stops = [s for s in mutated_route_list if s not in mandatory_set]
                if removable_stops: mutated_route_list.remove(random.choice(removable_stops))
            elif op == 'swap' and len(mutated_route_list) > 1: idx1, idx2 = random.sample(range(len(mutated_route_list)), 2); mutated_route_list[idx1], mutated_route_list[idx2] = mutated_route_list[idx2], mutated_route_list[idx1]
            elif op == 'reverse' and len(mutated_route_list) > 1: start, end = sorted(random.sample(range(len(mutated_route_list)), 2)); mutated_route_list[start:end+1] = reversed(mutated_route_list[start:end+1])
            
            new_route_tuple = tuple(mutated_route_list); new_fitness = get_fitness_score(new_route_tuple); current_fitness = get_fitness_score(current_route_tuple)
            if new_fitness > best_fitness: best_fitness = new_fitness; best_route = new_route_tuple
            if new_fitness > current_fitness or (temperature > 0.1 and random.random() < math.exp((new_fitness - current_fitness) / temperature)): bats[i] = new_route_tuple
        temperature *= temp_cooling_rate
    
    unique_routes = {}
    for route_tuple in bats:
        fitness, demand, time = eval_fitness(route_tuple)
        if fitness > -1:
            refined_path = two_opt_refine([source_id] + list(route_tuple) + [dest_id], duration_matrix_s)
            refined_tuple = tuple(refined_path[1:-1])
            if refined_tuple not in unique_routes or fitness > unique_routes[refined_tuple]['fitness']:
                # Recalculate fitness for the refined path to be accurate
                final_fitness, final_demand, final_time = eval_fitness(refined_tuple)
                if final_fitness > -1:
                    unique_routes[refined_tuple] = {"ids": refined_path, "demand": final_demand, "time": final_time, "fitness": final_fitness}

    if not unique_routes: return []
    return sorted(list(unique_routes.values()), key=lambda x: x['fitness'], reverse=True)


def get_detailed_path_from_kakao(route_stops, departure_time_str="09:00"):
    full_path_coords = []
    if not route_stops or len(route_stops) < 2: return []
    for i in range(len(route_stops) - 1):
        origin = route_stops[i]; destination = route_stops[i+1]
        url = "https://apis-navi.kakaomobility.com/v1/directions"; headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
        try:
            now = datetime.now(); sim_hour, sim_minute = map(int, departure_time_str.split(':')); departure_dt = now.replace(hour=sim_hour, minute=sim_minute, second=0, microsecond=0)
            if departure_dt <= now: departure_dt += timedelta(days=1)
            departure_time_formatted = departure_dt.strftime('%Y%m%d%H%M')
            params = {"origin": f"{origin['lng']},{origin['lat']}", "destination": f"{destination['lng']},{destination['lat']}", "departure_time": departure_time_formatted}
        except (ValueError, TypeError):
             print(f"Warning: Could not parse departure_time_str '{departure_time_str}'.", file=sys.stderr)
             params = {"origin": f"{origin['lng']},{origin['lat']}", "destination": f"{destination['lng']},{destination['lat']}"}
        try:
            response = requests.get(url, headers=headers, params=params); response.raise_for_status(); data = response.json()
            if 'routes' in data and len(data['routes']) > 0:
                for section in data['routes'][0]['sections']:
                    for road in section['roads']:
                        vertexes = road['vertexes']
                        for j in range(0, len(vertexes), 2): full_path_coords.append([vertexes[j+1], vertexes[j]])
            else: print(f"Kakao API returned no route path for {origin['name']} to {destination['name']}.", file=sys.stderr)
        except requests.exceptions.RequestException as e: print(f"Network error calling Kakao API for {origin['name']} to {destination['name']}: {e}", file=sys.stderr); continue
    return full_path_coords

@app.route('/api/find-best-route', methods=['POST'])
def find_best_route_api():
    data = request.get_json()
    try:
        source_name = data['start']
        dest_name = FIXED_DESTINATION_NAME
        sim_time = data.get('start_time', '09:00')
        user_max_duration = int(data['max_journey_time'])
        capacity = int(data['vehicle_capacity'])
        maximize_coverage = data.get('maximize_coverage', False)
        # MODIFIED: Get mandatory stops from the request payload
        mandatory_stop_ids = [int(sid) for sid in data.get('mandatory_stops', [])]
        source_id = NAME_TO_ID[source_name]
        dest_id = NAME_TO_ID[dest_name]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    all_stops_with_demand = simulate_demand(STOPS_BASE_DATA, sim_time)
    all_stops_map = {s['id']: s for s in all_stops_with_demand}
    
    # MODIFIED: Exclude source, destination, AND mandatory stops from the optional pool
    exclude_ids = {source_id, dest_id}.union(set(mandatory_stop_ids))
    intermediate_stops = [s for s in all_stops_with_demand if s['id'] not in exclude_ids]
    
    search_seed = f"{source_name}-{sim_time}-{capacity}"
    random.seed(search_seed)

    all_candidate_routes = find_optimized_routes(
        source_id, dest_id, intermediate_stops, all_stops_map,
        DURATION_MATRIX_S, SEARCH_TIME_HORIZON_MINS, capacity,
        maximize_coverage=maximize_coverage,
        mandatory_stop_ids=mandatory_stop_ids # Pass mandatory stops to the optimizer
    )

    valid_routes = [
        route for route in all_candidate_routes
        if route['time'] <= user_max_duration
    ]
    
    routes_to_display = valid_routes[:NUM_ROUTES_TO_FIND]

    if not routes_to_display:
        random.seed()
        error_message = "No valid route found. "
        if mandatory_stop_ids:
             error_message += "The selected mandatory stops may make the route impossible within the time limit. Try removing some stops or increasing the trip time."
        else:
             error_message += "Try increasing the trip time."
        return jsonify({"error": error_message})


    routes_for_json = []
    for route_result in routes_to_display:
        current_time_dt = datetime.strptime(sim_time, "%H:%M")
        passengers = 0
        route_ids = route_result['ids']
        route_stop_details = []
        for i, stop_id in enumerate(route_ids):
            stop_info = all_stops_map.get(stop_id)
            if not stop_info: continue
            if i > 0:
                travel_s = DURATION_MATRIX_S[route_ids[i-1]][stop_id]
                current_time_dt += timedelta(seconds=travel_s)
            
            # This logic should be that passengers from the CURRENT stop are added for the NEXT leg of the journey
            # The number of onboard passengers is what it is *after leaving* the stop
            if i < len(route_ids) - 1: # Don't add demand at the final destination
                 passengers += all_stops_map.get(stop_id, {}).get('demand', 0)

            route_stop_details.append({
                "id": stop_info['id'], "name": stop_info['name'], "lat": stop_info['lat'], "lng": stop_info['lng'],
                "demand": all_stops_map.get(stop_id, {}).get('demand', 0), # Show demand at every stop
                "eta": current_time_dt.strftime("%H:%M"),
                "passengers_onboard": passengers
            })
        total_passengers_collected = sum(s['demand'] for s in route_stop_details[:-1]) # Don't count demand at destination
        detailed_path = get_detailed_path_from_kakao(route_stop_details, sim_time)
        routes_for_json.append({
            "stops": route_stop_details, "total_demand": total_passengers_collected,
            "total_time": round(route_result['time'], 2), "detailed_path": detailed_path
        })

    random.seed()
    return jsonify({"routes": routes_for_json})

@app.route('/')
def index():
    selectable_stops = [s for s in STOPS_BASE_DATA if s['name'] != FIXED_DESTINATION_NAME]
    return render_template('index.html', 
                           all_stops=sorted(selectable_stops, key=lambda s: s['name']), destination_name=FIXED_DESTINATION_NAME)

@app.route('/api/all-stops')
def get_all_stops():
    """Provides all stop locations for initial map display."""
    return jsonify(STOPS_BASE_DATA)

if __name__ == '__main__':
    app.run(debug=True, port=5001)