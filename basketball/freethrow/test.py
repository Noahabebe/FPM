import numpy as np
import json
import os

def load_data(filepath):
    """Loads JSON data from a given filepath."""
    with open(filepath, 'r') as file:
        return json.load(file)

processed_data = {}

# Loop through all 125 trial files
for trial_number in range(1, 126): 
    trial_id = str(trial_number).zfill(4)
    
    data_file = f'basketball/freethrow/data/P0001/BB_FT_P0001_T{trial_id}.json'
    
    if os.path.isfile(data_file):
        # Load trial data
        data = load_data(data_file)
        trial_id = data['trial_id']
        result = data['result']
        tracking_data = data['tracking']
        
        distances = {
            "elbow_distance": [],
            "wrist_distance": [],
            "hand_distance": []
        }

        for frame in tracking_data:
            player_data = frame['data']['player']
            
            # Check if necessary points are available
            if all(k in player_data for k in ['L_SHOULDER', 'R_SHOULDER', 'R_ELBOW', 'R_WRIST', 'R_1STFINGER']):
                # Calculate shoulder midpoint and distances
                L_SHOULDER = np.array(player_data['L_SHOULDER'])
                R_SHOULDER = np.array(player_data['R_SHOULDER'])
                shoulder_midline = (L_SHOULDER + R_SHOULDER) / 2

                R_ELBOW = np.array(player_data['R_ELBOW'])
                R_WRIST = np.array(player_data['R_WRIST'])
                R_1STFINGER = np.array(player_data['R_1STFINGER'])

                elbow_distance = np.linalg.norm(R_ELBOW - shoulder_midline)
                wrist_distance = np.linalg.norm(R_WRIST - shoulder_midline)
                hand_distance = np.linalg.norm(R_1STFINGER - shoulder_midline)

                distances['elbow_distance'].append(elbow_distance)
                distances['wrist_distance'].append(wrist_distance)
                distances['hand_distance'].append(hand_distance)
            else:
                print(f"Skipping frame in trial {trial_id} due to missing points.")
        
        # Store distances and result for the trial if we have valid data
        if distances['elbow_distance']:
            processed_data[trial_id] = {
                "result": result,
                "distances": distances
            }
        else:
            print(f"No valid data for trial {trial_id}")
    else:
        print(f"File not found: {data_file}")

# Analyze results: Calculate average distances for successful and missed shots
success_distances = {
    "elbow_distance": [],
    "wrist_distance": [],
    "hand_distance": []
}
miss_distances = {
    "elbow_distance": [],
    "wrist_distance": [],
    "hand_distance": []
}

for trial_id, data in processed_data.items():
    result = data["result"]
    distances = data["distances"]

    # Calculate the mean distance for each type in this trial
    if distances["elbow_distance"]:
        avg_elbow_dist = np.mean(distances["elbow_distance"])
        avg_wrist_dist = np.mean(distances["wrist_distance"])
        avg_hand_dist = np.mean(distances["hand_distance"])

        # Categorize based on success or miss
        if result == "made":
            success_distances["elbow_distance"].append(avg_elbow_dist)
            success_distances["wrist_distance"].append(avg_wrist_dist)
            success_distances["hand_distance"].append(avg_hand_dist)
        else:
            miss_distances["elbow_distance"].append(avg_elbow_dist)
            miss_distances["wrist_distance"].append(avg_wrist_dist)
            miss_distances["hand_distance"].append(avg_hand_dist)

# Calculate overall averages for successful and missed shots if data is available
success_avg_distances = {k: np.mean(v) if v else np.nan for k, v in success_distances.items()}
miss_avg_distances = {k: np.mean(v) if v else np.nan for k, v in miss_distances.items()}

# Output the results
print("Average distances for successful shots:")
print(success_avg_distances)
print("\nAverage distances for missed shots:")
print(miss_avg_distances)
