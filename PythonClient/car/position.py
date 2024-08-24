import json
import unreal

# Modify the JSON file
file_path = 'C:/Users/gavin/OneDrive/Documents/AirSim/settings.json'

with open(file_path, 'r') as f:
    data = json.load(f)

# Modify X and Y values
data['Vehicles']['CPHusky']['X'] = -10.0  # Example new X value
data['Vehicles']['CPHusky']['Y'] = -10.0  # Example new Y value

with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)

# Start the Play-in-Editor (PIE) session in Unreal Engine
unreal.EditorLevelLibrary.editor_play_simulate()

