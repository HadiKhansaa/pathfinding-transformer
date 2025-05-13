import pickle

with open('C:/Users/pc/Desktop/AUB/CMPS396 - LLMs and RAGs/Project/pathfinding-transformer/evaluation_results/game_maps_latest/eval_results_game_grids.pkl', 'rb') as f:
    data = pickle.load(f)

# Print or inspect the loaded data
print(type(data))
print(data)