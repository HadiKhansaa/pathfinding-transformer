import pickle

with open('C:/Users/pc/Desktop/AUB/CMPS396 - LLMs and RAGs/Project/pathfinding-transformer/evaluation_results/run1/evaluation_results_100x100_maze.pkl', 'rb') as f:
    data = pickle.load(f)

# Print or inspect the loaded data
print(type(data))
print(data)