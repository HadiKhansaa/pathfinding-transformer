import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load the data ---
with open('C:/Users/pc/Desktop/AUB/CMPS396 - LLMs and RAGs/Project/pathfinding-transformer/evaluation_results/game_maps_latest/eval_results_game_grids.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['detailed_results']

# --- 2. Collect metrics only when both succeed ---
astar_lengths = []
astar_nodes   = []
astar_times   = []

xfm_lengths = []
xfm_steps   = []
xfm_times   = []

for case in results:
    a = case['astar']
    t = case['transformer']
    if a['status']=='success' and t['status']=='success':
        astar_lengths.append(a['length'])
        astar_nodes.append(  a['nodes'])
        astar_times.append( a['time'] * 1000 )  # s → ms

        # transformer stores 'length' only when success, and always 'steps'
        xfm_lengths.append(t.get('length', t['steps']))
        xfm_steps.append(  t['steps'])
        xfm_times.append(   t['time'] * 1000 )

# --- 3. Compute averages ---
astar_avg_length = np.mean(astar_lengths)
astar_avg_nodes  = np.mean(astar_nodes)
astar_avg_time   = np.mean(astar_times)

xfm_avg_length  = np.mean(xfm_lengths)
xfm_avg_steps   = np.mean(xfm_steps)
xfm_avg_time    = np.mean(xfm_times)

# --- 4. Plot ---
labels = ['Avg Path Length', 'Avg Nodes/Steps', 'Avg Time (ms)']
astar_vals     = [astar_avg_length, astar_avg_nodes, astar_avg_time]
transformer_vals = [xfm_avg_length, xfm_avg_steps, xfm_avg_time]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8,4))
rects1 = ax.bar(x - width/2, astar_vals,     width, label='A* (on mutual success)')
rects2 = ax.bar(x + width/2, transformer_vals, width, label='Transformer (on mutual success)')

ax.set_title('Comparison (Successful Runs) Game Maps')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for rect in rects1 + rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}',
                xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0,3),
                textcoords='offset points',
                ha='center', va='bottom')

plt.tight_layout()
plt.show()
