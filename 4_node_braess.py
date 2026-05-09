import networkx as nx
import matplotlib.pyplot as plt
import random
import os
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

NUM_AGENTS = 4000
VISUAL_AGENT_SCALE = 40
AGENT_DOT_SIZE = 1
AGENT_DOT_ALPHA = 0.5

latency_models = {
    ('Start', 'Top'):    {'type': 'local',   'f': lambda x: x / 100.0, 'label_base': 'x/100'},
    ('Top', 'Target'):   {'type': 'highway', 'f': lambda x: 40.0,      'label_base': '40'},
    ('Start', 'Bottom'): {'type': 'highway', 'f': lambda x: 40.0,      'label_base': '40'},
    ('Bottom', 'Target'):{'type': 'local',   'f': lambda x: x / 100.0, 'label_base': 'x/100'},
    ('Top', 'Bottom'):   {'type': 'shortcut','f': lambda x: 0.0,       'label_base': '0'}
}

traffic_cmap = LinearSegmentedColormap.from_list('traffic', [(0, 1, 0), (1, 1, 0), (1, 0, 0)])

fig_dir = 'figs/4-nodes'
os.makedirs(fig_dir, exist_ok=True)
print(f"Directory '{fig_dir}' prepared for PNG saving.")

G = nx.DiGraph()
edges_def = [('Start', 'Top'), ('Start', 'Bottom'), ('Top', 'Target'), ('Bottom', 'Target'), ('Top', 'Bottom')]
G.add_edges_from(edges_def)
pos = {'Start': (0, 1), 'Top': (1, 2), 'Bottom': (1, 0), 'Target': (2, 1)}

frames_data = []


def run_simulation(shortcut_open):
    available_paths = [0, 1] if not shortcut_open else [0, 1, 2]

    agent_paths = [random.choice(available_paths) for _ in range(NUM_AGENTS)]

    counts = {0: 0, 1: 0, 2: 0}
    for p in agent_paths: counts[p] += 1

    converged = False
    iterations = 0

    while not converged and iterations < 40:
        iterations += 1
        agents = list(range(NUM_AGENTS))
        random.shuffle(agents)

        vol_SA = counts[0] + counts[2]
        vol_AT = counts[0]
        vol_SB = counts[1]
        vol_BT = counts[1] + counts[2]
        vol_AB = counts[2]

        current_flows = {
            ('Start', 'Top'): vol_SA,
            ('Top', 'Target'): vol_AT,
            ('Start', 'Bottom'): vol_SB,
            ('Bottom', 'Target'): vol_BT,
            ('Top', 'Bottom'): vol_AB
        }

        total_time = (vol_SA * latency_models[('Start','Top')]['f'](vol_SA) +
                      vol_AT * latency_models[('Top','Target')]['f'](vol_AT) +
                      vol_SB * latency_models[('Start','Bottom')]['f'](vol_SB) +
                      vol_BT * latency_models[('Bottom','Target')]['f'](vol_BT) +
                      vol_AB * latency_models[('Top','Bottom')]['f'](vol_AB))
        avg_time = total_time / NUM_AGENTS

        frames_data.append({
            'flows': current_flows,
            'shortcut': shortcut_open,
            'iter': iterations,
            'time': avg_time,
            'phase': 2 if shortcut_open else 1
        })

        converged = True
        for agent_id in agents:
            current_path = agent_paths[agent_id]
            v_SA, v_AT = counts[0]+counts[2], counts[0]
            v_SB, v_BT = counts[1], counts[1]+counts[2]
            v_AB = counts[2]

            t0 = latency_models[('Start','Top')]['f'](v_SA) + latency_models[('Top','Target')]['f'](v_AT)
            t1 = latency_models[('Start','Bottom')]['f'](v_SB) + latency_models[('Bottom','Target')]['f'](v_BT)
            t2 = latency_models[('Start','Top')]['f'](v_SA) + latency_models[('Top','Bottom')]['f'](v_AB) + latency_models[('Bottom','Target')]['f'](v_BT)

            path_times = {0: t0, 1: t1, 2: t2}
            valid_times = {p: path_times[p] for p in available_paths}
            best_path = min(valid_times, key=valid_times.get)

            if valid_times[best_path] < valid_times[current_path] - 0.01:
                counts[current_path] -= 1
                counts[best_path] += 1
                agent_paths[agent_id] = best_path
                converged = False

print("Simulating Phase 1 (No Shortcut)...")
run_simulation(False)
for _ in range(5): frames_data.append(frames_data[-1].copy())
print("Simulating Phase 2 (Shortcut Open)...")
run_simulation(True)

for _ in range(8): frames_data.append(frames_data[-1].copy())
print(f"Simulation complete. Total frames generated: {len(frames_data)}")


def render_current_state(ax_plot, frame_data):
    ax_plot.clear()

    visible_edges = edges_def[:-1] if not frame_data['shortcut'] else edges_def

    edge_colors = []
    edge_widths = []
    edge_labels = {}

    for u, v in visible_edges:
        flow = frame_data['flows'].get((u, v), 0)

        traffic_ratio = flow / float(NUM_AGENTS)
        edge_colors.append(traffic_cmap(traffic_ratio))

        edge_widths.append(1 + (6 * traffic_ratio))

        model = latency_models[(u, v)]
        current_cost = model['f'](flow)
        edge_labels[(u, v)] = f"{model['label_base']}\n(T:{current_cost:.1f}, F:{flow})"

    nx.draw_networkx_nodes(G, pos, ax=ax_plot, node_color='lightblue', node_size=2500, edgecolors='black')
    nx.draw_networkx_labels(G, pos, ax=ax_plot, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, ax=ax_plot, edgelist=visible_edges, edge_color=edge_colors, width=edge_widths, arrows=True, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, ax=ax_plot, edge_labels=edge_labels, font_color='black', font_size=8, label_pos=0.5, verticalalignment='bottom')

    for u, v in visible_edges:
        flow = frame_data['flows'].get((u, v), 0)

        num_visual_agents = int(flow // VISUAL_AGENT_SCALE)

        if num_visual_agents > 0:
            start_coord = pos[u]
            end_coord = pos[v]

            t = np.linspace(0.05, 0.95, num_visual_agents)
            agent_x = start_coord[0] + (end_coord[0] - start_coord[0]) * t + np.random.normal(0, 0.02, num_visual_agents)
            agent_y = start_coord[1] + (end_coord[1] - start_coord[1]) * t + np.random.normal(0, 0.02, num_visual_agents)

            ax_plot.scatter(agent_x, agent_y, color='black', s=AGENT_DOT_SIZE, alpha=AGENT_DOT_ALPHA, marker='.')

    phase_title = "Phase 1: No Shortcut" if frame_data['phase'] == 1 else "Phase 2: Shortcut Opened"
    title_text = (f"Routing Efficiency and the Price of Anarchy: Braess's Paradox\n"
                  f"{phase_title} | Iteration: {frame_data['iter']} | Avg Commute: {frame_data['time']:.1f} mins")

    ax_plot.set_title(title_text, fontsize=14)
    ax_plot.axis('off')

main_fig, main_ax = plt.subplots(figsize=(10, 6))

print("\n[Part A: Saving milestone PNGs]...")
ph1_idx = [i for i, f in enumerate(frames_data) if f['phase'] == 1]
ph2_idx = [i for i, f in enumerate(frames_data) if f['phase'] == 2]
milestones = [
    (ph1_idx[-1],  'phase1_equilibrium'),
    (ph2_idx[0],   'phase2_start'),
    (ph2_idx[-1],  'phase2_equilibrium'),
]
for idx, label in milestones:
    render_current_state(main_ax, frames_data[idx])
    filename = f"{fig_dir}/{label}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"  Saved: {filename}")
print(f"Successfully saved {len(milestones)} milestone PNGs to the '{fig_dir}/' directory.")


def update_animation_frame(i):
    data = frames_data[i]
    render_current_state(main_ax, data)

print("\n[Part B: Generating complete GIF animation]...")
ani = animation.FuncAnimation(main_fig, update_animation_frame, frames=len(frames_data), interval=400)

gif_output_path = f'{fig_dir}/braess_paradox_complete_simulation.gif'
print(f" Writing complete GIF to: {gif_output_path} (This takes a moment)...")
ani.save(gif_output_path, writer='pillow', fps=2.5)
print(f"Successfully saved complete GIF to: {gif_output_path}.")
