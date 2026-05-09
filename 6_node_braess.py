import os
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


NUM_AGENTS = 4000
VISUAL_AGENT_SCALE = 40
AGENT_DOT_SIZE = 1
AGENT_DOT_ALPHA = 0.5


pos = {
    'Start':   (0.0, 1.5),
    'TopLeft': (2.0, 3.0),
    'Center':  (2.0, 1.5),
    'BotLeft': (2.0, 0.0),
    'Hub':     (4.0, 1.5),
    'Target':  (6.0, 1.5),
}


latency_models = {
    ('Start',   'TopLeft'):  {'type': 'local',   'f': lambda x: x / 50.0,  'label_base': 'x/50'},
    ('Start',   'Center'):   {'type': 'highway', 'f': lambda x: 80.0,      'label_base': '80'},
    ('Start',   'BotLeft'):  {'type': 'highway', 'f': lambda x: 80.0,      'label_base': '80'},
    ('TopLeft', 'Hub'):      {'type': 'highway', 'f': lambda x: 80.0,      'label_base': '80'},
    ('Center',  'Hub'):      {'type': 'highway', 'f': lambda x: 80.0,      'label_base': '80'},
    ('BotLeft', 'Hub'):      {'type': 'local',   'f': lambda x: x / 50.0,  'label_base': 'x/50'},
    ('Hub',     'Target'):   {'type': 'free',    'f': lambda x: 0.0,       'label_base': '0'},
    ('TopLeft', 'Center'):   {'type': 'shortcut','f': lambda x: 0.0,       'label_base': '0'},
    ('Center',  'BotLeft'):  {'type': 'shortcut','f': lambda x: 0.0,       'label_base': '0'},
}

standard_edges = [
    ('Start', 'TopLeft'), ('Start', 'Center'), ('Start', 'BotLeft'),
    ('TopLeft', 'Hub'), ('Center', 'Hub'), ('BotLeft', 'Hub'),
    ('Hub', 'Target'),
]
shortcut_edges = [('TopLeft', 'Center'), ('Center', 'BotLeft')]
all_edges = standard_edges + shortcut_edges


PATHS = {
    0: [('Start', 'TopLeft'), ('TopLeft', 'Hub'), ('Hub', 'Target')],
    1: [('Start', 'Center'),  ('Center',  'Hub'), ('Hub', 'Target')],
    2: [('Start', 'BotLeft'), ('BotLeft', 'Hub'), ('Hub', 'Target')],

    3: [('Start', 'TopLeft'), ('TopLeft', 'Center'), ('Center',  'Hub'),     ('Hub', 'Target')],
    4: [('Start', 'Center'),  ('Center',  'BotLeft'), ('BotLeft', 'Hub'),    ('Hub', 'Target')],
    5: [('Start', 'TopLeft'), ('TopLeft', 'Center'), ('Center',  'BotLeft'), ('BotLeft', 'Hub'), ('Hub', 'Target')],
}


traffic_cmap = LinearSegmentedColormap.from_list('traffic', [(0, 1, 0), (1, 1, 0), (1, 0, 0)])

fig_dir = 'figs/6-nodes'
os.makedirs(fig_dir, exist_ok=True)
print(f"Directory '{fig_dir}' prepared for PNG saving.")

G = nx.DiGraph()
G.add_edges_from(all_edges)


frames_data = []


def make_flows(agent_paths):
    """Return a dict mapping each edge to its current agent flow count."""
    flows = {e: 0 for e in latency_models}
    for p in agent_paths:
        for e in PATHS[p]:
            flows[e] += 1
    return flows


def path_cost(path_id, flows):
    """Sum the latency of every edge on the given route using current flows."""
    return sum(latency_models[e]['f'](flows[e]) for e in PATHS[path_id])


def run_simulation(shortcut_open):
    available = [0, 1, 2] if not shortcut_open else list(PATHS.keys())


    agent_paths = [random.choice(available) for _ in range(NUM_AGENTS)]

    flows = make_flows(agent_paths)

    converged = False
    iterations = 0

    while not converged and iterations < 60:
        iterations += 1


        total_t = sum(flows[e] * latency_models[e]['f'](flows[e]) for e in flows)
        frames_data.append({
            'flows':    dict(flows),
            'shortcut': shortcut_open,
            'iter':     iterations,
            'time':     total_t / NUM_AGENTS,
            'phase':    2 if shortcut_open else 1,
        })


        agents = list(range(NUM_AGENTS))
        random.shuffle(agents)

        converged = True
        for aid in agents:
            cur = agent_paths[aid]
            cur_cost = path_cost(cur, flows)


            best, best_cost = cur, cur_cost
            for p in available:
                c = path_cost(p, flows)
                if c < best_cost:
                    best_cost, best = c, p

            if best_cost < cur_cost - 0.01:

                for e in PATHS[cur]:
                    flows[e] -= 1
                for e in PATHS[best]:
                    flows[e] += 1
                agent_paths[aid] = best
                converged = False


print("Simulating Phase 1 (No Shortcuts)...")
run_simulation(False)

for _ in range(5):
    frames_data.append(frames_data[-1].copy())

print("Simulating Phase 2 (Shortcuts Opened)...")
run_simulation(True)

for _ in range(8):
    frames_data.append(frames_data[-1].copy())

print(f"Simulation complete. Total frames generated: {len(frames_data)}")


def render_frame(ax, frame_data):
    ax.clear()

    visible = standard_edges if not frame_data['shortcut'] else all_edges

    edge_colors, edge_widths, edge_labels = [], [], {}
    for edge in visible:
        flow  = frame_data['flows'].get(edge, 0)
        ratio = flow / float(NUM_AGENTS)
        edge_colors.append(traffic_cmap(ratio))
        edge_widths.append(1 + 6 * ratio)
        model = latency_models[edge]
        edge_labels[edge] = (
            f"{model['label_base']}\n"
            f"(T:{model['f'](flow):.1f}, F:{flow})"
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color='lightblue', node_size=2000, edgecolors='black',
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=visible,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=20,
        connectionstyle='arc3,rad=0.08',
    )
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=edge_labels,
        font_size=7,
        label_pos=0.45,
        verticalalignment='bottom',
    )


    for edge in visible:
        flow  = frame_data['flows'].get(edge, 0)
        n_vis = int(flow // VISUAL_AGENT_SCALE)
        if n_vis > 0:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            t = np.linspace(0.1, 0.9, n_vis)
            ax.scatter(
                x0 + (x1 - x0) * t + np.random.normal(0, 0.04, n_vis),
                y0 + (y1 - y0) * t + np.random.normal(0, 0.04, n_vis),
                color='black', s=AGENT_DOT_SIZE, alpha=AGENT_DOT_ALPHA, marker='.',
            )

    phase_str = (
        "Phase 1: No Shortcuts (Nash eq ≈ 120 min)"
        if frame_data['phase'] == 1
        else "Phase 2: Shortcuts Opened (Nash eq ≈ 160 min — WORSE!)"
    )
    ax.set_title(
        f"Extended Braess's Paradox — 6-Node Network\n"
        f"{phase_str}  |  Iter: {frame_data['iter']}  |  "
        f"Avg Commute: {frame_data['time']:.1f} min",
        fontsize=13,
    )
    ax.axis('off')


main_fig, main_ax = plt.subplots(figsize=(12, 7))

print("\n[Part A: Saving milestone PNGs]...")
ph1_idx = [i for i, f in enumerate(frames_data) if f['phase'] == 1]
ph2_idx = [i for i, f in enumerate(frames_data) if f['phase'] == 2]
milestones = [
    (ph1_idx[-1],  'phase1_equilibrium'),
    (ph2_idx[0],   'phase2_start'),
    (ph2_idx[-1],  'phase2_equilibrium'),
]
for idx, label in milestones:
    render_frame(main_ax, frames_data[idx])
    filename = f"{fig_dir}/{label}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"  Saved: {filename}")
print(f"Successfully saved {len(milestones)} milestone PNGs to '{fig_dir}/'.")


print("\n[Part B: Generating complete GIF animation]...")
ani = animation.FuncAnimation(
    main_fig,
    lambda i: render_frame(main_ax, frames_data[i]),
    frames=len(frames_data),
    interval=400,
)

gif_path = f"{fig_dir}/extended_braess_simulation.gif"
print(f"  Writing GIF to: {gif_path}  (this takes a moment)...")
ani.save(gif_path, writer='pillow', fps=2.5)
print(f"Successfully saved GIF to: {gif_path}")
