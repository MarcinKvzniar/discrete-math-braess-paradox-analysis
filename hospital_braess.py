import os
import random

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


NUM_PATIENTS = 4_000
VISUAL_PATIENT_SCALE = 50

traffic_cmap = LinearSegmentedColormap.from_list(
    'traffic', [(0.12, 0.72, 0.12), (0.95, 0.82, 0.0), (0.88, 0.10, 0.10)]
)

fig_dir = 'figs/hospital-braess'
os.makedirs(fig_dir, exist_ok=True)
print(f"Output directory '{fig_dir}' ready.")


EDGES = {
    ('Triage',  'Ward A'):     {'f': lambda x: x / 100.0, 'label': 'x/100  (load-sensitive)'},
    ('Ward A',  'Discharge'):  {'f': lambda x: 50.0,      'label': '50 min  (fixed)'},
    ('Triage',  'Ward B'):     {'f': lambda x: 50.0,      'label': '50 min  (fixed)'},
    ('Ward B',  'Discharge'):  {'f': lambda x: x / 100.0, 'label': 'x/100  (load-sensitive)'},
    ('Ward A',  'Ward B'):     {'f': lambda x: 0.0,       'label': '0 min   (fast-track)'},
}

standard_edges = [
    ('Triage', 'Ward A'), ('Ward A', 'Discharge'),
    ('Triage', 'Ward B'), ('Ward B', 'Discharge'),
]
shortcut_edge = ('Ward A', 'Ward B')

PATHS = {
    0: [('Triage', 'Ward A'), ('Ward A', 'Discharge')],
    1: [('Triage', 'Ward B'), ('Ward B', 'Discharge')],
    2: [('Triage', 'Ward A'), ('Ward A', 'Ward B'), ('Ward B', 'Discharge')],
}


frames_data = []
_history = {'phase1': [], 'phase2': []}


def _make_flows(agent_paths):
    flows = {e: 0 for e in EDGES}
    for p in agent_paths:
        for e in PATHS[p]:
            flows[e] += 1
    return flows


def _path_cost(path_id, flows):
    return sum(EDGES[e]['f'](flows[e]) for e in PATHS[path_id])


def run_simulation(shortcut_open):
    available = [0, 1] if not shortcut_open else list(PATHS.keys())
    agent_paths = [random.choice(available) for _ in range(NUM_PATIENTS)]
    flows = _make_flows(agent_paths)
    key = 'phase2' if shortcut_open else 'phase1'

    converged = False
    iterations = 0
    while not converged and iterations < 55:
        iterations += 1

        total_t = sum(flows[e] * EDGES[e]['f'](flows[e]) for e in flows)
        avg_wait = total_t / NUM_PATIENTS
        _history[key].append(avg_wait)

        frames_data.append({
            'flows':    dict(flows),
            'shortcut': shortcut_open,
            'iter':     iterations,
            'time':     avg_wait,
            'phase':    2 if shortcut_open else 1,
            'ph1_hist': list(_history['phase1']),
            'ph2_hist': list(_history['phase2']),
        })

        agents = list(range(NUM_PATIENTS))
        random.shuffle(agents)
        converged = True
        for aid in agents:
            cur = agent_paths[aid]
            cur_cost = _path_cost(cur, flows)
            best, best_cost = cur, cur_cost
            for p in available:
                c = _path_cost(p, flows)
                if c < best_cost:
                    best_cost, best = c, p
            if best_cost < cur_cost - 0.01:
                for e in PATHS[cur]:   flows[e] -= 1
                for e in PATHS[best]:  flows[e] += 1
                agent_paths[aid] = best
                converged = False


print("Simulating Phase 1  (No Fast-Track)...")
run_simulation(False)
_hold = frames_data[-1]
for _ in range(5):
    frames_data.append(_hold.copy())

print("Simulating Phase 2  (Fast-Track Opened)...")
run_simulation(True)
_hold = frames_data[-1]
for _ in range(8):
    frames_data.append(_hold.copy())

print(f"Simulation complete. {len(frames_data)} frames generated.")


ROOM_GEOM = {

    'Entrance':  (0.02, 0.40, 0.12, 0.15),
    'Triage':    (0.25, 0.40, 0.15, 0.15),
    'Ward A':    (0.55, 0.62, 0.16, 0.15),
    'Ward B':    (0.55, 0.18, 0.16, 0.15),
    'Discharge': (0.82, 0.40, 0.15, 0.15),
}

ROOM_LABELS = {
    'Entrance':  'ENTRANCE',
    'Triage':    'TRIAGE /\nREGISTRATION',
    'Ward A':    'WARD A\n(fast clerk)',
    'Ward B':    'WARD B\n(specialist)',
    'Discharge': 'DISCHARGE\n/ EXIT',
}

CORRIDORS = [

    ('Entrance',  'Triage',    None),
    ('Triage',    'Ward A',    ('Triage', 'Ward A')),
    ('Triage',    'Ward B',    ('Triage', 'Ward B')),
    ('Ward A',    'Discharge', ('Ward A', 'Discharge')),
    ('Ward B',    'Discharge', ('Ward B', 'Discharge')),
]


def _center(name):
    x, y, w, h = ROOM_GEOM[name]
    return x + w / 2, y + h / 2


def _room_utilization(name, flows):
    """Fraction of patients flowing through or into a room."""
    if name == 'Ward A':
        return flows.get(('Triage', 'Ward A'), 0) / NUM_PATIENTS
    if name == 'Ward B':
        return flows.get(('Ward B', 'Discharge'), 0) / NUM_PATIENTS
    if name == 'Triage':
        return (flows.get(('Triage', 'Ward A'), 0)
                + flows.get(('Triage', 'Ward B'), 0)) / NUM_PATIENTS

    return 1.0


def draw_floorplan(ax, fd):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    flows = fd['flows']


    ax.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle='round,pad=0.015',
        fc='#f2ede0', ec='#8a9bb0', lw=2.0, zorder=0,
    ))
    ax.text(0.5, 0.965, 'CITY GENERAL HOSPITAL — Emergency Department',
            ha='center', va='top', fontsize=8.5, fontweight='bold',
            color='#1a2c40', transform=ax.transAxes)


    for frm, to, edge_key in CORRIDORS:
        cx0, cy0 = _center(frm)
        cx1, cy1 = _center(to)
        flow = flows.get(edge_key, 0) if edge_key else NUM_PATIENTS
        ratio = flow / NUM_PATIENTS
        color = traffic_cmap(ratio)
        lw = 1.5 + 6.5 * ratio
        ax.annotate('', xy=(cx1, cy1), xytext=(cx0, cy0),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=lw, mutation_scale=14),
                    zorder=2)

        n_vis = int(flow // VISUAL_PATIENT_SCALE)
        if n_vis > 0:
            t = np.linspace(0.2, 0.8, n_vis)
            ax.scatter(
                cx0 + (cx1 - cx0) * t + np.random.normal(0, 0.008, n_vis),
                cy0 + (cy1 - cy0) * t + np.random.normal(0, 0.008, n_vis),
                s=9, c='#1c2e4a', alpha=0.55, zorder=3,
            )


    if fd['shortcut']:
        cx0, cy0 = _center('Ward A')
        cx1, cy1 = _center('Ward B')
        flow_ft = flows.get(shortcut_edge, 0)
        ratio_ft = flow_ft / NUM_PATIENTS
        ax.annotate('', xy=(cx1, cy1), xytext=(cx0, cy0),
                    arrowprops=dict(
                        arrowstyle='->',
                        color='#00b894',
                        lw=2.0 + 7 * ratio_ft,
                        mutation_scale=15,
                        connectionstyle='arc3,rad=-0.35',
                    ),
                    zorder=2)
        mid_x = (cx0 + cx1) / 2 + 0.10
        mid_y = (cy0 + cy1) / 2
        ax.text(mid_x, mid_y, f'⚡ FAST-TRACK\n{flow_ft} pts',
                fontsize=7.5, color='#00856a', fontweight='bold',
                ha='center', va='center', zorder=5)
        n_vis = int(flow_ft // VISUAL_PATIENT_SCALE)
        if n_vis > 0:
            t = np.linspace(0.2, 0.8, n_vis)
            ax.scatter(
                cx0 + (cx1 - cx0) * t + np.random.normal(0, 0.01, n_vis),
                cy0 + (cy1 - cy0) * t + np.random.normal(0, 0.01, n_vis),
                s=9, c='#00856a', alpha=0.65, zorder=3,
            )


    for name, (rx, ry, rw, rh) in ROOM_GEOM.items():
        util = _room_utilization(name, flows)
        base = np.array(traffic_cmap(util)[:3])
        pastel = base * 0.38 + np.ones(3) * 0.62
        ax.add_patch(mpatches.FancyBboxPatch(
            (rx, ry), rw, rh, boxstyle='round,pad=0.012',
            fc=pastel, ec='#2c3e50', lw=2.0, zorder=4,
        ))
        cx, cy = rx + rw / 2, ry + rh / 2
        ax.text(cx, cy + 0.015, ROOM_LABELS[name],
                ha='center', va='center', fontsize=7.5, fontweight='bold',
                color='#1a1a2e', zorder=5)

        if name in ('Ward A', 'Ward B'):
            load_edge = ('Triage', 'Ward A') if name == 'Ward A' else ('Ward B', 'Discharge')
            cnt = flows.get(load_edge, 0)
            ax.text(cx, cy - 0.030, f'{cnt} patients',
                    ha='center', va='center', fontsize=6.5, color='#444', zorder=5)


    phase_txt = ('  Phase 1: standard routing, no fast-track'
                 if fd['phase'] == 1
                 else '  Phase 2: FAST-TRACK OPEN — everyone uses it!')
    ax.text(0.5, 0.025, phase_txt, ha='center', va='bottom',
            fontsize=8, color='#2c3e50', style='italic', transform=ax.transAxes)


def draw_bars(ax, fd):
    ax.clear()
    flows = fd['flows']
    edges = standard_edges + ([shortcut_edge] if fd['shortcut'] else [])

    labels  = [f"{u} → {v}" for u, v in edges]
    values  = [flows.get(e, 0) for e in edges]
    colors  = [traffic_cmap(v / NUM_PATIENTS) for v in values]
    lat_txt = [f"T = {EDGES[e]['f'](flows.get(e, 0)):.0f} min" for e in edges]

    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor='#3a3a4a', lw=0.9, height=0.58)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, NUM_PATIENTS * 1.15)
    ax.set_xlabel('Number of patients in corridor', fontsize=8)
    ax.set_title('Live Patient Load per Corridor', fontsize=9, fontweight='bold', pad=5)

    ax.axvline(NUM_PATIENTS // 2, color='#aaa', lw=1, ls='--')
    ax.text(NUM_PATIENTS // 2 + 40, len(labels) - 0.65,
            'N/2', fontsize=6.5, color='#999', va='top')

    for bar, val, lt in zip(bars, values, lat_txt):
        ax.text(val + 50, bar.get_y() + bar.get_height() / 2,
                f'{val}    ({lt})', va='center', fontsize=7, color='#222')

    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7.5)


def draw_history(ax, fd):
    ax.clear()
    p1 = fd['ph1_hist']
    p2 = fd['ph2_hist']

    if p1:
        ax.plot(range(1, len(p1) + 1), p1,
                color='#2979FF', lw=2.2, marker='o', ms=3.5,
                label='Phase 1 — no fast-track')
    if p2:
        x2 = range(len(p1) + 1, len(p1) + len(p2) + 1)
        ax.plot(x2, p2,
                color='#FF5252', lw=2.2, marker='s', ms=3.5,
                label='Phase 2 — fast-track open')


    ax.axhline(70, color='#2979FF', lw=1.1, ls=':', alpha=0.65)
    ax.axhline(80, color='#FF5252', lw=1.1, ls=':', alpha=0.65)
    ax.text(0.995, 70, ' 70 min  (Phase 1 Nash eq.)', fontsize=6.5, color='#2979FF',
            ha='right', va='bottom', transform=ax.get_yaxis_transform())
    ax.text(0.995, 80, ' 80 min  (Phase 2 Nash eq.)', fontsize=6.5, color='#FF5252',
            ha='right', va='bottom', transform=ax.get_yaxis_transform())


    if p1 and p2:
        sep = len(p1) + 0.5
        ax.axvline(sep, color='#2ecc71', lw=1.8, ls='--', alpha=0.75)
        ylo = max(0, min(p1 + p2) - 5)
        ax.text(sep + 0.2, ylo + 0.3, ' Fast-Track\n   opens',
                fontsize=7, color='#27ae60', va='bottom')

    ax.set_xlabel('Iteration', fontsize=8.5)
    ax.set_ylabel('Avg wait (min)', fontsize=8.5)
    ax.set_title("Average Patient Wait Time — Braess's Paradox Emerges",
                 fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.85)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7.5)

    all_vals = p1 + p2
    if all_vals:
        ax.set_ylim(bottom=max(0, min(all_vals) - 5))


main_fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(
    2, 2, figure=main_fig, width_ratios=[1.3, 1],
    left=0.05, right=0.97, top=0.91, bottom=0.07,
    hspace=0.40, wspace=0.30,
)
ax_floorplan = main_fig.add_subplot(gs[0, 0])
ax_bars      = main_fig.add_subplot(gs[0, 1])
ax_history   = main_fig.add_subplot(gs[1, :])
_axes = (ax_floorplan, ax_bars, ax_history)


def _render(fd):
    draw_floorplan(ax_floorplan, fd)
    draw_bars(ax_bars, fd)
    draw_history(ax_history, fd)
    paradox_note = (
        '     BRAESS PARADOX: the Fast-Track INCREASED average wait time!'
        if fd['phase'] == 2 and fd['time'] > 71 else ''
    )
    main_fig.suptitle(
        f"Braess's Paradox — Hospital Emergency Dept. Queuing Network\n"
        f"Phase {fd['phase']}  |  Iteration {fd['iter']}  |  "
        f"Avg Wait: {fd['time']:.1f} min{paradox_note}",
        fontsize=11, fontweight='bold', y=0.975,
    )


print("\n[Part A: Saving milestone PNGs]...")
ph1_idx = [i for i, f in enumerate(frames_data) if f['phase'] == 1]
ph2_idx = [i for i, f in enumerate(frames_data) if f['phase'] == 2]
milestones = [
    (ph1_idx[-1],  'phase1_equilibrium'),
    (ph2_idx[0],   'phase2_start'),
    (ph2_idx[-1],  'phase2_equilibrium'),
]
for idx, label in milestones:
    _render(frames_data[idx])
    filename = f"{fig_dir}/{label}.png"
    main_fig.savefig(filename, dpi=180, bbox_inches='tight')
    print(f"  Saved: {filename}")
print(f"Done. {len(milestones)} milestone PNGs saved to '{fig_dir}/'.")


print("\n[Part B: Building GIF]...")
ani = animation.FuncAnimation(
    main_fig,
    lambda i: _render(frames_data[i]),
    frames=len(frames_data),
    interval=400,
)
gif_path = f"{fig_dir}/hospital_braess_simulation.gif"
print(f"  Writing: {gif_path}")
ani.save(gif_path, writer='pillow', fps=2.5)
print(f"Successfully saved GIF: {gif_path}")
