# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import time as _time
from collections import deque

# Reload modules to pick up recent changes
import importlib
import sys

if 'SpikingSystems' in sys.modules:
    importlib.reload(sys.modules['SpikingSystems'])
if 'Controller' in sys.modules:
    importlib.reload(sys.modules['Controller'])
if 'Plots' in sys.modules:
    importlib.reload(sys.modules['Plots'])
if 'Network' in sys.modules:
    importlib.reload(sys.modules['Network'])
if 'DataModels' in sys.modules:
    importlib.reload(sys.modules['DataModels'])

import SpikingSystems as ss
from Controller import Predictor, Controller
from Network import NetworkBuilder
from DataModels import Config, read_data_from_yaml
import Plots

# %%
# 1) Define the spring-mass-damper system and the spike encoder
dt = 0.001
SMD_params = {
    'mass': 1,
    'damping': 0.05,
    'stiffness': .4,
    'dt': dt,
}
# sys = ss.SpringMassDamper(**SMD_params, x0=[-0.1, -0.1])

# Ts = 0.05        # Sampling time
# R = 2.0           # Resistance
# L = 0.5           # Inductance
# Kb = 0.1          # Back-EMF constant
# Km = 0.5          # Motor constant
# Kf = 0.1         # Friction constant
# J = 0.3          # Inertia
motor_params = {
    'R': 2.0,
    'L': 0.5,
    'Kb': 0.1,
    'Km': 0.5,
    'Kf': 0.1,
    'J': 0.3,
    'Ts': dt,
}
sys = ss.DC_motor(**motor_params, x0=[0.2, 0.0])

encoder1 = ss.IFSpikeEncoder_absolute(threshold=0.05, dt=dt)
# encoder2 = IFSpikeEncoder_absolute(threshold=0.05, dt=dt)
encoder3 = ss.DifferentialEncoder(threshold=0.01, dt=dt)
spike_encoder = ss.EncoderArray([encoder1])#, encoder3])
config = 'config.yaml'
config = read_data_from_yaml(config, Config)

controllers, adjacency, ext_in, ext_out = NetworkBuilder.create_network(config, n_inputs_plant=spike_encoder.n_outputs)
# ext_out[1:, :] = 0  # Remove feedback from previous output
# ext_out = np.abs(ext_out) 
# ext_out[:, -1] = -1
# taus = [0.07, 0.1, 0.2, 0.5]
# for i, controller in enumerate(controllers):
#     controller.tau_decay = taus[i]

N = len(controllers)

time = np.arange(0, 100, dt)
u = np.zeros_like(time)
ref_signal = np.zeros_like(time)  # will be set live from slider (logged for later)

# --- Live plot setup (mirrors ss.plot_system_response, but interactive) ---
plt.ion()

fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[3.0, 2.0, 0.6], hspace=0.35)
ax_y = fig.add_subplot(gs[0, 0])
ax_spikes = fig.add_subplot(gs[1, 0], sharex=ax_y)
ax_slider = fig.add_subplot(gs[2, 0])

line_y, = ax_y.plot([], [], linewidth=1.5)
ax_y.set_title("Spring-Mass-Damper System Response")
ax_y.set_ylabel("Displacement")
ax_y.grid(True, alpha=0.3)
ax_y.set_ylim([-0.5, 1.5])

ax_spikes.set_title("Observed Spikes")
ax_spikes.set_ylabel("Input Channel")
ax_spikes.grid(True, alpha=0.3)

slider_ref = Slider(
    ax=ax_slider,
    label="Reference (0-1)",
    valmin=0.0,
    valmax=1.0,
    valinit=0.1,
)
ax_slider.grid(False)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


y = np.zeros_like(time)
spikes_plant = np.zeros((spike_encoder.n_outputs, len(time)))
spikes_network = np.zeros((N, len(time)))
u_prev = np.zeros((N, 1))  # Previous control outputs from all controllers

n_channels = spike_encoder.n_outputs + N
all_spike_times = [deque() for _ in range(n_channels)]

# Keep a constant 10s x-window
plot_window_seconds = 10.0

# Efficient spike raster: one lightweight marker artist per channel (no axis clearing)
spike_artists = []
spike_marker_size = 32  # visual height of each spike in points
spike_marker_edge_width = 2.5
for ch in range(n_channels):
    (artist,) = ax_spikes.plot(
        [],
        [],
        linestyle='None',
        marker='|',
        markersize=spike_marker_size,
        markeredgewidth=spike_marker_edge_width,
        color=colors[ch % len(colors)],
        label=f'x{ch+1}',
    )
    spike_artists.append(artist)

ax_spikes.set_ylim([0.5, n_channels + 0.5])
ax_spikes.set_yticks(range(1, n_channels + 1))
# ax_spikes.set_yticklabels([f'x{i+1}' for i in range(n_channels)])
# if n_channels <= 15:
#     ax_spikes.legend(loc='upper right', ncol=1)

# Smooth real-time pacing: render at a fixed FPS; simulate a fixed number of dt-steps per frame
fps = 60.0
frame_dt = 1.0 / fps
speed = 1.0  # 1.0 = real-time (sim seconds per wall second)
steps_remainder = 0.0
wall_next = _time.perf_counter() + frame_dt

plant_prev_spike = 0.0
step_idx = 0
while step_idx < len(time) and plt.fignum_exists(fig.number):
    # Simulate a fixed amount of sim-time per rendered frame (smooth pacing)
    steps_float = (speed * frame_dt) / dt + steps_remainder
    steps_to_run = int(steps_float)
    steps_remainder = steps_float - steps_to_run
    if steps_to_run < 1:
        steps_to_run = 1
    if step_idx + steps_to_run > len(time):
        steps_to_run = len(time) - step_idx

    for _ in range(steps_to_run):
        t = time[step_idx]

        u_current = np.zeros((N, 1))
        u[step_idx] = (u_prev.T @ ext_out.T).flatten()[0] * config.spiking_network.controller.spike_force

        # Step the plant
        y_val = sys.step(np.asarray(u[step_idx]).reshape(1,))
        y[step_idx] = float(np.asarray(y_val).squeeze())
        spike_plant = spike_encoder.step([y[step_idx] for _ in range(len(spike_encoder.encoders))])
        spikes_plant[:, step_idx] = spike_plant

        for ch in range(spike_encoder.n_outputs):
            if spike_plant[ch] == 1:
                all_spike_times[ch].append(t)

        if spike_plant[0] > 0:
            plant_prev_spike = 0.0
        else:
            plant_prev_spike += dt

        for j, controller in enumerate(controllers):
            connectivity_in_plant = ext_in[:, j]
            x_in_plant = spike_plant[connectivity_in_plant.astype(bool)]
            connectivity_in_neurons = adjacency[:, j]
            x_in_neurons = u_prev[connectivity_in_neurons.astype(bool)]

            x_in = np.vstack(
                (x_in_plant.reshape(-1, 1), x_in_neurons.reshape(-1, 1), u_prev[j].reshape(-1, 1))
            ).flatten()

            # Step the controller (reference from slider)
            ref_spike_period = float(slider_ref.val)
            ref_signal[step_idx] = ref_spike_period
            a_ref_val = np.exp(- (ref_spike_period - plant_prev_spike) / controller.tau_decay)
            a_ref_val = float(np.clip(a_ref_val, 0, 1))
            a_ref = np.zeros(controller.n_inputs)
            a_ref[0] = a_ref_val

            controller.update(x_in, a_ref)
            spike_controller = controller.spike(a_ref)
            spikes_network[j, step_idx] = spike_controller
            u_current[j, 0] = spike_controller

            if spike_controller == 1:
                all_spike_times[spike_encoder.n_outputs + j].append(t)

        u_prev = u_current.copy()
        step_idx += 1

    # Update plot once per frame
    tmax = time[step_idx - 1]
    tmin = max(0.0, tmax - plot_window_seconds)
    tmax_win = tmin + plot_window_seconds

    # Trim old spikes outside the window (keeps CPU/memory stable)
    for dq in all_spike_times:
        while dq and dq[0] < tmin:
            dq.popleft()

    idx0 = int(max(0, np.floor(tmin / dt)))
    line_y.set_data(time[idx0:step_idx], y[idx0:step_idx])
    ax_y.set_xlim([tmin, tmax_win])

    for ch, artist in enumerate(spike_artists):
        st = list(all_spike_times[ch])
        if st:
            artist.set_data(st, [ch + 1] * len(st))
        else:
            artist.set_data([], [])

    fig.suptitle(f"Live reference: {slider_ref.val:.3f}")
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    # Pace to FPS (smooth) and keep UI responsive
    now = _time.perf_counter()
    sleep_s = wall_next - now
    if sleep_s > 0:
        _time.sleep(sleep_s)
    else:
        wall_next = now
    wall_next += frame_dt
    plt.pause(0.001)

    # # Real-time pacing (keeps the simulation from running super fast)
    # if (i % pace_every_n_steps) == 0:
    #     target_wall_elapsed = t
    #     wall_elapsed = _time.perf_counter() - wall_start
    #     sleep_s = target_wall_elapsed - wall_elapsed
    #     if sleep_s > 0:
    #         _time.sleep(min(sleep_s, 0.05))


# Plots.compare_with_reference(time, spikes_plant[0, :], ref_signal)
plt.ioff()
# ss.plot_system_response(time, y[:], np.vstack((spikes_plant, spikes_network)), IF_integral=None)


        








