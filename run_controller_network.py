# %%
import matplotlib.pyplot as plt
import numpy as np

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
ref_signal = 0.1 * np.ones_like(time)

y = np.zeros_like(time)
spikes_plant = np.zeros((spike_encoder.n_outputs, len(time)))
spikes_network = np.zeros((N, len(time)))
u_prev = np.zeros((N, 1))  # Previous control outputs from all controllers

plant_prev_spike = 0.0
for i, t in enumerate(time):
    u_current = np.zeros((N, 1))
    u[i] = (u_prev.T @ ext_out.T).flatten()[0] * config.spiking_network.controller.spike_force
    # Step the plant
    y[i] = sys.step(u[i])
    spike_plant = spike_encoder.step([y[i] for _ in range(len(spike_encoder.encoders))])
    spikes_plant[:, i] = spike_plant

    if spike_plant[0] > 0:
        plant_prev_spike = 0.0
    else:
        plant_prev_spike += dt

    for j, controller in enumerate(controllers):
        # Get inputs from plant and other neurons
        # input_from_plant = ext_in[:, j].T @ spike_plant
        # print(input_from_plant)
        connectivity_in_plant = ext_in[:, j]
        x_in_plant = spike_plant[connectivity_in_plant.astype(bool)]
        connectivity_in_neurons = adjacency[:, j]
        x_in_neurons = u_prev[connectivity_in_neurons.astype(bool)]
    
        x_in = np.vstack((x_in_plant.reshape(-1,1), x_in_neurons.reshape(-1,1), u_prev[j].reshape(-1,1))).flatten()
        # Step the controller
        ref_spike_period = ref_signal[i]
        a_ref_val = np.exp(- (ref_spike_period - plant_prev_spike) / controller.tau_decay)
        a_ref_val = float(np.clip(a_ref_val, 0, 1))
        a_ref = np.zeros(controller.n_inputs)
        a_ref[0] = a_ref_val  # Reference applied to first input (plant)

        controller.update(x_in, a_ref)
        spike_controller = controller.spike(a_ref)
        spikes_network[j, i] = spike_controller
        u_current[j, 0] = spike_controller

    u_prev = u_current.copy()


Plots.compare_with_reference(time, spikes_plant[0, :], ref_signal)
ss.plot_system_response(time, y[:], np.vstack((spikes_plant, spikes_network)), IF_integral=None)


        








