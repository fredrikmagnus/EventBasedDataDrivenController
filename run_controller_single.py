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

from SpikingSystems import SpringMassDamper, IFSpikeEncoder_absolute, IFSpikeEncoder, plot_system_response, EncoderArray
from Controller import Predictor, Controller
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
smd = SpringMassDamper(**SMD_params, x0=[-0.1, -0.1])
encoder1 = IFSpikeEncoder(threshold=0.05, dt=dt)
encoder2 = IFSpikeEncoder_absolute(threshold=0.05, dt=dt)
spike_encoder = EncoderArray([encoder1])#, encoder2])

# 2) Define the spike-based predictor
n_inputs = spike_encoder.n_outputs
Q = np.zeros((n_inputs + 1, n_inputs + 1))
Q[0,0] = 1.0 # Only penalize deviation of first state
Q[1,1] = 1.0 # No penalty on second state
Q[-1,-1] = 3e-1 # Small penalty on control input deviation
tau = 0.07
controller = Controller(
    n_inputs=n_inputs,
    gamma_weights=0.99,
    tau_decay=tau,
    lambda_ridge=1e-4,
    Q=Q,
    dt=dt,
)


time = np.arange(0, 100, dt)
u = np.zeros_like(time)


ref_signal = np.ones_like(time)
ref_signal[:len(time)//2] = 0.1
ref_signal[len(time)//2:] = 0.1
# ref_signal = create_ref_signal(period=0.1, time=time)

# Log predictor attributes over time (keep one set, for controller)
Covs = np.zeros((controller.n_inputs, controller.n_inputs, len(time)))
CrossCovs = np.zeros((controller.n_inputs, controller.n_inputs, len(time)))

# Log system and encoder states over time
y = np.zeros(len(time))
spikes = np.zeros((n_inputs, len(time)))
controller_spikes = np.zeros((1, len(time)))
# predictions = np.zeros((n_inputs, len(time)))
predictions = np.zeros((controller.n_inputs, len(time)))
IF_integral = np.zeros((len(time),))
traces = np.zeros((controller.n_inputs, len(time)))

plant_prev_spike = 0.0

for i, t in enumerate(time[:-1]):
    # Step the spring-mass-damper system and spike encoder
    y[i] = smd.step(5 * u[i])
    spike = spike_encoder.step([y[i] for _ in range(n_inputs)])
    spikes[:, i] = spike
    IF_integral[i] = np.sum([encoder.integral for encoder in spike_encoder.encoders])
    # Single controller sees: [plant spikes..., previous output spike]
    x_in = np.concatenate([spike, np.array([u[i]])])

    if spike[0] > 0:
        plant_prev_spike = 0.0
    else:
        plant_prev_spike += dt

    ref_spike_period = ref_signal[i]
    a_ref = np.exp(- (ref_spike_period - plant_prev_spike) / tau)  # Desired spike timing reference
    # a_ref = np.exp(- (ref_signal[i]) / tau)  # Desired spike timing reference
    a_ref_val = float(np.clip(a_ref, 0, 1))
    # Reference: only for plant channels; no reference for output-feedback channel
    a_ref = np.zeros(controller.n_inputs)
    # a_ref[:n_inputs] = a_ref_val
    a_ref[0] = a_ref_val

    
    controller.update(x_in, a_ref)
    spike = controller.spike()

    u[i + 1] = spike
    controller_spikes[:, i] = np.array([spike])

    if x_in.sum() > 0:
        predictions[:, i] = controller.predict_optimal(a_ref)
    else:
        predictions[:, i] = 0.0

    traces[:, i] = controller.z

    # Keep logging the (shared-shape) covariance arrays for the positive controller only (simplest)
    Covs[:, :, i] = controller.Sigma
    CrossCovs[:, :, i] = controller.Psi

# %%
Plots.compare_with_reference(time, spikes[0, :], ref_signal)
# Plot controller's internal channels (plant + output)
a = np.vstack((spikes, controller_spikes))
Plots.compare_time_predictions(time, a, predictions, controller.tau_decay)
# Plots.compare_predictions(time, spikes, controller_spikes)
Plots.plot_traces(time, traces)
# Plots.plot_covariances(time, Covs, CrossCovs)

# %%
plot_system_response(time, y[:], np.vstack((spikes, controller_spikes)), IF_integral)


