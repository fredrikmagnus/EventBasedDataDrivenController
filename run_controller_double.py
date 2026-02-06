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
smd = SpringMassDamper(**SMD_params, x0=[0., 0.])
encoder1 = IFSpikeEncoder_absolute(threshold=0.05, dt=dt)
encoder2 = IFSpikeEncoder_absolute(threshold=0.05, dt=dt)
spike_encoder = EncoderArray([encoder1])#, encoder2])

# 2) Define the spike-based predictor
n_inputs = spike_encoder.n_outputs
# predictor = Predictor(
#     n_inputs=n_inputs,
#     gamma_weights=0.99,
#     tau_decay=0.2,
#     lambda_ridge=1e-4,
#     dt=dt,
#     estimate_mean=False
# )
tau = 0.07
controller_pos = Controller(
    n_inputs=n_inputs + 1,  # +1 for the other controller's output spike
    gamma_weights=0.99,
    tau_decay=tau,
    lambda_ridge=1e-4,
    dt=dt,
)
controller_neg = Controller(
    n_inputs=n_inputs + 1,  # +1 for the other controller's output spike
    gamma_weights=0.99,
    tau_decay=tau,
    lambda_ridge=1e-4,
    dt=dt,
)


time = np.arange(0, 50, dt)
u = np.zeros_like(time)


ref_signal = np.ones_like(time)
ref_signal[:len(time)//2] = 0.1
ref_signal[len(time)//2:] = 0.1

# Log predictor attributes over time (keep one set, for controller_pos)
Covs = np.zeros((controller_pos.n_inputs, controller_pos.n_inputs, len(time)))
CrossCovs = np.zeros((controller_pos.n_inputs, controller_pos.n_inputs, len(time)))

# Log system and encoder states over time
y = np.zeros(len(time))
spikes = np.zeros((n_inputs, len(time)))
controller_spikes = np.zeros((2, len(time)))
# predictions = np.zeros((n_inputs, len(time)))
predictions_pos = np.zeros((controller_pos.n_inputs, len(time)))
predictions_neg = np.zeros((controller_neg.n_inputs, len(time)))
IF_integral = np.zeros((len(time),))
traces_pos = np.zeros((controller_pos.n_inputs, len(time)))
traces_neg = np.zeros((controller_neg.n_inputs, len(time)))

plant_prev_spike = 0.0

# Impulse magnitudes to the plant (equal magnitude, opposite polarity)
impulse_gain = 1

for i, t in enumerate(time[:-1]):
    # Step the spring-mass-damper system and spike encoder
    y[i] = smd.step(5 * u[i])
    spike = spike_encoder.step([y[i] for _ in range(n_inputs)])
    spikes[:, i] = spike
    IF_integral[i] = np.sum([encoder.integral for encoder in spike_encoder.encoders])

    if spike[0] > 0:
        plant_prev_spike = 0.0
    else:
        plant_prev_spike += dt

    ref_spike_period = ref_signal[i]
    a_ref = np.exp(- (ref_spike_period - plant_prev_spike) / tau)  # Desired spike timing reference
    a_ref_val = float(np.clip(a_ref, 0, 1))

    # Each controller sees: [plant spikes..., other controller spike, own output spike]
    # NOTE: The Controller class automatically reserves the final channel for its own output feedback.
    spike_pos = controller_pos.spike()
    spike_neg = controller_neg.spike()

    x_in_pos = np.concatenate([spike, np.array([0*spike_neg, spike_pos])])
    x_in_neg = np.concatenate([spike, np.array([spike_pos, 0*spike_neg])])

    # Reference: only for plant channels; no reference for coupling/output channels
    a_ref_pos = np.zeros(controller_pos.n_inputs)
    a_ref_neg = np.zeros(controller_neg.n_inputs)
    a_ref_pos[:n_inputs] = a_ref_val
    a_ref_neg[:n_inputs] = a_ref_val

    controller_pos.update(x_in_pos, a_ref_pos)
    controller_neg.update(x_in_neg, a_ref_neg)

    # Opposite polarity impulses to the plant
    u[i + 1] = impulse_gain * (spike_pos - 0*spike_neg)
    controller_spikes[:, i] = np.array([spike_pos, spike_neg])

    if x_in_pos.sum() > 0:
        predictions_pos[:, i] = controller_pos.predict_optimal(a_ref_pos)
    else:
        predictions_pos[:, i] = 0.0

    if x_in_neg.sum() > 0:
        predictions_neg[:, i] = controller_neg.predict_optimal(a_ref_neg)
    else:
        predictions_neg[:, i] = 0.0

    traces_pos[:, i] = controller_pos.z
    traces_neg[:, i] = controller_neg.z

    # Keep logging the (shared-shape) covariance arrays for the positive controller only (simplest)
    Covs[:, :, i] = controller_pos.Sigma
    CrossCovs[:, :, i] = controller_pos.Psi

# %%
Plots.compare_with_reference(time, spikes[0, :], ref_signal)
# Plot each controller's internal channels separately (plant + coupling + output)
a_pos = np.vstack((spikes, controller_spikes[1:2, :], controller_spikes[0:1, :]))
a_neg = np.vstack((spikes, controller_spikes[0:1, :], controller_spikes[1:2, :]))
Plots.compare_time_predictions(time, a_pos, predictions_pos, controller_pos.tau_decay)
Plots.compare_time_predictions(time, a_neg, predictions_neg, controller_neg.tau_decay)
# Plots.compare_predictions(time, spikes, controller_spikes)
Plots.plot_traces(time, traces_pos)
Plots.plot_traces(time, traces_neg)
# Plots.plot_covariances(time, Covs, CrossCovs)

# %%
plot_system_response(time, y[:], np.vstack((spikes, controller_spikes)), IF_integral)


