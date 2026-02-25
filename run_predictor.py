import importlib
import sys
import numpy as np
import matplotlib.pyplot as plt
from Controller import Predictor, spike_signal
from SpikingSystems import IFSpikeEncoder_absolute
import Plots
importlib.reload(sys.modules['Controller'])
importlib.reload(sys.modules['Plots'])
from Controller import Predictor, spike_signal
import Plots

T = 100
dt = 0.001
time = np.arange(0, T, dt)
N = len(time)

periods = [0.5] # Spike periods for each input channel
phases = [0.05] # Phase offsets for each input channel
n_inputs = len(periods)
randomize = [0.0 for _ in range(n_inputs)] # Randomize spike times by adding uniform noise in [-randomize, randomize]

# print(np.exp(-0.05/0.2))

x = np.zeros((n_inputs, N), dtype=int)
for i in range(n_inputs):
    x[i, :] = spike_signal(time, periods[i], phases[i], randomize=randomize[i])
x[0, :] += spike_signal(time, periods[0], phases[0]+0.1, randomize=randomize[0]) # Add second spike train to first input channel

# Attempt a more complex signal
encoder = IFSpikeEncoder_absolute(threshold=0.05, dt=dt)
sine = np.sin(2 * np.pi * 0.5 *time)
for t in range(N):
    x[:, t] = encoder.step(sine[t])


predictor = Predictor(
    n_inputs=n_inputs,   # Number of input channels
    gamma_weights=0.99,   # Decay factor for covariance estimates
    tau_decay=0.2,      # Time constant for trace decay
    lambda_ridge=1e-4,   # Ridge regularization parameter
    dt=dt,               # Time step size
    affine=True   #
)

n_outputs = predictor.n_outputs
if predictor.affine:
    n_inputs += 1 # Account for bias input in input dimension
# Logs:
predictions = np.zeros_like(x, dtype=float)
traces = np.zeros((predictor.n_inputs, N))
Covs = np.zeros((n_inputs, n_inputs, N))
CrossCovs = np.zeros((n_outputs, n_inputs, N))
PredictionGains = np.zeros((n_outputs, n_inputs, N))


for k in range(1, N):
    predictor.update(x[:, k])
    predictions[:, k] = predictor.predict() #+ predictor.x
    
    traces[:, k] = predictor.z
    Covs[:, :, k] = predictor.Sigma
    CrossCovs[:, :, k] = predictor.Psi
    PredictionGains[:, :, k] = predictor.P

# print(predictor.z.shape, predictor.P.shape)
# print(predictions.shape)
print(predictor.P)

Plots.compare_time_predictions(time, x, predictions, predictor.tau_decay)
Plots.compare_predictions(time, x, predictions)
Plots.plot_traces(time, traces)
Plots.plot_covariances(time, Covs, CrossCovs)
Plots.plot_gains(time, PredictionGains)
