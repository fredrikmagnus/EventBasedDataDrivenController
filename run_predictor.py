import importlib
import sys
import numpy as np
import matplotlib.pyplot as plt
from Controller import Predictor, spike_signal
import Plots
importlib.reload(sys.modules['Controller'])
importlib.reload(sys.modules['Plots'])
from Controller import Predictor, spike_signal
import Plots

T = 100
dt = 0.001
time = np.arange(0, T, dt)
N = len(time)

periods = [1, 1, 1] # Spike periods for each input channel
phases = [0.05, 0.1, 0.15] # Phase offsets for each input channel
n_inputs = len(periods)
randomize = [0.0 for _ in range(n_inputs)] # Randomize spike times by adding uniform noise in [-randomize, randomize]

print(np.exp(-0.05/0.2))

x = np.zeros((n_inputs, N), dtype=int)
for i in range(n_inputs):
    x[i, :] = spike_signal(time, periods[i], phases[i], randomize=randomize[i])

predictor = Predictor(
    n_inputs=n_inputs,   # Number of input channels
    gamma_weights=0.995,   # Decay factor for covariance estimates
    tau_decay=0.2,      # Time constant for trace decay
    lambda_ridge=1e-4,   # Ridge regularization parameter
    dt=dt,               # Time step size
    estimate_mean=False   # Whether to center traces by event-based estimated mean
)

# Logs:
predictions = np.zeros_like(x, dtype=float)
means = np.zeros_like(x, dtype=float)
traces = np.zeros_like(x, dtype=float)
Covs = np.zeros((n_inputs, n_inputs, N))
CrossCovs = np.zeros((n_inputs, n_inputs, N))
PredictionGains = np.zeros((n_inputs, n_inputs, N))


for k in range(1, N):
    predictor.update(x[:, k])
    predictions[:, k] = predictor.predict() #+ predictor.x
    
    means[:, k] = predictor.mean
    traces[:, k] = predictor.x
    Covs[:, :, k] = predictor.Sigma
    CrossCovs[:, :, k] = predictor.Psi
    PredictionGains[:, :, k] = predictor.P

print(predictor.z.shape, predictor.P.shape)
print(predictions.shape)

Plots.compare_time_predictions(time, x, predictions, predictor.tau_decay)
Plots.compare_predictions(time, x, predictions)
Plots.plot_traces(time, traces-means)
# Plots.plot_covariances(time, Covs, CrossCovs)
# Plots.plot_gains(time, PredictionGains)
