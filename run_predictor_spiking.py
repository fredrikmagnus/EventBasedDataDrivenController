import importlib
import sys
import numpy as np
import matplotlib.pyplot as plt
from Controller_spiking import Predictor, spike_signal
from SpikingSystems import IFSpikeEncoder_absolute
import Plots
importlib.reload(sys.modules['Controller_spiking'])
importlib.reload(sys.modules['Plots'])
from Controller_spiking import Predictor, spike_signal
import Plots

T = 500
dt = 0.001
time = np.arange(0, T, dt)
N = len(time)

periods = [0.5, 0.5] # Spike periods for each input channel
phases = [0.05, 0.1] # Phase offsets for each input channel
n_inputs = len(periods)
randomize = [0.0 for _ in range(n_inputs)] # Randomize spike times by adding uniform noise in [-randomize, randomize]

# print(np.exp(-0.05/0.2))
# NOTE: Make it spike at correct time and not next timestep. 
x = np.zeros((n_inputs, N), dtype=int)
for i in range(n_inputs):
    x[i, :] = spike_signal(time, periods[i], phases[i], randomize=randomize[i])
# x[0, len(time)//2:] = 0 # Remove second half of spikes from first input channel to test prediction of missing spikes
x[0, :] += spike_signal(time, periods[0], phases[0]+0.08, randomize=randomize[0]) # Add second spike train to first input channel



# Attempt a more complex signal
# encoder = IFSpikeEncoder_absolute(threshold=0.05, dt=dt)
# sine = np.sin(2 * np.pi * 0.5 *time)
# for t in range(N):
#     x[0, t] = encoder.step(sine[t])


predictor = Predictor(
    n_inputs=n_inputs,   # Number of input channels
    gamma_weights=0.99,   # Decay factor for covariance estimates
    tau_decay=0.2,      # Time constant for trace decay
    lambda_ridge=1e-4,   # Ridge regularization parameter
    dt=dt,               # Time step size
    affine=True,   # Include affine term in predictor
    spiking=False
)

n_outputs = predictor.n_outputs
n_inputs = predictor.n_inputs

if predictor.spiking:
    x = np.vstack((np.zeros((1, N), dtype=int), x)) # Add row for feedback from previous output spike

# Logs:
predictions = np.zeros((n_outputs, N))
traces = np.zeros((n_inputs, N))
Covs = np.zeros((n_inputs, n_inputs, N))
CrossCovs = np.zeros((n_outputs, n_inputs, N))
PredictionGains = np.zeros((n_outputs, n_inputs, N))

for k in range(1, N):
    # predictor.update(x[:, k])
    if predictor.spiking:
        x_in = x[1:,k] # x[0, :] is used to store output spikes  
    else:
        x_in = x[:, k]
    predictions[:, k], x[0, k] = predictor.gradient_update(x_in)
    # predictions[:, k] = predictor.predict() #+ predictor.x
    #  predictor.spike()
    
    traces[:, k] = predictor.z
    Covs[:, :, k] = predictor.Sigma
    CrossCovs[:, :, k] = predictor.Psi
    PredictionGains[:, :, k] = predictor.W

# print(predictor.z.shape, predictor.P.shape)
# print(predictions.shape)
# print(predictor.P)
print("W:", predictor.W)
print("Psi:", predictor.Psi)
print("Sigma:", predictor.Sigma)

print(x.shape, predictions.shape)
Plots.compare_time_predictions(time, x, predictions, predictor.tau_decay)
Plots.compare_predictions(time, x, predictions)
# Plots.plot_traces(time, traces)
# Plots.plot_covariances(time, Covs, CrossCovs)
Plots.plot_gains(time, PredictionGains)
