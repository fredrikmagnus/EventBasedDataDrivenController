import importlib
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from PredictorEventBased import Predictor, spike_signal
from SpikingSystems import IFSpikeEncoder_absolute
from Plots import compare_event_time_predictions, plot_gains_event_based, plot_raw_predictions_event_based, plot_prediction_error_event_based
importlib.reload(sys.modules['PredictorEventBased'])
from PredictorEventBased import Predictor, spike_signal

# T = 100
# periods = [0.5, 0.5] # Spike periods for each input channel
# phases = [0.0, 0.2] # Phase offsets for each input channel
# n_inputs = len(periods)

# # Generate event-times for each input channel
# # For each signal, generate a list of event times based on the specified period and phase
# x = []
# for i in range(0, n_inputs):
#     spike_times = spike_signal(T, periods[i], phases[i])
#     x.append(spike_times)

# # x[0] = x[0][x[0] < T/2] # Remove second half of spikes from first input channel to test prediction of missing spikes
# # x[0] = np.hstack((x[0], spike_signal(T, periods[0], phases[0]+0.1))) # Add second spike train to first input channel
# # Sort spike times in each channel (in case they are not already sorted)
# for i in range(n_inputs):
#     x[i] = np.sort(x[i])


# x[0] = np.sort(x[0]) # Sort spike times in first input channel

# print("x:", [len(channel) for channel in x])

# # Attempt a more complex signal
# encoder = IFSpikeEncoder_absolute(threshold=0.1, dt=0.001)
# t = np.arange(0, T, 0.001)
# sine = np.sin(2 * np.pi * 0.5 * t)

# spike_times_sine = []
# for idx, t_k in enumerate(t):
#     if encoder.step(float(sine[idx])):
#         spike_times_sine.append(float(t_k))
# x[0] = spike_times_sine # Add spike times from sine wave input to third input channel

# Get global list of all spike times
# all_spike_times = sorted(set(time for channel in x for time in channel))
# N = len(all_spike_times) # Number of events
# print("Total number of events:", N)

n_inputs = 1 # Excluding feedback signal


predictor = Predictor(
    n_inputs=n_inputs,   # Number of input channels
    tau_decay=.4,      # Time constant for trace decay
    lambda_ridge=1e-2,   # Ridge regularization parameter
    eta=0.5,            # Learning rate for gradient update
    eta_cumulative=0.5,
    cumulative_channels=[], # Always accumulate covariance for first input channel
    reference_tracking_costs=[0., 0.], # Cost for tracking reference in each output channel (0 means no reference tracking)
    activation_enable=True,
    affine=True,   # Include affine term in predictor
    spiking=True,
    noise_std=0.01, # Standard deviation of noise added to trace at each event
    seed=1
)

predictor2 = Predictor(
    n_inputs=n_inputs,   # Number of input channels
    tau_decay=.4,      # Time constant for trace decay
    lambda_ridge=1e-3,   # Ridge regularization parameter
    eta=0.3,            # Learning rate for gradient update
    eta_cumulative=0.5,
    cumulative_channels=[], # Always accumulate covariance for first input channel
    reference_tracking_costs=[0., 0.], # Cost for tracking reference in each output channel (0 means no reference tracking)
    activation_enable=True,
    affine=True,   # Include affine term in predictor
    spiking=True,
    noise_std=0.01, # Standard deviation of noise added to trace at each event
    seed=2
)

predictors = [predictor, predictor2]

T = 1000.
x = [[] for _ in range(len(predictors))] # List of spike times for each input channel

n_outputs = predictor.n_outputs
n_inputs = predictor.n_inputs

# if predictor.spiking:
#     x.insert(0, []) # Add row for feedback from previous output spike

# Logs:
predictions = [[] for _ in predictors]
traces = [[] for _ in predictors]
PredictionGains = [[] for _ in predictors]

# if predictor.spiking and predictor.spike_threshold > 0:
#     next_spike_time = - predictor.tau_decay * np.log(predictor.spike_threshold)
# else:
#     next_spike_time = np.inf # predictor next spike time

# if next_spike_time < all_spike_times[0]:
#     all_spike_times.insert(0, next_spike_time) 

# time_to_spike = - predictor.tau_decay * np.log(predictor.z_post[-1])
print(predictor.W)
# times_to_next_spike = [- predictor.tau_decay * np.log(predictor.W[k, -1]) for k, predictor in enumerate(predictors)]
times_to_next_spike = [np.random.exponential(scale=predictor.tau_decay) for predictor in predictors] # Start with random initial spike times to break symmetry
print(times_to_next_spike)
time_to_spike = min(times_to_next_spike)
spiking_idx = np.argmin(times_to_next_spike)
t = 0.
start_time = time.time()
while t < T:
    if time_to_spike < 1e-6: 
        break
    t += time_to_spike
    print(time_to_spike)
    x[spiking_idx].append(t) 
    # Get input vector for this event
    # print(t)
    x_in = np.array([1.0 if t in channel else 0.0 for channel in x])
    # if predictor.spiking:
    #     x_in = x_in[1:] 
        # print("Input vector:", x_in)
    # PredictionGains[spiking_idx].append(predictors[spiking_idx].W.copy())
    # reference = np.zeros(predictors[0].n_outputs) # No reference tracking in this example
    
    # predictions_i = []
    time_to_spike = np.inf
    spiking_idx = -1
    for k, predictor in enumerate(predictors):
        predictor.update_state(t, x_in)
        pred = predictor.predict()

        time_to_spike_i = - predictor.tau_decay * np.log(pred[k]) if pred[k] > 0 else np.inf # Set a default time to next spike if prediction is zero or negative
        if time_to_spike_i < time_to_spike:
            time_to_spike = time_to_spike_i
            spiking_idx = k

        reference = np.ones(predictor.n_outputs) # No reference tracking in this example
        predictor.gradient_update(reference=reference)
        predictions[k].append(pred)
        traces[k].append(predictor.z_post[:n_inputs])
        PredictionGains[k].append(predictor.W.copy())
       
        # traces.append(predictor.z_post[:n_inputs])


    # time_to_spike = - predictor.tau_decay * np.log(predictor.spike_threshold)
    # times_to_next_spike = [- predictor.tau_decay * np.log(pred[i]) if pred[i] > 0 else np.inf for i in range(len(predictors))] # Set a default time to next spike if prediction is zero or negative
    # time_to_spike = min(times_to_next_spike)
    # spiking_idx = np.argmin(times_to_next_spike)
    # if k < len(all_spike_times)-1 and next_spike_time < all_spike_times[k+1] and next_spike_time <= T:
    #     all_spike_times.insert(k+1, next_spike_time)
end_time = time.time()
print(f"Runtime: {end_time - start_time:.4f} seconds")

idx = 1
predictions = np.array(predictions[idx]).T
traces = np.array(traces[idx]).T
PredictionGains = np.array(PredictionGains[idx]).transpose(1, 2, 0)
print(PredictionGains.shape)
# Now we need to make plots for the event-based predictor. 
# We start by comparing the predictions to the input spikes.
# The model outputs a prediction vector a_hat(t_k) at each event time t_k
# Element i: a_hat_i = exp(- (t_k+1 - t_k) / tau_decay) 
# We reconstruct the predicted time until next spike as:
# delta_t = - tau_decay * log(a_hat_i)
# all_spike_times = np.array(all_spike_times)
# print(predictions.shape)
# print(all_spike_times.shape)
# print([len(xi) for xi in x])

all_spike_times = sorted(set(time for channel in x for time in channel))
print(predictions.shape, len(all_spike_times), [len(xi) for xi in x])
compare_event_time_predictions(
    x_event_times=x,
    predictions=predictions,
    tau=predictors[idx].tau_decay,
)

plot_gains_event_based(
    event_times=all_spike_times,
    prediction_gains=PredictionGains,
)

# Plot raw predictions:
plot_raw_predictions_event_based(
    event_times=all_spike_times,
    predictions=predictions,
)

# Plot one-step-ahead prediction error:
plot_prediction_error_event_based(
    event_times=all_spike_times,
    x_event_times=x,
    predictions=predictions,
    tau=predictors[idx].tau_decay,
    cumulative_channels=predictors[idx].cumulative_channels,
)









