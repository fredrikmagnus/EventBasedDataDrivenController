import importlib
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from PredictorEventBased import Predictor, spike_signal
from SpikingSystems import IFSpikeEncoder_absolute
from Plots import compare_event_time_predictions, plot_gains_event_based, plot_raw_predictions_event_based
importlib.reload(sys.modules['PredictorEventBased'])
from PredictorEventBased import Predictor, spike_signal


# 1) We first need to create the network structure. 
# We now assume that the network is "closed". I.e. no external input or output, only internal feedback. 
# We can assume that the network structure is given as an adjacency matrix, where entry (i,j) indicates a connection from neuron i to j.

def fully_connected_adjacency(n):
    return np.ones((n, n)) - np.eye(n)

def create_network(adjacency, neuron_params, external_input=None):
    """
    
    external input is a mxn binary vector indicating which neurons receive external input,
    m is the number of input channels, n is the number of neurons. Element (i,j) is 1 if external input i to neuron j, 0 otherwise. 
    """
    n = adjacency.shape[0]
    neurons = []
    for i in range(n):
        n_inputs = int(np.sum(adjacency[:, i])) 
        if external_input is not None:
            n_inputs += int(np.sum(external_input[i, :])) # Count external inputs to neuron i
        print(f"Creating neuron {i} with {n_inputs} inputs.")
        neuron = Predictor(
            n_inputs=n_inputs,
            **neuron_params
        )
        neurons.append(neuron)
    return neurons


# def event_input_vector(adjacency: np.ndarray, spiking_neuron: int, target_neuron: int) -> np.ndarray:
#     """Create the compact input vector for `target_neuron` given a spike from `spiking_neuron`.

#     Each neuron only receives inputs from indices where adjacency[:, target_neuron] == 1.
#     The Predictor expects an input vector with length == number of such incoming connections.
#     """
#     incoming = np.where(adjacency[:, target_neuron] == 1)[0]
#     x_in = np.zeros(len(incoming), dtype=float)
#     pos = np.where(incoming == spiking_neuron)[0]
#     if pos.size:
#         x_in[int(pos[0])] = 1.0
#     return x_in
            
parameters = {
    'gamma_weights': 0.99,   # Decay factor for covariance estimates
    'tau_decay': 0.1,      # Time constant for trace decay
    'lambda_ridge': 1e-3,   # Ridge regularization parameter
    'eta': 0.1,            # Learning rate for gradient update
    'affine': True,   # Include affine term in predictor
    'spiking': True
}

adjacency = fully_connected_adjacency(4)
network = create_network(adjacency, parameters)


# Simulation data:
x = [[] for _ in range(adjacency.shape[0])] # No external input, only internal feedback
# x = []
spike_times = [[] for _ in range(adjacency.shape[0])] # Spike times for each neuron
predictions = [[] for _ in range(adjacency.shape[0])] # Predictions for each neuron
time_to_spike = lambda neuron: np.inf if neuron.spike_threshold <= 0 else - neuron.tau_decay * np.log(neuron.spike_threshold)

T = 10 # Simulation time
n_steps = 1000
n = 0
next_spike_times = [time_to_spike(neuron) for neuron in network]
t = 0
while t < T and n < n_steps:
    n += 1
    if t == 0:
        t = t + np.max(next_spike_times) # Advance to next spike time
    else:
        t = t + np.min(next_spike_times) # Advance to next spike time
    spiking_neuron_idx = np.argmin(next_spike_times)
    print(f"Time: {t:.3f}, Spiking neuron: {spiking_neuron_idx}")
    spike_times[spiking_neuron_idx].append(t)

    x[spiking_neuron_idx].append(t) # Store input spike time for spiking neuron

    spiking_neuron = network[spiking_neuron_idx]
    pred, _ = spiking_neuron.gradient_update(t, np.zeros(spiking_neuron.n_inputs-2)) # Subtract feedback and bias
    predictions[spiking_neuron_idx].append(pred)

    spike_targets = np.where(adjacency[spiking_neuron_idx, :] == 1)[0]


    for target_idx in spike_targets:
        target_neuron = network[target_idx]
        target_adjacency_in = adjacency[:, target_idx]
        spike_in = np.zeros_like(target_adjacency_in)
        spike_in[spiking_neuron_idx] = 1
        x_in_target_neuron = spike_in[target_adjacency_in == 1] # Only include inputs from connected neurons
        print(f"   x_in for target neuron {target_idx}:", x_in_target_neuron)
        pred, _ = target_neuron.gradient_update(t, x_in_target_neuron)
        predictions[target_idx].append(pred)
        next_spike_times[target_idx] = time_to_spike(target_neuron) # Update next spike time for target neuron



# Plot results
global_event_times = np.array(sorted(list(set([time for neuron_spike_times in spike_times for time in neuron_spike_times]))))
# predictions = [np.array(pred).T for pred in predictions]
# for p in predictions:
#     print(p.shape)
# print("Global event times:", global_event_times.shape)
# print("Global event times:", global_event_times)
compare_event_time_predictions(global_event_times, x, np.array(predictions[0]).T, tau=parameters['tau_decay'])






    

    # # Determine next spike time
    # # Higher threshold means earlier prediction.
    # thresholds = [neuron.spike_threshold for neuron in network]
    # spiking_neuron = np.argmax(thresholds) # neuron with greatest spike threshold will spike first

    # # Get input vector for next event
    # # We instantaneously give input to all neurons that receive input from the spiking neuron.
    # next_neurons = np.where(adjacency[spiking_neuron, :] == 1)[0] # Neurons that receive input from spiking neuron
    # print("test", next_neurons)
    # for target_idx in next_neurons:
    #     target_idx = int(target_idx)
    #     neuron_j = network[target_idx]
    #     x_in_neuron_j = event_input_vector(adjacency, spiking_neuron, target_idx)
    #     pred, spike_out = neuron_j.gradient_update(t, x_in_neuron_j)
    # x_in = np.zeros(adjacency.shape[0])
    # x_in[spiking_neuron] = 1.0

    # # Update each neuron with incoming spike
    # # Note: There is no explicit self-connection. Feedback is internal.
    # # Assume neuron i spikes at t.
    # # Neuron j receives this spike as input if adjacency[i, j] == 1.
    # # The input vector is x = {x_j : j -> i is connected}
    # next_neurons = np.where(adjacency[spiking_neuron, :] == 1)[0] # Neurons that receive input from spiking neuron
    # print(next_neurons)
    # for j in next_neurons:
    #     j = int(j)
    #     x_in_neuron_j = event_input_vector(adjacency, spiking_neuron, j)
    #     # Step neuron j
    #     pred, spike_out = network[j].gradient_update(t, x_in_neuron_j)

    #     # Go to next event time
    # t += time_to_spike(network[spiking_neuron])
    # spike_times[spiking_neuron].append(t)


    # for i, neuron in enumerate(network):
    #     connectivity_in_neuron_i = np.where(adjacency[:, i] == 1)[0]
    #     if spiking_neuron in connectivity_in_neuron_i:
    #         x_in_neuron_i = np.zeros(int(np.sum(adjacency[:, i])))
    #         x_in_neuron_i[connectivity_in_neuron_i.tolist().index(spiking_neuron)] = 1.0 # Set input from spiking neuron
    #     else:
    #         x_in_neuron_i = np.zeros(int(np.sum(adjacency[:, i])))
        
    #     pred, spike_out = neuron.gradient_update(t, x_in_neuron_i)

        




        # if adjacency[spiking_neuron, i] == 1:
        #     x_in = np.zeros_like([adjacency[:, i]==1])
        #     if spiking_neuron 

            

        

    # for i, neuron in enumerate(network):
    #     connectivity = adjacency[:, i]
    #     x_in = np.zeros(int(np.sum(connectivity)+1))
    #     pred, spike_out = neuron.gradient_update(t, x_in)
    #     x_next[i] = spike_out
    # x = x_next.copy() # Update input vector for next event






