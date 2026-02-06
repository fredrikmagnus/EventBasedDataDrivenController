import numpy as np
import matplotlib.pyplot as plt


# def compare_time_predictions(time, x, predictions, tau):
#     """
#     Compare true spike times with predicted next spike times.
#     Makes a raster plot showing true spikes and predicted spikes. Ideally they should align.

#     Parameters:
#     - time: 1D array of time points
#     - x: 2D binary array of shape (n_inputs, n_timepoints) indicating true spike events
#     - predictions: 2D array of shape (n_inputs, n_timepoints) with the model predictions at each time point (can be replaced with predictions at spike-times only)
#     - tau: synaptic trace time constant used in the prediction model
#     """
#     # Extract spike-times
#     n_inputs = x.shape[0]
#     spike_times = [time[np.where(x[i, :] == 1)[0]] for i in range(n_inputs)]
#     spike_times = np.unique(np.concatenate(spike_times))

#     spike_indices = [np.where(x[i, :] == 1)[0] for i in range(n_inputs)]
#     spike_indices = np.unique(np.concatenate(spike_indices))

#     next_spike_times_pred = np.zeros((n_inputs, len(spike_times)))

#     for i, spike_time in enumerate(spike_times):
#         spike_index = np.where(time == spike_time)[0][0]
#         delta_t_pred = -tau*np.log(predictions[:, spike_index] + 1e-12)  # Avoid log(0)
#         delta_t_pred[np.flip(np.argsort(delta_t_pred))[:-1]] = 0
#         next_spike_times_pred[:, i] = spike_time + delta_t_pred

#     # Plot true spikes and the predicted spikes in a raster plot
#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     plt.figure(figsize=(12, 6))
#     for i in range(n_inputs):
#         # Plot true spikes
#         plt.eventplot(time[x[i, :] == 1], lineoffsets=i*2, colors=colors[i], label=f'True x{i+1}', linewidths=2)
#         # Plot predicted spikes
#         # Remove times that are already in spike_times (not predicted new spikes)
#         mask = np.isin(next_spike_times_pred[i, :], spike_times, invert=True) # elements not in spike_times
#         masked = next_spike_times_pred[i, :][mask]
#         plt.eventplot(masked, lineoffsets=i*2, colors=colors[i], 
#                       linestyles='dashed', label=f'Predicted x{i+1}', linewidths=2)
#     plt.yticks(np.arange(0, n_inputs*2, 2), [f'x{i+1}' for i in range(n_inputs)])
#     plt.xlabel('Time')
#     plt.title('True and Predicted Next Spike Times')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

def compare_time_predictions(time, x, predictions, tau):
    """
    Compare true spike times with predicted next spike times.
    Makes a raster plot showing true spikes and predicted spikes. Ideally they should align.

    Parameters:
    - time: 1D array of time points
    - x: 2D binary array of shape (n_inputs, n_timepoints) indicating true spike events
    - predictions: 2D array of shape (n_inputs, n_timepoints) with the model predictions at each time point (can be replaced with predictions at spike-times only)
    - tau: synaptic trace time constant used in the prediction model
    """
    # Extract spike-times
    n_inputs = x.shape[0]
    spike_times = [time[np.where(x[i, :] == 1)[0]] for i in range(n_inputs)]
    spike_times = np.unique(np.concatenate(spike_times))

    spike_indices = [np.where(x[i, :] == 1)[0] for i in range(n_inputs)]
    spike_indices = np.unique(np.concatenate(spike_indices))

    next_spike_times_pred = np.zeros((n_inputs, len(spike_times)))

    for i, spike_time in enumerate(spike_times):
        spike_index = np.where(time == spike_time)[0][0]
        p = np.clip(predictions[:, spike_index], 1e-12, 1.0)
        delta_t_pred = -tau*np.log(p)
        
        next_spike_times_pred[:, i] = spike_time + delta_t_pred

    # Plot true spikes and the predicted spikes in a raster plot
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(12, 6))
    for i in range(n_inputs):
        # Plot true spikes
        plt.eventplot(time[x[i, :] == 1], lineoffsets=i*2, colors=colors[i], label=f'True x{i+1}', linewidths=2)
        # Plot predicted spikes
        # Remove times that are already in spike_times (not predicted new spikes)
        # mask = np.isin(next_spike_times_pred[i, :], spike_times, invert=True) # elements not in spike_times
        # masked = next_spike_times_pred[i, :][mask]
        plt.eventplot(next_spike_times_pred[i, :], lineoffsets=i*2, colors=colors[i+1], 
                      linestyles='dashed', label=f'Predicted x{i+1}', linewidths=2)
    plt.yticks(np.arange(0, n_inputs*2, 2), [f'x{i+1}' for i in range(n_inputs)])
    plt.xlabel('Time')
    plt.title('True and Predicted Next Spike Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_predictions(time, x, predictions):
    n_inputs = x.shape[0]
    
    # Standard color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # Top subplot: predictions over time
    for i in range(n_inputs):
        ax1.plot(time, predictions[i, :], label=f'$\\hat{{x}}_{{{i+1}}}$', linewidth=2, color=colors[i])
    ax1.set_ylabel('Prediction')
    ax1.set_title('Next-event Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom subplot: event plot of original spikes
    for i in range(n_inputs):
        spike_times = time[x[i, :] == 1]
        if len(spike_times) > 0:
            ax2.eventplot(spike_times, lineoffsets=i, linewidths=2, 
                        colors=colors[i], label=f'x{i+1}')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Input Channel')
    ax2.set_title('Observed Spikes')
    ax2.set_yticks(range(n_inputs))
    ax2.set_yticklabels([f'x{i+1}' for i in range(n_inputs)])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("SpikePredictionsAndEvents.pdf")
    plt.show()  

def plot_covariances(time, Covs, CrossCovs):
    # Plot all values of K over time
    n_inputs = Covs.shape[0]
    plt.figure(figsize=(12, 8))
    for i in range(n_inputs):
        for j in range(n_inputs):
            plt.step(time, Covs[i, j, :], label=f'[{i}, {j}]')
    plt.xlabel('Time')
    plt.ylabel('Covariance Values')
    plt.title(r'$\text{Cov}(z_k^+, z_k^+)$ Matrix Elements Over Time')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot all values of K_causal over time
    plt.figure(figsize=(12, 8))
    for i in range(n_inputs):
        for j in range(n_inputs):
            plt.step(time, CrossCovs[i, j, :], label=f'[{i}, {j}]')
    plt.xlabel('Time')
    plt.ylabel('Cross-Covariance Values')
    plt.title(r'$\text{Cov}(x_k, z_k^-)$ Matrix Elements Over Time')
    plt.legend()
    plt.grid()
    plt.show()  

def plot_traces(time, traces):
    n_inputs = traces.shape[0]
    plt.figure(figsize=(12, 6))
    for i in range(n_inputs):
        plt.plot(time, traces[i, :], label=f'x{i+1}')

    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Synaptic Traces over Time')
    plt.legend()
    plt.grid()
    plt.show()

def plot_gains(time, PredictionGains):
    n_inputs = PredictionGains.shape[0]
    plt.figure(figsize=(12, 8))
    for i in range(n_inputs):
        for j in range(n_inputs):
            plt.step(time, PredictionGains[i, j, :], label=f'[{i}, {j}]')
    plt.xlabel('Time')
    plt.ylabel('Prediction Gain Values')
    plt.title(r'Gain Matrix $\text{Cov}(x_k, z_k^-)\text{Cov}(z_k^+, z_k^+)^{-1}$ Elements Over Time')
    plt.legend()
    plt.grid()
    plt.show()

def compare_with_reference2(time, x, ref):
    """
    Compare true spike times with reference period.

    Parameters:
    - time: 1D array of time points
    - x: 1D binary array of shape (n_timepoints) indicating true spike events from a single channel
    - ref: 1D array of reference values over time
    Example reference signal:
        The signal counts the time until the next desired event, resetting to zero at each event.
        - period=1, dt=0.1
        - time = [0, 0.1, 0.2, ..., 2.0]
        - ref_signal = [1, 0.9, 0.8, ..., 0.0, 1.0, 0.9, ..., 0.0, ...]

    The function does: 
     1. Find spike times from x
     2. Create a reference spike train: 
        - Each time the reference signal reaches a minimum (resets to max),
        - Add a spike in the reference spike train at time
    3. Plot both true spikes and reference spikes in a raster plot at the same level.
    """
    # Extract spike-times
    spike_times = time[np.where(x == 1)[0]]

    ref_spike_times = []
    for i in range(0, len(ref)-1):
        if ref[i] < ref[i+1]:  # Detect reset to max (minimum point)
            ref_spike_times.append(time[i])
    ref_spike_times = np.array(ref_spike_times)

    # Plot true spikes and the reference spikes in a raster plot
    plt.figure(figsize=(12, 6))
    plt.eventplot(spike_times, lineoffsets=1, colors='C0', label='True Spikes', linewidths=2)
    plt.eventplot(ref_spike_times, lineoffsets=0, colors='C1', label='Reference Spikes', linewidths=2)
    
    plt.yticks([0, 1], ['Reference', 'True Spikes'])
    plt.xlabel('Time')
    plt.title('True Spikes and Reference Spike Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_with_reference(time, x, ref):
    """
    Compare true spike times with reference period.

    Parameters:
    - time: 1D array of time points
    - x: 1D binary array of shape (n_timepoints) indicating true spike events from a single channel
    - ref: 1D array of reference values over time

    The function does: 
     1. Find spike times from x
     2. Create a reference spike train: 
        - For each spike time, get the corresponding reference value at that time
        - Add a spike in the reference spike train at time + reference value
    3. Plot both true spikes and reference spikes in a raster plot at the same level.
    """
    # Extract spike-times
    spike_times = time[np.where(x == 1)[0]]

    ref_spike_times = []
    for spike_time in spike_times:
        spike_index = np.where(time == spike_time)[0][0]
        delta_t_ref = ref[spike_index]
        ref_spike_times.append(spike_time + delta_t_ref)
    ref_spike_times = np.array(ref_spike_times)

    # Plot true spikes and the reference spikes in a raster plot
    plt.figure(figsize=(12, 6))
    plt.eventplot(spike_times, lineoffsets=1, colors='C0', label='True Spikes', linewidths=2)
    plt.eventplot(ref_spike_times, lineoffsets=0, colors='C1', label='Reference Spikes', linewidths=2)
    
    plt.yticks([0, 1], ['Reference', 'True Spikes'])
    plt.xlabel('Time')
    plt.title('True Spikes and Reference Spike Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

def test(time, x, predictions, tau_decay):

    # predictions = -tau_decay * np.log(predictions + 1e-12)  # Avoid log(0)
    # predictions = np.cumsum(predictions, axis=1) * (time[1] - time[0])
    # Subtract the current value from respective trace at each event time


    n_inputs = x.shape[0]
    
    # Standard color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # Top subplot: predictions over time
    for i in range(n_inputs):
        ax1.plot(time, predictions[i, :], label=f'$\\hat{{x}}_{{{i+1}}}$', linewidth=2, color=colors[i])

    ax1.set_ylabel('Prediction')
    ax1.set_title('Next-event Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom subplot: event plot of original spikes
    for i in range(n_inputs):
        spike_times = time[x[i, :] == 1]
        if len(spike_times) > 0:
            ax2.eventplot(spike_times, lineoffsets=i, linewidths=2, 
                        colors=colors[i], label=f'x{i+1}')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Input Channel')
    ax2.set_title('Observed Spikes')
    ax2.set_yticks(range(n_inputs))
    ax2.set_yticklabels([f'x{i+1}' for i in range(n_inputs)])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("SpikePredictionsAndEvents.pdf")
    plt.show()  