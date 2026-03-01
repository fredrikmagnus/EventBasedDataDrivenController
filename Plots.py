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
        
        # Get indices of predictions that are above 1 or below 0
        error_indices = np.where((predictions[:, spike_index] < 0) | (predictions[:, spike_index] > 1))[0]
        
        delta_t_pred = -tau*np.log(p)
        # Set components that are above 1 or below 0 to -spike_time
        delta_t_pred[error_indices] = -spike_time
        
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


def compare_event_time_predictions(event_times, x_event_times, predictions, tau, title='True and Predicted Next Spike Times', figsize=(12, 6), ax=None, show=True):
    """Compare true spike event-times with predicted next spike times (event-based).

    Parameters:
    - event_times: 1D array-like of global event times t_k (length N)
    - x_event_times: list/sequence of length n_inputs, each element is an array-like of true spike times for that channel
    - predictions: 2D array of shape (n_inputs, N) with model predictions a_hat(t_k) at each global event time
    - tau: synaptic trace time constant used in the prediction model
    - title: plot title
    - figsize: figure size used when ax is None
    - ax: optional matplotlib Axes to draw into
    - show: whether to call plt.show()

    Returns:
    - fig, ax, next_spike_times_pred
    """
    event_times = np.asarray(event_times)
    predictions = np.asarray(predictions)
    n_inputs = len(x_event_times)
    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")
    if predictions.shape[0] != n_inputs:
        raise ValueError(
            f"predictions first dimension must match len(x_event_times)={n_inputs}, got {predictions.shape[0]}"
        )
    if predictions.shape[1] != event_times.shape[0]:
        raise ValueError(
            f"predictions second dimension must match len(event_times)={event_times.shape[0]}, got {predictions.shape[1]}"
        )

    next_spike_times_pred = np.zeros((n_inputs, event_times.shape[0]))
    for k, spike_time in enumerate(event_times):
        p = np.clip(predictions[:, k], 1e-12, 1.0)
        error_indices = np.where((predictions[:, k] < 0) | (predictions[:, k] > 1))[0]

        delta_t_pred = -tau * np.log(p)
        # Match compare_time_predictions behavior: invalid predictions plot at time 0
        delta_t_pred[error_indices] = -spike_time
        next_spike_times_pred[:, k] = spike_time + delta_t_pred

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    for i in range(n_inputs):
        true_color = colors[i % len(colors)]
        pred_color = colors[(i + 1) % len(colors)]

        ax.eventplot(
            x_event_times[i],
            lineoffsets=i * 2,
            colors=true_color,
            label=f'True x{i+1}',
            linewidths=2,
        )
        ax.eventplot(
            next_spike_times_pred[i, :],
            lineoffsets=i * 2,
            colors=pred_color,
            linestyles='dashed',
            label=f'Predicted x{i+1}',
            linewidths=2,
        )

    ax.set_yticks(np.arange(0, n_inputs * 2, 2))
    ax.set_yticklabels([f'x{i+1}' for i in range(n_inputs)])
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax, next_spike_times_pred


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
    n_outputs = CrossCovs.shape[0]
    plt.figure(figsize=(12, 8))
    for i in range(n_outputs):
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
    n_outputs = PredictionGains.shape[1]
    plt.figure(figsize=(12, 8))
    for i in range(n_inputs):
        for j in range(n_outputs):
            plt.step(time, PredictionGains[i, j, :], label=f'[{i}, {j}]')
    plt.xlabel('Time')
    plt.ylabel('Prediction Gain Values')
    plt.title(r'Gain Matrix $\text{Cov}(x_k, z_k^-)\text{Cov}(z_k^+, z_k^+)^{-1}$ Elements Over Time')
    plt.legend()
    plt.grid()
    plt.show()


def plot_gains_event_based(event_times, prediction_gains, title=None, figsize=(12, 8), ax=None, show=True):
    """Event-based equivalent of plot_gains.

    Parameters:
    - event_times: 1D array-like of global event times t_k (length N)
    - prediction_gains: 3D array of shape (n_outputs, n_inputs, N) or (n_inputs, n_outputs, N)
    - title: optional custom title
    - figsize: figure size used when ax is None
    - ax: optional matplotlib Axes to draw into
    - show: whether to call plt.show()

    Returns:
    - fig, ax
    """
    event_times = np.asarray(event_times)
    prediction_gains = np.asarray(prediction_gains)
    if prediction_gains.ndim != 3:
        raise ValueError(f"prediction_gains must be 3D, got shape {prediction_gains.shape}")
    if prediction_gains.shape[2] != event_times.shape[0]:
        raise ValueError(
            f"prediction_gains third dimension must match len(event_times)={event_times.shape[0]}, got {prediction_gains.shape[2]}"
        )

    if title is None:
        title = r'Gain Matrix $\text{Cov}(x_k, z_k^-)\text{Cov}(z_k^+, z_k^+)^{-1}$ Elements Over Event Times'

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    dim0, dim1, _ = prediction_gains.shape
    for i in range(dim0):
        for j in range(dim1):
            ax.step(event_times, prediction_gains[i, j, :], label=f'[{i}, {j}]')

    ax.set_xlabel('Time')
    ax.set_ylabel('Prediction Gain Values')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_raw_predictions_event_based(event_times, predictions, title='Raw Predictions (Event-Based)', figsize=(12, 6), ax=None, show=True, ylabel='Prediction'):
    """Plot raw model predictions over event times.

    Parameters:
    - event_times: 1D array-like of global event times t_k (length N)
    - predictions: 2D array of shape (n_outputs, N)
    - title: plot title
    - figsize: figure size used when ax is None
    - ax: optional matplotlib Axes to draw into
    - show: whether to call plt.show()
    - ylabel: y-axis label

    Returns:
    - fig, ax
    """
    event_times = np.asarray(event_times)
    predictions = np.asarray(predictions)
    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")
    if predictions.shape[1] != event_times.shape[0]:
        raise ValueError(
            f"predictions second dimension must match len(event_times)={event_times.shape[0]}, got {predictions.shape[1]}"
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(predictions.shape[0]):
        ax.plot(
            event_times,
            predictions[i, :],
            label=f'$\\hat{{x}}_{{{i+1}}}$',
            linewidth=2,
            color=colors[i % len(colors)],
        )

    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

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