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


def compare_event_time_predictions(x_event_times, predictions, tau, title='True and Predicted Next Spike Times', figsize=(12, 6), ax=None, show=True):
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
    # event_times = np.asarray(event_times)
    event_times = np.unique(np.concatenate(x_event_times))
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
        title = r'Predictor weights W'

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
    for i in reversed(range(predictions.shape[0])):
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


def discounted_future_counts(times_list, tau):
    """
    times_list: list of 1D array-likes, one per channel, each nondecreasing.
                Example: [t_ch0, t_ch1, ...]
    tau: positive decay constant

        Returns:
            y: shape (C, N) where C=len(times_list),
                 N = total number of events across all channels *including duplicates*.
                 Let T be the sorted merged list of all event times (not uniqued).
                 Then y[c, i] = sum_{t in channel c, t > T[i]} exp(-(t - T[i])/tau)
    """
    if tau <= 0:
        raise ValueError("tau must be positive.")

    C = len(times_list)

    # Build merged arrays
    all_times = []
    all_chans = []
    for c, t in enumerate(times_list):
        t = np.asarray(t, dtype=float)
        if t.size:
            if np.any(np.diff(t) < 0):
                print(t)
                print(np.diff(t))
                raise ValueError(f"times_list[{c}] must be nondecreasing.")
            all_times.append(t)
            all_chans.append(np.full(t.size, c, dtype=int))

    if not all_times:
        y = np.empty((C, 0), dtype=float)
        return y, np.array([]), np.array([], dtype=int)

    global_times = np.concatenate(all_times)
    global_chan  = np.concatenate(all_chans)

    # Sort by time (stable/deterministic tie-break by channel)
    order = np.lexsort((global_chan, global_times))
    global_times = global_times[order]
    global_chan  = global_chan[order]
    N = global_times.size

    # Unique time blocks to handle ties correctly (strictly future: t > T[i])
    uniq_times, inv = np.unique(global_times, return_inverse=True)
    M = uniq_times.size  # number of unique time points

    # counts[c, k] = number of events of channel c at exact time uniq_times[k]
    counts = np.zeros((C, M), dtype=float)
    np.add.at(counts, (global_chan, inv), 1.0)

    # Backward recursion over unique times:
    # S[:, k] = sum_{events at times > uniq_times[k]} exp(-(t - uniq_times[k])/tau)
    S = np.zeros((C, M), dtype=float)
    for k in range(M - 2, -1, -1):
        dt = uniq_times[k + 1] - uniq_times[k]
        a = np.exp(-dt / tau)
        S[:, k] = a * (counts[:, k + 1] + S[:, k + 1])

    # Evaluate at every global event time: all indices with same time share same S[:, k]
    y = S[:, inv]  # shape (C, N)

    return y


def discounted_future_counts_at_times(times_list, tau, query_times):
    """Compute discounted future spike counts evaluated at arbitrary times.

    For each channel c and query time t, returns
        y[c, i] = sum_{s in times_list[c], s > t} exp(-(s - t)/tau)
    i.e. strictly future spikes only.
    """
    if tau <= 0:
        raise ValueError("tau must be positive.")

    query_times = np.asarray(query_times, dtype=float)
    if query_times.ndim != 1:
        raise ValueError(f"query_times must be 1D, got shape {query_times.shape}")

    C = len(times_list)

    # Build merged arrays
    all_times = []
    all_chans = []
    for c, t in enumerate(times_list):
        t = np.asarray(t, dtype=float)
        if t.size:
            if np.any(np.diff(t) < 0):
                raise ValueError(f"times_list[{c}] must be nondecreasing.")
            all_times.append(t)
            all_chans.append(np.full(t.size, c, dtype=int))

    if not all_times:
        return np.zeros((C, query_times.size), dtype=float)

    global_times = np.concatenate(all_times)
    global_chan = np.concatenate(all_chans)

    # Sort by time (stable/deterministic tie-break by channel)
    order = np.lexsort((global_chan, global_times))
    global_times = global_times[order]
    global_chan = global_chan[order]

    uniq_times, inv = np.unique(global_times, return_inverse=True)
    M = uniq_times.size

    counts = np.zeros((C, M), dtype=float)
    np.add.at(counts, (global_chan, inv), 1.0)

    # Backward recursion over unique times:
    S = np.zeros((C, M), dtype=float)
    for k in range(M - 2, -1, -1):
        dt = uniq_times[k + 1] - uniq_times[k]
        a = np.exp(-dt / tau)
        S[:, k] = a * (counts[:, k + 1] + S[:, k + 1])

    # Evaluate at query times: use next event time strictly greater than t.
    idx_next = np.searchsorted(uniq_times, query_times, side='right')
    y = np.zeros((C, query_times.size), dtype=float)
    valid = idx_next < M
    if np.any(valid):
        dt_next = uniq_times[idx_next[valid]] - query_times[valid]
        a = np.exp(-dt_next / tau)
        y[:, valid] = a * (counts[:, idx_next[valid]] + S[:, idx_next[valid]])
    return y


def plot_prediction_error_event_based(
    event_times,
    x_event_times,
    predictions,
    tau,
    cumulative_channels=[],
    title='Prediction Error',
    figsize=(12, 6),
    ax=None,
    show=True,
    channel_labels=None,
    ylabel='Error',
):
    r"""Plot prediction error over event times using one-step-ahead (previous event) predictions.

    The model predictions are interpreted as
        \hat{a}_i(t_{k-1}) \approx \exp(-(t^{(i)}_{\text{next}} - t_{k-1})/\tau)
    so when a channel i actually spikes at time t_k, the corresponding "true" value
    for evaluating the previous-step prediction is
        a_i(t_k) = \exp(-(t_k - t_{k-1})/\tau).

    Error definition:
        e_{i,k} = a_i(t_k) - \hat{a}_i(t_{k-1})   (only defined when channel i spikes at t_k)

    Parameters:
    - event_times: 1D array-like of global event times t_k (length N)
    - x_event_times: list/sequence of length n_inputs, each element is an array-like of true spike times for that channel
    - predictions: 2D array of shape (n_outputs, N) holding model predictions at each t_k
        The first n_inputs rows are used.
    - tau: time constant used in the exp(-dt/tau) target
    - cumulative_channels: list of channel indices for which predictions are cumulative (i.e. precition at time t_k0 should be compared to sum of exp(-(t_k - t_{k0})/tau) for all k such that t_k > t_k0 and channel i spikes at t_k)
    - title: plot title
    - figsize: figure size used when ax is None
    - ax: optional matplotlib Axes. If provided, all channels are drawn in the same Axes.
          If None, creates one subplot per channel.
    - show: whether to call plt.show()
    - channel_labels: optional list of labels (length n_inputs)
    - ylabel: y-axis label

    Returns:
    - fig, axs, errors, true_values
        errors has shape (n_inputs, N-1) and corresponds to event_times[1:]
        true_values has the same shape, with NaN where a channel did not spike at that event.
    """
    event_times = np.asarray(event_times)
    predictions = np.asarray(predictions)
    n_inputs = len(x_event_times)

    if event_times.ndim != 1:
        raise ValueError(f"event_times must be 1D, got shape {event_times.shape}")
    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")
    if predictions.shape[1] != event_times.shape[0]:
        raise ValueError(
            f"predictions second dimension must match len(event_times)={event_times.shape[0]}, got {predictions.shape[1]}"
        )
    if predictions.shape[0] < n_inputs:
        raise ValueError(
            f"predictions must have at least {n_inputs} rows to match len(x_event_times), got {predictions.shape[0]}"
        )
    if event_times.shape[0] < 2:
        raise ValueError("Need at least 2 event times to compute one-step prediction error")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    # Build event indicator matrix x(t_k) for each channel i at each global event time t_k.
    x_matrix = np.zeros((n_inputs, event_times.shape[0]), dtype=float)
    for i in range(n_inputs):
        channel_times = np.asarray(x_event_times[i])
        if channel_times.size == 0:
            continue
        x_matrix[i, np.isin(event_times, channel_times)] = 1.0

    # One-step-ahead comparison: compare prediction made at t_{k-1} to the truth at t_k.
    # Truth is only defined for channels that actually spike at t_k.
    pred_prev = predictions[:n_inputs, :-1]  # \hat{a}(t_{k-1}) aligned with k=1..N-1
    dt = np.diff(event_times)  # dt[k-1] = t_k - t_{k-1}
    a_true_scalar = np.exp(-dt / tau)  # length N-1

    spiked_at_k = x_matrix[:, 1:].astype(bool)  # shape (n_inputs, N-1)
    true_values = np.full((n_inputs, event_times.shape[0] - 1), np.nan, dtype=float)
    true_values[spiked_at_k] = np.broadcast_to(a_true_scalar, true_values.shape)[spiked_at_k]
    errors = true_values - pred_prev
    t_err = event_times[1:]

    # True cumulative values for all channels, aligned with provided (unique) event_times
    x_cumulative = discounted_future_counts_at_times(x_event_times, tau, event_times)
    x_cumulative = np.asarray(x_cumulative, dtype=float)[:, :-1]
    errors_cumulative = x_cumulative - pred_prev

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if channel_labels is None:
        channel_labels = [f'x{i+1}' for i in range(n_inputs)]
    if len(channel_labels) != n_inputs:
        raise ValueError(f"channel_labels must have length {n_inputs}, got {len(channel_labels)}")

    if ax is None:
        fig, axs = plt.subplots(n_inputs, 1, figsize=(figsize[0], max(figsize[1], 2 * n_inputs)), sharex=True)
        if n_inputs == 1:
            axs = np.array([axs])
        for i in range(n_inputs):
            mask = ~np.isnan(errors[i, :])
            if i in cumulative_channels:
                axs[i].scatter(t_err[mask], errors_cumulative[i, mask], s=18, color=colors[i % len(colors)], label=f'{channel_labels[i]} (cumulative)')
            else:
                axs[i].scatter(t_err[mask], errors[i, mask], s=18, color=colors[i % len(colors)], label=channel_labels[i])
            axs[i].axhline(0.0, color='k', linewidth=1, alpha=0.25)
            axs[i].set_ylabel(ylabel)
            axs[i].grid(True, alpha=0.3)
            axs[i].legend(loc='upper right')
        axs[-1].set_xlabel('Time')
        fig.suptitle(title)
        fig.tight_layout()
    else:
        fig = ax.figure
        axs = ax
        for i in range(n_inputs):
            mask = ~np.isnan(errors[i, :])
            ax.scatter(t_err[mask], errors[i, mask], s=18, color=colors[i % len(colors)], label=channel_labels[i])
        ax.axhline(0.0, color='k', linewidth=1, alpha=0.25)
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    if show:
        plt.show()
    return fig, axs, errors, true_values

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