import importlib
import sys
import numpy as np
import matplotlib.pyplot as plt
from PredictorEventBased import Predictor, spike_signal
from SpikingSystems import IFSpikeEncoder_absolute, SpringMassDamper, spike_decoder, LeakyIntegrator
from Plots import compare_event_time_predictions, plot_gains_event_based, plot_raw_predictions_event_based, plot_prediction_error_event_based
importlib.reload(sys.modules['PredictorEventBased'])
importlib.reload(sys.modules['SpikingSystems'])
from PredictorEventBased import Predictor, spike_signal
from SpikingSystems import IFSpikeEncoder_absolute, SpringMassDamper, spike_decoder, LeakyIntegrator

def plot_trajectory_with_events(time, trajectory, event_times):
    """
    Plots a scalar trajectory over time and marks event times with dots.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(time, trajectory, label='Trajectory')
    # plt.scatter(event_times, np.interp(event_times, time, trajectory), color='red', label='Events')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Trajectory with Event Times')
    plt.legend()
    plt.grid()
    plt.show()


# 1) Set up system:
# Simulation parameters
dt = 0.01
# Create spring-mass-damper system
mass = 1.0
damping = 1.
stiffness = 1.
system = SpringMassDamper(dt, mass, damping, stiffness, x0=[0., 0.5])
# # Leaky integrator system:
# tau_leak = 10
# system = LeakyIntegrator(dt, tau_leak, x0=0.)

# Set up the spike-encoder
encoder = IFSpikeEncoder_absolute(dt=dt, threshold=0.05)

N_predictors = 2

predictor1 = Predictor(
    n_inputs=encoder.n_outputs + N_predictors-1,   # Number of input channels
    tau_decay=.1,      # Time constant for trace decay
    lambda_ridge=1e-6,   # Ridge regularization parameter
    eta=0.3,            # Learning rate for gradient update
    eta_cumulative=0.1,
    cumulative_channels=[-1], # Always accumulate covariance for first input channel
    reference_tracking_costs=[0., 0., 0.], # Cost for tracking reference in each output channel (0 means no reference tracking)
    activation_enable=True,
    affine=True,   # Include affine term in predictor
    spiking=True,
    noise_std=0.1, # Standard deviation of noise added to trace at each event
)

predictor2 = Predictor(
    n_inputs=encoder.n_outputs + N_predictors-1,   # Number of input channels
    tau_decay=.1,      # Time constant for trace decay
    lambda_ridge=1e-6,   # Ridge regularization parameter
    eta=0.3,            # Learning rate for gradient update
    eta_cumulative=0.1,
    cumulative_channels=[-1], # Always accumulate covariance for first input channel
    reference_tracking_costs=[0., 0., 0.], # Cost for tracking reference in each output channel (0 means no reference tracking)
    activation_enable=True,
    affine=True,   # Include affine term in predictor
    spiking=True,
    noise_std=0.1, # Standard deviation of noise added to trace at each event
    seed=1
)

predictors = [predictor1, predictor2]

reference = np.zeros(predictor1.n_outputs) 
reference[-1] = 0.2

T = 200.0
time = np.arange(0, T, dt)
sys_val = [system.y] # Position of spring-mass-damper system
sys_events = [] # Times of system-generated spike events
controller_events = [[] for _ in predictors] # Times of controller-generated spike events for each predictor

predictions = [[] for _ in predictors] # Predictions of each predictor at each event time
PredictionGains = [[] for _ in predictors] # Prediction gains of each predictor at each event time

# next_event_time = predictor1.time_to_spike() 
next_event_times = [predictor.time_to_spike() for predictor in predictors]
# Test simulation:
for t in time:
    x_in = np.zeros(N_predictors+1)
    u_in = 0.
    for i, spike_time in enumerate(next_event_times):
        if t >= spike_time:
            # Controller spikes
            controller_idx = i
            u_in += 1.
            # x_in = np.array([1, 0]) 
            x_in[controller_idx] = 1
            # controller_events.append(t)
            
            controller_events[controller_idx].append(t)
    
    system.step(u=u_in) # Step the system with control input
    sys_val.append(system.y)
    if encoder.step(system.y):
        # System generates spike event
        sys_events.append(t)
        x_in[-1] = 1 # Set input channel to 1 for system spike events

    if x_in.sum() > 0: # If there is an event
        for idx, predictor in enumerate(predictors):
            # Controller 
            PredictionGains[idx].append(predictor.W.copy())
            predictor.update_state(t, x_in)
            next_event_times[idx] = predictor.time_to_spike()
            pred = predictor.predict()
            predictions[idx].append(pred)
            predictor.gradient_update(reference) # Update predictor weights based on current state and reference (not used
        # PredictionGains.append(predictor1.W.copy())
        # predictor1.update_state(t, x_in) # Update predictor state with current input
        # next_event_time = t + predictor1.time_to_spike() # Get time of next predicted spike event based on current state
        # pred = predictor1.predict() # Get current prediction
        # predictions.append(pred)
        # predictor1.gradient_update(reference) # Update predictor weights based on current state and reference (not used in this example)



all_events = controller_events
all_events.append(sys_events)

event_times_global = np.unique(np.concatenate(all_events))

predictions = np.array(predictions[0]).T
print(predictions.shape)
PredictionGains = np.array(PredictionGains[0]).transpose(1, 2, 0)

plot_trajectory_with_events(time, sys_val[:-1], sys_events)

compare_event_time_predictions(
    x_event_times=all_events,
    predictions=predictions,
    tau=predictor1.tau_decay,
)

plot_gains_event_based(
    event_times=event_times_global,
    prediction_gains=PredictionGains,
)

# Plot raw predictions:
plot_raw_predictions_event_based(
    event_times=event_times_global,
    predictions=predictions,
)

# Plot one-step-ahead prediction error:
plot_prediction_error_event_based(
    event_times=event_times_global,
    x_event_times=all_events,
    predictions=predictions,
    tau=predictor1.tau_decay,
    cumulative_channels=predictor1.cumulative_channels,
)

