import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


class SpringMassDamper:
    def __init__(self, dt, mass, damping, stiffness, x0=[0.0, 0.0]):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness

        # 1. continuous-time matrices
        A_c = np.array([
            [0, 1],
            [-self.stiffness/self.mass, -self.damping/self.mass]
        ])
        B_c = np.array([
            [0],
            [1/self.mass]
        ])
        C   = np.array([[1, 0]])

        # 2. exact discretisation  (using matrix exponential)
        M = np.block([
            [A_c, B_c],
            [np.zeros((1,3))]
        ])      # augment for integral
        M_d = expm(M*dt)                       # expm([[A, B],[0,0]])
        self.A = M_d[:2, :2]                   # e^{A_c dt}
        self.B = M_d[:2,  2:3]                 # ∫_0^{dt} e^{A τ} B dτ
        self.C = C

        self.state = np.array(x0, dtype=float)               # initial state
        self.y = self.C @ self.state

    def step(self, u):
        self.state = self.A @ self.state + self.B.flatten() * u
        self.y = self.C @ self.state
        return self.y[0]


class IFSpikeEncoder_absolute:
    def __init__(self, threshold, dt):
        self.threshold = threshold
        self.dt = dt
        self.n_outputs = 1
        self.integral = 0.0

    def step(self, input_value):
        self.integral += abs(input_value) * self.dt
        if self.integral >= self.threshold:
            self.integral -= self.threshold
            return np.array([1])
        else:
            return np.array([0])
        
class IFSpikeEncoder:
    def __init__(self, threshold, dt):
        self.threshold = threshold
        self.dt = dt
        self.n_outputs = 2
        self.integral = 0.0
    
    def step(self, input_value):
        self.integral += input_value * self.dt
        if self.integral >= self.threshold:
            self.integral -= self.threshold
            return np.array([1,0])
        elif self.integral <= -self.threshold:
            self.integral += self.threshold
            return np.array([0,1])
        else:
            return np.array([0,0])
        
class DifferentialEncoder:
    def __init__(self, threshold, dt):
        self.threshold = threshold
        self.dt = dt
        self.n_outputs = 2
        self.prev_input = 0.0

    def step(self, input_value):
        delta = input_value - self.prev_input
        if delta >= self.threshold:
            self.prev_input = input_value
            return np.array([1, 0])
        elif delta <= -self.threshold:
            self.prev_input = input_value
            return np.array([0, 1])
        else:
            return np.array([0, 0])
        
class EncoderArray:
    def __init__(self, encoders):
        self.encoders = encoders
        self.n_outputs = sum(encoder.n_outputs for encoder in encoders)
    def step(self, input_values):
        spikes = []
        for encoder, input_value in zip(self.encoders, input_values):
            spike = encoder.step(input_value)
            spikes.append(spike)
        return np.concatenate(spikes)
    
    
def plot_system_response(time, y, spikes, IF_integral=None):
    """
    Plots the system response, spike raster, and IF neuron integral over time.
    
    Parameters:
    - time: array of time points
    - y: array of system outputs (continuous variable)
    - spikes: array of spike events (binary variable)
    - IF_integral: array of IF encoder integral values
    """
    if IF_integral is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)


    # Top subplot: displacement
    ax1.plot(time, y[:])
    ax1.set_title("Spring-Mass-Damper System Response")
    ax1.set_ylabel("Displacement")
    ax1.grid()

    # Middle subplot: spike raster
    n_inputs = spikes.shape[0]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(n_inputs):
        spike_times = time[spikes[i, :] == 1]
        if len(spike_times) > 0:
            ax2.eventplot(
                spike_times,
                lineoffsets=i + 1,
                linewidths=2,
                colors=colors[i % len(colors)],
                label=f'x{i+1}'
            )
    ax2.set_ylabel("Input Channel")
    ax2.set_title("Observed Spikes")
    ax2.set_ylim([0.3, n_inputs + 0.7])
    ax2.set_yticks(range(1, n_inputs + 1))
    ax2.set_yticklabels([f'x{i+1}' for i in range(n_inputs)])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom subplot: IF neuron integral
    # ax3.plot(np.insert(time, 0, 0), IF_integral)
    if IF_integral is not None:
        ax3.plot(time, IF_integral)
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("IF Integral")
        ax3.grid()

    plt.tight_layout()
    plt.show()


def create_ref_signal(period, time):
    """
    Create a reference signal with given period and phase over the specified time array.
    The signal counts the time until the next desired event, resetting to zero at each event.
    Example: 
        - period=1, dt=0.1
        - time = [0, 0.1, 0.2, ..., 2.0]
        - ref_signal = [1, 0.9, 0.8, ..., 0.0, 1.0, 0.9, ..., 0.0, ...]
    """
    dt = time[1] - time[0]
    ref_signal = np.zeros_like(time)
    time_until_event = period
    for i, t in enumerate(time):
        ref_signal[i] = time_until_event
        time_until_event -= dt
        if time_until_event <= 0:
            time_until_event = period
    return ref_signal