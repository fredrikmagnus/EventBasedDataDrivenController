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
    

class DC_motor:
    """
    DC motor (angular velocity output, voltage input) 

    State  x = [current; angular velocity]
    Input  u = voltage
    Output y = angular velocity

    Parameters
    ----------
    R : float
        Resistance.
    L : float
        Inductance.
    Kb : float
        Back-EMF constant.
    Km : float
        Motor constant.
    Kf : float
        Friction constant.
    J : float
        Inertia.
    Ts : float
        Sampling time interval.
    """
    def __init__(self, R=2.0, L=0.5, Kb=0.1, Km=0.1, Kf=0.02, J=0.02, Ts=0.01, x0=np.zeros((2,1))):
        self.R, self.L, self.Kb, self.Km, self.Kf, self.J, self.Ts = R, L, Kb, Km, Kf, J, Ts

        # 1. continuous-time matrices
        A_c = np.array([[-R/L,    -Kb/L],
                        [Km/J,  -Kf/J]])
        B_c = np.array([[1/L],
                        [0]])
        C   = np.array([[0, 1]])

        # 2. exact discretisation  (using matrix exponential)
        M = np.block([[A_c, B_c],
                      [np.zeros((1,3))]])      # augment for integral
        M_d = expm(M*Ts)                       # expm([[A, B],[0,0]])

        self.A = M_d[:2, :2]                   # e^{A_c Ts}
        self.B = M_d[:2,  2:3]                 # integral_0^{Ts} e^{A tau} B dtau
        self.C = C

        # # Euler discretisation
        # self.A = np.eye(2) + A_c * Ts
        # self.B = B_c * Ts
        # self.C = C

        self.x = np.asarray(x0, dtype=float).reshape(2, 1)
        self.y = self.C @ self.x  # initial output
 

    # 2. one simulation step -------------------------------------------------
    def step(self, u, input_noise_STD=0, process_noise_STD=0, measurement_noise_STD=0):
        """
        Parameters
        ----------
        u : ndarray shape (1,1)
            Voltage applied during the current interval [kTs,(k+1)Ts).

        Returns
        -------
        y : ndarray shape (1,1)
            Output after the state update (y_{k+1}).
        """
        u = u.reshape((1,1)) + np.random.normal(0, input_noise_STD, (1,1))
        # process_noise = 0.0 if not process_noise else 1.0
        # measurement_noise = 0.0 if not measurement_noise else 1.0
        self.x = self.A @ self.x + self.B @ u + np.random.normal(0, process_noise_STD, (2,1))
        self.y = self.C @ self.x + np.random.normal(0, measurement_noise_STD, (1,1))
        return self.y


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