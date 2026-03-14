import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softplus(x):
    return np.log(1 + np.exp(x))

class Predictor:
    def __init__(self, n_inputs, tau_decay, lambda_ridge, eta, eta_cumulative=None, activation_enable=True, cumulative_channels=[], reference_tracking_costs=[], affine=False, spiking=False, noise_std=0.0, seed=42):
        np.random.seed(seed)
        self.n_inputs = n_inputs
        self.n_outputs = n_inputs
        self.tau_decay = tau_decay
        self.lambda_ridge = lambda_ridge
        self.eta = eta
        self.eta_cumulative = eta_cumulative if eta_cumulative is not None else eta # Separate learning rate for cumulative channels
        self.affine = affine
        self.spiking = spiking
        self.activation_enable = activation_enable # Nonlinear activation
        self.cumulative_channels = cumulative_channels 
        self.reference_tracking_costs = reference_tracking_costs
        self.noise_std = noise_std

        if self.affine:
            self.n_inputs += 1 # Add bias input
            # self.z = np.hstack((self.z, np.array([1.0]))) # Bias input
        if self.spiking:
            # NOTE: Own output is fed back as input for next time step.
            # The trace for this feedback input is stored at index 0.
            self.n_inputs += 1 # Add feedback from previous output
            # self.z = np.hstack((np.array([0.0]), self.z)) # Feedback input
            self.n_outputs += 1 # Add output channel for predicting next output spike
        
        self.cumulative_channels_mask = np.zeros(self.n_outputs)
        self.cumulative_channels_mask[self.cumulative_channels] = 1

        self.z_post = np.zeros(self.n_inputs) # Trace vector after current event
        self.z_post[-1] = float(self.affine) # Set bias to 1 if affine is True
        self.z_pre = self.z_post.copy() # Trace vector before current event
        self.x_in = np.zeros(self.n_inputs - int(self.affine)) # Input vector for current event

        self.Q = np.diag(self.reference_tracking_costs) # Diagonal cost matrix for reference tracking
        self.R = np.eye(self.n_outputs) # Diagonal cost matrix for prediction

        # self.R[0,0] = 1
        # self.R[-1, -1] = 100

        # self.phi = np.zeros(self.n_outputs) # Storing sigmoid(W @ z)*(1-sigmoid(W @ z)) 
        self.activation_derivative = np.ones(self.n_outputs) # Storing sigmoid(W @ z)*(1-sigmoid(W @ z)) for sigmoid case, and 1 for linear case
        self.W = np.random.normal(0, 0.1, size=(self.n_outputs, self.n_inputs))

        if self.activation_enable:
            self.spike_threshold = sigmoid(self.W[0, -1]) 
        else:
            self.spike_threshold = self.W[0, -1] # Initial spike threshold set to bias weight

        self.t_prev = 0.0 # Time of previous event

    def update_state(self, t:float, x_in:np.ndarray):
        """
        Update internal state based on incoming spikes. 
        NOTE: Own spikes are assumed to be fed back through x_in (handled externally)
        """
        dt = t - self.t_prev # Time since previous event
        decay = np.exp(-dt/self.tau_decay)
        self.z_post *= decay
        self.z_post += np.random.normal(0, self.noise_std, size=self.n_inputs) # Add noise to trace
        self.z_pre = self.z_post.copy()
        self.x_in = x_in.copy()

        self.z_post[:len(x_in)] += x_in
        if self.affine:
            self.z_post[-1] = 1.0 # Bias input is always active

        self.t_prev = t

    def predict(self):
        """
        Make prediction based on current state. Should be called after update_state() in each event.
        Returns the prediction.
        """
        lin = self.W @ self.z_post
        if self.activation_enable:
            # Sigmoid for non-cumulative channels
            non_cumulative = sigmoid(lin) * (1 - self.cumulative_channels_mask)
            # Softplus for cumulative channels:
            cumulative = softplus(lin) * self.cumulative_channels_mask
            # ReLU for cumulative channels:
            # cumulative = np.maximum(0, lin) * self.cumulative_channels_mask
            return non_cumulative + cumulative
        else:
            return lin
    
    def set_activation_derivative(self):
        if self.activation_enable:
            lin = self.W @ self.z_post
            # Sigmoid derivative for non-cumulative channels:
            non_cumulative = ( sigmoid(lin) * (1 - sigmoid(lin)) ) * (1 - self.cumulative_channels_mask)
            # Softplus derivative for cumulative channels:
            cumulative = sigmoid(lin) * self.cumulative_channels_mask # Derivative of softplus is sigmoid
            # ReLU derivative for cumulative channels:
            # cumulative = (lin > 0) * self.cumulative_channels_mask # Derivative of ReLU is 1 for positive values, 0 otherwise
            self.activation_derivative = non_cumulative + cumulative
        else:
            self.activation_derivative = np.ones(self.n_outputs)

    def time_to_spike(self, feedback_channel_idx=0):
        """
        Return time until next spike based on current state and weights. 
        Only relevant if spiking=True. 
        Must be called after update_state() at each event.
        """
        pred = self.predict()
        return - self.tau_decay * np.log(pred[feedback_channel_idx])


    def gradient_update(self, reference:np.ndarray=None):
        D_bar_x = np.diag((1-self.x_in))
        for idx in self.cumulative_channels:
            D_bar_x[idx, idx] = 1 # Always accumulate for cumulative channels
        
        pred = self.predict()

        grad_W_a = - np.outer(self.R@D_bar_x @ pred*self.activation_derivative, self.z_pre) + self.lambda_ridge * self.W
        grad_W_a += - np.outer(self.R@self.x_in*self.activation_derivative, self.z_pre) 
        self.set_activation_derivative()
        grad_W_b = np.outer(self.R@pred*self.activation_derivative, self.z_post)

        grad = grad_W_a + grad_W_b
        if reference is not None:
            e = self.Q @ (pred - reference)
            grad_W_reference_tracking = np.outer(e * self.activation_derivative, self.z_post)  # (n_outputs, n_inputs)
            grad += grad_W_reference_tracking

        # Update weights
        eta = self.eta_cumulative * self.cumulative_channels_mask + self.eta * (1 - self.cumulative_channels_mask)
        self.W -= eta[:, np.newaxis] * grad
            


def spike_signal(t, period, phase, randomize=0):
    """
    Generate periodic spike signal.
    
    Args:
        t: duration of signal in seconds (float)
        period: spike period
        phase: phase offset
        randomize: add small random jitter to spike times. Uniform in [-randomize, randomize]
    """
    spike_times = np.arange(phase, t, period, dtype=np.float64) 
    spike_times += np.random.uniform(-randomize, randomize, size=len(spike_times))
    return spike_times