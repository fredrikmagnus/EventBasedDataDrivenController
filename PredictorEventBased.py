import numpy as np

class Predictor:
    def __init__(self, n_inputs, gamma_weights, tau_decay, lambda_ridge, eta, affine=False, spiking=False):
        self.n_inputs = n_inputs
        self.n_outputs = n_inputs
        self.gamma_weights = gamma_weights
        self.tau_decay = tau_decay
        self.lambda_ridge = lambda_ridge
        self.eta = eta
        self.affine = affine
        self.spiking = spiking

        self.z = np.zeros(self.n_inputs) # Trace vector
        
        if self.affine:
            self.n_inputs += 1 # Add bias input
            self.z = np.hstack((self.z, np.array([1.0]))) # Bias input
        if self.spiking:
            # NOTE: Own output is fed back as input for next time step.
            # The trace for this feedback input is stored at index 0.
            self.n_inputs += 1 # Add feedback from previous output
            self.z = np.hstack((np.array([0.0]), self.z)) # Feedback input
            self.n_outputs += 1 # Add output channel for predicting next output spike

        self.Sigma = self.lambda_ridge * np.eye(self.n_inputs)
        self.Psi = np.zeros((self.n_outputs, self.n_inputs))
        # self.W = np.zeros((self.n_outputs, self.n_inputs)) # Prediction weights
        self.W = 0.2*np.random.rand(self.n_outputs, self.n_inputs) # Random initial prediction weights
        # self.W[0, 0] = -0.1
        # self.W[0, 2] = 0.1
        # self.spike_threshold = 0.9 # Initial spike threshold
        self.spike_threshold = np.random.rand() # Random initial spike threshold

        self.t_prev = 0.0 # Time of previous event


    def gradient_update(self, t:float, x_in:np.ndarray):
        """
        Perform a gradient update of the prediction weights based on incoming spikes. 
        Then does a prediction. 
        Returns the prediction and the output spike (if spiking=True).

        Assumes that x_in != 0. I.e. that this function is only called at events. 
        """
        # 1) Decay traces:
        dt = t - self.t_prev
        decay = np.exp(-dt/self.tau_decay)
        self.z *= decay

        if self.spiking:
            if self.z[-1] <= self.spike_threshold + 1e-9:
                # print(self.z[-1], self.spike_threshold, t)
                self.z[-1] = self.spike_threshold 
                x_in = np.hstack((np.array([1.0]), x_in)) # Add feedback from previous output spike
            else:
                x_in = np.hstack((np.array([0.0]), x_in)) # Add feedback from previous output spike

        # 3) Update traces with incoming spikes
        z_pre = self.z.copy() 
        z_post = z_pre.copy() #+ x_in #* (1-self.decay)/self.dt

        z_post[:len(x_in)] += x_in
        if self.affine:
            z_post[-1] = 1.0 # Bias input is always active


        pred = self.W @ z_post
        if self.spiking:
            # Predict when to spike next:
            self.spike_threshold = pred[0] # Update spike threshold prediction
        # 4) Update covariance estimates and prediction weights when there is an input spike
        # Per-channel next-spike prediction:
        D_bar_x = np.diag((1-x_in))
        # grad_W = self.W @ np.outer(z_post, z_post) - np.outer(x_in, z_pre) - D_bar_x @ self.W @ np.outer(z_post, z_pre) + self.lambda_ridge * self.W
        grad_W_prev = self.W @ np.outer(z_post, z_post) + self.lambda_ridge * self.W
        grad_W_current = - np.outer(x_in, z_pre) - D_bar_x @ self.W @ np.outer(z_post, z_pre)
        
        # self.W -= self.eta * grad_W
        self.W -= self.eta * grad_W_prev
        self.W -= self.eta * grad_W_current
        
        
        

        # Standard global next-spike predictor, affine model:
        # self.Psi = self.gamma_weights*self.Psi + (1-self.gamma_weights)*np.outer(x_in, z_pre)
        # # Gradient = W @ Sigma_{k-1} - Psi_{k} + lambda_ridge * W
        # grad_W = self.W @ self.Sigma - self.Psi + self.lambda_ridge * self.W
        # self.W -= self.eta * grad_W
        # self.Sigma = self.gamma_weights*self.Sigma + (1-self.gamma_weights)*np.outer(z_post, z_post)
        # SGD update:
        # grad_W = self.W @ np.outer(z_post, z_post) - np.outer(x_in, z_pre) + self.lambda_ridge * self.W
        # self.W -= self.eta * grad_W

        # print("Trace values:", z_post)
        # print("Spike threshold:", self.spike_threshold)
        # print("Bias trace value:", self.z[-1])
        # print("Spike prediction:", -np.log(self.W[0, :] @ z_post)*self.tau_decay)

        self.z = z_post
        self.t_prev = t

        if self.spiking:
            return pred, x_in[0] # Return spike
        
        return pred, None # Return prediction without spike
    

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