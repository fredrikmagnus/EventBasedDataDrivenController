import numpy as np

class Predictor:
    def __init__(self, n_inputs, gamma_weights, tau_decay, lambda_ridge, dt, affine=False):
        self.n_inputs = n_inputs
        self.n_outputs = n_inputs
        self.gamma_weights = gamma_weights
        self.tau_decay = tau_decay
        self.lambda_ridge = lambda_ridge
        self.dt = dt
        self.decay = np.exp(-dt/self.tau_decay) # Per time-step trace decay
        self.affine = affine

        self.z = np.zeros(self.n_inputs) # Trace vector

        if self.affine:
            self.n_inputs += 1 # Add bias input
            self.z = np.hstack((self.z, np.array([0.0]))) # Bias input

        self.Sigma = self.lambda_ridge * np.eye(self.n_inputs)
        self.Psi = np.zeros((self.n_outputs, self.n_inputs))
        self.P = np.zeros((self.n_outputs, self.n_inputs)) # Prediction weights
        

    def update(self, x_in:np.ndarray):
        # 1) Decay traces:
        self.z *= self.decay

        # 3) Update traces with incoming spikes
        z_pre = self.z.copy() 
        z_post = z_pre + x_in #* (1-self.decay)/self.dt

        

        # 4) Update covariance estimates and prediction weights when there is an input spike
        if x_in.sum() > 0:
            if self.affine:
                z_post[-1] = 1.0 # Bias input is always active
            # x_k z_{k}^T = alpha_k x_{k+1} z_{k}^-T
            self.Psi = self.gamma_weights*self.Psi + (1-self.gamma_weights)*np.outer(x_in, z_pre)
            # self.CrossCov = self.gamma_weights*self.CrossCov + (1-self.gamma_weights)*np.outer(z_post, z_pre)
            # Standard global next-spike prediction:
            # self.Sigma = self.gamma_weights*self.Sigma + (1-self.gamma_weights)*(np.outer(z_post, z_post) + self.lambda_ridge*np.eye(self.n_inputs))
            # Bellman infinite-horizon prediction:
            # self.Sigma = self.gamma_weights*self.Sigma + (1-self.gamma_weights)*(np.outer(z_post, z_post) - np.outer(z_post, z_pre) + self.lambda_ridge*np.eye(self.n_inputs))

            # Per-channel next-spike prediction:
            # self.Sigma = self.gamma_weights*self.Sigma + (1-self.gamma_weights)*(np.outer(z_post, z_post) - np.outer(z_post, (1-x_in)*z_pre) + self.lambda_ridge*np.eye(self.n_inputs))
            if self.affine:
                x_in_expanded = np.zeros(self.n_inputs)
                x_in_expanded[:len(x_in)] = x_in
                x_in_expanded[-1] = 1.0 # Bias input is always active
            else:
                x_in_expanded = x_in

            self.Sigma = self.gamma_weights*self.Sigma + (1-self.gamma_weights)*(np.outer(z_post, z_post) - np.outer(z_post, (1-x_in_expanded)*z_pre) + self.lambda_ridge*np.eye(self.n_inputs))
            
            self.P = self.Psi @ np.linalg.inv(self.Sigma)

        self.z = z_post

    def predict(self):
        return self.P @ self.z
    

class Controller:
    def __init__(self, n_inputs, gamma_weights, tau_decay, lambda_ridge, Q, dt):
        self.n_inputs = n_inputs + 1 # Include feedback from previous output
        self.gamma_weights = gamma_weights
        self.tau_decay = tau_decay
        self.lambda_ridge = lambda_ridge
        self.dt = dt
        self.decay = np.exp(-dt/self.tau_decay) # Per time-step trace decay

        self.z = np.zeros(self.n_inputs) # Trace vector

        # self.Sigma = self.lambda_ridge * np.eye(self.n_inputs)
        self.Sigma = self.lambda_ridge * np.random.rand(self.n_inputs, self.n_inputs)
        self.Psi = np.zeros((self.n_inputs, self.n_inputs))
        self.P = self.lambda_ridge * np.eye(self.n_inputs) # Prediction weights

        # Control parameters
        self.rho = 0 # Control effort penalty
        # self.Q = np.eye(self.n_inputs-1) # Cost on state deviation
        self.Q = Q

        #Spike when z_u <= spike_threshold (z_u: output trace)
        self.spike_threshold = 1 # Induce initial spike

    def update(self, x_in:np.ndarray, a_ref:np.ndarray):
        # 1) Update predictor:
        # a) Decay traces:
        self.z *= self.decay
        if x_in.sum() == 0:
            return 
        # b) Update traces with incoming spikes
        z_pre = self.z.copy()
        z_post = z_pre + x_in #* (1-self.decay)/self.dt
        self.z = z_post.copy()

        # c) Update covariance estimates and prediction weights when there is an input spike

        # x_k z_{k}^T = alpha_k x_{k+1} z_{k}^-T
        self.Psi = self.gamma_weights*self.Psi + (1-self.gamma_weights)*np.outer(x_in, z_pre)
        # self.CrossCov = self.gamma_weights*self.CrossCov + (1-self.gamma_weights)*np.outer(z_post, z_pre)
        # Standard global next-spike prediction:
        # self.Sigma = self.gamma_weights*self.Sigma + (1-self.gamma_weights)*(np.outer(z_post, z_post) + self.lambda_ridge*np.eye(self.n_inputs))
        # Bellman infinite-horizon prediction:
        # self.Sigma = self.gamma_weights*self.Sigma + (1-self.gamma_weights)*(np.outer(z_post, z_post) - np.outer(z_post, z_pre) + self.lambda_ridge*np.eye(self.n_inputs))

        # Per-channel next-spike prediction:
        self.Sigma = self.gamma_weights*self.Sigma + (1-self.gamma_weights)*(np.outer(z_post, z_post) - np.outer(z_post, (1-x_in)*z_pre) + self.lambda_ridge*np.eye(self.n_inputs))
        self.P = self.Psi @ np.linalg.inv(self.Sigma)

        # 2) Compute optimal control input:
        alpha_star = self.cubic_control(a_ref)
        self.spike_threshold = alpha_star * self.z[-1]

    def cubic_control(self, a_ref):
        P = self.P #[:-1, :] # Prediction weights for state

        x_u = np.zeros(self.n_inputs) # Control input vector
        x_u[-1] = 1.0 # Control input weight

        p, q, r = P @ self.z, P @ x_u, a_ref
        Q = self.Q
        A = p@Q@p.T
        B = p@Q@q.T
        C = q@Q@q.T
        D = p@Q@r.T
        E = q@Q@r.T

        # Cost-derivative: 4*A*alpha**3 + 6*B*alpha**2 + 2*(C-2*D + self.rho)*alpha -2*E
        poly = np.polynomial.Polynomial([-2*E, 2*(C - 2*D + self.rho), 6*B, 4*A])
        roots = poly.roots()
        real_roots = roots[np.isreal(roots)].real
        feasible_roots = real_roots[(real_roots >= 0) & (real_roots <= 1)]

        candidates = np.unique(np.concatenate((feasible_roots, np.array([1e-3, 1.0]))))
        costs = []
        for alpha in candidates:
            y = (alpha**2) * p + alpha * q - r
            cost = y @ Q @ y.T + self.rho * alpha**2
            costs.append(cost)

        alpha_star = candidates[np.argmin(costs)]

        # Compare with cost of doing nothing (no spike)
        y_no_spike = self.P @ self.z - r
        cost_no_spike = y_no_spike @ self.Q @ y_no_spike.T
        if cost_no_spike < np.min(costs):
            alpha_star = 1e-3

        return alpha_star
    
    def predict(self):
        return self.P @ self.z
    
    def predict_optimal(self, a_ref:np.ndarray):
        alpha_star = self.cubic_control(a_ref)
        x_u = np.zeros(self.n_inputs) # Control input vector
        x_u[-1] = 1.0 # Control input weight
        a_opt = alpha_star * self.P[:-1, :] @ (alpha_star * self.z + x_u)
        a_opt = np.concatenate((a_opt, np.array([alpha_star])))
        return a_opt
    
    def spike(self, a_ref:np.ndarray):
        if self.z[-1] <= self.spike_threshold:
            # x_in = np.zeros(self.n_inputs)
            # x_in[-1] = 1.0
            # self.update(x_in, a_ref)
            return 1.
        else:
            return 0.
        # dJ/dalpha = z(t)^T@P^T@Q@P@z(t) + z(t)^T@P^T@Q(P@x_u - a_ref) 
        # x_u = np.zeros(self.n_inputs) # Control input vector
        # x_u[-1] = 1.0 # Control input weight
        # cost_derivative = self.z.T @ self.P.T @ self.Q @ self.P @ self.z + self.z.T @ self.P.T @ self.Q @ (self.P @ x_u - a_ref)
        # if cost_derivative <= 0:
        #     return 1.0
        # else:
        #     return 0.0
    


        

    

def spike_signal(t, period, phase, randomize=0):
    """
    Generate periodic spike signal.
    
    Args:
        t: time array
        period: spike period
        phase: phase offset
        randomize: add small random jitter to spike times. Uniform in [-randomize, randomize]
    
    Returns:
        Binary array with spikes at closest indices to theoretical spike times
    """
    spikes = np.zeros_like(t, dtype=int)
    
    # Generate theoretical spike times
    max_time = t[-1]
    n_spikes = int((max_time - phase) / period) + 1
    theoretical_spike_times = phase + np.arange(n_spikes) * period
    
    # Only keep spike times within the time range
    theoretical_spike_times = theoretical_spike_times[
        (theoretical_spike_times >= t[0]) & (theoretical_spike_times <= max_time)
    ]
    
    # Find closest indices for each theoretical spike time
    for spike_time in theoretical_spike_times:
        if randomize:
            jitter = np.random.uniform(-randomize, randomize)
            spike_time += jitter
        closest_idx = np.argmin(np.abs(t - spike_time))
        spikes[closest_idx] = 1
    
    return spikes