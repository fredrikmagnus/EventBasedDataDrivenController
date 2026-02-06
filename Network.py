import numpy as np
import DataModels as dm
from Controller import Controller

class NetworkBuilder:
    @staticmethod
    def create_connectivity(network_params: dm.SpikingNetwork, n_inputs_plant=1):
        """
        Creates:
            - adjacency: The adjacency matrix for inter-neuron connectivity based on the specified type.
                Shape: (total_neurons, total_neurons)
                Element ij: 1 if neuron i connects to neuron j, 0 otherwise.
            - ext_in: Connectivity matrix for external inputs to neurons (from plant to neurons).
                Shape: (plant_output_dim, total_neurons)
            - ext_out: Connectivity matrix for external outputs from neurons (from neurons to plant).
                Shape: (plant_input_dim, total_neurons)
        """
        n_neurons = network_params.n_neurons
        connectivity = network_params.connectivity
        
        # Initialize matrices
        adjacency = np.zeros((n_neurons, n_neurons))
        ext_in = np.zeros((n_inputs_plant, n_neurons))  # Plant output dimension is 1 (position)
        ext_out = np.zeros((1, n_neurons))  # Plant input dimension is 1 (force)

        if connectivity == 'full':
            adjacency.fill(1) # Fully connected network
            adjacency[np.arange(n_neurons), np.arange(n_neurons)] = 0 # No self-connections
            ext_in.fill(1)  # All neurons receive plant output
            ext_out[0, :n_neurons//2] = -1 # Half neurons apply negative force
            ext_out[0, n_neurons//2:] = 1  # Half neurons apply positive force
            # ext_out = np.array([(-1)**i for i in range(n_neurons)], dtype=float).reshape(1, -1)

        elif connectivity == 'none':
            ext_in.fill(1)  # All neurons receive plant output
            ext_out[0, :n_neurons//2] = -1 # Half neurons apply negative force
            ext_out[0, n_neurons//2:] = 1  # Half neurons apply positive force
            # ext_out = np.array([(-1)**i for i in range(n_neurons)], dtype=float).reshape(1, -1)

        elif connectivity == 'custom':
            custom_conn = network_params.custom_connectivity
            
            # Use custom adjacency matrix if provided
            if custom_conn.feedforward_connectivity is not None:
                adjacency = NetworkBuilder.feed_forward_adjacency(custom_conn.feedforward_connectivity)
            elif custom_conn.adjacency_matrix is not None:
                adjacency_array = np.array(custom_conn.adjacency_matrix)
                if adjacency_array.shape != (n_neurons, n_neurons):
                    raise ValueError(f"Custom adjacency matrix must be {n_neurons}x{n_neurons}, got {adjacency_array.shape}")
                adjacency = adjacency_array
            
            # Use custom external input connections if provided
            if custom_conn.ext_in is not None:
                ext_in_array = np.array(custom_conn.ext_in).reshape(1, -1)
                if ext_in_array.shape[1] != n_neurons:
                    raise ValueError(f"Custom ext_in must have {n_neurons} elements, got {ext_in_array.shape[1]}")
                ext_in = ext_in_array
            else:
                ext_in.fill(1)  # Default: all neurons receive plant output
            
            # Use custom external output connections if provided
            if custom_conn.ext_out is not None:
                ext_out_array = np.array(custom_conn.ext_out).reshape(1, -1)
                if ext_out_array.shape[1] != n_neurons:
                    raise ValueError(f"Custom ext_out must have {n_neurons} elements, got {ext_out_array.shape[1]}")
                ext_out = ext_out_array
            else:
                # Default: half neurons apply positive force, half apply negative force
                ext_out[0, :n_neurons//2] = 1
                ext_out[0, n_neurons//2:] = -1

        adjacency = adjacency.astype(int)  # Ensure adjacency is integer type

        return adjacency, ext_in, ext_out

    @staticmethod
    def create_network(config: dm.Config, n_inputs_plant=1):
        """
        Create a spiking network based on the provided parameters.
        
        Parameters:
        -----------
        network_params : SpikingNetwork
            Configuration object containing network parameters
        
        Returns:
        --------
        list of SpikingController
            List of controllers representing the spiking network
        """
        network_params = config.spiking_network
        adjacency, ext_in, ext_out = NetworkBuilder.create_connectivity(network_params, n_inputs_plant=n_inputs_plant)

        print("Adjacency matrix:")
        print(adjacency)
        print("External input connections (ext_in):")
        print(ext_in)
        print("External output connections (ext_out):")
        print(ext_out)

        controllers = []

        # Handle reference_tracking_cost as either float or list
        reference_costs = network_params.controller.reference_tracking_cost.cost
        if isinstance(reference_costs, (int, float)):
            # Single value for all controllers
            reference_cost_list = [reference_costs] * network_params.n_neurons
        else:
            # List of values
            reference_cost_list = reference_costs
            if len(reference_cost_list) != network_params.n_neurons:
                raise ValueError(f"reference_tracking_cost list must have {network_params.n_neurons} elements, got {len(reference_cost_list)}")

        # Handle reference_tracking_cost_enable as either string or list of strings
        reference_tracking_cost_enable = network_params.controller.reference_tracking_cost.enable
        if isinstance(reference_tracking_cost_enable, str):
            reference_tracking_cost_enable = [reference_tracking_cost_enable] * network_params.n_neurons

        for i in range(network_params.n_neurons):
            m_in_plant = int(ext_in[:, i].sum())  # Number of inputs from the plant

            m_in_spikes = int(adjacency[:, i].sum())  # Number of inputs from other neurons
            m_in = m_in_plant + m_in_spikes  # Total inputs for the controller
            
            Q = np.zeros((m_in+1, m_in+1))
            Q[0,0] = reference_cost_list[i]  # Penalize deviation of first state
            Q[-1,-1] = network_params.controller.mu  # Control input cost

            controller = Controller(
                n_inputs=m_in,
                gamma_weights=network_params.controller.gamma,
                tau_decay=network_params.controller.tau,
                lambda_ridge=network_params.controller.lambda_ridge,
                Q=Q,
                dt=config.simulation.dt,
            )
            controllers.append(controller)

        return controllers, adjacency, ext_in, ext_out
    
    @staticmethod
    def feed_forward_adjacency(layer_neurons):
        """
        Return the adjacency matrix of a fully connected feed-forward network.

        Parameters
        ----------
        layer_neurons : list[int]
            Number of neurons in each successive layer, e.g. [1, 2, 1].

        Returns
        -------
        adj : np.ndarray, shape (N, N), dtype=int
            Adjacency matrix with entry (i, j) = 1 when neuron i connects to j.
            Neurons are indexed layer-by-layer starting from 0.
        """
        if not layer_neurons:
            raise ValueError("layer_neurons must contain at least one layer")
        if any(n <= 0 for n in layer_neurons):
            raise ValueError("each layer must have at least one neuron")

        total = sum(layer_neurons)
        adj = np.zeros((total, total), dtype=int)

        idx = 0                              # running index of first neuron in layer ℓ
        for n_curr, n_next in zip(layer_neurons[:-1], layer_neurons[1:]):
            curr = np.arange(idx, idx + n_curr)              # neuron indices in layer ℓ
            nex  = np.arange(idx + n_curr, idx + n_curr + n_next)  # indices in layer ℓ+1
            adj[np.repeat(curr, n_next), np.tile(nex, n_curr)] = 1
            idx += n_curr

        return adj