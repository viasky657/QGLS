import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


class TopologyGenerator:
    """Defines the 3D knot structure for the network topology."""
    
    def __init__(self, knot_type='trefoil', node_density=100, strand_count=3, braid_depth=4):
        """
        Initialize the topology generator.
        
        Args:
            knot_type (str): Type of knot to generate ('trefoil', 'figure-eight')
            node_density (int): Number of nodes per strand
            strand_count (int): Number of strands in the braid
            braid_depth (int): Complexity of the braid
        """
        self.knot_type = knot_type
        self.node_density = node_density
        self.strand_count = strand_count
        self.braid_depth = braid_depth
        
        # Generate the knot structure
        self.nodes, self.paths = self._generate_topology()
        
    def _generate_topology(self):
        """Generate the knot topology based on specified parameters."""
        if self.knot_type == 'trefoil':
            return self._generate_trefoil_knot()
        elif self.knot_type == 'figure-eight':
            return self._generate_figure_eight_knot()
        else:
            raise ValueError(f"Unsupported knot type: {self.knot_type}")
    
    def _generate_trefoil_knot(self):
        """Generate a trefoil knot topology."""
        nodes = []
        t_values = np.linspace(0, 2*np.pi, self.node_density)
        
        # Parametric equations for a trefoil knot
        for t in t_values:
            x = np.sin(t) + 2 * np.sin(2*t)
            y = np.cos(t) - 2 * np.cos(2*t)
            z = -np.sin(3*t)
            nodes.append(np.array([x, y, z]))
        
        # Define entangled paths (connections between nodes)
        paths = []
        for i in range(len(nodes)):
            # Connect each node to several others based on spatial proximity and braid logic
            connections = []
            for j in range(1, self.braid_depth + 1):
                next_idx = (i + j) % len(nodes)
                prev_idx = (i - j) % len(nodes)
                connections.extend([next_idx, prev_idx])
                
                # Add some cross-strand connections for more complex entanglement
                cross_idx = (i + len(nodes)//3) % len(nodes)
                connections.append(cross_idx)
            
            paths.append(connections)
        
        return np.array(nodes), paths
    
    def _generate_figure_eight_knot(self):
        """Generate a figure-eight knot topology."""
        nodes = []
        t_values = np.linspace(0, 2*np.pi, self.node_density)
        
        # Parametric equations for a figure-eight knot
        for t in t_values:
            x = (2 + np.cos(2*t)) * np.cos(3*t)
            y = (2 + np.cos(2*t)) * np.sin(3*t)
            z = np.sin(4*t)
            nodes.append(np.array([x, y, z]))
        
        # Define entangled paths similar to trefoil but with different crossings
        paths = []
        for i in range(len(nodes)):
            connections = []
            for j in range(1, self.braid_depth + 1):
                next_idx = (i + j) % len(nodes)
                prev_idx = (i - j) % len(nodes)
                connections.extend([next_idx, prev_idx])
                
                # Figure eight has more complex crossings
                cross_idx1 = (i + len(nodes)//4) % len(nodes)
                cross_idx2 = (i + len(nodes)//2) % len(nodes)
                connections.extend([cross_idx1, cross_idx2])
            
            paths.append(connections)
        
        return np.array(nodes), paths
    
    def visualize_topology(self):
        """Visualize the generated knot topology."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the knot
        nodes = np.array(self.nodes)
        ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], 'b-', lw=2)
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='r', s=50)
        
        # Plot a subset of connections to avoid visual clutter
        for i in range(0, len(self.nodes), len(self.nodes)//20):
            for j in self.paths[i][:3]:  # Only show first 3 connections per node
                ax.plot([self.nodes[i][0], self.nodes[j][0]], 
                        [self.nodes[i][1], self.nodes[j][1]], 
                        [self.nodes[i][2], self.nodes[j][2]], 'g-', alpha=0.3)
        
        ax.set_title(f"{self.knot_type.capitalize()} Knot Topology")
        plt.tight_layout()
        return fig


class EntangledConnectionLayer(nn.Module):
    """
    Implements connections with entanglement coefficients and resonance phases.
    """
    
    def __init__(self, topology, in_features, out_features):
        """
        Initialize the entangled connection layer.
        
        Args:
            topology: The TopologyGenerator instance defining the structure
            in_features: Number of input features
            out_features: Number of output features
        """
        super(EntangledConnectionLayer, self).__init__()
        
        self.topology = topology
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Entanglement coefficients (ε)
        self.entanglement_coeff = nn.Parameter(torch.Tensor(len(topology.nodes), len(topology.nodes)))
        
        # Resonance phase (ϕ)
        self.resonance_phase = nn.Parameter(torch.Tensor(len(topology.nodes), len(topology.nodes)))
        
        # Knot tension (τ) - optional dynamic variable during training
        self.knot_tension = nn.Parameter(torch.Tensor(len(topology.nodes)))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the layer parameters."""
        # Standard initialization for weights
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))
        
        # Initialize entanglement coefficients
        nn.init.uniform_(self.entanglement_coeff, 0.1, 0.5)
        
        # Initialize resonance phase (between 0 and 2π)
        nn.init.uniform_(self.resonance_phase, 0, 2 * np.pi)
        
        # Initialize knot tension
        nn.init.ones_(self.knot_tension)
    
    def entangled_connection_function(self, i, j, signal):
        """
        Compute the signal transmission between nodes i and j based on entangled structure.
        
        Args:
            i, j: Node indices
            signal: Input signal
            
        Returns:
            Modified signal after entanglement effects
        """
        # Get entanglement parameters for this connection
        epsilon = self.entanglement_coeff[i, j]
        phi = self.resonance_phase[i, j]
        tau = self.knot_tension[i] * self.knot_tension[j]
        
        # Apply entanglement effects (using complex-valued operations to model interference)
        phase_factor = torch.exp(1j * phi)
        entangled_signal = signal * (1 + epsilon * phase_factor) / (1 + tau)
        
        # Extract the real component (could also use amplitude)
        return torch.real(entangled_signal)
    
    def forward(self, x):
        """
        Forward pass through the entangled connection layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor after applying entangled connections
        """
        # Standard linear transformation
        output = F.linear(x, self.weights)
        
        # Apply entanglement effects
        batch_size = x.shape[0]
        entangled_output = torch.zeros(batch_size, self.out_features, device=x.device)
        
        # Map input and output to topology nodes
        in_per_node = self.in_features // len(self.topology.nodes)
        out_per_node = self.out_features // len(self.topology.nodes)
        
        # Apply entanglement effects between connected nodes
        for i in range(len(self.topology.nodes)):
            in_start = i * in_per_node
            in_end = min((i+1) * in_per_node, self.in_features)
            out_start = i * out_per_node
            out_end = min((i+1) * out_per_node, self.out_features)
            
            # Process connections for this node
            for j in self.topology.paths[i]:
                j_out_start = j * out_per_node
                j_out_end = min((j+1) * out_per_node, self.out_features)
                
                if out_end > out_start and j_out_end > j_out_start:
                    # Apply entanglement function to modify the signal
                    signal_ij = output[:, out_start:out_end]
                    entangled_signal = self.entangled_connection_function(i, j, signal_ij)
                    
                    # Add entangled contribution to output
                    j_width = min(j_out_end - j_out_start, entangled_signal.shape[1])
                    entangled_output[:, j_out_start:j_out_start+j_width] += entangled_signal[:, :j_width]
        
        # Combine standard output with entangled effects
        return output + 0.5 * entangled_output


class EntanglementPropagator(nn.Module):
    """
    Propagates information across entangled paths instead of layer-by-layer.
    """
    
    def __init__(self, topology, feature_dim):
        """
        Initialize the entanglement propagator.
        
        Args:
            topology: The TopologyGenerator instance
            feature_dim: Dimension of features at each node
        """
        super(EntanglementPropagator, self).__init__()
        
        self.topology = topology
        self.feature_dim = feature_dim
        
        # Propagation weights
        self.propagation_weights = nn.Parameter(
            torch.Tensor(len(topology.nodes), len(topology.nodes), feature_dim)
        )
        
        # Phase factors for wave-based propagation
        self.phase_factors = nn.Parameter(
            torch.Tensor(len(topology.nodes), len(topology.nodes))
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the propagator parameters."""
        # Initialize propagation weights
        nn.init.xavier_uniform_(self.propagation_weights)
        
        # Initialize phase factors (between 0 and 2π)
        nn.init.uniform_(self.phase_factors, 0, 2 * np.pi)
    
    def forward(self, node_features):
        """
        Forward pass through the entanglement propagator.
        
        Args:
            node_features: Features for each node [batch_size, num_nodes, feature_dim]
            
        Returns:
            Propagated features after wave-based interference
        """
        batch_size = node_features.shape[0]
        num_nodes = len(self.topology.nodes)
        device = node_features.device
        
        # Initialize output tensor
        propagated_features = torch.zeros(
            batch_size, num_nodes, self.feature_dim, device=device
        )
        
        # Wave-based propagation with interference
        for i in range(num_nodes):
            # Get connected nodes from topology
            connections = self.topology.paths[i]
            
            # Propagate signals along entangled paths
            for j in connections:
                # Apply phase factor for wave-like propagation
                phase = self.phase_factors[i, j]
                complex_phase = torch.exp(1j * phase)
                
                # Propagate signal with wave characteristics
                propagated_signal = node_features[:, i, :] * self.propagation_weights[i, j, :]
                propagated_signal = propagated_signal * complex_phase
                
                # Add to the destination node (interference happens naturally through addition)
                propagated_features[:, j, :] += torch.real(propagated_signal)
        
        # Normalize by the number of incoming connections
        connection_counts = torch.tensor(
            [len(self.topology.paths[i]) for i in range(num_nodes)],
            device=device
        ).float()
        normalization = torch.maximum(connection_counts, torch.ones_like(connection_counts))
        
        # Apply normalization across nodes
        propagated_features = propagated_features / normalization.view(1, -1, 1)
        
        return propagated_features


class CollapseResolutionLayer(nn.Module):
    """
    Interprets multi-path propagation into a singular signal for decision-making.
    """
    
    def __init__(self, topology, feature_dim, output_dim, collapse_method='entropy'):
        """
        Initialize the collapse resolution layer.
        
        Args:
            topology: The TopologyGenerator instance
            feature_dim: Dimension of features at each node
            output_dim: Dimension of the output after collapse
            collapse_method: Method to use for collapse ('entropy', 'energy', 'tension')
        """
        super(CollapseResolutionLayer, self).__init__()
        
        self.topology = topology
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.collapse_method = collapse_method
        
        # Collapse operator weights
        self.collapse_weights = nn.Parameter(
            torch.Tensor(len(topology.nodes) * feature_dim, output_dim)
        )
        
        # Energy-based collapse parameters (if using energy method)
        if collapse_method == 'energy':
            self.energy_weights = nn.Parameter(
                torch.Tensor(len(topology.nodes))
            )
        
        # Tension-based collapse parameters (if using tension method)
        elif collapse_method == 'tension':
            self.tension_weights = nn.Parameter(
                torch.Tensor(len(topology.nodes))
            )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the collapse layer parameters."""
        # Initialize collapse weights
        nn.init.xavier_uniform_(self.collapse_weights)
        
        # Initialize method-specific parameters
        if self.collapse_method == 'energy':
            nn.init.uniform_(self.energy_weights, 0.5, 1.5)
        elif self.collapse_method == 'tension':
            nn.init.uniform_(self.tension_weights, 0.5, 1.5)
    
    def forward(self, propagated_features):
        """
        Forward pass through the collapse resolution layer.
        
        Args:
            propagated_features: Features after propagation [batch_size, num_nodes, feature_dim]
            
        Returns:
            Collapsed output for decision-making
        """
        batch_size = propagated_features.shape[0]
        num_nodes = len(self.topology.nodes)
        
        # Apply collapse method to resolve superimposed states
        if self.collapse_method == 'entropy':
            # Entropy-based collapse: focus on most uncertain nodes
            node_entropy = -torch.sum(
                F.softmax(propagated_features, dim=2) * 
                F.log_softmax(propagated_features, dim=2),
                dim=2
            )
            collapse_weights = F.softmax(node_entropy, dim=1).unsqueeze(2)
            weighted_features = propagated_features * collapse_weights
            
        elif self.collapse_method == 'energy':
            # Energy-based collapse: weight by energy distribution
            node_energy = torch.sum(propagated_features**2, dim=2)
            energy_weights = F.softmax(node_energy * self.energy_weights, dim=1).unsqueeze(2)
            weighted_features = propagated_features * energy_weights
            
        elif self.collapse_method == 'tension':
            # Tension-based collapse: minimize topological strain
            tension_weights = F.softmax(self.tension_weights, dim=0).unsqueeze(0).unsqueeze(2)
            weighted_features = propagated_features * tension_weights
            
        else:
            # Default: equal weighting
            weighted_features = propagated_features / num_nodes
        
        # Flatten and project to output dimension
        collapsed_features = weighted_features.reshape(batch_size, -1)
        output = F.linear(collapsed_features, self.collapse_weights.T)  # <== Fix here

        
        return output


class ResonanceLoss(nn.Module):
    """
    Custom loss function that includes a resonance loss component.
    """
    
    def __init__(self, topology, base_criterion=nn.CrossEntropyLoss(), resonance_weight=0.1):
        """
        Initialize the resonance loss.
        
        Args:
            topology: The TopologyGenerator instance
            base_criterion: Base loss criterion (e.g., CrossEntropyLoss)
            resonance_weight: Weight for the resonance component
        """
        super(ResonanceLoss, self).__init__()
        
        self.topology = topology
        self.base_criterion = base_criterion
        self.resonance_weight = resonance_weight
    
    def forward(self, outputs, targets, entanglement_layer):
        """
        Compute the loss with resonance component.
        
        Args:
            outputs: Model outputs
            targets: Target labels
            entanglement_layer: The EntangledConnectionLayer instance
            
        Returns:
            Combined loss value
        """
        # Base loss (e.g., cross-entropy)
        base_loss = self.base_criterion(outputs, targets)
        
        # Resonance loss component: penalize disharmony in signal propagation
        resonance_loss = 0.0
        
        # Compute phase disharmony across connections
        for i in range(len(self.topology.nodes)):
            for j in self.topology.paths[i]:
                # Phase difference between connected nodes
                phase_i = entanglement_layer.resonance_phase[i, j]
                phase_j = entanglement_layer.resonance_phase[j, i]
                
                # Penalize large phase differences (disharmony)
                phase_diff = torch.abs(phase_i - phase_j) % (2 * np.pi)
                if phase_diff > np.pi:
                    phase_diff = 2 * np.pi - phase_diff
                    
                resonance_loss += phase_diff
        
        # Normalize by number of connections
        total_connections = sum(len(paths) for paths in self.topology.paths)
        resonance_loss = resonance_loss / total_connections
        
        # Combine losses
        total_loss = base_loss + self.resonance_weight * resonance_loss
        
        return total_loss


class TopologicalRegularizer:
    """
    Regularizer that encourages conservation of knot topology while optimizing function.
    """
    
    def __init__(self, topology, regularization_strength=0.01):
        """
        Initialize the topological regularizer.
        
        Args:
            topology: The TopologyGenerator instance
            regularization_strength: Strength of the regularization
        """
        self.topology = topology
        self.regularization_strength = regularization_strength
        
        # Store the original node positions for reference
        self.original_nodes = topology.nodes.copy()
    
    def compute_regularization(self, entanglement_layer):
        """
        Compute the topological regularization term.
        
        Args:
            entanglement_layer: The EntangledConnectionLayer instance
            
        Returns:
            Regularization loss term
        """
        # Compute a proxy for topological distortion using tension parameters
        tension_variation = torch.var(entanglement_layer.knot_tension)
        
        # Compute entanglement coefficient variation
        entanglement_variation = torch.var(entanglement_layer.entanglement_coeff)
        
        # Combine for total regularization
        regularization = tension_variation + entanglement_variation
        
        return self.regularization_strength * regularization


class DatasetAdapter:
    """
    Maps standard input data onto the knot structure.
    """
    
    def __init__(self, topology, input_shape):
        """
        Initialize the dataset adapter.
        
        Args:
            topology: The TopologyGenerator instance
            input_shape: Shape of the input data (e.g., [28, 28] for MNIST)
        """
        self.topology = topology
        self.input_shape = input_shape
        
        # Precompute mapping from input space to knot coordinates
        self.mapping = self._compute_mapping()
    
    def _compute_mapping(self):
        """
        Compute mapping from input space to knot coordinates.
        
        Returns:
            Mapping indices and weights
        """
        # Flatten the input shape
        input_size = np.prod(self.input_shape)
        
        # Number of nodes in the topology
        num_nodes = len(self.topology.nodes)
        
        # Create mapping (for simplicity, we'll use a modulo mapping)
        mapping = {}
        for i in range(input_size):
            node_idx = i % num_nodes
            if node_idx not in mapping:
                mapping[node_idx] = []
            mapping[node_idx].append(i)
        
        return mapping
    
    def adapt(self, input_data):
        """
        Map input data onto the knot structure.
        
        Args:
            input_data: Input data tensor [batch_size, *input_shape]
            
        Returns:
            Data mapped to knot nodes [batch_size, num_nodes, features_per_node]
        """
        batch_size = input_data.shape[0]
        num_nodes = len(self.topology.nodes)
        
        # Flatten input data
        flat_input = input_data.view(batch_size, -1)
        input_size = flat_input.shape[1]
        
        # Determine features per node
        features_per_node = max(1, input_size // num_nodes)
        
        # Initialize output tensor
        adapted_data = torch.zeros(batch_size, num_nodes, features_per_node, device=input_data.device)
        
        # Map input features to knot nodes
        for node_idx, input_indices in self.mapping.items():
            if node_idx < num_nodes:
                # Get input features mapped to this node
                node_features = []
                for i in input_indices:
                    if i < input_size:
                        node_features.append(flat_input[:, i])
                
                # Average the features if we have more than we need
                if len(node_features) > features_per_node:
                    chunks = [node_features[i:i+len(node_features)//features_per_node] 
                             for i in range(0, len(node_features), len(node_features)//features_per_node)]
                    chunks = chunks[:features_per_node]  # Ensure we don't exceed features_per_node
                    
                    for f_idx, chunk in enumerate(chunks):
                        if f_idx < features_per_node:
                            feature_value = torch.stack(chunk, dim=1).mean(dim=1)
                            adapted_data[:, node_idx, f_idx] = feature_value
                
                # Or pad with zeros if we have fewer
                elif len(node_features) > 0:
                    for f_idx, feature in enumerate(node_features):
                        if f_idx < features_per_node:
                            adapted_data[:, node_idx, f_idx] = feature
        
        return adapted_data


class EDTNN(nn.Module):
    """
    Entanglement-Driven Topological Neural Network (ED-TNN) model.
    """
    
    def __init__(self, input_shape, num_classes, knot_type='trefoil', 
                 node_density=64, features_per_node=8, collapse_method='entropy'):
        """
        Initialize the ED-TNN model.
        
        Args:
            input_shape: Shape of the input data (e.g., [28, 28] for MNIST)
            num_classes: Number of output classes
            knot_type: Type of knot topology ('trefoil', 'figure-eight')
            node_density: Number of nodes in the topology
            features_per_node: Number of features per node
            collapse_method: Method for the collapse layer
        """
        super(EDTNN, self).__init__()
        
        # Generate the topology
        self.topology = TopologyGenerator(
            knot_type=knot_type,
            node_density=node_density,
            strand_count=3,
            braid_depth=4
        )
        
        # Compute dimensions
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.num_classes = num_classes
        self.features_per_node = features_per_node
        
        # Dataset adapter
        self.dataset_adapter = DatasetAdapter(self.topology, input_shape)
        
        # Network layers
        self.input_mapping = nn.Linear(self.input_size, node_density * features_per_node)
        
        self.entangled_layer = EntangledConnectionLayer(
            self.topology, 
            node_density * features_per_node, 
            node_density * features_per_node
        )
        
        self.propagator = EntanglementPropagator(
            self.topology,
            features_per_node
        )
        
        self.collapse_layer = CollapseResolutionLayer(
            self.topology,
            features_per_node,
            num_classes,
            collapse_method=collapse_method
        )
    
    def forward(self, x):
        """
        Forward pass through the ED-TNN model.
        
        Args:
            x: Input data [batch_size, *input_shape]
            
        Returns:
            Output predictions
        """
        batch_size = x.shape[0]
        
        # Flatten input
        x_flat = x.view(batch_size, -1)
        
        # Initial linear mapping
        x = self.input_mapping(x_flat)
        
        # Apply entangled connections
        x = self.entangled_layer(x)
        
        # Reshape for propagator
        x = x.view(batch_size, len(self.topology.nodes), self.features_per_node)
        
        # Apply entanglement propagation
        x = self.propagator(x)
        
        # Apply collapse resolution
        x = self.collapse_layer(x)
        
        return x


class TrainerEngine:
    """
    Training and optimization system for ED-TNN.
    """
    
    def __init__(self, model, device, learning_rate=0.001, resonance_weight=0.1, 
                 topology_reg_strength=0.01):
        """
        Initialize the trainer engine.
        
        Args:
            model: The ED-TNN model instance
            device: Device to train on (cpu or cuda)
            learning_rate: Learning rate for optimization
            resonance_weight: Weight for resonance loss component
            topology_reg_strength: Strength of topological regularization
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        
        # Loss function with resonance component
        self.criterion = ResonanceLoss(
            model.topology,
            base_criterion=nn.CrossEntropyLoss(),
            resonance_weight=resonance_weight
        )
        
        # Topological regularizer
        self.regularizer = TopologicalRegularizer(
            model.topology,
            regularization_strength=topology_reg_strength
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Compute loss
            loss = self.criterion(output, target, self.model.entangled_layer)
            
            # Add topological regularization
            reg_loss = self.regularizer.compute_regularization(self.model.entangled_layer)
            loss += reg_loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return epoch_loss, accuracy
    
    def evaluate(self, test_loader):
        """
        Evaluate the model.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Test accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Calculate accuracy
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        
        return accuracy


class VisualizationSuite:
    """
    Visualization and monitoring tools for ED-TNN.
    """
    
    def __init__(self, model):
        """
        Initialize the visualization suite.
        
        Args:
            model: The ED-TNN model instance
        """
        self.model = model
        self.topology = model.topology

    def visualize_topology(self):
        """Visualizes the 3D topology structure and entangled paths."""
        self.topology.visualize_topology()
        plt.show()

    def visualize_node_tension(self):
        """Visualizes the knot tension values on each node as a color map."""
        node_positions = np.array(self.topology.nodes)
        tensions = self.model.entangled_layer.knot_tension.detach().cpu().numpy()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
            c=tensions, cmap='viridis', s=60
        )
        ax.set_title("Node Knot Tensions")
        fig.colorbar(scatter, ax=ax, label="Tension")
        plt.tight_layout()
        plt.show()

    def visualize_phase_interference(self):
        """Displays a heatmap of resonance phase differences between nodes."""
        phase_matrix = self.model.entangled_layer.resonance_phase.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(phase_matrix, cmap='twilight', interpolation='nearest')
        fig.colorbar(cax, label="Resonance Phase (ϕ)")
        ax.set_title("Resonance Phase Interference Matrix")
        plt.xlabel("Node j")
        plt.ylabel("Node i")
        plt.tight_layout()
        plt.show()

    def plot_loss_accuracy(self, loss_history, accuracy_history):
        """Plots loss and accuracy curves over epochs."""
        epochs = range(1, len(loss_history) + 1)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(epochs, loss_history, color='tab:red', label='Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (%)', color='tab:blue')
        ax2.plot(epochs, accuracy_history, color='tab:blue', label='Accuracy')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        fig.tight_layout()
        plt.title("Training Loss and Accuracy Over Epochs")
        plt.show()
