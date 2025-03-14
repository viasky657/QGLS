"""
Quantum Circuit Optimization Module

This module provides advanced optimization techniques for quantum circuits,
including gate synthesis, circuit depth reduction, qubit mapping, measurement
optimization, and more.
"""

import numpy as np
import time
from collections import defaultdict, deque

class QuantumCircuitOptimizations:
    """
    A class that implements various quantum circuit optimization techniques.
    
    Each optimization can be enabled or disabled individually for benchmarking purposes.
    """
    
    def __init__(self):
        """Initialize the quantum circuit optimizer."""
        # Initialize optimization statistics
        self.reset_stats()
        
        # Hardware parameters
        self.hardware_params = {
            "topology": "linear",  # Options: linear, grid, full
            "qubit_connectivity": {},  # Will be initialized based on topology
            "native_gates": ["X", "Y", "Z", "H", "CNOT", "RZ"],  # Native gates for the hardware
            "gate_fidelities": {  # Example gate fidelities
                "X": 0.998,
                "Y": 0.998,
                "Z": 0.999,
                "H": 0.997,
                "CNOT": 0.985,
                "RZ": 0.998
            },
            "measurement_fidelity": 0.96,
            "coherence_time": 100  # in arbitrary time units
        }
    
    def reset_stats(self):
        """Reset the optimization statistics."""
        self.stats = {
            "original_gate_count": 0,
            "optimized_gate_count": 0,
            "original_depth": 0,
            "optimized_depth": 0,
            "gates_removed": 0,
            "gates_replaced": 0,
            "swaps_inserted": 0,
            "measurements_optimized": 0,
            "memory_optimizations": 0,
            "execution_time": 0
        }
    
    def optimize_circuit(self, circuit, num_qubits, techniques=None):
        """
        Apply optimization techniques to a quantum circuit.
        
        Args:
            circuit: The quantum circuit to optimize, represented as a list of gate operations.
            num_qubits: The number of qubits in the circuit.
            techniques: A list of optimization techniques to apply. If None, all techniques are applied.
                Options: "gate_synthesis", "circuit_depth_reduction", "qubit_mapping",
                         "measurement_optimization", "compiler_optimization",
                         "hardware_specific", "memory_management"
        
        Returns:
            The optimized quantum circuit.
        """
        start_time = time.time()
        
        # If no specific techniques are provided, apply all of them
        if techniques is None:
            techniques = [
                "gate_synthesis",
                "circuit_depth_reduction",
                "qubit_mapping",
                "measurement_optimization",
                "compiler_optimization",
                "hardware_specific",
                "memory_management"
            ]
        
        # Record original circuit statistics
        self.stats["original_gate_count"] = len(circuit)
        self.stats["original_depth"] = self.calculate_circuit_depth(circuit, num_qubits)
        
        # Make a copy of the circuit to avoid modifying the original
        optimized_circuit = circuit.copy()
        
        # Apply each optimization technique in sequence
        for technique in techniques:
            if technique == "gate_synthesis":
                optimized_circuit = self._optimize_gate_synthesis(optimized_circuit)
            elif technique == "circuit_depth_reduction":
                optimized_circuit = self._optimize_circuit_depth(optimized_circuit, num_qubits)
            elif technique == "qubit_mapping":
                self._initialize_qubit_connectivity(num_qubits)
                optimized_circuit = self._optimize_qubit_mapping(optimized_circuit, num_qubits)
            elif technique == "measurement_optimization":
                optimized_circuit = self._optimize_measurements(optimized_circuit, num_qubits)
            elif technique == "compiler_optimization":
                optimized_circuit = self._apply_compiler_optimizations(optimized_circuit)
            elif technique == "hardware_specific":
                optimized_circuit = self._apply_hardware_specific_optimizations(optimized_circuit)
            elif technique == "memory_management":
                optimized_circuit = self._optimize_memory_management(optimized_circuit, num_qubits)
        
        # Record optimized circuit statistics
        self.stats["optimized_gate_count"] = len(optimized_circuit)
        self.stats["optimized_depth"] = self.calculate_circuit_depth(optimized_circuit, num_qubits)
        self.stats["execution_time"] = time.time() - start_time
        
        return optimized_circuit
    
    def _optimize_gate_synthesis(self, circuit):
        """
        Optimize the circuit through gate synthesis and decomposition.
        
        This includes:
        - Cancelling consecutive identical gates that cancel out
        - Replacing common gate sequences with more efficient equivalents
        - Combining rotation gates
        
        Args:
            circuit: The quantum circuit to optimize.
            
        Returns:
            The optimized quantum circuit.
        """
        optimized_circuit = []
        i = 0
        gates_removed = 0
        gates_replaced = 0
        
        while i < len(circuit):
            # Check for cancellable gates (e.g., two consecutive X gates)
            if i + 1 < len(circuit) and self._are_cancellable_gates(circuit[i], circuit[i+1]):
                i += 2
                gates_removed += 2
                continue
            
            # Check for H-Z-H pattern (equivalent to X gate)
            if i + 2 < len(circuit) and self._is_h_z_h_pattern(circuit[i:i+3]):
                # Replace with X gate
                target_qubit = circuit[i][1]
                optimized_circuit.append(("X", target_qubit))
                i += 3
                gates_replaced += 2  # Replaced 3 gates with 1
                continue
            
            # Check for consecutive rotation gates that can be combined
            if i + 1 < len(circuit) and self._are_combinable_rotations(circuit[i], circuit[i+1]):
                combined_gate = self._combine_rotations(circuit[i], circuit[i+1])
                optimized_circuit.append(combined_gate)
                i += 2
                gates_replaced += 1  # Replaced 2 gates with 1
                continue
            
            # Check for CNOT-CNOT-CNOT pattern (equivalent to SWAP)
            if i + 2 < len(circuit) and self._is_cnot_swap_pattern(circuit[i:i+3]):
                control, target = circuit[i][1], circuit[i][2]
                optimized_circuit.append(("SWAP", control, target))
                i += 3
                gates_replaced += 2  # Replaced 3 gates with 1
                continue
            
            # No optimization applicable, keep the gate
            optimized_circuit.append(circuit[i])
            i += 1
        
        self.stats["gates_removed"] += gates_removed
        self.stats["gates_replaced"] += gates_replaced
        
        return optimized_circuit
    
    def _are_cancellable_gates(self, gate1, gate2):
        """Check if two gates cancel each other out."""
        # Single-qubit gates that are self-inverse (X, Y, Z, H)
        self_inverse_gates = ["X", "Y", "Z", "H"]
        
        # Check if both gates are the same self-inverse gate on the same qubit
        if (len(gate1) == 2 and len(gate2) == 2 and
            gate1[0] in self_inverse_gates and gate1[0] == gate2[0] and
            gate1[1] == gate2[1]):
            return True
        
        return False
    
    def _is_h_z_h_pattern(self, gates):
        """Check if three gates form an H-Z-H pattern on the same qubit."""
        if (len(gates) == 3 and
            len(gates[0]) == 2 and len(gates[1]) == 2 and len(gates[2]) == 2 and
            gates[0][0] == "H" and gates[1][0] == "Z" and gates[2][0] == "H" and
            gates[0][1] == gates[1][1] == gates[2][1]):
            return True
        return False
    
    def _are_combinable_rotations(self, gate1, gate2):
        """Check if two rotation gates can be combined."""
        rotation_gates = ["RX", "RY", "RZ"]
        
        # Check if both gates are the same rotation type on the same qubit
        if (len(gate1) == 3 and len(gate2) == 3 and
            gate1[0] in rotation_gates and gate1[0] == gate2[0] and
            gate1[1] == gate2[1]):
            return True
        
        return False
    
    def _combine_rotations(self, gate1, gate2):
        """Combine two rotation gates into a single rotation."""
        gate_type = gate1[0]
        qubit = gate1[1]
        angle1 = gate1[2]
        angle2 = gate2[2]
        
        # Add the rotation angles
        combined_angle = angle1 + angle2
        
        # Normalize the angle to [-π, π]
        combined_angle = (combined_angle + np.pi) % (2 * np.pi) - np.pi
        
        return (gate_type, qubit, combined_angle)
    
    def _is_cnot_swap_pattern(self, gates):
        """Check if three CNOT gates form a SWAP pattern."""
        if (len(gates) == 3 and
            len(gates[0]) == 3 and len(gates[1]) == 3 and len(gates[2]) == 3 and
            gates[0][0] == "CNOT" and gates[1][0] == "CNOT" and gates[2][0] == "CNOT" and
            gates[0][1] == gates[2][1] and gates[0][2] == gates[2][2] and
            gates[1][1] == gates[0][2] and gates[1][2] == gates[0][1]):
            return True
        return False
    
    def _optimize_circuit_depth(self, circuit, num_qubits):
        """
        Optimize the circuit depth by rearranging gates to maximize parallelism.
        
        Args:
            circuit: The quantum circuit to optimize.
            num_qubits: The number of qubits in the circuit.
            
        Returns:
            The depth-optimized quantum circuit.
        """
        # Convert the circuit to a dependency graph
        dependency_graph = self._build_dependency_graph(circuit, num_qubits)
        
        # Topologically sort the graph to get a schedule that respects dependencies
        schedule = self._topological_sort(dependency_graph)
        
        # Group operations into layers that can be executed in parallel
        layers = self._group_operations_into_layers(schedule, circuit, num_qubits)
        
        # Reconstruct the circuit from the layers
        optimized_circuit = []
        for layer in layers:
            for op_idx in layer:
                optimized_circuit.append(circuit[op_idx])
        
        return optimized_circuit
    
    def _build_dependency_graph(self, circuit, num_qubits):
        """Build a dependency graph for the circuit operations."""
        # Initialize the graph: node -> list of nodes that depend on it
        graph = {i: [] for i in range(len(circuit))}
        
        # Track the last operation on each qubit
        last_op_on_qubit = [-1] * num_qubits
        
        for i, op in enumerate(circuit):
            # Get qubits involved in this operation
            qubits = self._get_qubits_in_operation(op)
            
            # Add dependencies from the last operations on these qubits
            for q in qubits:
                if last_op_on_qubit[q] != -1:
                    graph[last_op_on_qubit[q]].append(i)
            
            # Update the last operation for these qubits
            for q in qubits:
                last_op_on_qubit[q] = i
        
        return graph
    
    def _get_qubits_in_operation(self, op):
        """Get the qubits involved in an operation."""
        if len(op) == 2:  # Single-qubit gate or measurement
            return [int(op[1])]
        elif len(op) == 3:  # Two-qubit gate
            return [int(op[1]), int(op[2])]
        elif len(op) == 4:  # Three-qubit gate or controlled rotation
            return [int(op[1]), int(op[2]), int(op[3])]
        return []
    
    def _topological_sort(self, graph):
        """Perform a topological sort of the dependency graph."""
        # Count incoming edges for each node
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
        
        # Start with nodes that have no dependencies
        queue = deque([node for node in in_degree if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # Remove this node's outgoing edges
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _group_operations_into_layers(self, schedule, circuit, num_qubits):
        """Group operations into layers that can be executed in parallel."""
        layers = []
        qubit_busy_until = [-1] * num_qubits  # Layer index until which each qubit is busy
        
        for op_idx in schedule:
            op = circuit[op_idx]
            qubits = self._get_qubits_in_operation(op)
            
            # Find the earliest layer where all qubits are available
            earliest_layer = max([qubit_busy_until[q] for q in qubits]) + 1
            
            # Ensure we have enough layers
            while len(layers) <= earliest_layer:
                layers.append([])
            
            # Add the operation to the layer
            layers[earliest_layer].append(op_idx)
            
            # Mark the qubits as busy for this layer
            for q in qubits:
                qubit_busy_until[q] = earliest_layer
        
        return layers
    
    def _initialize_qubit_connectivity(self, num_qubits):
        """Initialize the qubit connectivity based on the hardware topology."""
        connectivity = {}
        
        if self.hardware_params["topology"] == "linear":
            # Linear topology: each qubit is connected to its neighbors
            for i in range(num_qubits):
                connectivity[i] = []
                if i > 0:
                    connectivity[i].append(i - 1)
                if i < num_qubits - 1:
                    connectivity[i].append(i + 1)
        
        elif self.hardware_params["topology"] == "grid":
            # Grid topology: assume a square grid
            grid_size = int(np.ceil(np.sqrt(num_qubits)))
            for i in range(num_qubits):
                connectivity[i] = []
                row, col = i // grid_size, i % grid_size
                
                # Connect to neighbors in the grid
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                        neighbor = new_row * grid_size + new_col
                        if neighbor < num_qubits:
                            connectivity[i].append(neighbor)
        
        elif self.hardware_params["topology"] == "full":
            # Fully connected topology
            for i in range(num_qubits):
                connectivity[i] = [j for j in range(num_qubits) if j != i]
        
        self.hardware_params["qubit_connectivity"] = connectivity
    
    def _optimize_qubit_mapping(self, circuit, num_qubits):
        """
        Optimize the mapping of logical qubits to physical qubits.
        
        Args:
            circuit: The quantum circuit to optimize.
            num_qubits: The number of qubits in the circuit.
            
        Returns:
            The optimized quantum circuit with SWAP operations inserted as needed.
        """
        # Ensure connectivity is initialized
        if not self.hardware_params["qubit_connectivity"]:
            self._initialize_qubit_connectivity(num_qubits)
        
        # Initialize the logical to physical qubit mapping
        # Start with identity mapping
        logical_to_physical = {i: i for i in range(num_qubits)}
        physical_to_logical = {i: i for i in range(num_qubits)}
        
        optimized_circuit = []
        swaps_inserted = 0
        
        for op in circuit:
            if len(op) == 3 and op[0] in ["CNOT", "CZ", "SWAP"]:  # Two-qubit gate
                control, target = op[1], op[2]
                p_control, p_target = logical_to_physical[control], logical_to_physical[target]
                
                # Check if qubits are adjacent in the hardware
                if p_target not in self.hardware_params["qubit_connectivity"][p_control]:
                    # Find the shortest path between the qubits
                    path = self._find_shortest_path(p_control, p_target)
                    
                    if path:
                        # Insert SWAP gates to move qubits closer
                        for i in range(len(path) - 1):
                            q1, q2 = path[i], path[i + 1]
                            
                            # Add SWAP gate to the circuit
                            optimized_circuit.append(("SWAP", physical_to_logical[q1], physical_to_logical[q2]))
                            swaps_inserted += 1
                            
                            # Update the mappings
                            l1, l2 = physical_to_logical[q1], physical_to_logical[q2]
                            logical_to_physical[l1], logical_to_physical[l2] = q2, q1
                            physical_to_logical[q1], physical_to_logical[q2] = l2, l1
                            
                            # Update control and target if they were affected
                            p_control = logical_to_physical[control]
                            p_target = logical_to_physical[target]
                
                # Add the original gate with updated qubit indices
                optimized_circuit.append((op[0], physical_to_logical[p_control], physical_to_logical[p_target]))
            else:
                # Single-qubit gate or measurement, no mapping needed
                optimized_circuit.append(op)
        
        self.stats["swaps_inserted"] += swaps_inserted
        return optimized_circuit
    
    def _find_shortest_path(self, start, end):
        """Find the shortest path between two qubits in the hardware topology."""
        # Breadth-first search
        visited = set([start])
        queue = deque([(start, [start])])
        
        while queue:
            node, path = queue.popleft()
            
            if node == end:
                return path
            
            for neighbor in self.hardware_params["qubit_connectivity"][node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # No path found
    
    def _optimize_measurements(self, circuit, num_qubits):
        """
        Optimize measurements in the circuit.
        
        This includes:
        - Deferring measurements to the end of the circuit when possible
        - Removing unnecessary measurements
        
        Args:
            circuit: The quantum circuit to optimize.
            num_qubits: The number of qubits in the circuit.
            
        Returns:
            The optimized quantum circuit.
        """
        # Track which qubits are measured and when
        measurements = {}  # qubit -> (index in circuit, is_used_after)
        
        # First pass: identify measurements and their usage
        for i, op in enumerate(circuit):
            if len(op) == 2 and op[0] == "M":  # Measurement
                qubit = op[1]
                measurements[qubit] = (i, False)
            elif len(op) >= 3 and op[0].startswith("COND_"):  # Conditional operation
                # Mark the controlling qubit as used after measurement
                control_qubit = op[1]
                if control_qubit in measurements:
                    measurements[control_qubit] = (measurements[control_qubit][0], True)
        
        # Second pass: optimize measurements
        optimized_circuit = []
        deferred_measurements = []
        measurements_optimized = 0
        
        for i, op in enumerate(circuit):
            if len(op) == 2 and op[0] == "M":  # Measurement
                qubit = op[1]
                if qubit in measurements and not measurements[qubit][1]:
                    # This measurement can be deferred
                    deferred_measurements.append(op)
                    measurements_optimized += 1
                else:
                    # This measurement affects later operations, keep it
                    optimized_circuit.append(op)
            else:
                optimized_circuit.append(op)
        
        # Add deferred measurements at the end
        optimized_circuit.extend(deferred_measurements)
        
        self.stats["measurements_optimized"] += measurements_optimized
        return optimized_circuit
    
    def _apply_compiler_optimizations(self, circuit):
        """
        Apply advanced compiler optimizations to the circuit.
        
        This includes:
        - Peephole optimizations for common patterns
        - Constant folding for known inputs
        
        Args:
            circuit: The quantum circuit to optimize.
            
        Returns:
            The optimized quantum circuit.
        """
        # Implement peephole optimizations
        optimized_circuit = self._apply_peephole_optimizations(circuit)
        
        # Apply constant folding
        optimized_circuit = self._apply_constant_folding(optimized_circuit)
        
        return optimized_circuit
    
    def _apply_peephole_optimizations(self, circuit):
        """Apply peephole optimizations to the circuit."""
        optimized_circuit = []
        i = 0
        
        while i < len(circuit):
            # Pattern: H-H cancellation
            if (i + 1 < len(circuit) and
                len(circuit[i]) == 2 and len(circuit[i+1]) == 2 and
                circuit[i][0] == "H" and circuit[i+1][0] == "H" and
                circuit[i][1] == circuit[i+1][1]):
                i += 2
                self.stats["gates_removed"] += 2
                continue
            
            # Pattern: X-X cancellation
            if (i + 1 < len(circuit) and
                len(circuit[i]) == 2 and len(circuit[i+1]) == 2 and
                circuit[i][0] == "X" and circuit[i+1][0] == "X" and
                circuit[i][1] == circuit[i+1][1]):
                i += 2
                self.stats["gates_removed"] += 2
                continue
            
            # Pattern: Z-Z cancellation
            if (i + 1 < len(circuit) and
                len(circuit[i]) == 2 and len(circuit[i+1]) == 2 and
                circuit[i][0] == "Z" and circuit[i+1][0] == "Z" and
                circuit[i][1] == circuit[i+1][1]):
                i += 2
                self.stats["gates_removed"] += 2
                continue
            
            # Pattern: CNOT-CNOT cancellation
            if (i + 1 < len(circuit) and
                len(circuit[i]) == 3 and len(circuit[i+1]) == 3 and
                circuit[i][0] == "CNOT" and circuit[i+1][0] == "CNOT" and
                circuit[i][1] == circuit[i+1][1] and circuit[i][2] == circuit[i+1][2]):
                i += 2
                self.stats["gates_removed"] += 2
                continue
            
            # No optimization applicable, keep the gate
            optimized_circuit.append(circuit[i])
            i += 1
        
        return optimized_circuit
    
    def _apply_constant_folding(self, circuit):
        """Apply constant folding to the circuit."""
        # This is a simplified implementation
        # In a real system, this would analyze the circuit for constant inputs
        # and pre-compute parts of the circuit that depend only on constants
        return circuit
    
    def _apply_hardware_specific_optimizations(self, circuit):
        """
        Apply hardware-specific optimizations to the circuit.
        
        This includes:
        - Adapting to the native gate set
        - Optimizing for hardware-specific constraints
        
        Args:
            circuit: The quantum circuit to optimize.
            
        Returns:
            The optimized quantum circuit.
        """
        # Convert to native gates
        optimized_circuit = self._convert_to_native_gates(circuit)
        
        # Optimize for hardware-specific constraints
        optimized_circuit = self._optimize_for_hardware_constraints(optimized_circuit)
        
        return optimized_circuit
    
    def _convert_to_native_gates(self, circuit):
        """Convert the circuit to use only native gates."""
        native_gates = self.hardware_params["native_gates"]
        optimized_circuit = []
        
        for op in circuit:
            if len(op) >= 2 and op[0] in native_gates:
                # Gate is already native, keep it
                optimized_circuit.append(op)
            else:
                # Decompose non-native gates
                decomposed = self._decompose_to_native_gates(op, native_gates)
                optimized_circuit.extend(decomposed)
        
        return optimized_circuit
    
    def _decompose_to_native_gates(self, op, native_gates):
        """Decompose a non-native gate into native gates."""
        # This is a simplified implementation
        # In a real system, this would have decomposition rules for various gates
        
        if op[0] == "S" and "Z" in native_gates:
            # S gate is equivalent to sqrt(Z)
            return [("RZ", op[1], np.pi/2)]
        
        if op[0] == "T" and "Z" in native_gates:
            # T gate is equivalent to Z^(1/4)
            return [("RZ", op[1], np.pi/4)]
        
        if op[0] == "Y" and "X" in native_gates and "Z" in native_gates:
            # Y = Z-X-Z decomposition
            return [("Z", op[1]), ("X", op[1]), ("Z", op[1])]
        
        # Default: keep the original gate
        return [op]
    
    def _optimize_for_hardware_constraints(self, circuit):
        """Optimize the circuit for hardware-specific constraints."""
        # This is a simplified implementation
        # In a real system, this would adapt to various hardware constraints
        return circuit
    
    def _optimize_memory_management(self, circuit, num_qubits):
        """
        Optimize quantum memory management in the circuit.
        
        This includes:
        - Reusing qubits when possible
        - Optimizing qubit allocation and deallocation
        
        Args:
            circuit: The quantum circuit to optimize.
            num_qubits: The number of qubits in the circuit.
            
        Returns:
            The optimized quantum circuit.
        """
        # Analyze qubit lifetimes
        qubit_first_use = {i: -1 for i in range(num_qubits)}
        qubit_last_use = {i: -1 for i in range(num_qubits)}
        
        for i, op in enumerate(circuit):
            qubits = self._get_qubits_in_operation(op)
            for q in qubits:
                if qubit_first_use[q] == -1:
                    qubit_first_use[q] = i
                qubit_last_use[q] = i
        
        # Identify qubits that can be reused
        available_qubits = []
        logical_to_physical = {i: i for i in range(num_qubits)}
        physical_to_logical = {i: i for i in range(num_qubits)}
        memory_optimizations = 0
        
        optimized_circuit = []
        
        for i, op in enumerate(circuit):
            # Check if any qubits are no longer needed
            for q in range(num_qubits):
                if qubit_last_use[q] == i - 1:
                    available_qubits.append(logical_to_physical[q])
            
            # Check if any new qubits are needed
            qubits = self._get_qubits_in_operation(op)
            for q in qubits:
                if qubit_first_use[q] == i and available_qubits:
                    # Reuse an available qubit
                    physical_q = available_qubits.pop(0)
                    old_logical = physical_to_logical[physical_q]
                    
                    # Update mappings
                    logical_to_physical[q] = physical_q
                    logical_to_physical[old_logical] = -1  # Mark as unmapped
                    physical_to_logical[physical_q] = q
                    
                    memory_optimizations += 1
            
            # Add the operation with mapped qubits
            if len(op) == 2:  # Single-qubit gate or measurement
                mapped_op = (op[0], logical_to_physical[op[1]])
                optimized_circuit.append(mapped_op)
            elif len(op) == 3:  # Two-qubit gate
                mapped_op = (op[0], logical_to_physical[op[1]], logical_to_physical[op[2]])
                optimized_circuit.append(mapped_op)
            else:
                optimized_circuit.append(op)
        
        self.stats["memory_optimizations"] += memory_optimizations
        return optimized_circuit
    
    def calculate_circuit_depth(self, circuit, num_qubits):
        """
        Calculate the depth of a quantum circuit.
        
        Args:
            circuit: The quantum circuit.
            num_qubits: The number of qubits in the circuit.
            
        Returns:
            The depth of the circuit.
        """
        # Track the layer index for each qubit
        qubit_layer = [0] * num_qubits
        
        for op in circuit:
            qubits = self._get_qubits_in_operation(op)
            
            # Find the maximum layer among the qubits involved
            max_layer = max([qubit_layer[q] for q in qubits])
            
            # Place the operation in the next layer
            for q in qubits:
                qubit_layer[q] = max_layer + 1
        
        # The circuit depth is the maximum layer across all qubits
        return max(qubit_layer)
    
    def print_optimization_stats(self):
        """Print the optimization statistics."""
        print(f"Original gate count: {self.stats['original_gate_count']}")
        print(f"Optimized gate count: {self.stats['optimized_gate_count']}")
        print(f"Gate reduction: {1 - self.stats['optimized_gate_count']/self.stats['original_gate_count']:.2%}")
        
        print(f"Original circuit depth: {self.stats['original_depth']}")
        print(f"Optimized circuit depth: {self.stats['optimized_depth']}")
        print(f"Depth reduction: {1 - self.stats['optimized_depth']/self.stats['original_depth']:.2%}")
        
        print(f"Gates removed: {self.stats['gates_removed']}")
        print(f"Gates replaced: {self.stats['gates_replaced']}")
        print(f"SWAP gates inserted: {self.stats['swaps_inserted']}")
        print(f"Measurements optimized: {self.stats['measurements_optimized']}")
        print(f"Memory optimizations: {self.stats['memory_optimizations']}")
        print(f"Execution time: {self.stats['execution_time']:.4f} seconds")