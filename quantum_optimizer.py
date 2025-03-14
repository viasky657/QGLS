import numpy as np
import time
from collections import defaultdict
import copy

class QuantumCircuitOptimizer:
    """
    A quantum circuit optimizer that implements various optimization techniques
    that can be individually enabled or disabled for benchmarking purposes.
    """
    
    def __init__(self, enable_all=False):
        """Initialize the quantum circuit optimizer."""
        # Configuration flags for each optimization technique
        self.config = {
            "gate_synthesis": enable_all,
            "circuit_depth_reduction": enable_all,
            "qubit_mapping": enable_all,
            "measurement_optimization": enable_all,
            "compiler_techniques": enable_all,
            "hardware_specific": enable_all,
            "memory_management": enable_all
        }
        
        # Default hardware configuration
        self.hardware_config = {
            "connectivity": "all-to-all",
            "native_gates": ["X", "Y", "Z", "H", "CNOT"],
            "gate_fidelities": {
                "X": 0.998, "Y": 0.998, "Z": 0.999, "H": 0.997, "CNOT": 0.985
            }
        }
        
        # Statistics for benchmarking
        self.reset_stats()
    
    def enable_optimization(self, technique, enable=True):
        """Enable or disable a specific optimization technique."""
        if technique in self.config:
            self.config[technique] = enable
        elif technique == "all":
            for key in self.config:
                self.config[key] = enable
        else:
            raise ValueError(f"Unknown optimization technique: {technique}")
    
    def is_enabled(self, technique):
        """Check if a specific optimization technique is enabled."""
        return self.config.get(technique, False)
    
    def set_hardware_config(self, config):
        """Set the hardware configuration for hardware-specific optimizations."""
        self.hardware_config.update(config)
    
    def optimize_circuit(self, circuit, num_qubits):
        """Apply all enabled optimization techniques to the circuit."""
        start_time = time.time()
        
        # Make a copy of the original circuit
        optimized_circuit = copy.deepcopy(circuit)
        
        # Calculate original statistics
        self.stats["original_gate_count"] = len(circuit)
        self.stats["original_depth"] = self._calculate_circuit_depth(circuit, num_qubits)
        
        # Apply enabled optimization techniques
        if self.is_enabled("gate_synthesis"):
            technique_start = time.time()
            optimized_circuit = self._optimize_gate_sequences(optimized_circuit)
            self.stats["technique_stats"]["gate_synthesis"]["time"] += time.time() - technique_start
            self.stats["technique_stats"]["gate_synthesis"]["count"] += 1
        
        if self.is_enabled("circuit_depth_reduction"):
            technique_start = time.time()
            optimized_circuit = self._reduce_circuit_depth(optimized_circuit, num_qubits)
            self.stats["technique_stats"]["circuit_depth_reduction"]["time"] += time.time() - technique_start
            self.stats["technique_stats"]["circuit_depth_reduction"]["count"] += 1
        
        if self.is_enabled("qubit_mapping"):
            technique_start = time.time()
            optimized_circuit = self._optimize_qubit_mapping(optimized_circuit, num_qubits)
            self.stats["technique_stats"]["qubit_mapping"]["time"] += time.time() - technique_start
            self.stats["technique_stats"]["qubit_mapping"]["count"] += 1
        
        if self.is_enabled("measurement_optimization"):
            technique_start = time.time()
            optimized_circuit = self._optimize_measurements(optimized_circuit, num_qubits)
            self.stats["technique_stats"]["measurement_optimization"]["time"] += time.time() - technique_start
            self.stats["technique_stats"]["measurement_optimization"]["count"] += 1
        
        if self.is_enabled("compiler_techniques"):
            technique_start = time.time()
            optimized_circuit = self._apply_compiler_optimizations(optimized_circuit, num_qubits)
            self.stats["technique_stats"]["compiler_techniques"]["time"] += time.time() - technique_start
            self.stats["technique_stats"]["compiler_techniques"]["count"] += 1
        
        if self.is_enabled("hardware_specific"):
            technique_start = time.time()
            optimized_circuit = self._apply_hardware_specific_optimizations(optimized_circuit, num_qubits)
            self.stats["technique_stats"]["hardware_specific"]["time"] += time.time() - technique_start
            self.stats["technique_stats"]["hardware_specific"]["count"] += 1
        
        if self.is_enabled("memory_management"):
            technique_start = time.time()
            optimized_circuit = self._optimize_qubit_allocation(optimized_circuit, num_qubits)
            self.stats["technique_stats"]["memory_management"]["time"] += time.time() - technique_start
            self.stats["technique_stats"]["memory_management"]["count"] += 1
        
        # Calculate optimized statistics
        self.stats["optimized_gate_count"] = len(optimized_circuit)
        self.stats["optimized_depth"] = self._calculate_circuit_depth(optimized_circuit, num_qubits)
        self.stats["optimization_time"] = time.time() - start_time
        
        return optimized_circuit
    
    def _calculate_circuit_depth(self, circuit, num_qubits):
        """Calculate the depth of a quantum circuit."""
        qubit_times = [0] * num_qubits
        
        for op in circuit:
            if len(op) == 2:  # Single-qubit gate
                gate, target = op
                qubit_times[target] += 1
            elif len(op) == 3:  # Two-qubit gate
                gate, control, target = op
                latest_time = max(qubit_times[control], qubit_times[target])
                qubit_times[control] = latest_time + 1
                qubit_times[target] = latest_time + 1
        
        return max(qubit_times) if qubit_times else 0
    
    # 1. Gate Synthesis and Decomposition
    def _optimize_gate_sequences(self, circuit):
        """Optimize sequences of gates using advanced synthesis techniques."""
        optimized_circuit = []
        i = 0
        
        while i < len(circuit):
            # Check for self-cancelling gates (e.g., X-X, Y-Y, Z-Z)
            if i + 1 < len(circuit) and self._is_self_cancelling(circuit[i], circuit[i+1]):
                i += 2  # Skip both gates
                continue
            
            # Add the current gate to the optimized circuit
            optimized_circuit.append(circuit[i])
            i += 1
        
        return optimized_circuit
    
    def _is_self_cancelling(self, op1, op2):
        """Check if two operations cancel each other."""
        # Both must be single-qubit gates on the same qubit
        if len(op1) != 2 or len(op2) != 2:
            return False
        
        gate1, qubit1 = op1
        gate2, qubit2 = op2
        
        # Must be on the same qubit
        if qubit1 != qubit2:
            return False
        
        # Check if gates are self-inverse (like X, Y, Z) and identical
        # In a real implementation, we would check if gate1 @ gate2 = Identity
        return np.array_equal(gate1, gate2)
    
    # 2. Circuit Depth Reduction
    def _reduce_circuit_depth(self, circuit, num_qubits):
        """Reduce the depth of a quantum circuit by rearranging gates."""
        # Build dependency graph
        dependencies = self._build_dependency_graph(circuit, num_qubits)
        
        # Schedule gates into layers
        layers = self._schedule_gates_in_layers(circuit, dependencies)
        
        # Flatten layers back into a circuit
        optimized_circuit = []
        for layer in layers:
            optimized_circuit.extend(layer)
        
        return optimized_circuit
    
    def _build_dependency_graph(self, circuit, num_qubits):
        """Build a dependency graph for the circuit."""
        last_op_on_qubit = [-1] * num_qubits
        dependencies = {i: [] for i in range(len(circuit))}
        
        for i, op in enumerate(circuit):
            qubits = [op[1]] if len(op) == 2 else [op[1], op[2]]
            
            # Add dependencies from the last operations on these qubits
            for q in qubits:
                if last_op_on_qubit[q] >= 0:
                    dependencies[i].append(last_op_on_qubit[q])
            
            # Update last operation for these qubits
            for q in qubits:
                last_op_on_qubit[q] = i
        
        return dependencies
    
    def _schedule_gates_in_layers(self, circuit, dependency_graph):
        """Schedule gates into layers using topological sort."""
        in_degree = {i: len(deps) for i, deps in dependency_graph.items()}
        queue = [i for i, count in in_degree.items() if count == 0]
        layers = []
        visited = set()
        
        while queue:
            current_layer = []
            next_queue = []
            
            for node in queue:
                if node not in visited:
                    current_layer.append(circuit[node])
                    visited.add(node)
                    
                    # Decrease in-degree of neighbors
                    for i in range(len(circuit)):
                        if node in dependency_graph.get(i, []):
                            in_degree[i] -= 1
                            if in_degree[i] == 0:
                                next_queue.append(i)
            
            if current_layer:
                layers.append(current_layer)
            queue = next_queue
        
        return layers
    
    # 3. Qubit Mapping and Routing
    def _optimize_qubit_mapping(self, circuit, num_qubits):
        """Optimize the mapping of logical qubits to physical qubits."""
        # If hardware has all-to-all connectivity, no mapping needed
        if self.hardware_config["connectivity"] == "all-to-all":
            return circuit
        
        # Get connectivity graph
        connectivity = self._get_connectivity_graph(num_qubits)
        
        # Find initial mapping
        mapping = {i: i for i in range(num_qubits)}  # Identity mapping
        
        # Apply mapping and insert SWAP gates as needed
        mapped_circuit = []
        
        for op in circuit:
            if len(op) == 2:  # Single-qubit gate
                gate, logical_qubit = op
                physical_qubit = mapping[logical_qubit]
                mapped_circuit.append((gate, physical_qubit))
            elif len(op) == 3:  # Two-qubit gate
                gate, logical_control, logical_target = op
                physical_control = mapping[logical_control]
                physical_target = mapping[logical_target]
                
                # Check if qubits are connected
                if physical_target in connectivity[physical_control]:
                    mapped_circuit.append((gate, physical_control, physical_target))
                else:
                    # Insert SWAP gates (simplified)
                    mapped_circuit.append(("SWAP", physical_control, physical_target))
                    
                    # Update mapping after SWAP
                    mapping[logical_control], mapping[logical_target] = mapping[logical_target], mapping[logical_control]
                    
                    # Add the original gate with updated mapping
                    mapped_circuit.append((gate, mapping[logical_control], mapping[logical_target]))
                    
                    self.stats["swap_count"] += 1
        
        return mapped_circuit
    
    def _get_connectivity_graph(self, num_qubits):
        """Get the connectivity graph based on hardware configuration."""
        connectivity = self.hardware_config["connectivity"]
        graph = {i: [] for i in range(num_qubits)}
        
        if connectivity == "all-to-all":
            for i in range(num_qubits):
                graph[i] = [j for j in range(num_qubits) if j != i]
        elif connectivity == "linear":
            for i in range(num_qubits):
                if i > 0:
                    graph[i].append(i-1)
                if i < num_qubits - 1:
                    graph[i].append(i+1)
        elif connectivity == "grid":
            size = int(np.ceil(np.sqrt(num_qubits)))
            for i in range(num_qubits):
                row, col = i // size, i % size
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < size and 0 <= new_col < size:
                        neighbor = new_row * size + new_col
                        if neighbor < num_qubits:
                            graph[i].append(neighbor)
        
        return graph
    
    # 4. Measurement-Based Optimizations
    def _optimize_measurements(self, circuit, num_qubits):
        """Optimize circuits based on measurement results."""
        # Identify measurement operations
        measurements = []
        for i, op in enumerate(circuit):
            if len(op) == 2 and op[0] == 'M':
                measurements.append((i, op[1]))  # (index, qubit)
        
        if not measurements:
            return circuit
        
        # Check which measurements can be deferred
        optimized_circuit = []
        skip_indices = set()
        deferred_measurements = []
        
        for i, op in enumerate(circuit):
            if i in skip_indices:
                continue
                
            if len(op) == 2 and op[0] == 'M':
                qubit = op[1]
                
                # Check if this qubit is used after measurement
                used_after = False
                for j in range(i+1, len(circuit)):
                    next_op = circuit[j]
                    if (len(next_op) == 2 and next_op[1] == qubit) or \
                       (len(next_op) == 3 and (next_op[1] == qubit or next_op[2] == qubit)):
                        used_after = True
                        break
                
                if not used_after:
                    # Defer this measurement
                    skip_indices.add(i)
                    deferred_measurements.append(qubit)
                    continue
            
            optimized_circuit.append(op)
        
        # Add deferred measurements at the end
        for qubit in deferred_measurements:
            optimized_circuit.append(('M', qubit))
        
        return optimized_circuit
    
    # 5. Advanced Compiler Techniques
    def _apply_compiler_optimizations(self, circuit, num_qubits):
        """Apply advanced compiler optimization techniques."""
        # Simplified implementation - just return the original circuit
        return circuit
    
    # 6. Hardware-Specific Optimizations
    def _apply_hardware_specific_optimizations(self, circuit, num_qubits):
        """Apply optimizations tailored to specific quantum hardware."""
        # Convert to native gate set (simplified)
        native_gates = self.hardware_config["native_gates"]
        native_circuit = []
        
        for op in circuit:
            # In a real implementation, we would decompose non-native gates
            native_circuit.append(op)
        
        return native_circuit
    
    # 7. Quantum Memory Management
    def _optimize_qubit_allocation(self, circuit, num_qubits):
        """Optimize the allocation and deallocation of qubits."""
        # Analyze qubit lifetimes
        qubit_lifetimes = self._analyze_qubit_lifetimes(circuit, num_qubits)
        
        # Create an optimized allocation plan (simplified)
        allocation_plan = {i: i for i in range(num_qubits)}  # Identity mapping
        
        # Rewrite the circuit with the optimized allocation
        optimized_circuit = []
        
        for op in circuit:
            if len(op) == 2:  # Single-qubit gate
                gate, target = op
                physical_target = allocation_plan[target]
                optimized_circuit.append((gate, physical_target))
            elif len(op) == 3:  # Two-qubit gate
                gate, control, target = op
                physical_control = allocation_plan[control]
                physical_target = allocation_plan[target]
                optimized_circuit.append((gate, physical_control, physical_target))
        
        return optimized_circuit
    
    def _analyze_qubit_lifetimes(self, circuit, num_qubits):
        """Analyze the lifetimes of qubits in the circuit."""
        first_use = {i: float('inf') for i in range(num_qubits)}
        last_use = {i: -1 for i in range(num_qubits)}
        
        for i, op in enumerate(circuit):
            qubits = [op[1]] if len(op) == 2 else [op[1], op[2]]
            
            for q in qubits:
                first_use[q] = min(first_use[q], i)
                last_use[q] = max(last_use[q], i)
        
        return {i: (first_use[i], last_use[i]) for i in range(num_qubits) if first_use[i] <= last_use[i]}
    
    # Utility Methods
    def get_stats(self):
        """Get optimization statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset optimization statistics."""
        self.stats = {
            "original_gate_count": 0,
            "optimized_gate_count": 0,
            "original_depth": 0,
            "optimized_depth": 0,
            "optimization_time": 0,
            "swap_count": 0,
            "technique_stats": {
                "gate_synthesis": {"count": 0, "time": 0},
                "circuit_depth_reduction": {"count": 0, "time": 0},
                "qubit_mapping": {"count": 0, "time": 0},
                "measurement_optimization": {"count": 0, "time": 0},
                "compiler_techniques": {"count": 0, "time": 0},
                "hardware_specific": {"count": 0, "time": 0},
                "memory_management": {"count": 0, "time": 0}
            }
        }
    
    def print_stats(self):
        """Print optimization statistics."""
        print("\nQuantum Circuit Optimization Statistics:")
        print(f"Original gate count: {self.stats['original_gate_count']}")
        print(f"Optimized gate count: {self.stats['optimized_gate_count']}")
        print(f"Gate reduction: {self.stats['original_gate_count'] - self.stats['optimized_gate_count']} gates ({(1 - self.stats['optimized_gate_count']/max(1, self.stats['original_gate_count']))*100:.1f}%)")
        print(f"Original circuit depth: {self.stats['original_depth']}")
        print(f"Optimized circuit depth: {self.stats['optimized_depth']}")
        print(f"Depth reduction: {self.stats['original_depth'] - self.stats['optimized_depth']} layers ({(1 - self.stats['optimized_depth']/max(1, self.stats['original_depth']))*100:.1f}%)")
        print(f"SWAP gates inserted: {self.stats['swap_count']}")
        print(f"Total optimization time: {self.stats['optimization_time']:.4f} seconds")
        
        print("\nOptimization Technique Statistics:")
        for technique, stats in self.stats["technique_stats"].items():
            if stats["count"] > 0:
                print(f"  {technique}: {stats['count']} applications, {stats['time']:.4f} seconds")