import time
import numpy as np
import matplotlib.pyplot as plt
from quantum_circuit_optimizations import QuantumCircuitOptimizations

class OptimizationBenchmark:
    """Benchmark framework for quantum circuit optimizations."""
    
    def __init__(self):
        """Initialize the benchmark framework."""
        self.optimizer = QuantumCircuitOptimizations()
        self.results = {}
        
    def generate_random_circuit(self, num_qubits, circuit_depth, two_qubit_gate_ratio=0.3):
        """Generate a random quantum circuit for benchmarking.
        
        Args:
            num_qubits: Number of qubits in the circuit
            circuit_depth: Approximate depth of the circuit
            two_qubit_gate_ratio: Ratio of two-qubit gates to include
            
        Returns:
            A randomly generated quantum circuit
        """
        circuit = []
        single_qubit_gates = ["X", "Y", "Z", "H", "S", "T"]
        two_qubit_gates = ["CNOT", "CZ", "SWAP"]
        
        # Generate random gates
        for _ in range(circuit_depth * num_qubits):
            # Decide if this is a single-qubit or two-qubit gate
            if np.random.random() < two_qubit_gate_ratio and num_qubits > 1:
                # Two-qubit gate
                gate = np.random.choice(two_qubit_gates)
                control = np.random.randint(0, num_qubits)
                target = np.random.randint(0, num_qubits)
                # Ensure control and target are different
                while target == control:
                    target = np.random.randint(0, num_qubits)
                circuit.append((gate, control, target))
            else:
                # Single-qubit gate
                gate = np.random.choice(single_qubit_gates)
                qubit = np.random.randint(0, num_qubits)
                circuit.append((gate, qubit))
        
        # Add some measurements at the end
        for q in range(min(num_qubits, 3)):  # Measure a few qubits
            circuit.append(("M", q))
            
        return circuit
    
    def benchmark_optimization(self, optimization_technique, num_qubits_range, circuit_depths, trials=3):
        """Benchmark a specific optimization technique.
        
        Args:
            optimization_technique: Name of the optimization technique to benchmark
            num_qubits_range: Range of qubit counts to test
            circuit_depths: List of circuit depths to test
            trials: Number of trials for each configuration
            
        Returns:
            Dictionary of benchmark results
        """
        results = {
            'gate_count_reduction': [],
            'depth_reduction': [],
            'execution_time': [],
            'qubit_counts': [],
            'circuit_depths': []
        }
        
        for num_qubits in num_qubits_range:
            for depth in circuit_depths:
                gate_reductions = []
                depth_reductions = []
                execution_times = []
                
                for _ in range(trials):
                    # Generate a random circuit
                    circuit = self.generate_random_circuit(num_qubits, depth)
                    
                    # Reset optimizer stats
                    self.optimizer.reset_stats()
                    
                    # Measure original circuit depth
                    original_depth = self.optimizer.calculate_circuit_depth(circuit, num_qubits)
                    
                    # Time the optimization
                    start_time = time.time()
                    optimized_circuit = self.optimizer.optimize_circuit(
                        circuit, num_qubits, [optimization_technique]
                    )
                    end_time = time.time()
                    
                    # Calculate optimized circuit depth
                    optimized_depth = self.optimizer.calculate_circuit_depth(optimized_circuit, num_qubits)
                    
                    # Record results
                    gate_reduction = 1 - (len(optimized_circuit) / len(circuit))
                    depth_reduction = 1 - (optimized_depth / original_depth)
                    execution_time = end_time - start_time
                    
                    gate_reductions.append(gate_reduction)
                    depth_reductions.append(depth_reduction)
                    execution_times.append(execution_time)
                
                # Average the results
                results['gate_count_reduction'].append(np.mean(gate_reductions))
                results['depth_reduction'].append(np.mean(depth_reductions))
                results['execution_time'].append(np.mean(execution_times))
                results['qubit_counts'].append(num_qubits)
                results['circuit_depths'].append(depth)
        
        self.results[optimization_technique] = results
        return results
    
    def benchmark_all_optimizations(self, num_qubits_range, circuit_depths, trials=3):
        """Benchmark all optimization techniques.
        
        Args:
            num_qubits_range: Range of qubit counts to test
            circuit_depths: List of circuit depths to test
            trials: Number of trials for each configuration
        """
        optimization_techniques = [
            "gate_synthesis",
            "circuit_depth_reduction",
            "qubit_mapping",
            "measurement_optimization",
            "compiler_optimization",
            "hardware_specific",
            "memory_management"
        ]
        
        for technique in optimization_techniques:
            print(f"Benchmarking {technique}...")
            self.benchmark_optimization(technique, num_qubits_range, circuit_depths, trials)
    
    def plot_results(self, metric='gate_count_reduction'):
        """Plot benchmark results for a specific metric.
        
        Args:
            metric: The metric to plot ('gate_count_reduction', 'depth_reduction', or 'execution_time')
        """
        plt.figure(figsize=(12, 8))
        
        for technique, results in self.results.items():
            # Create a unique marker for each technique
            qubit_counts = results['qubit_counts']
            circuit_depths = results['circuit_depths']
            metric_values = results[metric]
            
            # Create a scatter plot with size proportional to circuit depth
            for i in range(len(qubit_counts)):
                plt.scatter(
                    qubit_counts[i], 
                    metric_values[i], 
                    s=circuit_depths[i]*10, 
                    label=f"{technique} (depth={circuit_depths[i]})" if i % len(circuit_depths) == 0 else "",
                    alpha=0.7
                )
        
        plt.xlabel('Number of Qubits')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} vs. Number of Qubits')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{metric}_benchmark.png")
        plt.close()
    
    def plot_all_metrics(self):
        """Plot all benchmark metrics."""
        metrics = ['gate_count_reduction', 'depth_reduction', 'execution_time']
        for metric in metrics:
            self.plot_results(metric)
    
    def print_summary(self):
        """Print a summary of the benchmark results."""
        print("\nBenchmark Summary")
        print("=================")
        
        for technique, results in self.results.items():
            print(f"\n{technique.replace('_', ' ').title()}:")
            print(f"  Average gate count reduction: {np.mean(results['gate_count_reduction']):.2%}")
            print(f"  Average depth reduction: {np.mean(results['depth_reduction']):.2%}")
            print(f"  Average execution time: {np.mean(results['execution_time']):.4f} seconds")


def benchmark_gate_synthesis():
    """Benchmark the gate synthesis optimization."""
    print("\nBenchmarking Gate Synthesis and Decomposition")
    print("============================================")
    
    # Create benchmark instance
    benchmark = OptimizationBenchmark()
    
    # Example 1: Small circuit with cancellable gates
    circuit1 = [
        ("X", 0), ("X", 0),  # These cancel out
        ("H", 1), ("Z", 1), ("H", 1),  # This is equivalent to X
        ("X", 2), ("Y", 2),  # These can be combined
        ("CNOT", 0, 1)
    ]
    
    # Example 2: Circuit with rotation gates
    circuit2 = [
        ("RX", 0, 0.1), ("RX", 0, 0.2),  # These can be combined
        ("RY", 1, 0.3), ("RZ", 1, 0.4),  # These can be optimized
        ("H", 2), ("T", 2), ("T", 2), ("H", 2)  # This has a known optimization
    ]
    
    # Test the optimization
    optimizer = QuantumCircuitOptimizations()
    
    print("\nExample 1: Circuit with cancellable gates")
    print("Original circuit:", circuit1)
    optimized1 = optimizer.optimize_circuit(circuit1, 3, ["gate_synthesis"])
    print("Optimized circuit:", optimized1)
    print(f"Gate count reduction: {1 - len(optimized1)/len(circuit1):.2%}")
    
    print("\nExample 2: Circuit with rotation gates")
    print("Original circuit:", circuit2)
    optimized2 = optimizer.optimize_circuit(circuit2, 3, ["gate_synthesis"])
    print("Optimized circuit:", optimized2)
    print(f"Gate count reduction: {1 - len(optimized2)/len(circuit2):.2%}")
    
    # Run full benchmark
    print("\nRunning full benchmark...")
    benchmark.benchmark_optimization(
        "gate_synthesis", 
        num_qubits_range=[2, 4, 6, 8], 
        circuit_depths=[5, 10, 20],
        trials=2  # Reduced for demonstration
    )
    
    # Print results
    benchmark.print_summary()
    
    # Example questions for further testing:
    print("\nExample questions for further testing:")
    print("1. How does the effectiveness of gate synthesis vary with circuit depth?")
    print("2. Which gate patterns show the highest optimization potential?")
    print("3. How does the optimization time scale with circuit size?")


def benchmark_circuit_depth_reduction():
    """Benchmark the circuit depth reduction optimization."""
    print("\nBenchmarking Circuit Depth Reduction")
    print("==================================")
    
    # Create benchmark instance
    benchmark = OptimizationBenchmark()
    
    # Example 1: Circuit with parallelizable operations
    circuit1 = [
        ("H", 0), ("X", 1),  # These can be parallelized
        ("Y", 2), ("Z", 3),  # These can be parallelized
        ("CNOT", 0, 1),
        ("CNOT", 2, 3)  # This can be parallelized with the previous CNOT
    ]
    
    # Example 2: Circuit with sequential dependencies
    circuit2 = [
        ("H", 0),
        ("CNOT", 0, 1),  # Depends on H gate
        ("X", 1),  # Depends on CNOT
        ("CNOT", 1, 2),  # Depends on X gate
        ("Z", 0)  # Can be parallelized with CNOT(1,2)
    ]
    
    # Test the optimization
    optimizer = QuantumCircuitOptimizations()
    
    print("\nExample 1: Circuit with parallelizable operations")
    original_depth1 = optimizer.calculate_circuit_depth(circuit1, 4)
    optimized1 = optimizer.optimize_circuit(circuit1, 4, ["circuit_depth_reduction"])
    optimized_depth1 = optimizer.calculate_circuit_depth(optimized1, 4)
    print(f"Original depth: {original_depth1}")
    print(f"Optimized depth: {optimized_depth1}")
    print(f"Depth reduction: {1 - optimized_depth1/original_depth1:.2%}")
    
    print("\nExample 2: Circuit with sequential dependencies")
    original_depth2 = optimizer.calculate_circuit_depth(circuit2, 3)
    optimized2 = optimizer.optimize_circuit(circuit2, 3, ["circuit_depth_reduction"])
    optimized_depth2 = optimizer.calculate_circuit_depth(optimized2, 3)
    print(f"Original depth: {original_depth2}")
    print(f"Optimized depth: {optimized_depth2}")
    print(f"Depth reduction: {1 - optimized_depth2/original_depth2:.2%}")
    
    # Run full benchmark
    print("\nRunning full benchmark...")
    benchmark.benchmark_optimization(
        "circuit_depth_reduction", 
        num_qubits_range=[3, 6, 9], 
        circuit_depths=[10, 20],
        trials=2  # Reduced for demonstration
    )
    
    # Print results
    benchmark.print_summary()
    
    # Example questions for further testing:
    print("\nExample questions for further testing:")
    print("1. How does the circuit topology affect depth reduction potential?")
    print("2. What is the trade-off between circuit depth and gate count?")
    print("3. How does the optimization perform on circuits with high two-qubit gate density?")


def benchmark_qubit_mapping():
    """Benchmark the qubit mapping and routing optimization."""
    print("\nBenchmarking Qubit Mapping and Routing")
    print("====================================")
    
    # Create benchmark instance
    benchmark = OptimizationBenchmark()
    
    # Example 1: Circuit with non-adjacent interactions on a linear topology
    circuit1 = [
        ("H", 0),
        ("CNOT", 0, 3),  # Non-adjacent in linear topology
        ("X", 1),
        ("CNOT", 1, 4),  # Non-adjacent in linear topology
        ("CNOT", 2, 3)   # Adjacent in linear topology
    ]
    
    # Example 2: Circuit with a mix of adjacent and non-adjacent interactions
    circuit2 = [
        ("H", 0), ("H", 1), ("H", 2), ("H", 3),
        ("CNOT", 0, 1),  # Adjacent
        ("CNOT", 2, 3),  # Adjacent
        ("CNOT", 0, 2),  # Non-adjacent in linear topology
        ("CNOT", 1, 3)   # Non-adjacent in linear topology
    ]
    
    # Test the optimization
    optimizer = QuantumCircuitOptimizations()
    optimizer.hardware_params["topology"] = "linear"
    
    print("\nExample 1: Circuit with non-adjacent interactions")
    optimizer._initialize_qubit_connectivity(5)
    print("Original circuit:", circuit1)
    optimized1 = optimizer.optimize_circuit(circuit1, 5, ["qubit_mapping"])
    print("Optimized circuit:", optimized1)
    print(f"SWAP gates inserted: {optimizer.stats['swaps_inserted']}")
    
    print("\nExample 2: Circuit with mixed interactions")
    optimizer._initialize_qubit_connectivity(4)
    print("Original circuit:", circuit2)
    optimized2 = optimizer.optimize_circuit(circuit2, 4, ["qubit_mapping"])
    print("Optimized circuit:", optimized2)
    print(f"SWAP gates inserted: {optimizer.stats['swaps_inserted']}")
    
    # Run full benchmark with different topologies
    print("\nRunning full benchmark with linear topology...")
    optimizer.hardware_params["topology"] = "linear"
    benchmark.optimizer.hardware_params["topology"] = "linear"
    benchmark.benchmark_optimization(
        "qubit_mapping", 
        num_qubits_range=[4, 6, 8], 
        circuit_depths=[5, 10],
        trials=2  # Reduced for demonstration
    )
    
    # Print results
    benchmark.print_summary()
    
    # Example questions for further testing:
    print("\nExample questions for further testing:")
    print("1. How does the effectiveness of qubit mapping vary with different hardware topologies?")
    print("2. What is the overhead of SWAP insertion for different circuit connectivities?")
    print("3. How does the initial qubit placement affect the optimization results?")


def benchmark_measurement_optimization():
    """Benchmark the measurement-based optimization."""
    print("\nBenchmarking Measurement-Based Optimization")
    print("========================================")
    
    # Create benchmark instance
    benchmark = OptimizationBenchmark()
    
    # Example 1: Circuit with deferrable measurements
    circuit1 = [
        ("H", 0), ("H", 1),
        ("CNOT", 0, 1),
        ("M", 0),  # This measurement can be deferred
        ("X", 1),
        ("M", 1)
    ]
    
    # Example 2: Circuit with measurements that affect subsequent operations
    circuit2 = [
        ("H", 0), ("H", 1),
        ("CNOT", 0, 1),
        ("M", 0),
        ("COND_X", 0, 1),  # Conditional operation based on measurement of qubit 0
        ("H", 2),
        ("CNOT", 1, 2),
        ("M", 1), ("M", 2)
    ]
    
    # Test the optimization
    optimizer = QuantumCircuitOptimizations()
    
    print("\nExample 1: Circuit with deferrable measurements")
    print("Original circuit:", circuit1)
    optimized1 = optimizer.optimize_circuit(circuit1, 2, ["measurement_optimization"])
    print("Optimized circuit:", optimized1)
    print(f"Measurements optimized: {optimizer.stats['measurements_optimized']}")
    
    print("\nExample 2: Circuit with conditional operations")
    print("Original circuit:", circuit2)
    optimized2 = optimizer.optimize_circuit(circuit2, 3, ["measurement_optimization"])
    print("Optimized circuit:", optimized2)
    print(f"Measurements optimized: {optimizer.stats['measurements_optimized']}")
    
    # Run full benchmark
    print("\nRunning full benchmark...")
    benchmark.benchmark_optimization(
        "measurement_optimization", 
        num_qubits_range=[2, 4, 6], 
        circuit_depths=[5, 10],
        trials=2  # Reduced for demonstration
    )
    
    # Print results
    benchmark.print_summary()
    
    # Example questions for further testing:
    print("\nExample questions for further testing:")
    print("1. How does the presence of conditional operations affect measurement optimization?")
    print("2. What percentage of measurements can typically be deferred in quantum algorithms?")
    print("3. How does measurement optimization affect the overall circuit fidelity?")


def benchmark_memory_management():
    """Benchmark the quantum memory management optimization."""
    print("\nBenchmarking Quantum Memory Management")
    print("====================================")
    
    # Create benchmark instance
    benchmark = OptimizationBenchmark()
    
    # Example 1: Circuit where qubits can be reused
    circuit1 = [
        ("H", 0), ("X", 1),
        ("CNOT", 0, 1),
        ("M", 0),  # Qubit 0 is no longer used
        ("H", 2),  # New qubit allocation
        ("CNOT", 1, 2),
        ("M", 1), ("M", 2)
    ]
    
    # Example 2: Circuit with interleaved qubit usage
    circuit2 = [
        ("H", 0), ("H", 1), ("H", 2),
        ("CNOT", 0, 1),
        ("M", 0),  # Qubit 0 is measured early
        ("X", 3),  # New qubit allocation
        ("CNOT", 1, 3),
        ("M", 1),  # Qubit 1 is measured
        ("CNOT", 2, 3),
        ("M", 2), ("M", 3)
    ]
    
    # Test the optimization
    optimizer = QuantumCircuitOptimizations()
    
    print("\nExample 1: Circuit with reusable qubits")
    print("Original circuit:", circuit1)
    optimized1 = optimizer.optimize_circuit(circuit1, 3, ["memory_management"])
    print("Optimized circuit:", optimized1)
    print(f"Memory optimizations: {optimizer.stats['memory_optimizations']}")
    
    print("\nExample 2: Circuit with interleaved qubit usage")
    print("Original circuit:", circuit2)
    optimized2 = optimizer.optimize_circuit(circuit2, 4, ["memory_management"])
    print("Optimized circuit:", optimized2)
    print(f"Memory optimizations: {optimizer.stats['memory_optimizations']}")
    
    # Run full benchmark
    print("\nRunning full benchmark...")
    benchmark.benchmark_optimization(
        "memory_management", 
        num_qubits_range=[4, 8, 12], 
        circuit_depths=[10, 20],
        trials=2  # Reduced for demonstration
    )
    
    # Print results
    benchmark.print_summary()
    
    # Example questions for further testing:
    print("\nExample questions for further testing:")
    print("1. How does the qubit reuse potential vary with different quantum algorithms?")
    print("2. What is the relationship between circuit depth and memory optimization potential?")
    print("3. How does memory optimization affect the overall resource requirements?")


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark of all optimization techniques."""
    print("\nRunning Comprehensive Benchmark")
    print("==============================")
    
    # Create benchmark instance
    benchmark = OptimizationBenchmark()
    
    # Run benchmark for all optimization techniques
    benchmark.benchmark_all_optimizations(
        num_qubits_range=[4, 8, 12, 16], 
        circuit_depths=[10, 20, 30],
        trials=3
    )
    
    # Plot results
    benchmark.plot_all_metrics()
    
    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    # Run individual benchmarks
    benchmark_gate_synthesis()
    benchmark_circuit_depth_reduction()
    benchmark_qubit_mapping()
    benchmark_measurement_optimization()
    benchmark_memory_management()
    
    # Uncomment to run comprehensive benchmark (takes longer)
    # run_comprehensive_benchmark()