#!/usr/bin/env python3
"""
Specific Benchmarks for Quantum Circuit Optimizations

This script implements specific benchmarks for each optimization technique
as described in the task. Each benchmark is designed to test a specific
optimization technique with circuits that are tailored to trigger that
optimization.
"""

import numpy as np
import time
from quantum_circuit_optimizations import QuantumCircuitOptimizations

class SpecificBenchmarks:
    """
    A class that implements specific benchmarks for each optimization technique.
    """
    
    def __init__(self):
        """Initialize the benchmark framework."""
        self.optimizer = QuantumCircuitOptimizations()
        self.results = {}
    
    def run_all_benchmarks(self, verbose=True):
        """Run all specific benchmarks."""
        self.benchmark_gate_synthesis(verbose)
        self.benchmark_circuit_depth_reduction(verbose)
        self.benchmark_qubit_mapping(verbose)
        self.benchmark_measurement_optimization(verbose)
        self.benchmark_compiler_optimization(verbose)
        self.benchmark_hardware_specific(verbose)
        self.benchmark_memory_management(verbose)
        
        return self.results
    
    def benchmark_gate_synthesis(self, verbose=True):
        """
        Benchmark the gate synthesis optimization.
        
        Test Circuit:
        - Two consecutive X gates on qubit 0 (should cancel out).
        - H-Z-H sequence on qubit 0 (should become X).
        - Two RZ gates with angle π/4 on qubit 1 (should combine into RZ with angle π/2).
        """
        if verbose:
            print("\n=== Gate Synthesis Benchmark ===")
        
        # Create the test circuit
        circuit = [
            ("X", 0), ("X", 0),  # Two consecutive X gates (should cancel out)
            ("H", 0), ("Z", 0), ("H", 0),  # H-Z-H sequence (should become X)
            ("RZ", 1, np.pi/4), ("RZ", 1, np.pi/4)  # Two RZ gates (should combine)
        ]
        
        num_qubits = 2
        
        # Expected optimized circuit
        expected_circuit = [
            ("X", 0),  # From H-Z-H
            ("RZ", 1, np.pi/2)  # From combined RZ gates
        ]
        
        # Run the optimization
        self.optimizer.reset_stats()
        optimized_circuit = self.optimizer.optimize_circuit(circuit, num_qubits, ["gate_synthesis"])
        
        # Verify results
        gates_removed = self.optimizer.stats["gates_removed"]
        gates_replaced = self.optimizer.stats["gates_replaced"]
        
        # Check if the optimized circuit matches the expected circuit
        is_correct = self._compare_circuits(optimized_circuit, expected_circuit)
        
        # Record results
        self.results["gate_synthesis"] = {
            "original_circuit": circuit,
            "optimized_circuit": optimized_circuit,
            "expected_circuit": expected_circuit,
            "gates_removed": gates_removed,
            "gates_replaced": gates_replaced,
            "is_correct": is_correct
        }
        
        if verbose:
            print(f"Original circuit: {circuit}")
            print(f"Optimized circuit: {optimized_circuit}")
            print(f"Expected circuit: {expected_circuit}")
            print(f"Gates removed: {gates_removed} (expected: 2)")
            print(f"Gates replaced: {gates_replaced} (expected: 3)")
            print(f"Correct optimization: {is_correct}")
        
        return self.results["gate_synthesis"]
    
    def benchmark_circuit_depth_reduction(self, verbose=True):
        """
        Benchmark the circuit depth reduction optimization.
        
        Test Circuit:
        - H(0) and X(1) can be parallel (no qubit overlap).
        - CNOT(0,1) depends on H(0) and X(1).
        - Y(2) is independent of CNOT(0,1).
        - CNOT(1,2) depends on CNOT(0,1) and Y(2).
        """
        if verbose:
            print("\n=== Circuit Depth Reduction Benchmark ===")
        
        # Create the test circuit
        circuit = [
            ("H", 0), ("X", 1), ("CNOT", 0, 1), ("Y", 2), ("CNOT", 1, 2)
        ]
        
        num_qubits = 3
        
        # Run the optimization
        self.optimizer.reset_stats()
        original_depth = self.optimizer.calculate_circuit_depth(circuit, num_qubits)
        optimized_circuit = self.optimizer.optimize_circuit(circuit, num_qubits, ["circuit_depth_reduction"])
        optimized_depth = self.optimizer.calculate_circuit_depth(optimized_circuit, num_qubits)
        
        # Expected depth is 3 (instead of 5 in sequential execution)
        expected_depth = 3
        
        # Record results
        self.results["circuit_depth_reduction"] = {
            "original_circuit": circuit,
            "optimized_circuit": optimized_circuit,
            "original_depth": original_depth,
            "optimized_depth": optimized_depth,
            "expected_depth": expected_depth,
            "is_correct": optimized_depth == expected_depth
        }
        
        if verbose:
            print(f"Original circuit: {circuit}")
            print(f"Original depth: {original_depth}")
            print(f"Optimized circuit: {optimized_circuit}")
            print(f"Optimized depth: {optimized_depth}")
            print(f"Expected depth: {expected_depth}")
            print(f"Correct optimization: {optimized_depth == expected_depth}")
        
        return self.results["circuit_depth_reduction"]
    
    def benchmark_qubit_mapping(self, verbose=True):
        """
        Benchmark the qubit mapping optimization.
        
        Test Circuit:
        - CNOT(0,2) in a linear topology (0-1-2-3), qubits 0 and 2 are not adjacent.
        """
        if verbose:
            print("\n=== Qubit Mapping Benchmark ===")
        
        # Create the test circuit
        circuit = [
            ("CNOT", 0, 2)  # Non-adjacent in linear topology
        ]
        
        num_qubits = 4
        
        # Set linear topology
        self.optimizer.hardware_params["topology"] = "linear"
        self.optimizer._initialize_qubit_connectivity(num_qubits)
        
        # Run the optimization
        self.optimizer.reset_stats()
        optimized_circuit = self.optimizer.optimize_circuit(circuit, num_qubits, ["qubit_mapping"])
        
        # Expected: SWAP(1,2) followed by CNOT(0,1)
        # Note: The exact implementation might vary, but should include at least one SWAP
        swaps_inserted = self.optimizer.stats["swaps_inserted"]
        
        # Record results
        self.results["qubit_mapping"] = {
            "original_circuit": circuit,
            "optimized_circuit": optimized_circuit,
            "swaps_inserted": swaps_inserted,
            "is_correct": swaps_inserted >= 1  # At least one SWAP should be inserted
        }
        
        if verbose:
            print(f"Original circuit: {circuit}")
            print(f"Optimized circuit: {optimized_circuit}")
            print(f"SWAP gates inserted: {swaps_inserted} (expected: at least 1)")
            print(f"Correct optimization: {swaps_inserted >= 1}")
        
        return self.results["qubit_mapping"]
    
    def benchmark_measurement_optimization(self, verbose=True):
        """
        Benchmark the measurement optimization.
        
        Test Circuit:
        - H(0), M(0), X(1), M(1)
        - M(0) can be deferred since X(1) doesn't depend on it.
        - M(1) is already at the end.
        """
        if verbose:
            print("\n=== Measurement Optimization Benchmark ===")
        
        # Create the test circuit
        circuit = [
            ("H", 0), ("M", 0), ("X", 1), ("M", 1)
        ]
        
        num_qubits = 2
        
        # Expected optimized circuit
        expected_circuit = [
            ("H", 0), ("X", 1), ("M", 0), ("M", 1)
        ]
        
        # Run the optimization
        self.optimizer.reset_stats()
        optimized_circuit = self.optimizer.optimize_circuit(circuit, num_qubits, ["measurement_optimization"])
        
        # Verify results
        measurements_optimized = self.optimizer.stats["measurements_optimized"]
        
        # Check if the optimized circuit matches the expected circuit
        is_correct = self._compare_circuits(optimized_circuit, expected_circuit)
        
        # Record results
        self.results["measurement_optimization"] = {
            "original_circuit": circuit,
            "optimized_circuit": optimized_circuit,
            "expected_circuit": expected_circuit,
            "measurements_optimized": measurements_optimized,
            "is_correct": is_correct
        }
        
        if verbose:
            print(f"Original circuit: {circuit}")
            print(f"Optimized circuit: {optimized_circuit}")
            print(f"Expected circuit: {expected_circuit}")
            print(f"Measurements optimized: {measurements_optimized} (expected: 1)")
            print(f"Correct optimization: {is_correct}")
        
        return self.results["measurement_optimization"]
    
    def benchmark_compiler_optimization(self, verbose=True):
        """
        Benchmark the compiler optimization.
        
        Test Circuit:
        - H-H, X-X, and CNOT-CNOT pairs that should cancel out.
        """
        if verbose:
            print("\n=== Compiler Optimization Benchmark ===")
        
        # Create the test circuit
        circuit = [
            ("H", 0), ("H", 0),  # H-H cancels
            ("X", 1), ("X", 1),  # X-X cancels
            ("CNOT", 0, 1), ("CNOT", 0, 1)  # CNOT-CNOT cancels
        ]
        
        num_qubits = 2
        
        # Expected optimized circuit (empty, as all gates cancel out)
        expected_circuit = []
        
        # Run the optimization
        self.optimizer.reset_stats()
        optimized_circuit = self.optimizer.optimize_circuit(circuit, num_qubits, ["compiler_optimization"])
        
        # Verify results
        gates_removed = self.optimizer.stats["gates_removed"]
        
        # Check if the optimized circuit matches the expected circuit
        is_correct = self._compare_circuits(optimized_circuit, expected_circuit)
        
        # Record results
        self.results["compiler_optimization"] = {
            "original_circuit": circuit,
            "optimized_circuit": optimized_circuit,
            "expected_circuit": expected_circuit,
            "gates_removed": gates_removed,
            "is_correct": is_correct
        }
        
        if verbose:
            print(f"Original circuit: {circuit}")
            print(f"Optimized circuit: {optimized_circuit}")
            print(f"Expected circuit: {expected_circuit}")
            print(f"Gates removed: {gates_removed} (expected: 6)")
            print(f"Correct optimization: {is_correct}")
        
        return self.results["compiler_optimization"]
    
    def benchmark_hardware_specific(self, verbose=True):
        """
        Benchmark the hardware-specific optimization.
        
        Test Circuit:
        - S and T gates, which are not native and should be decomposed.
        """
        if verbose:
            print("\n=== Hardware-Specific Optimization Benchmark ===")
        
        # Create the test circuit
        circuit = [
            ("S", 0),  # Should be decomposed to RZ(π/2)
            ("T", 1)   # Should be decomposed to RZ(π/4)
        ]
        
        num_qubits = 2
        
        # Expected optimized circuit
        expected_circuit = [
            ("RZ", 0, np.pi/2),  # S gate decomposed
            ("RZ", 1, np.pi/4)   # T gate decomposed
        ]
        
        # Run the optimization
        self.optimizer.reset_stats()
        optimized_circuit = self.optimizer.optimize_circuit(circuit, num_qubits, ["hardware_specific"])
        
        # Check if the optimized circuit matches the expected circuit
        is_correct = self._compare_circuits(optimized_circuit, expected_circuit)
        
        # Record results
        self.results["hardware_specific"] = {
            "original_circuit": circuit,
            "optimized_circuit": optimized_circuit,
            "expected_circuit": expected_circuit,
            "is_correct": is_correct
        }
        
        if verbose:
            print(f"Original circuit: {circuit}")
            print(f"Optimized circuit: {optimized_circuit}")
            print(f"Expected circuit: {expected_circuit}")
            print(f"Correct optimization: {is_correct}")
        
        return self.results["hardware_specific"]
    
    def benchmark_memory_management(self, verbose=True):
        """
        Benchmark the memory management optimization.
        
        Test Circuit:
        - H(0), CNOT(0,1), M(0), H(2), M(2)
        - Qubit 0 is measured after CNOT and not used again.
        - Qubit 2 is used later, potentially reusing qubit 0's physical qubit.
        """
        if verbose:
            print("\n=== Memory Management Benchmark ===")
        
        # Create the test circuit
        circuit = [
            ("H", 0), ("CNOT", 0, 1), ("M", 0),  # Qubit 0 is no longer used
            ("H", 2), ("M", 2)  # Qubit 2 could reuse qubit 0's physical qubit
        ]
        
        num_qubits = 3
        
        # Run the optimization
        self.optimizer.reset_stats()
        optimized_circuit = self.optimizer.optimize_circuit(circuit, num_qubits, ["memory_management"])
        
        # Verify results
        memory_optimizations = self.optimizer.stats["memory_optimizations"]
        
        # Record results
        self.results["memory_management"] = {
            "original_circuit": circuit,
            "optimized_circuit": optimized_circuit,
            "memory_optimizations": memory_optimizations,
            "is_correct": memory_optimizations >= 1  # At least one memory optimization
        }
        
        if verbose:
            print(f"Original circuit: {circuit}")
            print(f"Optimized circuit: {optimized_circuit}")
            print(f"Memory optimizations: {memory_optimizations} (expected: at least 1)")
            print(f"Correct optimization: {memory_optimizations >= 1}")
        
        return self.results["memory_management"]
    
    def _compare_circuits(self, circuit1, circuit2):
        """
        Compare two circuits for functional equivalence.
        
        This is a simplified comparison that checks if the circuits have the same gates
        in the same order. In a real system, a more sophisticated comparison would be used
        that accounts for equivalent gate sequences.
        """
        if len(circuit1) != len(circuit2):
            return False
        
        for i in range(len(circuit1)):
            gate1 = circuit1[i]
            gate2 = circuit2[i]
            
            # Check gate type
            if gate1[0] != gate2[0]:
                return False
            
            # Check qubit indices
            if len(gate1) != len(gate2):
                return False
            
            for j in range(1, len(gate1) - 1):  # Skip the last element if it's a parameter
                if gate1[j] != gate2[j]:
                    return False
            
            # Check parameters (if any)
            if len(gate1) > 2 and isinstance(gate1[-1], (int, float)) and isinstance(gate2[-1], (int, float)):
                # For floating point parameters, use approximate equality
                if abs(gate1[-1] - gate2[-1]) > 1e-6:
                    return False
        
        return True

def run_benchmarks():
    """Run all specific benchmarks and print a summary."""
    print("\n=== Running Specific Benchmarks for Quantum Circuit Optimizations ===\n")
    
    benchmarks = SpecificBenchmarks()
    results = benchmarks.run_all_benchmarks()
    
    print("\n=== Benchmark Summary ===\n")
    
    all_correct = True
    for technique, result in results.items():
        is_correct = result.get("is_correct", False)
        all_correct = all_correct and is_correct
        
        print(f"{technique.replace('_', ' ').title()}: {'✓' if is_correct else '✗'}")
    
    print(f"\nOverall: {'All optimizations working correctly' if all_correct else 'Some optimizations need adjustment'}")
    
    return results

if __name__ == "__main__":
    run_benchmarks()