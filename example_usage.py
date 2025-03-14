#!/usr/bin/env python3
"""
Example Usage of Quantum Circuit Optimizations

This script demonstrates how to use the quantum circuit optimization techniques
in practical quantum computing scenarios.
"""

import numpy as np
import time
from quantum_circuit_optimizations import QuantumCircuitOptimizations

def print_circuit_stats(circuit, num_qubits, name="Circuit"):
    """Print statistics about a quantum circuit."""
    optimizer = QuantumCircuitOptimizations()
    depth = optimizer.calculate_circuit_depth(circuit, num_qubits)
    
    # Count gate types
    gate_counts = {}
    for op in circuit:
        gate_type = op[0]
        if gate_type not in gate_counts:
            gate_counts[gate_type] = 0
        gate_counts[gate_type] += 1
    
    print(f"\n{name} Statistics:")
    print(f"  Total gates: {len(circuit)}")
    print(f"  Circuit depth: {depth}")
    print("  Gate breakdown:")
    for gate, count in sorted(gate_counts.items()):
        print(f"    {gate}: {count}")

def example_quantum_fourier_transform():
    """Example of optimizing a Quantum Fourier Transform circuit."""
    print("\n=== Quantum Fourier Transform Optimization ===\n")
    
    def create_qft_circuit(num_qubits):
        """Create a Quantum Fourier Transform circuit."""
        circuit = []
        
        # QFT implementation
        for i in range(num_qubits):
            # Hadamard gate
            circuit.append(("H", i))
            
            # Controlled phase rotations
            for j in range(i + 1, num_qubits):
                # Controlled phase rotation with angle pi/2^(j-i)
                circuit.append(("CP", i, j, 1.0 / (2 ** (j - i))))
        
        # Swap qubits
        for i in range(num_qubits // 2):
            circuit.append(("SWAP", i, num_qubits - i - 1))
        
        return circuit
    
    # Create QFT circuit for 6 qubits
    num_qubits = 6
    qft_circuit = create_qft_circuit(num_qubits)
    
    print(f"Created QFT circuit for {num_qubits} qubits")
    print_circuit_stats(qft_circuit, num_qubits, "Original QFT")
    
    # Initialize optimizer
    optimizer = QuantumCircuitOptimizations()
    
    # Apply individual optimizations
    print("\nApplying individual optimizations:")
    
    # Gate synthesis
    gate_optimized = optimizer.optimize_circuit(qft_circuit, num_qubits, ["gate_synthesis"])
    print("\nAfter gate synthesis optimization:")
    print_circuit_stats(gate_optimized, num_qubits, "Gate-Optimized QFT")
    
    # Circuit depth reduction
    depth_optimized = optimizer.optimize_circuit(qft_circuit, num_qubits, ["circuit_depth_reduction"])
    print("\nAfter circuit depth reduction:")
    print_circuit_stats(depth_optimized, num_qubits, "Depth-Optimized QFT")
    
    # Apply all optimizations
    print("\nApplying all optimizations:")
    fully_optimized = optimizer.optimize_circuit(qft_circuit, num_qubits)
    print_circuit_stats(fully_optimized, num_qubits, "Fully Optimized QFT")
    
    # Print optimization statistics
    print("\nOptimization Statistics:")
    optimizer.print_optimization_stats()

def example_grover_search():
    """Example of optimizing Grover's search algorithm circuit."""
    print("\n=== Grover's Search Algorithm Optimization ===\n")
    
    def create_grover_circuit(num_qubits, num_iterations):
        """Create a circuit for Grover's search algorithm."""
        circuit = []
        
        # Initialize with Hadamard gates
        for i in range(num_qubits):
            circuit.append(("H", i))
        
        # Apply Grover iterations
        for _ in range(num_iterations):
            # Oracle (assume we're searching for the all-1s state)
            for i in range(num_qubits - 1):
                circuit.append(("CNOT", i, num_qubits - 1))
            
            circuit.append(("Z", num_qubits - 1))
            
            for i in range(num_qubits - 1, 0, -1):
                circuit.append(("CNOT", i - 1, i))
            
            # Diffusion operator
            for i in range(num_qubits):
                circuit.append(("H", i))
            
            for i in range(num_qubits):
                circuit.append(("X", i))
            
            # Multi-controlled Z gate (implemented with CNOTs and single-qubit gates)
            for i in range(num_qubits - 1):
                circuit.append(("CNOT", i, num_qubits - 1))
            
            circuit.append(("Z", num_qubits - 1))
            
            for i in range(num_qubits - 1, 0, -1):
                circuit.append(("CNOT", i - 1, i))
            
            for i in range(num_qubits):
                circuit.append(("X", i))
            
            for i in range(num_qubits):
                circuit.append(("H", i))
        
        # Measure all qubits
        for i in range(num_qubits):
            circuit.append(("M", i))
        
        return circuit
    
    # Create Grover circuit for 4 qubits with 2 iterations
    num_qubits = 4
    num_iterations = 2
    grover_circuit = create_grover_circuit(num_qubits, num_iterations)
    
    print(f"Created Grover's search circuit for {num_qubits} qubits with {num_iterations} iterations")
    print_circuit_stats(grover_circuit, num_qubits, "Original Grover")
    
    # Initialize optimizer
    optimizer = QuantumCircuitOptimizations()
    
    # Apply individual optimizations
    print("\nApplying individual optimizations:")
    
    # Gate synthesis
    gate_optimized = optimizer.optimize_circuit(grover_circuit, num_qubits, ["gate_synthesis"])
    print("\nAfter gate synthesis optimization:")
    print_circuit_stats(gate_optimized, num_qubits, "Gate-Optimized Grover")
    
    # Circuit depth reduction
    depth_optimized = optimizer.optimize_circuit(grover_circuit, num_qubits, ["circuit_depth_reduction"])
    print("\nAfter circuit depth reduction:")
    print_circuit_stats(depth_optimized, num_qubits, "Depth-Optimized Grover")
    
    # Qubit mapping with linear topology
    optimizer.hardware_params["topology"] = "linear"
    optimizer._initialize_qubit_connectivity(num_qubits)
    mapping_optimized = optimizer.optimize_circuit(grover_circuit, num_qubits, ["qubit_mapping"])
    print("\nAfter qubit mapping optimization (linear topology):")
    print_circuit_stats(mapping_optimized, num_qubits, "Mapping-Optimized Grover")
    
    # Apply all optimizations
    print("\nApplying all optimizations:")
    fully_optimized = optimizer.optimize_circuit(grover_circuit, num_qubits)
    print_circuit_stats(fully_optimized, num_qubits, "Fully Optimized Grover")
    
    # Print optimization statistics
    print("\nOptimization Statistics:")
    optimizer.print_optimization_stats()

def example_quantum_error_correction():
    """Example of optimizing a quantum error correction circuit."""
    print("\n=== Quantum Error Correction Circuit Optimization ===\n")
    
    def create_bit_flip_code_circuit():
        """Create a circuit for the 3-qubit bit flip code."""
        circuit = []
        
        # Encode logical qubit (qubit 0) into three physical qubits (0, 1, 2)
        circuit.append(("CNOT", 0, 1))  # Spread the state to ancilla qubits
        circuit.append(("CNOT", 0, 2))
        
        # Simulate a bit flip error on qubit 1
        circuit.append(("X", 1))
        
        # Error detection
        circuit.append(("CNOT", 0, 3))  # Use qubits 3 and 4 as syndrome qubits
        circuit.append(("CNOT", 1, 3))
        circuit.append(("CNOT", 1, 4))
        circuit.append(("CNOT", 2, 4))
        
        # Measure syndrome qubits
        circuit.append(("M", 3))
        circuit.append(("M", 4))
        
        # Error correction (conditional on syndrome measurements)
        # In a real circuit, these would be conditional operations
        circuit.append(("X", 1))  # Correct the error
        
        # Decode
        circuit.append(("CNOT", 0, 1))
        circuit.append(("CNOT", 0, 2))
        
        # Measure data qubits
        circuit.append(("M", 0))
        circuit.append(("M", 1))
        circuit.append(("M", 2))
        
        return circuit
    
    # Create bit flip code circuit
    qec_circuit = create_bit_flip_code_circuit()
    num_qubits = 5  # 3 data qubits + 2 syndrome qubits
    
    print("Created 3-qubit bit flip code circuit")
    print_circuit_stats(qec_circuit, num_qubits, "Original QEC")
    
    # Initialize optimizer
    optimizer = QuantumCircuitOptimizations()
    
    # Apply measurement optimization
    measurement_optimized = optimizer.optimize_circuit(qec_circuit, num_qubits, ["measurement_optimization"])
    print("\nAfter measurement optimization:")
    print_circuit_stats(measurement_optimized, num_qubits, "Measurement-Optimized QEC")
    
    # Apply memory management optimization
    memory_optimized = optimizer.optimize_circuit(qec_circuit, num_qubits, ["memory_management"])
    print("\nAfter memory management optimization:")
    print_circuit_stats(memory_optimized, num_qubits, "Memory-Optimized QEC")
    
    # Apply all optimizations
    print("\nApplying all optimizations:")
    fully_optimized = optimizer.optimize_circuit(qec_circuit, num_qubits)
    print_circuit_stats(fully_optimized, num_qubits, "Fully Optimized QEC")
    
    # Print optimization statistics
    print("\nOptimization Statistics:")
    optimizer.print_optimization_stats()

def example_optimization_comparison():
    """Compare the effectiveness of different optimization techniques on various circuits."""
    print("\n=== Optimization Technique Comparison ===\n")
    
    # Define circuit types and sizes
    circuit_types = ["QFT", "Grover", "QEC"]
    qubit_counts = [4, 6, 8]
    
    # Initialize results table
    results = {
        "circuit_type": [],
        "num_qubits": [],
        "original_gates": [],
        "original_depth": [],
        "gate_synthesis_gates": [],
        "gate_synthesis_depth": [],
        "depth_reduction_gates": [],
        "depth_reduction_depth": [],
        "qubit_mapping_gates": [],
        "qubit_mapping_depth": [],
        "measurement_opt_gates": [],
        "measurement_opt_depth": [],
        "memory_opt_gates": [],
        "memory_opt_depth": [],
        "all_opt_gates": [],
        "all_opt_depth": []
    }
    
    # Initialize optimizer
    optimizer = QuantumCircuitOptimizations()
    optimizer.hardware_params["topology"] = "linear"
    
    # Generate and optimize circuits
    for circuit_type in circuit_types:
        for num_qubits in qubit_counts:
            # Skip larger QEC circuits
            if circuit_type == "QEC" and num_qubits > 6:
                continue
                
            print(f"\nProcessing {circuit_type} circuit with {num_qubits} qubits...")
            
            # Create circuit
            if circuit_type == "QFT":
                circuit = create_qft_circuit(num_qubits)
            elif circuit_type == "Grover":
                iterations = max(1, num_qubits // 2)
                circuit = create_grover_circuit(num_qubits, iterations)
            elif circuit_type == "QEC":
                # For QEC, we scale the number of logical qubits
                logical_qubits = max(1, num_qubits // 3)
                circuit = create_bit_flip_code_circuit()
                # Adjust the circuit for different sizes (simplified)
                num_qubits = 5  # Fixed for the bit flip code
            
            # Calculate original statistics
            original_depth = optimizer.calculate_circuit_depth(circuit, num_qubits)
            
            # Apply different optimizations
            gate_opt = optimizer.optimize_circuit(circuit, num_qubits, ["gate_synthesis"])
            gate_opt_depth = optimizer.calculate_circuit_depth(gate_opt, num_qubits)
            
            depth_opt = optimizer.optimize_circuit(circuit, num_qubits, ["circuit_depth_reduction"])
            depth_opt_depth = optimizer.calculate_circuit_depth(depth_opt, num_qubits)
            
            optimizer._initialize_qubit_connectivity(num_qubits)
            mapping_opt = optimizer.optimize_circuit(circuit, num_qubits, ["qubit_mapping"])
            mapping_opt_depth = optimizer.calculate_circuit_depth(mapping_opt, num_qubits)
            
            meas_opt = optimizer.optimize_circuit(circuit, num_qubits, ["measurement_optimization"])
            meas_opt_depth = optimizer.calculate_circuit_depth(meas_opt, num_qubits)
            
            mem_opt = optimizer.optimize_circuit(circuit, num_qubits, ["memory_management"])
            mem_opt_depth = optimizer.calculate_circuit_depth(mem_opt, num_qubits)
            
            all_opt = optimizer.optimize_circuit(circuit, num_qubits)
            all_opt_depth = optimizer.calculate_circuit_depth(all_opt, num_qubits)
            
            # Record results
            results["circuit_type"].append(circuit_type)
            results["num_qubits"].append(num_qubits)
            results["original_gates"].append(len(circuit))
            results["original_depth"].append(original_depth)
            results["gate_synthesis_gates"].append(len(gate_opt))
            results["gate_synthesis_depth"].append(gate_opt_depth)
            results["depth_reduction_gates"].append(len(depth_opt))
            results["depth_reduction_depth"].append(depth_opt_depth)
            results["qubit_mapping_gates"].append(len(mapping_opt))
            results["qubit_mapping_depth"].append(mapping_opt_depth)
            results["measurement_opt_gates"].append(len(meas_opt))
            results["measurement_opt_depth"].append(meas_opt_depth)
            results["memory_opt_gates"].append(len(mem_opt))
            results["memory_opt_depth"].append(mem_opt_depth)
            results["all_opt_gates"].append(len(all_opt))
            results["all_opt_depth"].append(all_opt_depth)
    
    # Print results table
    print("\nOptimization Results Summary:")
    print("-----------------------------")
    print(f"{'Circuit':<8} {'Qubits':<6} {'Original':<16} {'Gate Synth':<16} {'Depth Red':<16} {'Qubit Map':<16} {'Meas Opt':<16} {'Mem Opt':<16} {'All Opt':<16}")
    print(f"{'Type':<8} {'Count':<6} {'Gates':<8} {'Depth':<7} {'Gates':<8} {'Depth':<7} {'Gates':<8} {'Depth':<7} {'Gates':<8} {'Depth':<7} {'Gates':<8} {'Depth':<7} {'Gates':<8} {'Depth':<7} {'Gates':<8} {'Depth':<7}")
    print("-" * 120)
    
    for i in range(len(results["circuit_type"])):
        print(f"{results['circuit_type'][i]:<8} {results['num_qubits'][i]:<6} "
              f"{results['original_gates'][i]:<8} {results['original_depth'][i]:<7} "
              f"{results['gate_synthesis_gates'][i]:<8} {results['gate_synthesis_depth'][i]:<7} "
              f"{results['depth_reduction_gates'][i]:<8} {results['depth_reduction_depth'][i]:<7} "
              f"{results['qubit_mapping_gates'][i]:<8} {results['qubit_mapping_depth'][i]:<7} "
              f"{results['measurement_opt_gates'][i]:<8} {results['measurement_opt_depth'][i]:<7} "
              f"{results['memory_opt_gates'][i]:<8} {results['memory_opt_depth'][i]:<7} "
              f"{results['all_opt_gates'][i]:<8} {results['all_opt_depth'][i]:<7}")
    
    # Calculate average improvements
    print("\nAverage Improvements:")
    print("--------------------")
    
    techniques = [
        ("Gate Synthesis", "gate_synthesis_gates", "gate_synthesis_depth"),
        ("Depth Reduction", "depth_reduction_gates", "depth_reduction_depth"),
        ("Qubit Mapping", "qubit_mapping_gates", "qubit_mapping_depth"),
        ("Measurement Opt", "measurement_opt_gates", "measurement_opt_depth"),
        ("Memory Opt", "memory_opt_gates", "memory_opt_depth"),
        ("All Optimizations", "all_opt_gates", "all_opt_depth")
    ]
    
    for name, gates_key, depth_key in techniques:
        gate_reductions = [1 - (results[gates_key][i] / results["original_gates"][i]) for i in range(len(results["circuit_type"]))]
        depth_reductions = [1 - (results[depth_key][i] / results["original_depth"][i]) for i in range(len(results["circuit_type"]))]
        
        avg_gate_reduction = np.mean(gate_reductions)
        avg_depth_reduction = np.mean(depth_reductions)
        
        print(f"{name}: Gate reduction: {avg_gate_reduction:.2%}, Depth reduction: {avg_depth_reduction:.2%}")

if __name__ == "__main__":
    # Run examples
    example_quantum_fourier_transform()
    example_grover_search()
    example_quantum_error_correction()
    example_optimization_comparison()