import numpy as np
from quantum_circuit_optimizations import QuantumCircuitOptimizations

def test_gate_synthesis_optimization():
    """Test the gate synthesis and decomposition optimization."""
    print("\nTesting Gate Synthesis and Decomposition Optimization")
    print("----------------------------------------------------")
    
    # Create a test circuit with patterns that can be optimized
    circuit = [
        # Pattern 1: Two X gates cancel out
        ("X", 0),
        ("X", 0),
        
        # Pattern 2: H-Z-H is equivalent to X
        ("H", 1),
        ("Z", 1),
        ("H", 1),
        
        # Pattern 3: Some other gates
        ("H", 2),
        ("X", 3),
        
        # Pattern 4: CNOT-CNOT-CNOT is a SWAP
        ("CNOT", 4, 5),
        ("CNOT", 5, 4),
        ("CNOT", 4, 5)
    ]
    
    # Create the optimizer
    optimizer = QuantumCircuitOptimizations()
    
    # Apply gate synthesis optimization
    optimized_circuit = optimizer.optimize_circuit(circuit, 6, ["gate_synthesis"])
    
    # Print results
    print(f"Original circuit: {len(circuit)} gates")
    print(f"Optimized circuit: {len(optimized_circuit)} gates")
    print(f"Gates removed: {optimizer.stats['gates_removed']}")
    print(f"Gates replaced: {optimizer.stats['gates_replaced']}")
    
    print("\nOriginal circuit:")
    for op in circuit:
        print(f"  {op}")
    
    print("\nOptimized circuit:")
    for op in optimized_circuit:
        print(f"  {op}")

def test_circuit_depth_reduction():
    """Test the circuit depth reduction optimization."""
    print("\nTesting Circuit Depth Reduction")
    print("------------------------------")
    
    # Create a test circuit with operations that can be parallelized
    circuit = [
        ("X", 0),
        ("H", 1),
        ("X", 2),
        ("CNOT", 0, 1),
        ("Z", 2),
        ("H", 3),
        ("CNOT", 2, 3),
        ("X", 1)
    ]
    
    # Create the optimizer
    optimizer = QuantumCircuitOptimizations()
    
    # Apply circuit depth reduction
    optimized_circuit = optimizer.optimize_circuit(circuit, 4, ["circuit_depth_reduction"])
    
    # Print results
    original_depth = optimizer.stats["original_depth"]
    optimized_depth = optimizer.stats["optimized_depth"]
    
    print(f"Original circuit depth: {original_depth}")
    print(f"Optimized circuit depth: {optimized_depth}")
    
    print("\nOriginal circuit:")
    for op in circuit:
        print(f"  {op}")
    
    print("\nOptimized circuit:")
    for op in optimized_circuit:
        print(f"  {op}")

def test_qubit_mapping_optimization():
    """Test the qubit mapping and routing optimization."""
    print("\nTesting Qubit Mapping and Routing")
    print("--------------------------------")
    
    # Create a test circuit with two-qubit gates between non-adjacent qubits
    circuit = [
        ("H", 0),
        ("H", 3),
        ("CNOT", 0, 3),  # Non-adjacent in linear topology
        ("X", 1),
        ("CNOT", 1, 2)   # Adjacent in linear topology
    ]
    
    # Create the optimizer with linear topology
    optimizer = QuantumCircuitOptimizations()
    optimizer.hardware_params["topology"] = "linear"
    optimizer._initialize_qubit_connectivity(4)
    
    # Apply qubit mapping optimization
    optimized_circuit = optimizer.optimize_circuit(circuit, 4, ["qubit_mapping"])
    
    # Print results
    print(f"Original circuit: {len(circuit)} gates")
    print(f"Optimized circuit: {len(optimized_circuit)} gates")
    print(f"SWAP gates inserted: {optimizer.stats['swaps_inserted']}")
    
    print("\nHardware connectivity:")
    for qubit, neighbors in optimizer.hardware_params["qubit_connectivity"].items():
        print(f"  Qubit {qubit} connected to: {neighbors}")
    
    print("\nOriginal circuit:")
    for op in circuit:
        print(f"  {op}")
    
    print("\nOptimized circuit:")
    for op in optimized_circuit:
        print(f"  {op}")

def test_measurement_optimization():
    """Test the measurement optimization."""
    print("\nTesting Measurement Optimization")
    print("-------------------------------")
    
    # Create a test circuit with measurements that can be deferred
    circuit = [
        ("H", 0),
        ("CNOT", 0, 1),
        ("M", 0),  # This measurement can be deferred
        ("X", 1),
        ("M", 1),  # This measurement affects later operations
        ("CNOT", 1, 2),
        ("X", 2),
        ("M", 2)   # This measurement can be deferred
    ]
    
    # Create the optimizer
    optimizer = QuantumCircuitOptimizations()
    
    # Apply measurement optimization
    optimized_circuit = optimizer.optimize_circuit(circuit, 3, ["measurement_optimization"])
    
    # Print results
    print(f"Original circuit: {len(circuit)} gates")
    print(f"Optimized circuit: {len(optimized_circuit)} gates")
    print(f"Measurements optimized: {optimizer.stats['measurements_optimized']}")
    
    print("\nOriginal circuit:")
    for op in circuit:
        print(f"  {op}")
    
    print("\nOptimized circuit:")
    for op in optimized_circuit:
        print(f"  {op}")

def test_memory_optimization():
    """Test the quantum memory management optimization."""
    print("\nTesting Quantum Memory Management")
    print("--------------------------------")
    
    # Create a test circuit where some qubits can be reused
    circuit = [
        ("H", 0),
        ("X", 1),
        ("CNOT", 0, 1),
        ("M", 0),  # Qubit 0 is no longer used after this
        ("H", 2),  # Qubit 2 starts being used here
        ("CNOT", 1, 2),
        ("M", 1),
        ("M", 2)
    ]
    
    # Create the optimizer
    optimizer = QuantumCircuitOptimizations()
    
    # Apply memory optimization
    optimized_circuit = optimizer.optimize_circuit(circuit, 3, ["memory_management"])
    
    # Print results
    print(f"Original circuit: {len(circuit)} gates")
    print(f"Optimized circuit: {len(optimized_circuit)} gates")
    print(f"Memory optimizations: {optimizer.stats['memory_optimizations']}")
    
    print("\nOriginal circuit:")
    for op in circuit:
        print(f"  {op}")
    
    print("\nOptimized circuit:")
    for op in optimized_circuit:
        print(f"  {op}")

def test_all_optimizations():
    """Test all optimizations together."""
    print("\nTesting All Optimizations Combined")
    print("--------------------------------")
    
    # Create a complex test circuit with various optimization opportunities
    circuit = [
        # Some initial gates
        ("H", 0),
        ("H", 1),
        ("X", 2),
        ("H", 3),
        
        # Cancellable gates
        ("X", 4),
        ("X", 4),
        
        # H-Z-H pattern
        ("H", 5),
        ("Z", 5),
        ("H", 5),
        
        # Two-qubit gates
        ("CNOT", 0, 1),
        ("CNOT", 2, 3),
        
        # Non-adjacent CNOT in linear topology
        ("CNOT", 0, 3),
        
        # More gates
        ("X", 1),
        ("Z", 2),
        
        # Measurements
        ("M", 0),
        ("M", 1),
        
        # Operations after measurements
        ("H", 4),
        ("CNOT", 4, 5),
        
        # Final measurements
        ("M", 2),
        ("M", 3),
        ("M", 4),
        ("M", 5)
    ]
    
    # Create the optimizer with linear topology
    optimizer = QuantumCircuitOptimizations()
    optimizer.hardware_params["topology"] = "linear"
    optimizer._initialize_qubit_connectivity(6)
    
    # Apply all optimizations
    optimized_circuit = optimizer.optimize_circuit(circuit, 6)
    
    # Print results
    print(f"Original circuit: {len(circuit)} gates")
    print(f"Optimized circuit: {len(optimized_circuit)} gates")
    
    print("\nOptimization statistics:")
    optimizer.print_optimization_stats()
    
    print("\nOriginal circuit:")
    for op in circuit:
        print(f"  {op}")
    
    print("\nOptimized circuit:")
    for op in optimized_circuit:
        print(f"  {op}")

if __name__ == "__main__":
    # Run the tests
    test_gate_synthesis_optimization()
    test_circuit_depth_reduction()
    test_qubit_mapping_optimization()
    test_measurement_optimization()
    test_memory_optimization()
    test_all_optimizations()