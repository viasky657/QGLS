# Quantum Geometry Learning Systems (QGLS)

![image](https://github.com/user-attachments/assets/2c6125d3-aea6-4795-b7cf-b39821206895)


**Quantum Geometry Learning Systems (QGLS)** is a novel AI architecture that brings the principles of quantum mechanics into classical deep learning models through topological and entanglement-inspired design. Developed by **Moonshot Labs**, QGLS is fully open-source and released under the **MIT License**.

---

## 🌌 Overview
QGLS models intelligence as an emergent phenomenon of **geometry, resonance, and entanglement**, using knot-based structures, interference patterns, and wave-driven propagation. This architecture is a step toward topological intelligence systems shaped by the physics of quantum behavior—without the need for quantum hardware.

---

## 🔬 Key Features
- **Entangled Connection Layer:** Simulates interference using entanglement coefficients (ε), resonance phases (ϕ), and knot tension (τ).
- **Topological Network Structure:** Nodes are organized in trefoil or figure-eight knots to shape signal flow.
- **Wave-Based Propagation:** Information moves non-linearly across entangled paths.
- **Collapse Resolution Layer:** Resolves signal superposition using entropy, energy, or tension-based collapse mechanisms.
- **Resonance Loss Function:** Penalizes disharmonic phase interference to encourage coherent learning.
- **Dataset Adapter:** Maps classical input data onto the knot structure.


---

## 📈 Results
QGLS has shown competitive performance on Fashion MNIST with enhanced learning dynamics, high coherence, and smooth generalization under noise.

---

## 📜 License
This project is released under the **MIT License**.

---

## 🤝 Credits
QGLS is a research project by **Moonshot Labs**, founded on the principle that AI should evolve through the shape of nature’s laws.


---

Research Paper: https://docs.google.com/document/d/1mZzgz7C_R4kewDWzKwi-aLm9-3jrW4ZVs1sSB0e76jg/edit?usp=sharing 

## ⭐ Support the Project
If this research inspires you, consider starring the repo or contributing ideas. Let's reshape AI together.

> "Intelligence is not just learned — it’s shaped."

# "True" Quantum encoding and Transformer Network Learning added by changing the original code (This is experimental and may not work as intended; Use with Caution!)

How the Tensor Neural Network Learns and Trains in app.py
The app.py file implements an advanced neural network architecture called the Entanglement-Driven Topological Neural Network (EDTNN), which incorporates quantum computing concepts into a classical neural network framework. Here's how it learns and trains:

# Core Architecture
The EDTNN model (defined around line 1779) is built on a tensor-based architecture that uses quantum-inspired concepts:

# Quantum-Inspired Layers: The network uses several specialized layers:

QuantumGateLayer: Maps classical data to quantum states, applies quantum operations, and maps back to classical outputs
EntangledConnectionLayer: Implements connections with entanglement coefficients and resonance phases
EntanglementPropagator: Propagates information across entangled paths instead of layer-by-layer
CollapseResolutionLayer: Interprets multi-path propagation into a singular signal
Topological Structure: The network is organized around a 3D knot structure (generated by TopologyGenerator) that defines how information flows through the network.

# Training Process
The training process is handled by the TrainerEngine class (line 4472) and involves:

# Forward Pass (Inference):

Input data is mapped to quantum parameters through an encoder
These parameters are processed through quantum-inspired layers
Information propagates through the entangled topology
The CollapseResolutionLayer resolves the multi-path propagation into final outputs
Loss Calculation:

The network uses a specialized ResonanceLoss (line 1582) that combines:
A standard loss function (e.g., CrossEntropyLoss)
A resonance component that penalizes disharmony in signal propagation
Additionally, a TopologicalRegularizer (line 1644) encourages conservation of knot topology
Backpropagation:

Gradients flow backward through the network
The optimizer updates weights to minimize the combined loss
The quantum-inspired parameters are updated to improve performance
Training Loop:

The train_epoch method processes batches of data
For each batch, it performs forward pass, loss calculation, and backpropagation
It tracks statistics like loss and accuracy
Quantum-Inspired Learning Mechanisms
The network incorporates several quantum-inspired learning mechanisms:

# Superposition:

The QuantumGateLayer encodes classical data into quantum-like states
Parameters like rx_params, ry_params, and rz_params represent rotation angles for quantum gates
This allows the network to explore multiple states simultaneously
Entanglement:

The EntangledConnectionLayer models quantum entanglement through:
Entanglement coefficients (ε) that determine connection strength
Resonance phase (ϕ) that models interference effects
Knot tension (τ) that affects signal transmission
Wave-Based Propagation:

The EntanglementPropagator uses phase factors for wave-like information propagation
This creates interference patterns that affect how information flows
Collapse Resolution:

The CollapseResolutionLayer resolves the multi-path propagation using different methods:
Entropy-based collapse: focuses on most uncertain nodes
Energy-based collapse: weights by energy distribution
Tension-based collapse: minimizes topological strain
Advanced Implementation: QuantumEDTNN
The file also includes a more advanced implementation called QuantumEDTNN (line 1888) that:

Uses true qubit representation and quantum gates for computation

# Implements a hybrid quantum-classical neural network with:

Quantum encoding of classical data
Parameterized quantum circuits (PQCs) for quantum processing
Quantum measurement and classical post-processing
Topological structure for enhanced information propagation
Includes comprehensive error mitigation techniques for large qubit systems:

Zero-noise extrapolation
Readout error mitigation
Dynamical decoupling
Error-aware circuit optimization
Training Optimization
The training process is optimized through:

# Resonance-Based Learning:

The ResonanceLoss encourages harmonious signal propagation
Phase differences between connected nodes are penalized to maintain coherence

# Topological Regularization:

The TopologicalRegularizer preserves the knot structure during training
This prevents the network from distorting its topology too drastically
Parallel Processing:

For large qubit systems, the model implements distributed processing
The parallel_quantum_processing_pipeline method orchestrates multiple parallelization techniques




# Quantum Circuit Optimization

This project implements advanced quantum circuit optimization techniques that can be individually enabled or disabled for benchmarking purposes.

## Optimization Techniques

The system includes the following optimization categories:

1. **Gate Synthesis and Decomposition**
   - Optimizes the decomposition of complex gates into simpler ones
   - Uses more efficient gate sequences for common operations
   - Combines consecutive rotation gates

2. **Circuit Depth Reduction**
   - Minimizes the depth of quantum circuits to reduce decoherence effects
   - Rearranges gates to maximize parallelism
   - Identifies and merges layers of gates that can be executed simultaneously

3. **Qubit Mapping and Routing**
   - Optimizes the mapping of logical qubits to physical qubits
   - Minimizes SWAP operations needed for connectivity constraints
   - Handles different hardware topologies (linear, grid, etc.)

4. **Measurement-Based Optimizations**
   - Defers measurements to the end of the circuit when possible
   - Removes unnecessary measurements
   - Uses measurement results to simplify subsequent operations

5. **Advanced Compiler Techniques**
   - Implements peephole optimizations for common patterns
   - Applies constant folding for known inputs
   - Uses pattern-based optimizations

6. **Hardware-Specific Optimizations**
   - Tailors the circuit to specific quantum hardware characteristics
   - Exploits native gate sets more efficiently
   - Adapts to hardware-specific constraints and capabilities

7. **Quantum Memory Management**
   - Optimizes the allocation and deallocation of qubits
   - Reuses qubits when possible
   - Implements efficient qubit allocation strategies

## Usage

You can apply these optimizations to your quantum circuits using the `QuantumRegister` class:

```python
# Create a quantum register
qreg = QuantumRegister(num_qubits=4)

# Create a quantum circuit
circuit = [
    (Qubit.H_GATE, 0),  # Hadamard on qubit 0
    (Qubit.X_GATE, 1),  # X gate on qubit 1
    (Qubit.CNOT_GATE, 0, 1),  # CNOT with control 0, target 1
    ('M', 0),  # Measure qubit 0
    ('M', 1)   # Measure qubit 1
]

# Apply all optimizations
optimized_circuit = qreg.apply_advanced_optimizations(circuit)

# Apply specific optimizations
optimized_circuit = qreg.apply_advanced_optimizations(
    circuit, 
    techniques=["gate_synthesis", "circuit_depth_reduction"]
)
```

You can also use the `QuantumCircuitOptimizations` class directly for more control:

```python
from quantum_circuit_optimizations import QuantumCircuitOptimizations

# Create the optimizer
optimizer = QuantumCircuitOptimizations()

# Apply specific optimizations
optimized_circuit = optimizer.optimize_circuit(
    circuit, 
    num_qubits=4,
    techniques=["gate_synthesis", "qubit_mapping"]
)

# Get optimization statistics
optimizer.print_optimization_stats()
```

## Testing

You can test the optimization techniques using the provided test script:

```bash
python test_optimizer.py
```

This will run each optimization technique individually and show the results.


# Benchmarking for Quantum Circuit Optimizations:

I've implemented specific benchmarks for testing the quantum circuit optimizations as requested. The implementation includes:

A specific_benchmarks.py file that contains targeted test circuits for each of the seven optimization techniques:

Gate Synthesis: Tests cancellation of consecutive gates, H-Z-H pattern replacement, and rotation combining
Circuit Depth Reduction: Tests parallelization of independent operations
Qubit Mapping: Tests SWAP insertion for non-adjacent qubits in linear topology
Measurement Optimization: Tests deferral of measurements not used in conditionals
Compiler Optimization: Tests cancellation of gate pairs like H-H, X-X, CNOT-CNOT
Hardware-Specific Optimization: Tests decomposition of S and T gates to native gates
Memory Management: Tests qubit reuse after measurement
A run_specific_benchmarks.py script that executes all benchmarks and generates comprehensive reports with:

Detailed text summaries of each optimization's performance
Visualizations comparing original and optimized circuits
Clear pass/fail indicators for each technique
These benchmarks will accurately determine if the optimizations are working as intended or need adjustment. Each test is designed to isolate a specific optimization technique and verify its correct behavior against expected outcomes.

To test the optimizations, simply run:

python run_specific_benchmarks.py
The results will be saved to a timestamped directory under specific_benchmark_results/ for easy reference and comparison between runs.

# Quantum Parallization Techniques for Large Qubit Counts

I've implemented advanced parallel processing enhancements to the quantum computing implementation in app.py. The improvements include:

Added thread management with automatic and manual thread allocation options via the new num_threads parameter in key methods:

implement_grovers_search
_parallel_oracle
_parallel_diffusion
apply_hadamard_all
create_superposition
Implemented multi-level parallelism for extremely large qubit systems (>30 qubits) through the new _implement_advanced_parallel_grovers method that divides qubits into groups that can be processed in parallel.

Added adaptive batch processing with dynamic batch size adjustment that monitors execution time and automatically adjusts batch sizes to optimize performance.

Implemented cache-optimized memory access patterns in the new _advanced_parallel_oracle and _advanced_parallel_diffusion methods for better CPU cache utilization.

Added specialized parallel matrix multiplication for medium-sized systems (10-15 qubits) through the new _parallel_matrix_grovers method.

Implemented performance tracking that monitors execution times of oracle and diffusion operations to optimize resource allocation during execution.

These enhancements make the quantum simulation more efficient, allowing it to handle larger qubit systems with better performance. The code now scales more effectively with increasing qubit counts by automatically detecting optimal thread count, using hierarchical parallelism for very large systems, and dynamically adjusting processing parameters based on performance metrics.

These enhancements would allow the model to run at 50 qubits but it is calculated in hierarchical format that calculates one qubit at a time and calculates all neccessary entangelment, error correction, etcetera, in a row in a vector-based format. The qubit calculations are corrected and formatted into vector representation after all qubit calculations so there may be some simplification of the quantum states so it isn't entirely 100% accurate quantum simulation (which is why that I put air quotes next to the "true" quantum title). A true quantum system would use matrix representations instead of vector and would more precisely compute the quantum computatoins. However, it is a really close approximation. 

# Training Time and Hardware Requirements

# GPUs: 2 - 4 Recommended (Parallel Processing over multiple GPUS allows the training to take a couple of hours with just 2 GPUs instead of 2 days)

# 4X A100s GPUs Recommended for best experience

# GPU Requirements According to GROK

To determine the GPU make and count required to run the `app.py` file with 50 qubits, all optimizations enabled, and train on a million characters of data within two days, we need to assess the computational and memory demands of the `QuantumEDTNN` model as implemented in the provided code. The goal is to ensure the training completes in 48 hours (172,800 seconds). Below, we analyze the problem step-by-step, leveraging the code's features and making reasonable assumptions based on typical deep learning and quantum simulation workloads.

---

### Understanding the Problem

- **50 Qubits**: The model uses 50 qubits, but the `large_qubit_mode` optimization caps the effective simulation at 20 qubits, using a sparse representation for efficiency. This significantly reduces the computational burden from an exponential \(2^{50}\) state vector (approximately 18 petabytes) to a manageable subset.
- **All Optimizations On**: The code includes optimizations like sparse quantum representations, parallel processing, distributed computing, and error mitigation techniques (e.g., Zero-Noise Extrapolation, Dynamical Decoupling). These reduce memory and compute requirements while enabling multi-GPU parallelism.
- **Million Characters of Data**: We interpret this as a dataset of 1,000,000 samples (e.g., images or text encoded into an input shape like [28,28] for MNIST-like data). Training on this dataset in two days constrains the required compute power.
- **Two Days**: The total time is 172,800 seconds, driving the need for sufficient GPU resources to process the workload efficiently.

---

### Model Architecture and Memory Requirements

The `QuantumEDTNN` model integrates classical neural networks with quantum-inspired layers. Key components include:

1. **Input Encoder**:
   - Layers: \(784 \to 512 \to 256 \to 100\) (where 100 = 50 qubits * 2 parameters per qubit).
   - Parameters: \((784 \times 512 + 512) + (512 \times 256 + 256) + (256 \times 100 + 100) = 401,408 + 131,328 + 25,700 = 558,436\).

2. **Superposition Layer**:
   - Parameters: 50 (one per qubit).

3. **Output Decoder**:
   - Layers: \(50 \to 128 \to 10\).
   - Parameters: \((50 \times 128 + 128) + (128 \times 10 + 10) = 6,528 + 1,290 = 7,818\).

4. **Total Parameters**:
   - \(558,436 + 50 + 7,818 = 566,304\).
   - Memory (float32, 4 bytes each): \(566,304 \times 4 \approx 2.26 \, \text{MB}\).

During training, additional memory is needed for:
- **Gradients and Optimizer States**: Using Adam, this typically triples the parameter memory (approximately \(2.26 \times 3 = 6.78 \, \text{MB}\)).
- **Activations**: For a batch size of 32 and input shape [28,28]:
  - Input data: \(32 \times 784 \times 4 \approx 0.1 \, \text{MB}\).
  - Encoder activations: e.g., \(32 \times 512 \times 4 \approx 0.065 \, \text{MB}\), \(32 \times 256 \times 4 \approx 0.032 \, \text{MB}\).
- **Quantum Layer**: With `large_qubit_mode`, it simulates 20 effective qubits using sparse representations and batch processing (`batch_process_quantum_states`), keeping memory usage low.

Total memory per GPU is conservatively under 16 GB, fitting within modern GPUs like the NVIDIA A100 (40 GB) or V100 (16 GB).

---

### Computational Requirements

#### Operations per Batch
For a batch size of 32:
- **Input Encoder**:
  - \(784 \to 512\): \(32 \times 512 \times (2 \times 784 - 1) \approx 25.7 \, \text{MFLOPs}\).
  - \(512 \to 256\): \(32 \times 256 \times (2 \times 512 - 1) \approx 8.4 \, \text{MFLOPs}\).
  - \(256 \to 100\): \(32 \times 100 \times (2 \times 256 - 1) \approx 1.6 \, \text{MFLOPs}\).
  - Total: \(\approx 35.7 \, \text{MFLOPs}\).
- **Output Decoder**:
  - \(50 \to 128\): \(32 \times 128 \times (2 \times 50 - 1) \approx 0.4 \, \text{MFLOPs}\).
  - \(128 \to 10\): \(32 \times 10 \times (2 \times 128 - 1) \approx 0.08 \, \text{MFLOPs}\).
  - Total: \(\approx 0.5 \, \text{MFLOPs}\).
- **Quantum Layer**: Approximated with 20 effective qubits, assume a complexity of \(32 \times 20 \times 1000 \approx 0.64 \, \text{MFLOPs}\) (adjustable based on implementation).

**Forward Pass**: \(\approx 35.7 + 0.5 + 0.64 = 36.84 \, \text{MFLOPs}\).

**Training Pass**: Backward pass typically triples FLOPs, so \(\approx 3 \times 36.84 = 110.52 \, \text{MFLOPs per batch}\).

#### Total Compute
- **Batches per Epoch**: \(1,000,000 / 32 = 31,250\).
- **Epochs**: Assume 10 epochs (typical for convergence), total batches = \(31,250 \times 10 = 312,500\).
- **Total FLOPs**: \(312,500 \times 110.52 \times 10^6 \approx 3.45 \times 10^{13}\).
- **Required FLOPS**: \(3.45 \times 10^{13} / 172,800 \approx 2 \times 10^8 \, \text{FLOPS} = 200 \, \text{MFLOPS}\).

However, the quantum layer’s approximation may underestimate compute needs. For 50 qubits with optimizations (e.g., distributed processing), let’s assume a higher complexity, say 10x, adjusting total FLOPs to \(\approx 3.45 \times 10^{14}\), requiring \(\approx 2 \, \text{GFLOPS}\).

#### GPU Performance
An NVIDIA A100 offers 19.5 TFLOPS (FP32), or \(1.95 \times 10^{13} \, \text{FLOPS}\). One A100 could complete \(3.45 \times 10^{14} / 1.95 \times 10^{13} \approx 17.7 \, \text{seconds}\), far under two days. However, practical overheads (data loading, synchronization) and quantum simulation complexities suggest a multi-GPU setup for robustness.

---

### Multi-GPU Optimization

The code supports:
- **Data Parallelism**: Via `batch_parallel_processing`.
- **Model Parallelism**: Via `distributed_forward`, splitting quantum computations across `num_partitions`.
- **Demonstration**: Shows up to 8 partitions, implying scalability with GPU count.

For 50 qubits, `large_qubit_mode` reduces simulation to 20 qubits, but distributed processing across multiple GPUs accelerates training. With 31,250 batches per epoch, 10 epochs, and assuming 0.1 seconds per batch on one GPU (8.7 hours total), 4 GPUs could reduce this to \(\approx 2.2 \, \text{hours}\), well within two days, accounting for overheads.

---

### Recommendation

Considering:
- **Memory**: Fits within 16 GB per GPU.
- **Compute**: Enhanced by parallelization, needing 2-4 GPUs for speed and reliability.
- **Code Features**: Optimized for NVIDIA GPUs (PyTorch, CUDA).

We recommend **4 NVIDIA A100 GPUs**. This ensures training completes within two days, leveraging distributed processing and providing margin for unmodeled overheads or higher computational demands.

--- 

**Final Answer**: 4 NVIDIA A100 GPUs