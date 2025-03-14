#!/usr/bin/env python3
"""
Run Specific Benchmarks for Quantum Circuit Optimizations

This script runs the specific benchmarks for each optimization technique
and generates a detailed report on the results.
"""

import os
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from specific_benchmarks import SpecificBenchmarks

def create_output_directory():
    """Create an output directory for benchmark results."""
    output_dir = "specific_benchmark_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Create a timestamped subdirectory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir)
    
    return run_dir

def generate_report(results, output_dir):
    """Generate a detailed report of the benchmark results."""
    # Save raw results
    with open(os.path.join(output_dir, "benchmark_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    with open(os.path.join(output_dir, "summary_report.txt"), 'w') as f:
        f.write("Quantum Circuit Optimization Specific Benchmarks Summary\n")
        f.write("====================================================\n\n")
        
        all_correct = True
        for technique, result in results.items():
            is_correct = result.get("is_correct", False)
            all_correct = all_correct and is_correct
            
            f.write(f"\n{technique.replace('_', ' ').title()}:\n")
            f.write(f"  Correct optimization: {'Yes' if is_correct else 'No'}\n")
            
            # Write technique-specific metrics
            if "gates_removed" in result:
                f.write(f"  Gates removed: {result['gates_removed']}\n")
            if "gates_replaced" in result:
                f.write(f"  Gates replaced: {result['gates_replaced']}\n")
            if "swaps_inserted" in result:
                f.write(f"  SWAP gates inserted: {result['swaps_inserted']}\n")
            if "measurements_optimized" in result:
                f.write(f"  Measurements optimized: {result['measurements_optimized']}\n")
            if "memory_optimizations" in result:
                f.write(f"  Memory optimizations: {result['memory_optimizations']}\n")
            if "original_depth" in result and "optimized_depth" in result:
                f.write(f"  Original depth: {result['original_depth']}\n")
                f.write(f"  Optimized depth: {result['optimized_depth']}\n")
                f.write(f"  Expected depth: {result.get('expected_depth', 'N/A')}\n")
                f.write(f"  Depth reduction: {1 - result['optimized_depth']/result['original_depth']:.2%}\n")
            
            # Write circuit details
            f.write("\n  Original circuit:\n")
            for gate in result["original_circuit"]:
                f.write(f"    {gate}\n")
            
            f.write("\n  Optimized circuit:\n")
            for gate in result["optimized_circuit"]:
                f.write(f"    {gate}\n")
            
            if "expected_circuit" in result:
                f.write("\n  Expected circuit:\n")
                for gate in result["expected_circuit"]:
                    f.write(f"    {gate}\n")
        
        f.write("\n\nOverall Assessment:\n")
        f.write("------------------\n")
        if all_correct:
            f.write("All optimization techniques are working correctly.\n")
        else:
            f.write("Some optimization techniques need adjustment:\n")
            for technique, result in results.items():
                if not result.get("is_correct", False):
                    f.write(f"  - {technique.replace('_', ' ').title()}\n")
    
    # Generate visualization
    generate_visualization(results, output_dir)

def generate_visualization(results, output_dir):
    """Generate visualizations of the benchmark results."""
    # Create a bar chart showing which optimizations passed/failed
    techniques = list(results.keys())
    correctness = [1 if results[t].get("is_correct", False) else 0 for t in techniques]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        [t.replace('_', ' ').title() for t in techniques],
        correctness,
        color=['green' if c else 'red' for c in correctness]
    )
    
    # Add labels on top of bars
    for bar, correct in zip(bars, correctness):
        label = "✓" if correct else "✗"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            label,
            ha='center',
            fontsize=14,
            fontweight='bold'
        )
    
    plt.ylim(0, 1.2)
    plt.title('Optimization Technique Correctness')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_correctness.png"))
    plt.close()
    
    # Create a detailed comparison for each optimization
    for technique, result in results.items():
        if "original_circuit" in result and "optimized_circuit" in result:
            plt.figure(figsize=(12, 6))
            
            # Plot gate counts
            original_count = len(result["original_circuit"])
            optimized_count = len(result["optimized_circuit"])
            expected_count = len(result.get("expected_circuit", []))
            
            labels = ['Original', 'Optimized']
            counts = [original_count, optimized_count]
            colors = ['blue', 'green' if result.get("is_correct", False) else 'red']
            
            if "expected_circuit" in result:
                labels.append('Expected')
                counts.append(expected_count)
                colors.append('orange')
            
            plt.bar(labels, counts, color=colors)
            
            # Add count labels
            for i, count in enumerate(counts):
                plt.text(i, count + 0.1, str(count), ha='center')
            
            plt.title(f'{technique.replace("_", " ").title()} - Gate Count Comparison')
            plt.ylabel('Number of Gates')
            plt.savefig(os.path.join(output_dir, f"{technique}_comparison.png"))
            plt.close()

def run_specific_benchmarks():
    """Run the specific benchmarks and generate a report."""
    print("\n=== Running Specific Benchmarks for Quantum Circuit Optimizations ===\n")
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Run benchmarks
    start_time = time.time()
    benchmarks = SpecificBenchmarks()
    results = benchmarks.run_all_benchmarks()
    end_time = time.time()
    
    # Generate report
    generate_report(results, output_dir)
    
    # Print summary
    all_correct = all(result.get("is_correct", False) for result in results.values())
    
    print("\n=== Benchmark Summary ===\n")
    for technique, result in results.items():
        is_correct = result.get("is_correct", False)
        print(f"{technique.replace('_', ' ').title()}: {'✓' if is_correct else '✗'}")
    
    print(f"\nOverall: {'All optimizations working correctly' if all_correct else 'Some optimizations need adjustment'}")
    print(f"\nBenchmarks completed in {end_time - start_time:.2f} seconds")
    print(f"Results saved to {output_dir}")
    
    return results, all_correct

if __name__ == "__main__":
    run_specific_benchmarks()