import time
import os
from collections import defaultdict

class Profiler:
    """Simple profiling utility to track execution times of operations"""
    
    def __init__(self):
        self.operation_times = defaultdict(float)
        self.operation_counts = defaultdict(int)
        self.start_times = {}
        
    def start(self, operation):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        
    def end(self, operation):
        """End timing an operation and record the elapsed time"""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            self.operation_times[operation] += elapsed
            self.operation_counts[operation] += 1
            del self.start_times[operation]
            return elapsed
        return 0
        
    def __str__(self):
        """Generate a summary report of all operations"""
        output = ["\n=== Profiling Report ==="]
        
        # Sort operations by total time (descending)
        sorted_ops = sorted(
            self.operation_times.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Calculate total tracked time
        total_time = sum(self.operation_times.values())
        
        # Print header
        output.append(f"{'Operation':<30} {'Count':<10} {'Total Time':<15} {'Avg Time':<15} {'%':<10}")
        output.append("-" * 80)
        
        # Add each operation
        for op, total in sorted_ops:
            count = self.operation_counts[op]
            avg = total / count if count > 0 else 0
            percentage = (total / total_time) * 100 if total_time > 0 else 0
            
            output.append(
                f"{op:<30} {count:<10} {total:.4f}s {avg:.4f}s {percentage:.2f}%"
            )
            
        return "\n".join(output)

# Global profiler instance
profiler = Profiler()
