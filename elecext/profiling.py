# Performance profiling utilities for parallel numerical gradients
import time
import threading
import functools
from collections import defaultdict, deque
from contextlib import contextmanager
import sys
import os
from datetime import datetime

class PerformanceProfiler:
    """Thread-safe performance profiler for diagnosing parallel execution bottlenecks."""
    
    def __init__(self, log_file="parallel_performance.log"):
        self.timings = defaultdict(list)
        self.thread_operations = defaultdict(deque)
        self.io_operations = defaultdict(int)
        self.blocking_operations = defaultdict(list)
        self.lock = threading.Lock()
        self.start_time = time.perf_counter()
        self.log_file = log_file
        
        # Track thread creation and lifecycle
        self.thread_lifecycle = {}
        self.active_threads = set()
        
        # Initialize log file
        self._log(f"=== PERFORMANCE PROFILING SESSION STARTED ===")
        self._log(f"Session ID: {datetime.now().isoformat()}")
        self._log(f"Python GIL Status: Active (CPython)")
        self._log("=" * 60)
    
    def _log(self, message):
        """Thread-safe logging to file."""
        timestamp = time.perf_counter() - self.start_time
        thread_name = threading.current_thread().name
        thread_id = threading.current_thread().ident
        
        log_entry = f"[{timestamp:8.3f}s] [{thread_name:>12}:{thread_id:>6}] {message}"
        
        # Write to file immediately for real-time monitoring
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
            f.flush()  # Force immediate write
    
    @contextmanager
    def measure_operation(self, operation_name, track_blocking=True):
        """Context manager to measure operation timing and detect blocking."""
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        start_time = time.perf_counter()
        
        # Track operation start
        with self.lock:
            self.active_threads.add(thread_id)
            self.thread_operations[thread_id].append(f"START: {operation_name}")
        
        self._log(f"🚀 START: {operation_name}")
        
        try:
            yield
        except Exception as e:
            self._log(f"💥 ERROR in {operation_name}: {str(e)}")
            raise
        finally:
            duration = time.perf_counter() - start_time
            
            # Record timing
            with self.lock:
                self.timings[operation_name].append(duration)
                self.thread_operations[thread_id].append(f"END: {operation_name} ({duration:.3f}s)")
                
                # Detect potentially blocking operations
                if track_blocking and duration > 1.0:
                    self.blocking_operations[operation_name].append({
                        'duration': duration,
                        'thread': thread_name,
                        'timestamp': time.perf_counter() - self.start_time
                    })
            
            # Log completion with performance flags
            flag = ""
            if duration > 5.0:
                flag = " 🐌 VERY SLOW"
            elif duration > 2.0:
                flag = " ⏰ SLOW"
            elif duration > 1.0:
                flag = " ⚠️  MODERATE"
            
            self._log(f"✅ DONE: {operation_name} ({duration:.3f}s){flag}")
    
    def track_thread_start(self, thread_name):
        """Track when a new thread starts."""
        thread_id = threading.current_thread().ident
        with self.lock:
            self.thread_lifecycle[thread_id] = {
                'name': thread_name,
                'start_time': time.perf_counter() - self.start_time,
                'operations': []
            }
        self._log(f"🧵 THREAD STARTED: {thread_name}")
    
    def track_thread_end(self, thread_name):
        """Track when a thread ends."""
        thread_id = threading.current_thread().ident
        end_time = time.perf_counter() - self.start_time
        
        with self.lock:
            if thread_id in self.thread_lifecycle:
                duration = end_time - self.thread_lifecycle[thread_id]['start_time']
                self.thread_lifecycle[thread_id]['end_time'] = end_time
                self.thread_lifecycle[thread_id]['total_duration'] = duration
        
        self._log(f"🏁 THREAD ENDED: {thread_name} (total: {duration:.3f}s)")
    
    def track_io_operation(self, operation_type, file_path=None):
        """Track I/O operations for contention analysis."""
        with self.lock:
            self.io_operations[operation_type] += 1
        
        file_info = f" -> {os.path.basename(file_path)}" if file_path else ""
        self._log(f"💾 I/O: {operation_type}{file_info}")
    
    def detect_thread_contention(self):
        """Analyze current thread contention."""
        with self.lock:
            active_count = len(self.active_threads)
            total_threads = threading.active_count()
        
        if active_count > 2:
            self._log(f"⚡ CONTENTION: {active_count} threads actively running (total: {total_threads})")
            return True
        return False
    
    def print_summary(self):
        """Generate comprehensive performance summary."""
        total_session_time = time.perf_counter() - self.start_time
        
        self._log("\n" + "=" * 80)
        self._log("COMPREHENSIVE PERFORMANCE ANALYSIS")
        self._log("=" * 80)
        self._log(f"Total session time: {total_session_time:.2f}s")
        self._log(f"Active threads peak: {len(self.thread_lifecycle)}")
        
        # Analyze operation timings
        self._log("\n📊 OPERATION TIMINGS (sorted by total impact):")
        operation_analysis = []
        
        for op_name, times in self.timings.items():
            if times:
                total_time = sum(times)
                avg_time = total_time / len(times)
                max_time = max(times)
                min_time = min(times)
                count = len(times)
                
                operation_analysis.append({
                    'name': op_name,
                    'total': total_time,
                    'avg': avg_time,
                    'max': max_time,
                    'min': min_time,
                    'count': count,
                    'impact': total_time  # For sorting
                })
        
        # Sort by total impact
        operation_analysis.sort(key=lambda x: x['impact'], reverse=True)
        
        for i, op in enumerate(operation_analysis[:15]):  # Top 15 operations
            self._log(f"  {i+1:2d}. {op['name']:<30} | "
                     f"Total: {op['total']:6.2f}s | "
                     f"Avg: {op['avg']:6.3f}s | "
                     f"Max: {op['max']:6.3f}s | "
                     f"Count: {op['count']:3d}")
        
        # Analyze blocking operations
        if self.blocking_operations:
            self._log(f"\n🚫 BLOCKING OPERATIONS (>1.0s):")
            for op_name, blocks in self.blocking_operations.items():
                total_blocking = sum(b['duration'] for b in blocks)
                max_block = max(b['duration'] for b in blocks)
                self._log(f"  {op_name}: {len(blocks)} blocks, "
                         f"total: {total_blocking:.2f}s, max: {max_block:.2f}s")
        
        # I/O Analysis
        if self.io_operations:
            self._log(f"\n💾 I/O OPERATIONS:")
            total_io = sum(self.io_operations.values())
            for op_type, count in sorted(self.io_operations.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_io) * 100 if total_io > 0 else 0
                self._log(f"  {op_type:<20}: {count:4d} operations ({percentage:5.1f}%)")
        
        # Thread Analysis
        self._log(f"\n🧵 THREAD ANALYSIS:")
        for thread_id, info in self.thread_lifecycle.items():
            if 'total_duration' in info:
                self._log(f"  {info['name']:<15}: {info['total_duration']:6.2f}s")
        
        self._log("=" * 80)
        self._log("💡 PERFORMANCE RECOMMENDATIONS:")
        
        # Generate recommendations based on analysis
        recommendations = []
        
        # Check for I/O heavy operations
        if any(op['avg'] > 0.5 for op in operation_analysis if 'read' in op['name'].lower() or 'write' in op['name'].lower()):
            recommendations.append("• Consider I/O buffering for file operations")
        
        # Check for many small operations
        high_frequency_ops = [op for op in operation_analysis if op['count'] > 10 and op['avg'] < 0.1]
        if high_frequency_ops:
            recommendations.append("• Consider batching small frequent operations")
        
        # Check for thread contention indicators
        if len(self.thread_lifecycle) > 4:
            recommendations.append("• High thread count detected - consider ProcessPoolExecutor")
        
        # Check for blocking operations
        if self.blocking_operations:
            recommendations.append("• Blocking operations detected - consider async alternatives")
        
        if not recommendations:
            recommendations.append("• No obvious bottlenecks detected - performance seems optimal")
        
        for rec in recommendations:
            self._log(rec)
        
        self._log("=" * 80)

# Global profiler instance
profiler = PerformanceProfiler()

def profile_function(operation_name=None, track_blocking=True):
    """Decorator to profile function execution time."""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.measure_operation(operation_name, track_blocking):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def profile_io(operation_type):
    """Decorator to track I/O operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract file path from args
            file_path = None
            if args and isinstance(args[0], str) and ('.' in args[0] or '/' in args[0]):
                file_path = args[0]
            
            profiler.track_io_operation(operation_type, file_path)
            return func(*args, **kwargs)
        return wrapper
    return decorator