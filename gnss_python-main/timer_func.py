import time
from functools import wraps

_total_times = {}
_call_counts = {}

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        
        if func.__name__ not in _total_times:
            _total_times[func.__name__] = 0
            _call_counts[func.__name__] = 0
        
        _total_times[func.__name__] += elapsed_time
        _call_counts[func.__name__] += 1
        
        return result
    return wrapper

def print_timing_stats():
    print("\n" + "="*50)
    print("FUNCTION TIMING SUMMARY")
    print("="*50)
    for func_name in sorted(_total_times.keys()):
        total = _total_times[func_name]
        calls = _call_counts[func_name]
        avg = total / calls
        print(f"{func_name}: {calls} calls, {total:.4f}s total, {avg:.6f}s avg")
    print("="*50)