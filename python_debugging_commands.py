# Python Debugging Commands and Techniques

# PRINT DEBUGGING - BASIC OUTPUT
print("Debug: Variable value is", variable)        # Basic print statement
print(f"Debug: x={x}, y={y}")                     # F-string formatting
print("Debug:", variable, type(variable))          # Print with type
print("Debug: Function called with args:", args)   # Function debugging

# PRINT WITH SEPARATORS AND END CHARACTERS
print("Value1", "Value2", "Value3", sep=" | ")    # Custom separator
print("Processing...", end=" ")                   # No newline
print("Done!")                                    # Continues on same line

# PRETTY PRINTING
import pprint
data = {'users': [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]}
pprint.pprint(data)                               # Pretty print complex data
pprint.pprint(data, width=40, depth=2)           # With formatting options

# INSPECT MODULE - OBJECT INTROSPECTION
import inspect

def debug_function(obj):
    """Debug function to inspect objects"""
    print(f"Type: {type(obj)}")
    print(f"Dir: {dir(obj)}")
    if hasattr(obj, '__dict__'):
        print(f"Attributes: {obj.__dict__}")
    if inspect.isfunction(obj):
        print(f"Signature: {inspect.signature(obj)}")
        print(f"Source file: {inspect.getfile(obj)}")

# TRACEBACK MODULE - EXCEPTION INFORMATION
import traceback
import sys

def debug_exception():
    """Function to demonstrate exception debugging"""
    try:
        result = 10 / 0
    except Exception as e:
        print("Exception occurred:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()                     # Print full traceback
        
        # Get traceback as string
        tb_str = traceback.format_exc()
        print("Traceback as string:", tb_str)

# LOGGING MODULE - STRUCTURED DEBUGGING
import logging

# Basic logging setup
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Logging levels
logging.debug("Debug message - detailed information")
logging.info("Info message - general information")
logging.warning("Warning message - something unexpected")
logging.error("Error message - serious problem")
logging.critical("Critical message - program may abort")

# Custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# PDB DEBUGGER - INTERACTIVE DEBUGGING
import pdb

def buggy_function(x, y):
    """Function with a bug for debugging"""
    pdb.set_trace()                              # Set breakpoint
    result = x + y
    if result > 10:
        result = result * 2
    return result

# PDB COMMANDS (used in interactive debugger)
"""
n (next)        - Execute next line
s (step)        - Step into function calls
c (continue)    - Continue execution
l (list)        - Show current code
p <var>         - Print variable value
pp <var>        - Pretty print variable
w (where)       - Show current stack trace
u (up)          - Move up in stack
d (down)        - Move down in stack
b <line>        - Set breakpoint at line
cl <num>        - Clear breakpoint
q (quit)        - Quit debugger
h (help)        - Show help
"""

# BREAKPOINT() FUNCTION (Python 3.7+)
def modern_debugging():
    """Using the built-in breakpoint() function"""
    x = 10
    y = 20
    breakpoint()                                 # Modern way to set breakpoint
    result = x + y
    return result

# ASSERT STATEMENTS - DEBUGGING ASSUMPTIONS
def validate_input(value):
    """Function using assertions for debugging"""
    assert isinstance(value, int), f"Expected int, got {type(value)}"
    assert value > 0, f"Expected positive value, got {value}"
    assert value < 100, f"Expected value < 100, got {value}"
    return value * 2

# WARNINGS MODULE - NON-FATAL ISSUES
import warnings

def deprecated_function():
    """Function that issues a deprecation warning"""
    warnings.warn("This function is deprecated", DeprecationWarning)
    return "old functionality"

# Custom warning
warnings.warn("Custom warning message", UserWarning)

# Filter warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# TIMEIT MODULE - PERFORMANCE DEBUGGING
import timeit

# Time a simple statement
time_taken = timeit.timeit('sum([1, 2, 3, 4, 5])', number=100000)
print(f"Time taken: {time_taken}")

# Time a function
def test_function():
    return sum(range(100))

time_taken = timeit.timeit(test_function, number=10000)
print(f"Function time: {time_taken}")

# PROFILE MODULE - CODE PROFILING
import cProfile
import pstats

def profile_example():
    """Function to demonstrate profiling"""
    total = 0
    for i in range(1000000):
        total += i
    return total

# Profile the function
cProfile.run('profile_example()', 'profile_stats')

# Analyze profile results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(10)                           # Show top 10 functions

# MEMORY PROFILING
try:
    from memory_profiler import profile
    
    @profile
    def memory_intensive_function():
        """Function to demonstrate memory profiling"""
        big_list = [i for i in range(1000000)]
        return sum(big_list)
    
except ImportError:
    print("memory_profiler not installed: pip install memory-profiler")

# DEBUGGING DECORATORS
def debug_decorator(func):
    """Decorator to debug function calls"""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

@debug_decorator
def add_numbers(a, b):
    """Function with debug decorator"""
    return a + b

# CONTEXT MANAGER FOR DEBUGGING
from contextlib import contextmanager
import time

@contextmanager
def debug_timer(name):
    """Context manager to time code blocks"""
    start = time.time()
    print(f"Starting {name}")
    try:
        yield
    finally:
        end = time.time()
        print(f"{name} took {end - start:.4f} seconds")

# Usage: with debug_timer("my operation"):
#            # code to time

# VARIABLE INSPECTION FUNCTIONS
def debug_vars(**kwargs):
    """Function to debug multiple variables at once"""
    for name, value in kwargs.items():
        print(f"{name}: {value} (type: {type(value).__name__})")

# Usage: debug_vars(x=x, y=y, result=result)

def debug_locals():
    """Function to debug all local variables"""
    frame = inspect.currentframe().f_back
    local_vars = frame.f_locals
    for name, value in local_vars.items():
        if not name.startswith('__'):
            print(f"{name}: {value}")

# EXCEPTION HANDLING WITH DEBUGGING
def safe_divide(a, b):
    """Function with comprehensive error handling"""
    try:
        result = a / b
        logging.info(f"Division successful: {a} / {b} = {result}")
        return result
    except ZeroDivisionError as e:
        logging.error(f"Division by zero: {a} / {b}")
        logging.debug(f"Exception details: {e}")
        return None
    except TypeError as e:
        logging.error(f"Type error in division: {type(a)} / {type(b)}")
        logging.debug(f"Exception details: {e}")
        return None
    except Exception as e:
        logging.critical(f"Unexpected error: {e}")
        logging.debug(f"Full traceback: {traceback.format_exc()}")
        return None

# DEBUGGING CLASS METHODS
class DebuggableClass:
    """Class with built-in debugging methods"""
    
    def __init__(self, value):
        self.value = value
        self._debug = True
    
    def _log(self, message):
        """Internal logging method"""
        if self._debug:
            print(f"[{self.__class__.__name__}] {message}")
    
    def process(self, x):
        """Method with debugging"""
        self._log(f"Processing with x={x}, self.value={self.value}")
        result = self.value + x
        self._log(f"Result: {result}")
        return result
    
    def __repr__(self):
        """Debug-friendly string representation"""
        return f"{self.__class__.__name__}(value={self.value})"

# CONDITIONAL DEBUGGING
DEBUG = True  # Global debug flag

def conditional_debug(message):
    """Print debug message only if DEBUG is True"""
    if DEBUG:
        print(f"DEBUG: {message}")

# Environment-based debugging
import os
DEBUG_MODE = os.environ.get('DEBUG', 'False').lower() == 'true'

if DEBUG_MODE:
    logging.getLogger().setLevel(logging.DEBUG)

# DEBUGGING GENERATORS
def debug_generator(iterable):
    """Generator that debugs each yielded value"""
    for i, item in enumerate(iterable):
        print(f"Yielding item {i}: {item}")
        yield item

# DEBUGGING ASYNC CODE
import asyncio

async def debug_async_function():
    """Async function with debugging"""
    print("Starting async operation")
    await asyncio.sleep(1)
    print("Async operation completed")
    return "result"

# DEBUGGING IMPORTS
def debug_import(module_name):
    """Debug module imports"""
    try:
        module = __import__(module_name)
        print(f"Successfully imported {module_name}")
        print(f"Module file: {getattr(module, '__file__', 'Built-in')}")
        return module
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
        return None

# DEBUGGING COMMAND LINE ARGUMENTS
import sys

def debug_cli_args():
    """Debug command line arguments"""
    print(f"Script name: {sys.argv[0]}")
    print(f"Arguments: {sys.argv[1:]}")
    print(f"Number of arguments: {len(sys.argv) - 1}")

# DEBUGGING ENVIRONMENT VARIABLES
def debug_environment():
    """Debug environment variables"""
    import os
    print("Environment variables:")
    for key, value in sorted(os.environ.items()):
        print(f"  {key}: {value}")

# DEBUGGING FILE OPERATIONS
def debug_file_operation(filename, operation="read"):
    """Debug file operations"""
    try:
        if operation == "read":
            with open(filename, 'r') as f:
                content = f.read()
                print(f"Successfully read {len(content)} characters from {filename}")
                return content
        elif operation == "write":
            with open(filename, 'w') as f:
                f.write("test content")
                print(f"Successfully wrote to {filename}")
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except PermissionError:
        print(f"Permission denied: {filename}")
    except Exception as e:
        print(f"Unexpected error with {filename}: {e}")

# DEBUGGING NETWORK REQUESTS
def debug_http_request(url):
    """Debug HTTP requests"""
    try:
        import requests
        print(f"Making request to: {url}")
        response = requests.get(url, timeout=5)
        print(f"Status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response size: {len(response.content)} bytes")
        return response
    except Exception as e:
        print(f"Request failed: {e}")
        return None

# DEBUGGING JSON DATA
def debug_json_data(data):
    """Debug JSON serialization/deserialization"""
    import json
    try:
        json_str = json.dumps(data, indent=2)
        print("JSON serialization successful:")
        print(json_str)
        
        parsed = json.loads(json_str)
        print("JSON parsing successful")
        return parsed
    except TypeError as e:
        print(f"JSON serialization error: {e}")
        print(f"Problematic data type: {type(data)}")
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")

# EXAMPLES OF USAGE:

# Basic debugging
x = 10
y = 20
print(f"Debug: x={x}, y={y}, sum={x+y}")

# Using logger
logger.info("Application started")
logger.debug("Processing user input")

# Using assertions
try:
    validate_input(-5)
except AssertionError as e:
    print(f"Assertion failed: {e}")

# Using debug decorator
result = add_numbers(5, 3)

# Using debug timer
with debug_timer("list comprehension"):
    squares = [i**2 for i in range(1000)]

# Using debug class
obj = DebuggableClass(10)
result = obj.process(5)