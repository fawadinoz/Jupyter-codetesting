# Python Functions and Generators with Examples

# BASIC FUNCTION DEFINITIONS
def simple_function():
    """Function with no parameters"""
    return "Hello World"

def function_with_params(a, b):
    """Function with required parameters"""
    return a + b

def function_with_defaults(name, greeting="Hello"):
    """Function with default parameters"""
    return f"{greeting}, {name}!"

def function_with_varargs(*args):
    """Function with variable positional arguments"""
    return sum(args)

def function_with_kwargs(**kwargs):
    """Function with variable keyword arguments"""
    return kwargs

def function_with_all(required, default="value", *args, **kwargs):
    """Function with all parameter types"""
    return required, default, args, kwargs

def function_with_annotations(x: int, y: int) -> int:
    """Function with type annotations"""
    return x + y

# LAMBDA FUNCTIONS (ANONYMOUS FUNCTIONS)
add = lambda x, y: x + y                    # Simple lambda
square = lambda x: x ** 2                   # Single parameter lambda
greet = lambda name: f"Hello, {name}"       # String formatting lambda
no_params = lambda: "No parameters"         # Lambda with no parameters
conditional = lambda x: "positive" if x > 0 else "negative"  # Conditional lambda

# NESTED FUNCTIONS (FUNCTIONS INSIDE FUNCTIONS)
def outer_function(x):
    """Function containing another function"""
    def inner_function(y):
        return x + y
    return inner_function

def closure_example(multiplier):
    """Function demonstrating closures"""
    def multiply(number):
        return number * multiplier  # Accesses outer variable
    return multiply

# HIGHER-ORDER FUNCTIONS (FUNCTIONS THAT TAKE/RETURN FUNCTIONS)
def apply_operation(func, x, y):
    """Function that takes another function as parameter"""
    return func(x, y)

def create_adder(n):
    """Function that returns another function"""
    def adder(x):
        return x + n
    return adder

def decorator_function(func):
    """Basic decorator function"""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result
    return wrapper

# RECURSIVE FUNCTIONS
def factorial(n):
    """Recursive function to calculate factorial"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """Recursive function to calculate Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def countdown(n):
    """Recursive countdown function"""
    if n <= 0:
        print("Done!")
    else:
        print(n)
        countdown(n - 1)

# GENERATOR FUNCTIONS (FUNCTIONS WITH YIELD)
def simple_generator():
    """Basic generator function"""
    yield 1
    yield 2
    yield 3

def count_up_to(max_count):
    """Generator that counts up to a number"""
    count = 1
    while count <= max_count:
        yield count
        count += 1

def fibonacci_generator():
    """Generator for infinite Fibonacci sequence"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def read_file_lines(filename):
    """Generator to read file line by line (memory efficient)"""
    with open(filename, 'r') as file:
        for line in file:
            yield line.strip()

def yield_from_example():
    """Generator using yield from"""
    yield from range(3)      # Yields 0, 1, 2
    yield from [10, 20, 30]  # Yields 10, 20, 30

# GENERATOR EXPRESSIONS (COMPACT GENERATORS)
squares = (x**2 for x in range(10))         # Generator expression for squares
even_numbers = (x for x in range(20) if x % 2 == 0)  # Filtered generator
file_lengths = (len(line) for line in open('file.txt'))  # Generator from file

# ASYNC FUNCTIONS (COROUTINES)
async def async_function():
    """Basic async function"""
    await some_async_operation()
    return "Async result"

async def async_generator():
    """Async generator function"""
    for i in range(5):
        await asyncio.sleep(1)
        yield i

# FUNCTION DECORATORS
@decorator_function
def decorated_function():
    """Function with decorator applied"""
    return "This function is decorated"

def timer_decorator(func):
    """Decorator to time function execution"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer_decorator
def slow_function():
    """Function that takes time to execute"""
    import time
    time.sleep(1)
    return "Done"

# BUILT-IN HIGHER-ORDER FUNCTIONS
numbers = [1, 2, 3, 4, 5]

# map() - applies function to each item
squared = map(lambda x: x**2, numbers)      # Returns map object (iterator)
squared_list = list(map(lambda x: x**2, numbers))  # Convert to list

# filter() - filters items based on function
evens = filter(lambda x: x % 2 == 0, numbers)      # Returns filter object
evens_list = list(filter(lambda x: x % 2 == 0, numbers))  # Convert to list

# reduce() - reduces sequence to single value
from functools import reduce
sum_all = reduce(lambda x, y: x + y, numbers)      # Sum all numbers
product = reduce(lambda x, y: x * y, numbers)      # Multiply all numbers

# zip() - combines multiple iterables
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
combined = zip(names, ages)                         # Returns zip object
combined_list = list(zip(names, ages))              # Convert to list

# enumerate() - adds counter to iterable
items = ['apple', 'banana', 'cherry']
indexed = enumerate(items)                          # Returns enumerate object
indexed_list = list(enumerate(items))               # Convert to list

# sorted() - sorts with custom key function
words = ['apple', 'pie', 'banana']
by_length = sorted(words, key=len)                  # Sort by length
by_last_char = sorted(words, key=lambda x: x[-1])  # Sort by last character

# FUNCTION ATTRIBUTES AND INTROSPECTION
def example_function(x, y=10):
    """Example function for introspection"""
    return x + y

# Function attributes
print(example_function.__name__)        # Function name
print(example_function.__doc__)         # Function docstring
print(example_function.__defaults__)    # Default parameter values
print(example_function.__code__.co_varnames)  # Variable names

# PARTIAL FUNCTIONS
from functools import partial

def multiply(x, y, z):
    """Function to demonstrate partial application"""
    return x * y * z

# Create partial function with some arguments pre-filled
double = partial(multiply, 2)           # Pre-fill first argument with 2
result = double(3, 4)                   # Equivalent to multiply(2, 3, 4)

# FUNCTION CACHING
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(n):
    """Function with caching to improve performance"""
    print(f"Computing for {n}")
    return n ** 2

# VARIABLE SCOPE EXAMPLES
global_var = "I'm global"

def scope_example():
    """Function demonstrating variable scope"""
    local_var = "I'm local"
    global global_var
    global_var = "Modified global"
    
    def nested():
        nonlocal local_var
        local_var = "Modified local"
        nested_var = "I'm nested"
        return nested_var
    
    return nested()

# FUNCTION AS FIRST-CLASS OBJECTS
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

# Functions can be stored in data structures
operations = {
    'add': add,
    'subtract': subtract,
    'multiply': lambda x, y: x * y
}

# Functions can be passed as arguments
def calculate(operation, x, y):
    return operation(x, y)

# GENERATOR METHODS AND OPERATIONS
def generator_with_methods():
    """Generator demonstrating send(), throw(), close()"""
    try:
        value = yield "First value"
        while True:
            if value is not None:
                yield f"Received: {value}"
            value = yield "Next value"
    except GeneratorExit:
        print("Generator closed")
    except Exception as e:
        yield f"Exception: {e}"

# COROUTINE EXAMPLE (GENERATOR-BASED)
def coroutine_example():
    """Generator-based coroutine"""
    print("Coroutine started")
    try:
        while True:
            value = yield
            print(f"Received: {value}")
    except GeneratorExit:
        print("Coroutine closed")

# FUNCTION FACTORIES
def create_multiplier(factor):
    """Factory function that creates multiplier functions"""
    def multiplier(number):
        return number * factor
    return multiplier

# Create specific multiplier functions
double = create_multiplier(2)
triple = create_multiplier(3)

# FUNCTION COMPOSITION
def compose(f, g):
    """Function composition: returns f(g(x))"""
    return lambda x: f(g(x))

# Example usage
add_one = lambda x: x + 1
square = lambda x: x ** 2
add_one_then_square = compose(square, add_one)

# MEMOIZATION DECORATOR
def memoize(func):
    """Decorator for memoization (caching function results)"""
    cache = {}
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

@memoize
def fibonacci_memoized(n):
    """Memoized Fibonacci function"""
    if n <= 1:
        return n
    return fibonacci_memoized(n-1) + fibonacci_memoized(n-2)

# EXAMPLES OF USAGE:

# Basic function calls
result1 = simple_function()                    # "Hello World"
result2 = function_with_params(5, 3)          # 8
result3 = function_with_defaults("Alice")     # "Hello, Alice!"

# Lambda usage
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16, 25]

# Generator usage
gen = count_up_to(3)
for num in gen:
    print(num)  # Prints 1, 2, 3

# Higher-order function usage
multiply_by_2 = create_multiplier(2)
result4 = multiply_by_2(5)                     # 10

# Decorator usage
@timer_decorator
def test_function():
    return sum(range(1000000))

result5 = test_function()  # Prints execution time