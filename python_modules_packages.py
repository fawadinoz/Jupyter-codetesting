# Python Modules and Packages with Examples

# IMPORTING MODULES - BASIC SYNTAX
import os                           # Import entire module
import sys                          # Import system module
import math                         # Import math module
import random                       # Import random module
import datetime                     # Import datetime module

# IMPORTING WITH ALIASES
import numpy as np                  # Import with alias (common convention)
import pandas as pd                 # Import with alias
import matplotlib.pyplot as plt     # Import submodule with alias
import os.path as path             # Import submodule with alias

# IMPORTING SPECIFIC ITEMS
from math import pi, sqrt, sin      # Import specific functions/constants
from os import getcwd, listdir      # Import specific functions
from datetime import datetime, date # Import specific classes
from collections import Counter     # Import specific class

# IMPORTING ALL (NOT RECOMMENDED)
from math import *                  # Imports everything from math module
from os import *                    # Imports everything from os module (dangerous)

# CONDITIONAL IMPORTS
try:
    import numpy as np              # Try to import optional dependency
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("NumPy not available")

# RELATIVE IMPORTS (WITHIN PACKAGES)
from . import sibling_module        # Import from same package
from .subpackage import module      # Import from subpackage
from .. import parent_module        # Import from parent package
from ..other_package import module  # Import from sibling package

# BUILT-IN MODULES EXAMPLES

# OS MODULE - Operating system interface
import os
current_dir = os.getcwd()           # Get current working directory
files = os.listdir('.')             # List files in directory
os.mkdir('new_folder')              # Create directory
os.path.join('folder', 'file.txt')  # Join path components
os.path.exists('file.txt')          # Check if path exists
os.environ['HOME']                  # Access environment variables

# SYS MODULE - System-specific parameters
import sys
sys.version                         # Python version
sys.platform                       # Platform identifier
sys.path                           # Module search path
sys.argv                           # Command line arguments
sys.exit(0)                        # Exit program

# MATH MODULE - Mathematical functions
import math
math.pi                            # Pi constant
math.e                             # Euler's number
math.sqrt(16)                      # Square root
math.sin(math.pi/2)               # Sine function
math.log(10)                      # Natural logarithm
math.factorial(5)                 # Factorial

# RANDOM MODULE - Generate random numbers
import random
random.random()                    # Random float between 0 and 1
random.randint(1, 10)             # Random integer between 1 and 10
random.choice(['a', 'b', 'c'])    # Random choice from list
random.shuffle([1, 2, 3, 4])      # Shuffle list in place
random.sample([1, 2, 3, 4, 5], 3) # Random sample without replacement

# DATETIME MODULE - Date and time handling
import datetime
now = datetime.datetime.now()      # Current date and time
today = datetime.date.today()      # Current date
time_obj = datetime.time(14, 30)   # Time object
delta = datetime.timedelta(days=7) # Time difference
formatted = now.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime

# JSON MODULE - JSON encoder and decoder
import json
data = {'name': 'Alice', 'age': 30}
json_string = json.dumps(data)     # Convert to JSON string
parsed_data = json.loads(json_string)  # Parse JSON string
with open('data.json', 'w') as f:
    json.dump(data, f)             # Write JSON to file
with open('data.json', 'r') as f:
    loaded_data = json.load(f)     # Read JSON from file

# RE MODULE - Regular expressions
import re
pattern = r'\d+'                   # Pattern for digits
text = "I have 5 apples and 3 oranges"
matches = re.findall(pattern, text)  # Find all matches
match = re.search(r'(\d+) apples', text)  # Search for pattern
replaced = re.sub(r'\d+', 'X', text)  # Replace matches

# COLLECTIONS MODULE - Specialized container datatypes
from collections import Counter, defaultdict, deque, namedtuple

# Counter - count hashable objects
counter = Counter(['a', 'b', 'a', 'c', 'b', 'a'])  # Counter({'a': 3, 'b': 2, 'c': 1})

# defaultdict - dict with default values
dd = defaultdict(list)
dd['key'].append('value')          # Automatically creates list

# deque - double-ended queue
dq = deque([1, 2, 3])
dq.appendleft(0)                   # Add to left
dq.append(4)                       # Add to right

# namedtuple - tuple with named fields
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)                    # p.x = 1, p.y = 2

# ITERTOOLS MODULE - Iterator functions
import itertools

# Infinite iterators
count = itertools.count(10, 2)     # 10, 12, 14, 16, ...
cycle = itertools.cycle(['A', 'B', 'C'])  # A, B, C, A, B, C, ...
repeat = itertools.repeat('hello', 3)  # 'hello', 'hello', 'hello'

# Finite iterators
chain = itertools.chain([1, 2], [3, 4])  # 1, 2, 3, 4
combinations = itertools.combinations([1, 2, 3, 4], 2)  # (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
permutations = itertools.permutations([1, 2, 3], 2)  # (1,2), (1,3), (2,1), (2,3), (3,1), (3,2)

# FUNCTOOLS MODULE - Higher-order functions and operations on callable objects
from functools import reduce, partial, wraps, lru_cache

# reduce - apply function cumulatively
numbers = [1, 2, 3, 4, 5]
sum_all = reduce(lambda x, y: x + y, numbers)  # 15

# partial - partial function application
def multiply(x, y, z):
    return x * y * z
double = partial(multiply, 2)      # Pre-fill first argument
result = double(3, 4)              # multiply(2, 3, 4) = 24

# lru_cache - least-recently-used cache decorator
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# PATHLIB MODULE - Object-oriented filesystem paths
from pathlib import Path

# Path operations
current_path = Path.cwd()          # Current working directory
home_path = Path.home()            # Home directory
file_path = Path('folder/file.txt')
file_path.exists()                 # Check if path exists
file_path.is_file()               # Check if it's a file
file_path.parent                  # Get parent directory
file_path.suffix                  # Get file extension
file_path.stem                    # Get filename without extension

# URLLIB MODULE - URL handling modules
from urllib.request import urlopen
from urllib.parse import urljoin, urlparse

# URL operations
response = urlopen('https://httpbin.org/json')
data = response.read()
parsed_url = urlparse('https://example.com/path?query=value')
joined_url = urljoin('https://example.com/', 'path/to/resource')

# CREATING CUSTOM MODULES

# Example: math_utils.py
"""
def add(x, y):
    '''Add two numbers'''
    return x + y

def multiply(x, y):
    '''Multiply two numbers'''
    return x * y

PI = 3.14159

class Calculator:
    '''Simple calculator class'''
    def __init__(self):
        self.result = 0
    
    def add(self, x):
        self.result += x
        return self.result
"""

# Using custom module
# import math_utils
# result = math_utils.add(5, 3)
# calc = math_utils.Calculator()

# PACKAGE STRUCTURE EXAMPLE
"""
mypackage/
    __init__.py          # Makes it a package
    module1.py           # Module in package
    module2.py           # Another module
    subpackage/
        __init__.py      # Makes subpackage
        submodule.py     # Module in subpackage
"""

# __init__.py examples
"""
# Empty __init__.py - minimal package

# __init__.py with imports
from .module1 import function1
from .module2 import Class2
__all__ = ['function1', 'Class2']  # Define what * imports

# __init__.py with package initialization
print("Package initialized")
VERSION = "1.0.0"
"""

# IMPORTING FROM PACKAGES
# from mypackage import module1              # Import module from package
# from mypackage.module1 import function1   # Import function from module
# from mypackage.subpackage import submodule # Import from subpackage
# import mypackage.module1 as m1            # Import with alias

# MODULE SEARCH PATH
import sys
print(sys.path)                    # List of directories Python searches for modules

# Add directory to path
sys.path.append('/path/to/modules')
sys.path.insert(0, '/priority/path')

# RELOADING MODULES (FOR DEVELOPMENT)
import importlib
# importlib.reload(module_name)     # Reload module after changes

# MODULE ATTRIBUTES
import math
print(math.__name__)               # Module name
print(math.__file__)               # Module file path
print(math.__doc__)                # Module docstring
print(dir(math))                   # List module contents

# CONDITIONAL MODULE LOADING
def load_module_safely(module_name):
    """Safely load a module with error handling"""
    try:
        module = __import__(module_name)
        return module
    except ImportError as e:
        print(f"Could not import {module_name}: {e}")
        return None

# LAZY IMPORTS (IMPORT WHEN NEEDED)
def get_numpy():
    """Import numpy only when needed"""
    global np
    if 'np' not in globals():
        import numpy as np
    return np

# NAMESPACE PACKAGES (PEP 420)
"""
Packages without __init__.py files
namespace_package/
    subpackage1/
        module1.py
    subpackage2/
        module2.py
"""

# ENTRY POINTS AND CONSOLE SCRIPTS
"""
# setup.py example
from setuptools import setup

setup(
    name='mypackage',
    entry_points={
        'console_scripts': [
            'mycmd=mypackage.cli:main',
        ],
    },
)
"""

# MODULE EXECUTION
"""
# In module file
if __name__ == '__main__':
    # Code that runs when module is executed directly
    main()
"""

# THIRD-PARTY PACKAGE MANAGEMENT
"""
# Installing packages
pip install package_name
pip install package_name==1.2.3    # Specific version
pip install -r requirements.txt    # From requirements file

# requirements.txt example
numpy>=1.19.0
pandas==1.3.0
matplotlib
requests>=2.25.0,<3.0.0
"""

# VIRTUAL ENVIRONMENTS
"""
# Create virtual environment
python -m venv myenv

# Activate (Windows)
myenv\Scripts\activate

# Activate (Unix/macOS)
source myenv/bin/activate

# Deactivate
deactivate

# Install packages in virtual environment
pip install package_name
"""

# PACKAGE DISTRIBUTION
"""
# setup.py for package distribution
from setuptools import setup, find_packages

setup(
    name='mypackage',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'requests>=2.25.0',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A short description',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/mypackage',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
"""

# EXAMPLES OF USAGE:

# Basic module usage
import math
result = math.sqrt(16)             # 4.0

# Package usage
from collections import Counter
counts = Counter(['a', 'b', 'a'])  # Counter({'a': 2, 'b': 1})

# Custom module usage (if math_utils.py exists)
# import math_utils
# result = math_utils.add(5, 3)    # 8

# Conditional import usage
if HAS_NUMPY:
    arr = np.array([1, 2, 3])
else:
    arr = [1, 2, 3]

# Path operations
from pathlib import Path
config_file = Path.home() / '.config' / 'app.conf'
if config_file.exists():
    print("Config file found")