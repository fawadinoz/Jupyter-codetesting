# All Python Built-in Objects Examples

# Built-in Functions
abs(-5)  # Returns absolute value: 5
all([True, True])  # True if all elements are true: True
any([False, True])  # True if any element is true: True
ascii('ñ')  # ASCII representation of object: '\\xf1'
bin(10)  # Binary representation of integer: '0b1010'
bool(1)  # Convert to boolean: True
bytearray(b'hi')  # Mutable bytes array: bytearray(b'hi')
bytes([72, 105])  # Immutable bytes object: b'Hi'
callable(len)  # Check if object is callable: True
chr(65)  # Character from Unicode code point: 'A'
classmethod  # Decorator for class methods
compile('1+1', '', 'eval')  # Compile source into code object
complex(1, 2)  # Create complex number: (1+2j)
dict([('a', 1)])  # Create dictionary: {'a': 1}
dir([])  # List object attributes
divmod(10, 3)  # Division and modulo: (3, 1)
enumerate(['a', 'b'])  # Add counter to iterable
eval('2+2')  # Evaluate expression: 4
filter(None, [0, 1, 2])  # Filter elements
float(3.14)  # Convert to floating point: 3.14
format(42, 'b')  # Format value: '101010'
frozenset([1, 2])  # Immutable set: frozenset({1, 2})
getattr([], 'append')  # Get attribute value
globals()  # Global symbol table
hasattr([], 'append')  # Check if attribute exists: True
hash('hello')  # Hash value of object
hex(255)  # Hexadecimal representation: '0xff'
id([])  # Identity of object
int('42')  # Convert to integer: 42
isinstance(1, int)  # Check instance type: True
issubclass(bool, int)  # Check subclass relationship: True
iter([1, 2])  # Create iterator
len('hello')  # Length of object: 5
list((1, 2, 3))  # Create list: [1, 2, 3]
locals()  # Local symbol table
map(str, [1, 2])  # Apply function to iterable
max([1, 3, 2])  # Maximum value: 3
memoryview(b'hi')  # Memory view object
min([1, 3, 2])  # Minimum value: 1
next(iter([1, 2]))  # Next item from iterator: 1
object()  # Base object instance
oct(8)  # Octal representation: '0o10'
ord('A')  # Unicode code point: 65
pow(2, 3)  # Power operation: 8
property  # Property decorator
range(3)  # Range of numbers: range(0, 3)
repr('hi')  # String representation: "'hi'"
reversed([1, 2, 3])  # Reverse iterator
round(3.14159, 2)  # Round number: 3.14
set([1, 2, 2])  # Create set: {1, 2}
setattr  # Set attribute value
slice(1, 3)  # Slice object: slice(1, 3, None)
sorted([3, 1, 2])  # Sort iterable: [1, 2, 3]
staticmethod  # Static method decorator
str(42)  # Convert to string: '42'
sum([1, 2, 3])  # Sum of iterable: 6
super  # Access parent class
tuple([1, 2, 3])  # Create tuple: (1, 2, 3)
type(42)  # Type of object: <class 'int'>
vars({})  # Object's __dict__ attribute: {}
zip([1, 2], ['a', 'b'])  # Combine iterables

# Built-in Constants
None  # Null value
True  # Boolean true
False  # Boolean false
Ellipsis  # ... literal
NotImplemented  # Special value for comparisons

# Built-in Exceptions (examples)
ZeroDivisionError  # Division by zero: 1/0
ValueError  # Invalid conversion: int('abc')
IndexError  # Index out of range: [1, 2][5]
KeyError  # Key not found: {'a': 1}['b']
NameError  # Variable not defined: undefined_var
AttributeError  # Attribute doesn't exist: ''.nonexistent
TypeError  # Type mismatch: int('5') + 'string'

# Exception hierarchy examples
BaseException  # Root of exception hierarchy
Exception  # Base for most exceptions
ArithmeticError  # Math operation errors
LookupError  # Key/index lookup errors
RuntimeError  # Runtime errors
SystemError  # Internal Python errors
OSError  # Operating system errors
ImportError  # Import failures
SyntaxError  # Syntax errors
SystemExit  # sys.exit() exception
KeyboardInterrupt  # Ctrl+C interrupt