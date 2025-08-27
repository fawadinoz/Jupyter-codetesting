# Python Built-in Classes Explained

# ROOT CLASS
class object:
    """Base class for all Python objects - everything inherits from this"""
    pass

# BASIC TYPES
class type:
    """Metaclass - the class that creates other classes"""
    # When you define a class, type creates it
    pass

class NoneType:
    """Type of None - Python's null value"""
    # None is the single instance of NoneType
    pass

# NUMERIC CLASSES
class int:
    """Integer numbers - whole numbers"""
    # Examples: 42, -17, 0
    pass

class float:
    """Floating point numbers - decimal numbers"""
    # Examples: 3.14, -2.5, 1.0
    pass

class complex:
    """Complex numbers - numbers with real and imaginary parts"""
    # Examples: 3+4j, 1-2j
    pass

class bool:
    """Boolean values - True or False (inherits from int)"""
    # True = 1, False = 0 internally
    pass

# SEQUENCE CLASSES
class str:
    """String - sequence of characters"""
    # Examples: "hello", 'world', "123"
    pass

class list:
    """Mutable sequence - can change after creation"""
    # Examples: [1, 2, 3], ['a', 'b'], []
    pass

class tuple:
    """Immutable sequence - cannot change after creation"""
    # Examples: (1, 2, 3), ('a', 'b'), ()
    pass

class range:
    """Sequence of numbers - memory efficient"""
    # Examples: range(10), range(1, 5), range(0, 10, 2)
    pass

class bytes:
    """Immutable sequence of bytes"""
    # Examples: b'hello', b'\x00\x01'
    pass

class bytearray:
    """Mutable sequence of bytes"""
    # Like bytes but can be modified
    pass

# SET CLASSES
class set:
    """Mutable collection of unique items"""
    # Examples: {1, 2, 3}, {'a', 'b'}, set()
    pass

class frozenset:
    """Immutable collection of unique items"""
    # Like set but cannot be modified
    pass

# MAPPING CLASS
class dict:
    """Key-value pairs - mutable mapping"""
    # Examples: {'a': 1, 'b': 2}, {}, dict()
    pass

# CALLABLE CLASSES
class function:
    """Regular functions defined with def"""
    # Created when you use def keyword
    pass

class method:
    """Bound methods - functions attached to instances"""
    # Created when you access instance.method
    pass

class builtin_function_or_method:
    """Built-in functions like len(), print()"""
    # C-implemented functions
    pass

class generator:
    """Objects that yield values on-demand"""
    # Created by functions with yield or generator expressions
    pass

class iterator:
    """Objects that can be iterated over"""
    # Objects with __iter__ and __next__ methods
    pass

# DESCRIPTOR CLASSES
class classmethod:
    """Descriptor that makes methods receive class as first argument"""
    # Used with @classmethod decorator
    pass

class staticmethod:
    """Descriptor that makes methods not receive self/cls"""
    # Used with @staticmethod decorator
    pass

class property:
    """Descriptor that makes methods accessible like attributes"""
    # Used with @property decorator
    pass

# MODULE CLASS
class module:
    """Represents imported modules"""
    # Created when you import something
    pass

# EXCEPTION CLASSES HIERARCHY
class BaseException:
    """Root of all exceptions - catches everything including system exits"""
    pass

class Exception(BaseException):
    """Base for most user-defined exceptions"""
    pass

class ArithmeticError(Exception):
    """Base for math-related errors"""
    pass

class LookupError(Exception):
    """Base for key/index lookup errors"""
    pass

class ValueError(Exception):
    """Wrong value (right type, wrong value)"""
    # Example: int('abc') - string is right type, wrong value
    pass

class TypeError(Exception):
    """Wrong type of argument"""
    # Example: len(42) - 42 is wrong type for len()
    pass

class AttributeError(Exception):
    """Attribute doesn't exist"""
    # Example: 'hello'.nonexistent_method()
    pass

class NameError(Exception):
    """Name not found in scope"""
    # Example: print(undefined_variable)
    pass

class IndexError(LookupError):
    """List/tuple index out of range"""
    # Example: [1, 2][5] - index 5 doesn't exist
    pass

class KeyError(LookupError):
    """Dictionary key not found"""
    # Example: {'a': 1}['b'] - key 'b' doesn't exist
    pass

class ZeroDivisionError(ArithmeticError):
    """Division by zero"""
    # Example: 5 / 0
    pass

class OverflowError(ArithmeticError):
    """Numeric result too large"""
    # Rare in Python due to arbitrary precision integers
    pass

class RuntimeError(Exception):
    """Generic runtime error"""
    pass

class SystemError(Exception):
    """Internal Python error"""
    pass

class OSError(Exception):
    """Operating system related errors"""
    # File not found, permission denied, etc.
    pass

class ImportError(Exception):
    """Module import failed"""
    # Example: import nonexistent_module
    pass

class SyntaxError(Exception):
    """Invalid Python syntax"""
    # Example: if True print("hello") - missing colon
    pass

class IndentationError(SyntaxError):
    """Incorrect indentation"""
    # Example: mixing tabs and spaces
    pass

class TabError(IndentationError):
    """Inconsistent tab/space usage"""
    pass

# SYSTEM EXCEPTIONS (inherit from BaseException, not Exception)
class SystemExit(BaseException):
    """Raised by sys.exit() - program termination"""
    pass

class KeyboardInterrupt(BaseException):
    """Raised by Ctrl+C - user interruption"""
    pass

class GeneratorExit(BaseException):
    """Raised when generator is closed"""
    pass

# FILE CLASSES
class TextIOWrapper:
    """Text file objects from open()"""
    # Created by open('file.txt', 'r')
    pass

class BufferedReader:
    """Binary file objects for reading"""
    # Created by open('file.bin', 'rb')
    pass

class BufferedWriter:
    """Binary file objects for writing"""
    # Created by open('file.bin', 'wb')
    pass

# ITERATOR CLASSES
class enumerate:
    """Adds counter to iterable"""
    # enumerate(['a', 'b']) -> (0, 'a'), (1, 'b')
    pass

class zip:
    """Combines multiple iterables"""
    # zip([1, 2], ['a', 'b']) -> (1, 'a'), (2, 'b')
    pass

class filter:
    """Filters items based on function"""
    # filter(lambda x: x > 0, [-1, 0, 1]) -> [1]
    pass

class map:
    """Applies function to each item"""
    # map(str, [1, 2, 3]) -> ['1', '2', '3']
    pass

class reversed:
    """Reverses a sequence"""
    # reversed([1, 2, 3]) -> [3, 2, 1]
    pass

# INTERNAL/SPECIAL CLASSES
class code:
    """Compiled Python code objects"""
    # Created by compile() function
    pass

class slice:
    """Slice objects for indexing"""
    # Created by [start:stop:step] syntax
    pass

class ellipsis:
    """The ... (Ellipsis) object"""
    # Single instance used as placeholder
    pass

class memoryview:
    """Memory-efficient view of binary data"""
    # Allows access to internal data without copying
    pass

class super:
    """Proxy object for accessing parent class methods"""
    # Used to call parent class methods in inheritance
    pass

# EXAMPLES OF USAGE:

# Creating instances (objects) from these classes:
my_int = int(42)                    # Integer object
my_str = str("hello")               # String object  
my_list = list([1, 2, 3])          # List object
my_dict = dict({'a': 1})           # Dictionary object

# Everything is an object:
print(type(42))         # <class 'int'>
print(type("hello"))    # <class 'str'>
print(type([1, 2]))     # <class 'list'>
print(type(len))        # <class 'builtin_function_or_method'>

# All classes inherit from object:
print(issubclass(int, object))      # True
print(issubclass(str, object))      # True
print(issubclass(Exception, object)) # True