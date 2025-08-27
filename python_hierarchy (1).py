# Python Hierarchy

# Root
object  # Base class for all Python objects

# Types
type(object)  # Metaclass for all classes
type(None)  # NoneType - type of None

# Numbers
int  # Integer type
float  # Floating point type
complex  # Complex number type
bool  # Boolean type (inherits from int)

# Sequences
str  # String type
list  # List type
tuple  # Tuple type
range  # Range type
bytes  # Bytes type
bytearray  # Bytearray type

# Sets
set  # Set type
frozenset  # Frozenset type

# Mappings
dict  # Dictionary type

# Callables
type(lambda: None)  # Function type
type(str.upper)  # Method type
type(len)  # Builtin function type
type((x for x in []))  # Generator type
type(iter([]))  # Iterator type

# Classes
classmethod  # Class method descriptor
staticmethod  # Static method descriptor
property  # Property descriptor

# Modules
type(__import__('sys'))  # Module type

# Exceptions
BaseException  # Root exception class
Exception  # Base for most exceptions
ArithmeticError  # Math operation errors
LookupError  # Key/index lookup errors
ValueError  # Value-related errors
TypeError  # Type-related errors
AttributeError  # Attribute access errors
NameError  # Name lookup errors
IndexError  # Index out of range (inherits from LookupError)
KeyError  # Key not found (inherits from LookupError)
ZeroDivisionError  # Division by zero (inherits from ArithmeticError)
OverflowError  # Numeric overflow (inherits from ArithmeticError)
RuntimeError  # Runtime errors
SystemError  # Internal Python errors
OSError  # Operating system errors
ImportError  # Import failures
SyntaxError  # Syntax errors
IndentationError  # Indentation errors (inherits from SyntaxError)
TabError  # Tab/space mixing (inherits from IndentationError)
SystemExit  # sys.exit() exception (inherits from BaseException)
KeyboardInterrupt  # Ctrl+C interrupt (inherits from BaseException)
GeneratorExit  # Generator close (inherits from BaseException)

# Files
type(open(__file__))  # TextIOWrapper type

# Iterators
type(enumerate([]))  # Enumerate type
type(zip([], []))  # Zip type
type(filter(None, []))  # Filter type
type(map(str, []))  # Map type
type(reversed([]))  # Reversed type

# Internal Types
type(compile('1', '', 'eval'))  # Code type
type(slice(1))  # Slice type
type(...)  # Ellipsis type
type(memoryview(b''))  # Memoryview type
super  # Super type