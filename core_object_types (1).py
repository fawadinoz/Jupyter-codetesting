# Python's Core Object Types

# Numbers
42  # int - integer number
3.14  # float - floating point number
2 + 3j  # complex - complex number with real and imaginary parts
True  # bool - boolean value (subclass of int)

# Sequences
"Hello"  # str - string of characters
[1, 2, 3]  # list - mutable ordered collection
(1, 2, 3)  # tuple - immutable ordered collection
range(5)  # range - sequence of numbers
b"hello"  # bytes - immutable sequence of bytes
bytearray(b"hello")  # bytearray - mutable sequence of bytes

# Sets
{1, 2, 3}  # set - mutable unordered collection of unique items
frozenset([1, 2, 3])  # frozenset - immutable set

# Mappings
{"key": "value", "num": 42}  # dict - key-value mapping

# Callables
def function(): pass  # function - user-defined function
class C:
    def method(self): pass  # method - bound method
len  # builtin_function_or_method - built-in function
type  # type - metaclass for creating classes

# Internal Types
compile("x = 1", "<string>", "exec")  # code - compiled code object
slice(1, 5, 2)  # slice - slice object for indexing
Ellipsis  # ellipsis - ... literal
None  # NoneType - null value

# Iterators
(x for x in [1, 2])  # generator - generator object
iter([1, 2, 3])  # iterator - iterator object

# Other
object()  # object - base class for all objects
property(lambda self: None)  # property - property descriptor
super  # super - access parent class methods
classmethod(lambda cls: None)  # classmethod - class method descriptor
staticmethod(lambda: None)  # staticmethod - static method descriptor
memoryview(b"hello")  # memoryview - memory view of bytes-like object

# Module
import sys; sys  # module - imported module object

# File
open(__file__)  # file - file object (TextIOWrapper)