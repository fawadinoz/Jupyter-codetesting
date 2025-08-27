# NumPy Comprehensive Guide - Data Types, Arrays, Operations

import numpy as np

# NUMPY DATA TYPES

# INTEGER TYPES
np.int8      # 8-bit signed integer (-128 to 127)
np.int16     # 16-bit signed integer (-32,768 to 32,767)
np.int32     # 32-bit signed integer (-2^31 to 2^31-1)
np.int64     # 64-bit signed integer (-2^63 to 2^63-1)
np.uint8     # 8-bit unsigned integer (0 to 255)
np.uint16    # 16-bit unsigned integer (0 to 65,535)
np.uint32    # 32-bit unsigned integer (0 to 2^32-1)
np.uint64    # 64-bit unsigned integer (0 to 2^64-1)

# FLOATING POINT TYPES
np.float16   # Half precision float (16-bit)
np.float32   # Single precision float (32-bit)
np.float64   # Double precision float (64-bit) - default
np.float128  # Extended precision float (128-bit)

# COMPLEX TYPES
np.complex64   # Complex number with float32 real and imaginary parts
np.complex128  # Complex number with float64 real and imaginary parts
np.complex256  # Complex number with float128 real and imaginary parts

# BOOLEAN TYPE
np.bool_     # Boolean type (True or False)

# STRING TYPES
np.str_      # Unicode string
np.bytes_    # Byte string

# DATETIME TYPES
np.datetime64  # Date and time
np.timedelta64 # Time differences

# CREATING ARRAYS WITH SPECIFIC DATA TYPES
arr_int8 = np.array([1, 2, 3], dtype=np.int8)
arr_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
arr_complex = np.array([1+2j, 3+4j], dtype=np.complex128)
arr_bool = np.array([True, False, True], dtype=np.bool_)

# ARRAY TYPES AND CREATION

# 1D ARRAYS (VECTORS)
arr_1d = np.array([1, 2, 3, 4, 5])                    # From list
arr_range = np.arange(0, 10, 2)                       # Range: [0, 2, 4, 6, 8]
arr_linspace = np.linspace(0, 1, 5)                   # 5 evenly spaced values from 0 to 1
arr_zeros_1d = np.zeros(5)                            # [0, 0, 0, 0, 0]
arr_ones_1d = np.ones(5)                              # [1, 1, 1, 1, 1]
arr_full_1d = np.full(5, 7)                          # [7, 7, 7, 7, 7]
arr_empty_1d = np.empty(5)                            # Uninitialized values

# 2D ARRAYS (MATRICES)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])            # From nested list
arr_zeros_2d = np.zeros((3, 4))                       # 3x4 matrix of zeros
arr_ones_2d = np.ones((2, 3))                         # 2x3 matrix of ones
arr_full_2d = np.full((2, 3), 5)                     # 2x3 matrix filled with 5
arr_eye = np.eye(3)                                    # 3x3 identity matrix
arr_diag = np.diag([1, 2, 3])                        # Diagonal matrix
arr_random_2d = np.random.random((3, 3))              # 3x3 random values [0,1)

# 3D ARRAYS (TENSORS)
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # From nested list
arr_zeros_3d = np.zeros((2, 3, 4))                    # 2x3x4 tensor of zeros
arr_ones_3d = np.ones((2, 2, 2))                      # 2x2x2 tensor of ones
arr_random_3d = np.random.random((2, 3, 4))           # 2x3x4 random tensor

# SPECIAL ARRAY CREATION
arr_like = np.zeros_like(arr_2d)                       # Same shape as arr_2d, filled with zeros
arr_ones_like = np.ones_like(arr_2d)                   # Same shape as arr_2d, filled with ones
arr_full_like = np.full_like(arr_2d, 9)               # Same shape as arr_2d, filled with 9

# ARRAY ATTRIBUTES

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# SHAPE AND DIMENSIONS
print(arr.shape)        # (3, 4) - dimensions of array
print(arr.ndim)         # 2 - number of dimensions
print(arr.size)         # 12 - total number of elements
print(len(arr))         # 3 - length of first dimension

# DATA TYPE INFORMATION
print(arr.dtype)        # Data type of elements
print(arr.itemsize)     # Size of each element in bytes
print(arr.nbytes)       # Total bytes consumed by array

# MEMORY LAYOUT
print(arr.flags)        # Memory layout information
print(arr.strides)      # Bytes to step in each dimension
print(arr.data)         # Buffer containing actual data

# ARRAY INDEXING

# 1D ARRAY INDEXING
arr_1d = np.array([10, 20, 30, 40, 50])
print(arr_1d[0])        # 10 - first element
print(arr_1d[-1])       # 50 - last element
print(arr_1d[2])        # 30 - third element

# 2D ARRAY INDEXING
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr_2d[0, 0])     # 1 - first row, first column
print(arr_2d[1, 2])     # 6 - second row, third column
print(arr_2d[-1, -1])   # 9 - last row, last column
print(arr_2d[0])        # [1, 2, 3] - entire first row
print(arr_2d[:, 0])     # [1, 4, 7] - entire first column

# 3D ARRAY INDEXING
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr_3d[0, 1, 0])  # 3 - first matrix, second row, first column
print(arr_3d[1, 0, 1])  # 6 - second matrix, first row, second column

# BOOLEAN INDEXING
arr = np.array([1, 2, 3, 4, 5, 6])
mask = arr > 3          # [False, False, False, True, True, True]
print(arr[mask])        # [4, 5, 6] - elements greater than 3
print(arr[arr % 2 == 0]) # [2, 4, 6] - even elements

# FANCY INDEXING (ARRAY INDEXING)
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])
print(arr[indices])     # [10, 30, 50] - elements at indices 0, 2, 4

# ARRAY SLICING

# 1D ARRAY SLICING
arr_1d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr_1d[2:7])      # [2, 3, 4, 5, 6] - elements from index 2 to 6
print(arr_1d[:5])       # [0, 1, 2, 3, 4] - first 5 elements
print(arr_1d[5:])       # [5, 6, 7, 8, 9] - from index 5 to end
print(arr_1d[::2])      # [0, 2, 4, 6, 8] - every second element
print(arr_1d[::-1])     # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] - reversed
print(arr_1d[1:8:2])    # [1, 3, 5, 7] - from 1 to 7, step 2

# 2D ARRAY SLICING
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(arr_2d[0:2, 1:3]) # [[2, 3], [6, 7]] - rows 0-1, columns 1-2
print(arr_2d[:, 1])     # [2, 6, 10] - all rows, column 1
print(arr_2d[1, :])     # [5, 6, 7, 8] - row 1, all columns
print(arr_2d[::2, ::2]) # [[1, 3], [9, 11]] - every second row and column
print(arr_2d[::-1, :])  # Reverse rows
print(arr_2d[:, ::-1])  # Reverse columns

# 3D ARRAY SLICING
arr_3d = np.zeros((4, 3, 2))
print(arr_3d[1:3, :, 0]) # Matrices 1-2, all rows, first column
print(arr_3d[:, 1:, :])  # All matrices, rows 1+, all columns

# ELLIPSIS SLICING
arr_4d = np.zeros((2, 3, 4, 5))
print(arr_4d[0, ..., 1].shape)  # (3, 4) - equivalent to arr_4d[0, :, :, 1]
print(arr_4d[..., 0].shape)     # (2, 3, 4) - equivalent to arr_4d[:, :, :, 0]

# ARRAY RESHAPING

# BASIC RESHAPING
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
reshaped_2d = arr.reshape(3, 4)        # 3x4 matrix
reshaped_3d = arr.reshape(2, 2, 3)     # 2x2x3 tensor
reshaped_auto = arr.reshape(-1, 4)     # Auto-calculate rows: 3x4
reshaped_auto2 = arr.reshape(3, -1)    # Auto-calculate columns: 3x4

# FLATTENING
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
flattened = arr_2d.flatten()           # [1, 2, 3, 4, 5, 6] - copy
raveled = arr_2d.ravel()               # [1, 2, 3, 4, 5, 6] - view if possible

# TRANSPOSE
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
transposed = arr_2d.T                  # [[1, 4], [2, 5], [3, 6]]
transposed2 = np.transpose(arr_2d)     # Same as .T
transposed3 = arr_2d.transpose()       # Same as .T

# SWAPPING AXES
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
swapped = np.swapaxes(arr_3d, 0, 2)    # Swap axis 0 and 2

# EXPANDING DIMENSIONS
arr_1d = np.array([1, 2, 3])
expanded = np.expand_dims(arr_1d, axis=0)  # [[1, 2, 3]] - add axis at position 0
expanded2 = np.expand_dims(arr_1d, axis=1) # [[1], [2], [3]] - add axis at position 1
expanded3 = arr_1d[np.newaxis, :]         # Same as expand_dims(axis=0)
expanded4 = arr_1d[:, np.newaxis]         # Same as expand_dims(axis=1)

# SQUEEZING DIMENSIONS
arr_with_single_dims = np.array([[[1], [2], [3]]])  # Shape: (1, 3, 1)
squeezed = np.squeeze(arr_with_single_dims)         # [1, 2, 3] - remove single dimensions

# ARRAY JOINING

# CONCATENATION
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concatenated = np.concatenate([arr1, arr2])         # [1, 2, 3, 4, 5, 6]

arr1_2d = np.array([[1, 2], [3, 4]])
arr2_2d = np.array([[5, 6], [7, 8]])
concat_axis0 = np.concatenate([arr1_2d, arr2_2d], axis=0)  # Vertical: [[1,2],[3,4],[5,6],[7,8]]
concat_axis1 = np.concatenate([arr1_2d, arr2_2d], axis=1)  # Horizontal: [[1,2,5,6],[3,4,7,8]]

# STACKING
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
vstacked = np.vstack([arr1, arr2])      # [[1,2,3],[4,5,6]] - vertical stack
hstacked = np.hstack([arr1, arr2])      # [1,2,3,4,5,6] - horizontal stack
dstacked = np.dstack([arr1, arr2])      # [[[1,4],[2,5],[3,6]]] - depth stack

# GENERAL STACKING
stacked_axis0 = np.stack([arr1, arr2], axis=0)  # Same as vstack for 1D
stacked_axis1 = np.stack([arr1, arr2], axis=1)  # [[1,4],[2,5],[3,6]]

# COLUMN AND ROW STACKING
arr1_2d = np.array([[1, 2], [3, 4]])
arr2_2d = np.array([[5, 6], [7, 8]])
col_stacked = np.column_stack([arr1_2d, arr2_2d])  # [[1,2,5,6],[3,4,7,8]]
row_stacked = np.row_stack([arr1_2d, arr2_2d])     # [[1,2],[3,4],[5,6],[7,8]]

# ARRAY SPLITTING

# BASIC SPLITTING
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
split_equal = np.split(arr, 4)          # [array([1,2]), array([3,4]), array([5,6]), array([7,8])]
split_indices = np.split(arr, [3, 6])   # Split at indices 3 and 6

# 2D ARRAY SPLITTING
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
hsplit_result = np.hsplit(arr_2d, 2)    # Split horizontally into 2 parts
vsplit_result = np.vsplit(arr_2d, 3)    # Split vertically into 3 parts

# ARRAY SPLITTING WITH INDICES
arr_2d = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
hsplit_indices = np.hsplit(arr_2d, [2, 4])  # Split at columns 2 and 4

# DEPTH SPLITTING (3D)
arr_3d = np.random.random((2, 3, 4))
dsplit_result = np.dsplit(arr_3d, 2)    # Split along depth (3rd axis)

# ARRAY SPLITTING FUNCTIONS
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
array_split = np.array_split(arr, 4)    # Split into 4 parts (unequal if necessary)

# EXAMPLES OF USAGE:

# Create arrays with different data types
int_array = np.array([1, 2, 3], dtype=np.int32)
float_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
complex_array = np.array([1+2j, 3+4j], dtype=np.complex128)

# Array attributes
print(f"Shape: {int_array.shape}")
print(f"Data type: {int_array.dtype}")
print(f"Size: {int_array.size}")

# Indexing and slicing
matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Element at [1,2]: {matrix[1, 2]}")
print(f"First row: {matrix[0, :]}")
print(f"Last column: {matrix[:, -1]}")

# Reshaping
original = np.arange(12)
reshaped = original.reshape(3, 4)
print(f"Original shape: {original.shape}")
print(f"Reshaped: {reshaped.shape}")

# Joining arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
joined = np.concatenate([arr1, arr2])
stacked = np.vstack([arr1, arr2])

# Splitting arrays
big_array = np.arange(12).reshape(3, 4)
split_arrays = np.hsplit(big_array, 2)
print(f"Split into {len(split_arrays)} arrays")