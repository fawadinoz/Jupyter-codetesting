# NumPy Advanced Computations - Broadcasting, Comparisons, Masks, Boolean Logic, Fancy Indexing, Sorting, Structured Arrays

import numpy as np

# BROADCASTING - OPERATIONS BETWEEN ARRAYS OF DIFFERENT SHAPES

# BASIC BROADCASTING RULES
# 1. Arrays are aligned from the rightmost dimension
# 2. Dimensions of size 1 can be "stretched" to match
# 3. Missing dimensions are assumed to be size 1

# SCALAR WITH ARRAY
arr = np.array([1, 2, 3, 4])
result = arr + 10                    # [11, 12, 13, 14] - scalar broadcasts to all elements

# 1D ARRAY WITH 2D ARRAY
arr_1d = np.array([1, 2, 3])        # Shape: (3,)
arr_2d = np.array([[10], [20], [30]]) # Shape: (3, 1)
result = arr_1d + arr_2d             # Shape: (3, 3) - broadcasts to [[11,12,13],[21,22,23],[31,32,33]]

# DIFFERENT BROADCASTING EXAMPLES
a = np.array([1, 2, 3])              # Shape: (3,)
b = np.array([[1], [2], [3]])        # Shape: (3, 1)
c = np.array([[1, 2, 3]])            # Shape: (1, 3)

result_ab = a + b                    # Shape: (3, 3)
result_ac = a + c                    # Shape: (1, 3) -> (3,)
result_bc = b + c                    # Shape: (3, 3)

# 3D BROADCASTING
arr_3d = np.random.random((2, 3, 4)) # Shape: (2, 3, 4)
arr_2d = np.random.random((3, 4))    # Shape: (3, 4)
arr_1d = np.random.random((4,))      # Shape: (4,)

result_3d_2d = arr_3d + arr_2d       # 2D broadcasts to (1, 3, 4) then (2, 3, 4)
result_3d_1d = arr_3d + arr_1d       # 1D broadcasts to (1, 1, 4) then (2, 3, 4)

# BROADCASTING WITH NEWAXIS
arr = np.array([1, 2, 3, 4])
col_vector = arr[:, np.newaxis]      # Shape: (4, 1)
row_vector = arr[np.newaxis, :]      # Shape: (1, 4)
matrix = col_vector + row_vector     # Shape: (4, 4) - outer sum

# BROADCASTING FUNCTIONS
np.broadcast_arrays(a, b)            # Return arrays broadcasted to common shape
np.broadcast_to(arr, (3, 4))         # Broadcast array to specific shape

# COMPARISONS - ELEMENT-WISE COMPARISON OPERATIONS

# BASIC COMPARISONS
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([1, 3, 2, 4, 6])

equal = arr1 == arr2                 # [True, False, False, True, False]
not_equal = arr1 != arr2             # [False, True, True, False, True]
less_than = arr1 < arr2              # [False, True, False, False, True]
less_equal = arr1 <= arr2            # [True, True, False, True, True]
greater_than = arr1 > arr2           # [False, False, True, False, False]
greater_equal = arr1 >= arr2         # [True, False, True, True, False]

# COMPARISON WITH SCALARS
arr = np.array([1, 2, 3, 4, 5])
greater_than_3 = arr > 3             # [False, False, False, True, True]
equal_to_2 = arr == 2                # [False, True, False, False, False]

# COMPARISON FUNCTIONS
np.equal(arr1, arr2)                 # Same as arr1 == arr2
np.not_equal(arr1, arr2)             # Same as arr1 != arr2
np.less(arr1, arr2)                  # Same as arr1 < arr2
np.greater(arr1, arr2)               # Same as arr1 > arr2

# ELEMENT-WISE MIN/MAX
np.maximum(arr1, arr2)               # Element-wise maximum
np.minimum(arr1, arr2)               # Element-wise minimum

# FLOATING POINT COMPARISONS
arr_float = np.array([1.0, 2.0, 3.0])
np.isclose(arr_float, [1.0001, 2.0001, 3.0001], atol=1e-3)  # Close within tolerance
np.allclose(arr_float, [1.0001, 2.0001, 3.0001], atol=1e-3) # All elements close

# MASKS AND BOOLEAN LOGIC

# CREATING BOOLEAN MASKS
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask_even = data % 2 == 0            # [False, True, False, True, False, True, False, True, False, True]
mask_gt_5 = data > 5                 # [False, False, False, False, False, True, True, True, True, True]
mask_range = (data >= 3) & (data <= 7)  # [False, False, True, True, True, True, True, False, False, False]

# BOOLEAN INDEXING WITH MASKS
even_numbers = data[mask_even]       # [2, 4, 6, 8, 10]
numbers_gt_5 = data[mask_gt_5]       # [6, 7, 8, 9, 10]
numbers_in_range = data[mask_range]  # [3, 4, 5, 6, 7]

# COMBINING MASKS WITH LOGICAL OPERATIONS
np.logical_and(mask_even, mask_gt_5) # Even AND greater than 5
np.logical_or(mask_even, mask_gt_5)  # Even OR greater than 5
np.logical_xor(mask_even, mask_gt_5) # Even XOR greater than 5
np.logical_not(mask_even)            # NOT even (odd numbers)

# SHORTHAND LOGICAL OPERATIONS
combined_mask = mask_even & mask_gt_5    # Even AND greater than 5
combined_mask = mask_even | mask_gt_5    # Even OR greater than 5
combined_mask = ~mask_even               # NOT even

# BOOLEAN ARRAY OPERATIONS
bool_arr = np.array([True, False, True, False, True])
np.any(bool_arr)                     # True - at least one True
np.all(bool_arr)                     # False - not all True
np.sum(bool_arr)                     # 3 - count of True values
np.count_nonzero(bool_arr)           # 3 - count of True values

# CONDITIONAL SELECTION
np.where(mask_even, data, -1)        # Return data where mask is True, -1 otherwise
np.where(data > 5, data, 0)          # Return data if > 5, else 0

# MULTIPLE CONDITIONS WITH SELECT
conditions = [data < 3, (data >= 3) & (data < 7), data >= 7]
choices = ['small', 'medium', 'large']
np.select(conditions, choices, default='unknown')

# MASKING 2D ARRAYS
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask_2d = matrix > 5                 # Boolean mask for elements > 5
matrix[mask_2d] = 0                  # Set elements > 5 to 0

# FANCY INDEXING - ADVANCED ARRAY INDEXING

# BASIC FANCY INDEXING
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
indices = np.array([0, 2, 4, 6])
selected = arr[indices]              # [10, 30, 50, 70]

# NEGATIVE INDICES
indices_neg = np.array([-1, -2, -3])
selected_neg = arr[indices_neg]      # [90, 80, 70]

# 2D FANCY INDEXING
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# Select specific rows
row_indices = np.array([0, 2])
selected_rows = matrix[row_indices]  # [[1, 2, 3, 4], [9, 10, 11, 12]]

# Select specific elements
row_idx = np.array([0, 1, 2])
col_idx = np.array([1, 2, 3])
selected_elements = matrix[row_idx, col_idx]  # [2, 7, 12] - diagonal elements

# FANCY INDEXING WITH BROADCASTING
matrix = np.random.random((4, 4))
row_indices = np.array([[0, 1], [2, 3]])    # Shape: (2, 2)
col_indices = np.array([[0, 1], [2, 3]])    # Shape: (2, 2)
selected = matrix[row_indices, col_indices]  # Shape: (2, 2)

# FANCY INDEXING FOR ASSIGNMENT
arr = np.zeros(10)
indices = np.array([1, 3, 5, 7])
arr[indices] = 99                    # Set specific indices to 99

# FANCY INDEXING WITH BOOLEAN ARRAYS
data = np.array([1, 2, 3, 4, 5])
bool_indices = np.array([True, False, True, False, True])
selected = data[bool_indices]        # [1, 3, 5]

# COMBINING FANCY AND REGULAR INDEXING
matrix = np.random.random((5, 5))
selected = matrix[np.array([0, 2, 4]), 1:4]  # Rows 0,2,4 and columns 1-3

# SORTING ARRAYS

# BASIC SORTING
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
sorted_arr = np.sort(arr)            # [1, 1, 2, 3, 4, 5, 6, 9] - returns sorted copy
arr.sort()                           # Sort in-place

# SORTING ALONG AXES
matrix = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
sorted_rows = np.sort(matrix, axis=1)     # Sort each row
sorted_cols = np.sort(matrix, axis=0)     # Sort each column

# ARGSORT - INDICES THAT WOULD SORT THE ARRAY
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
sort_indices = np.argsort(arr)       # [1, 3, 6, 0, 2, 4, 7, 5]
sorted_arr = arr[sort_indices]       # Use indices to get sorted array

# SORTING IN DESCENDING ORDER
sorted_desc = np.sort(arr)[::-1]     # Reverse sorted array
argsort_desc = np.argsort(arr)[::-1] # Indices for descending sort

# LEXICOGRAPHIC SORTING (MULTIPLE KEYS)
names = np.array(['Alice', 'Bob', 'Charlie', 'Alice'])
ages = np.array([25, 30, 35, 22])
# Sort by name first, then by age
sort_indices = np.lexsort([ages, names])  # Note: keys in reverse order
sorted_names = names[sort_indices]
sorted_ages = ages[sort_indices]

# PARTIAL SORTING
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
k = 3
partitioned = np.partition(arr, k)   # kth element in correct position
partition_indices = np.argpartition(arr, k)  # Indices for partition

# SEARCHING IN SORTED ARRAYS
sorted_arr = np.array([1, 2, 3, 5, 8, 13, 21])
indices = np.searchsorted(sorted_arr, [4, 7, 15])  # Where to insert values

# STRUCTURED ARRAYS - NUMPY'S STRUCTURED DATA

# DEFINING STRUCTURED ARRAY DTYPE
# Method 1: List of tuples
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])

# Method 2: Dictionary
dt = np.dtype({'names': ['name', 'age', 'weight'],
               'formats': ['U10', 'i4', 'f4']})

# Method 3: Comma-separated string
dt = np.dtype('U10,i4,f4')

# CREATING STRUCTURED ARRAYS
# Empty structured array
people = np.empty(3, dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])

# From list of tuples
data = [('Alice', 25, 55.5), ('Bob', 30, 70.2), ('Charlie', 35, 80.1)]
people = np.array(data, dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])

# Using np.zeros with structured dtype
people = np.zeros(3, dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])

# ACCESSING STRUCTURED ARRAY FIELDS
print(people['name'])                # Access all names
print(people['age'])                 # Access all ages
print(people[0])                     # Access first record
print(people[0]['name'])             # Access name of first record

# MODIFYING STRUCTURED ARRAYS
people['age'] += 1                   # Increment all ages
people[0]['name'] = 'Alexander'      # Change first person's name

# STRUCTURED ARRAY WITH NESTED FIELDS
dt = np.dtype([('name', 'U10'),
               ('info', [('age', 'i4'), ('weight', 'f4')])])
people = np.array([('Alice', (25, 55.5)), ('Bob', (30, 70.2))], dtype=dt)
print(people['info']['age'])         # Access nested field

# RECORD ARRAYS (ALTERNATIVE TO STRUCTURED ARRAYS)
people_rec = np.rec.array(data, dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
print(people_rec.name)               # Access field as attribute
print(people_rec.age)                # Access field as attribute

# STRUCTURED ARRAY OPERATIONS
# Sorting by field
sorted_people = np.sort(people, order='age')  # Sort by age
sorted_people = np.sort(people, order=['age', 'name'])  # Sort by age, then name

# Filtering structured arrays
young_people = people[people['age'] < 30]
heavy_people = people[people['weight'] > 60]

# EXAMPLES WITH REAL DATA

# Example 1: Broadcasting with different shapes
print("=== BROADCASTING EXAMPLE ===")
temperatures = np.array([[20, 25, 30],    # 3 cities
                        [22, 27, 32],
                        [18, 23, 28]])     # 3 days
adjustment = np.array([2, -1, 0])          # Adjustment per city
adjusted_temps = temperatures + adjustment[:, np.newaxis]  # Broadcast adjustment
print("Original temperatures:\n", temperatures)
print("Adjusted temperatures:\n", adjusted_temps)

# Example 2: Boolean masking for data filtering
print("\n=== BOOLEAN MASKING EXAMPLE ===")
scores = np.array([85, 92, 78, 96, 88, 73, 91, 84])
passing_mask = scores >= 80
failing_mask = scores < 80
print("All scores:", scores)
print("Passing scores:", scores[passing_mask])
print("Failing scores:", scores[failing_mask])
print("Number passing:", np.sum(passing_mask))

# Example 3: Fancy indexing for data selection
print("\n=== FANCY INDEXING EXAMPLE ===")
data_matrix = np.random.randint(1, 100, (5, 4))
selected_rows = np.array([0, 2, 4])
selected_cols = np.array([1, 3])
print("Original matrix:\n", data_matrix)
print("Selected rows (0,2,4):\n", data_matrix[selected_rows])
print("Selected intersection:\n", data_matrix[np.ix_(selected_rows, selected_cols)])

# Example 4: Structured array for mixed data
print("\n=== STRUCTURED ARRAY EXAMPLE ===")
student_dtype = [('name', 'U20'), ('student_id', 'i4'), ('gpa', 'f4'), ('graduated', '?')]
students = np.array([
    ('Alice Johnson', 12345, 3.8, True),
    ('Bob Smith', 12346, 3.2, False),
    ('Charlie Brown', 12347, 3.9, True),
    ('Diana Prince', 12348, 3.6, False)
], dtype=student_dtype)

print("All students:\n", students)
print("Names:", students['name'])
print("GPAs:", students['gpa'])
print("Graduated students:", students[students['graduated']])
print("High GPA students (>3.5):", students[students['gpa'] > 3.5]['name'])

# Sorting students by GPA
sorted_by_gpa = np.sort(students, order='gpa')
print("Students sorted by GPA:\n", sorted_by_gpa[['name', 'gpa']])