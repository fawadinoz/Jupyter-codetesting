# NumPy Aggregation Functions - Complete Reference

import numpy as np

# BASIC STATISTICAL AGGREGATIONS

# SUM FUNCTIONS
np.sum(arr)                    # Sum of all elements
np.sum(arr, axis=0)            # Sum along axis 0 (columns for 2D)
np.sum(arr, axis=1)            # Sum along axis 1 (rows for 2D)
np.sum(arr, axis=(0, 1))       # Sum along multiple axes
np.sum(arr, keepdims=True)     # Keep dimensions in result
np.nansum(arr)                 # Sum ignoring NaN values
np.cumsum(arr)                 # Cumulative sum
np.cumsum(arr, axis=0)         # Cumulative sum along axis

# PRODUCT FUNCTIONS
np.prod(arr)                   # Product of all elements
np.prod(arr, axis=0)           # Product along axis 0
np.prod(arr, axis=1)           # Product along axis 1
np.nanprod(arr)                # Product ignoring NaN values
np.cumprod(arr)                # Cumulative product
np.cumprod(arr, axis=0)        # Cumulative product along axis

# MEAN FUNCTIONS
np.mean(arr)                   # Arithmetic mean of all elements
np.mean(arr, axis=0)           # Mean along axis 0
np.mean(arr, axis=1)           # Mean along axis 1
np.nanmean(arr)                # Mean ignoring NaN values
np.average(arr)                # Weighted average (same as mean if no weights)
np.average(arr, weights=w)     # Weighted average with weights

# MEDIAN AND PERCENTILE FUNCTIONS
np.median(arr)                 # Median (50th percentile)
np.median(arr, axis=0)         # Median along axis 0
np.nanmedian(arr)              # Median ignoring NaN values
np.percentile(arr, 25)         # 25th percentile
np.percentile(arr, [25, 50, 75]) # Multiple percentiles
np.percentile(arr, 90, axis=0) # Percentile along axis
np.nanpercentile(arr, 50)      # Percentile ignoring NaN values
np.quantile(arr, 0.25)         # Quantile (same as percentile but 0-1 scale)
np.nanquantile(arr, 0.75)      # Quantile ignoring NaN values

# VARIANCE AND STANDARD DEVIATION
np.var(arr)                    # Variance
np.var(arr, axis=0)            # Variance along axis 0
np.var(arr, ddof=1)            # Sample variance (N-1 denominator)
np.nanvar(arr)                 # Variance ignoring NaN values
np.std(arr)                    # Standard deviation
np.std(arr, axis=1)            # Standard deviation along axis 1
np.nanstd(arr)                 # Standard deviation ignoring NaN values

# MINIMUM AND MAXIMUM FUNCTIONS
np.min(arr)                    # Minimum value
np.min(arr, axis=0)            # Minimum along axis 0
np.nanmin(arr)                 # Minimum ignoring NaN values
np.max(arr)                    # Maximum value
np.max(arr, axis=1)            # Maximum along axis 1
np.nanmax(arr)                 # Maximum ignoring NaN values
np.ptp(arr)                    # Peak-to-peak (max - min)
np.ptp(arr, axis=0)            # Peak-to-peak along axis

# ARGMIN AND ARGMAX FUNCTIONS
np.argmin(arr)                 # Index of minimum value (flattened)
np.argmin(arr, axis=0)         # Indices of minimum along axis 0
np.nanargmin(arr)              # Index of minimum ignoring NaN
np.argmax(arr)                 # Index of maximum value (flattened)
np.argmax(arr, axis=1)         # Indices of maximum along axis 1
np.nanargmax(arr)              # Index of maximum ignoring NaN
np.unravel_index(np.argmax(arr), arr.shape)  # Convert flat index to coordinates

# SORTING AND ORDERING
np.sort(arr)                   # Sort elements
np.sort(arr, axis=0)           # Sort along axis 0
np.sort(arr, axis=-1)          # Sort along last axis
np.argsort(arr)                # Indices that would sort the array
np.argsort(arr, axis=0)        # Sorting indices along axis
np.lexsort([arr2, arr1])       # Sort by multiple keys
np.partition(arr, k)           # Partial sort (kth element in correct position)
np.argpartition(arr, k)        # Indices for partial sort

# UNIQUE AND COUNTING FUNCTIONS
np.unique(arr)                 # Unique elements
np.unique(arr, return_counts=True)     # Unique elements with counts
np.unique(arr, return_index=True)      # Unique elements with first indices
np.unique(arr, return_inverse=True)    # Unique elements with inverse indices
np.bincount(arr)               # Count occurrences of each value (non-negative ints)
np.histogram(arr, bins=10)     # Histogram counts and bin edges
np.histogram2d(x, y, bins=10)  # 2D histogram

# LOGICAL AGGREGATIONS
np.all(arr)                    # True if all elements are True
np.all(arr, axis=0)            # All along axis 0
np.any(arr)                    # True if any element is True
np.any(arr, axis=1)            # Any along axis 1

# CORRELATION AND COVARIANCE
np.corrcoef(x, y)              # Correlation coefficient matrix
np.cov(x, y)                   # Covariance matrix
np.correlate(x, y)             # Cross-correlation

# ADVANCED STATISTICAL FUNCTIONS
np.histogram_bin_edges(arr, bins=10)   # Bin edges for histogram
np.digitize(arr, bins)         # Indices of bins to which each value belongs
np.searchsorted(arr, values)   # Indices where values should be inserted
np.count_nonzero(arr)          # Count non-zero elements
np.count_nonzero(arr, axis=0)  # Count non-zero along axis

# ARRAY COMPARISON AGGREGATIONS
np.array_equal(arr1, arr2)     # True if arrays are element-wise equal
np.array_equiv(arr1, arr2)     # True if arrays are equivalent (broadcasting)
np.allclose(arr1, arr2)        # True if arrays are close (within tolerance)
np.isclose(arr1, arr2)         # Element-wise close comparison

# GRADIENT AND DIFFERENCES
np.gradient(arr)               # Gradient using central differences
np.gradient(arr, axis=0)       # Gradient along specific axis
np.diff(arr)                   # Discrete difference along last axis
np.diff(arr, n=2)              # Second-order differences
np.diff(arr, axis=0)           # Differences along axis 0
np.ediff1d(arr)                # Differences between consecutive elements

# INTERPOLATION AGGREGATIONS
np.interp(x, xp, fp)           # Linear interpolation
np.trapz(y, x)                 # Trapezoidal integration
np.trapz(y, dx=1.0)            # Trapezoidal integration with uniform spacing

# WINDOW FUNCTIONS (MOVING AGGREGATIONS)
def moving_average(arr, window_size):
    """Moving average using convolution"""
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def rolling_sum(arr, window_size):
    """Rolling sum using convolution"""
    return np.convolve(arr, np.ones(window_size), mode='valid')

# CONDITIONAL AGGREGATIONS
np.where(condition, x, y)      # Element-wise selection based on condition
np.select([cond1, cond2], [choice1, choice2], default=0)  # Multiple conditions
np.choose(indices, choices)    # Choose elements from multiple arrays

# MASKED ARRAY AGGREGATIONS (for arrays with missing data)
masked_arr = np.ma.masked_array(arr, mask=mask)
np.ma.sum(masked_arr)          # Sum of masked array
np.ma.mean(masked_arr)         # Mean of masked array
np.ma.std(masked_arr)          # Standard deviation of masked array
np.ma.min(masked_arr)          # Minimum of masked array
np.ma.max(masked_arr)          # Maximum of masked array

# REDUCTION WITH CUSTOM FUNCTIONS
np.apply_along_axis(func, axis, arr)    # Apply function along axis
np.apply_over_axes(func, arr, axes)     # Apply function over multiple axes

# EXAMPLES WITH SAMPLE DATA

# Create sample arrays
arr_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
arr_2d = np.array([[1, 2, 3, 4], 
                   [5, 6, 7, 8], 
                   [9, 10, 11, 12]])
arr_3d = np.random.random((3, 4, 5))

# Basic aggregations
print("1D Array:", arr_1d)
print("Sum:", np.sum(arr_1d))                    # 55
print("Mean:", np.mean(arr_1d))                  # 5.5
print("Std:", np.std(arr_1d))                    # 2.87
print("Min:", np.min(arr_1d))                    # 1
print("Max:", np.max(arr_1d))                    # 10
print("Median:", np.median(arr_1d))              # 5.5

# 2D Array aggregations
print("\n2D Array:")
print(arr_2d)
print("Sum all:", np.sum(arr_2d))                # 78
print("Sum axis 0:", np.sum(arr_2d, axis=0))     # [15 18 21 24] - column sums
print("Sum axis 1:", np.sum(arr_2d, axis=1))     # [10 26 42] - row sums
print("Mean axis 0:", np.mean(arr_2d, axis=0))   # [5. 6. 7. 8.] - column means
print("Max axis 1:", np.max(arr_2d, axis=1))     # [4 8 12] - row maxima

# Cumulative operations
print("\nCumulative operations:")
print("Cumsum:", np.cumsum(arr_1d))              # [1 3 6 10 15 21 28 36 45 55]
print("Cumprod:", np.cumprod([1, 2, 3, 4]))      # [1 2 6 24]

# Percentiles and quantiles
print("\nPercentiles:")
print("25th percentile:", np.percentile(arr_1d, 25))    # 3.25
print("75th percentile:", np.percentile(arr_1d, 75))    # 7.75
print("Multiple percentiles:", np.percentile(arr_1d, [25, 50, 75]))  # [3.25 5.5 7.75]

# Sorting and ordering
print("\nSorting:")
unsorted = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("Original:", unsorted)
print("Sorted:", np.sort(unsorted))              # [1 1 2 3 4 5 6 9]
print("Argsort:", np.argsort(unsorted))          # [1 3 6 0 2 4 7 5]

# Unique values
print("\nUnique values:")
repeated = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
unique_vals, counts = np.unique(repeated, return_counts=True)
print("Unique:", unique_vals)                    # [1 2 3 4]
print("Counts:", counts)                         # [1 2 3 4]

# Logical aggregations
bool_arr = np.array([True, False, True, True, False])
print("\nLogical aggregations:")
print("All:", np.all(bool_arr))                  # False
print("Any:", np.any(bool_arr))                  # True

# NaN handling
arr_with_nan = np.array([1, 2, np.nan, 4, 5])
print("\nNaN handling:")
print("Sum with NaN:", np.sum(arr_with_nan))     # nan
print("Sum ignoring NaN:", np.nansum(arr_with_nan))  # 12.0
print("Mean ignoring NaN:", np.nanmean(arr_with_nan))  # 3.0

# Conditional aggregations
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
condition = data > 5
print("\nConditional operations:")
print("Values > 5:", data[condition])            # [6 7 8 9 10]
print("Sum where > 5:", np.sum(data[condition])) # 40
print("Count > 5:", np.count_nonzero(condition)) # 5

# Multi-dimensional aggregations
print("\n3D Array aggregations:")
print("Shape:", arr_3d.shape)                    # (3, 4, 5)
print("Sum all:", np.sum(arr_3d))                # Sum of all elements
print("Sum axis 0 shape:", np.sum(arr_3d, axis=0).shape)  # (4, 5)
print("Sum axis (0,1) shape:", np.sum(arr_3d, axis=(0,1)).shape)  # (5,)

# Custom aggregation function
def custom_aggregation(arr):
    """Custom aggregation: sum of squares"""
    return np.sum(arr ** 2)

print("\nCustom aggregation:")
print("Sum of squares:", custom_aggregation(arr_1d))  # 385

# Weighted operations
values = np.array([1, 2, 3, 4, 5])
weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
print("\nWeighted operations:")
print("Weighted average:", np.average(values, weights=weights))  # 3.0

# Correlation example
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
print("\nCorrelation:")
print("Correlation coefficient:", np.corrcoef(x, y)[0, 1])  # 1.0 (perfect correlation)