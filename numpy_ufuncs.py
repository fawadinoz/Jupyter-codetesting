# NumPy Universal Functions (ufuncs) - Complete Reference

import numpy as np

# ARITHMETIC UFUNCS

# BASIC ARITHMETIC
np.add(x, y)           # x + y - element-wise addition
np.subtract(x, y)      # x - y - element-wise subtraction
np.multiply(x, y)      # x * y - element-wise multiplication
np.divide(x, y)        # x / y - element-wise division
np.true_divide(x, y)   # x / y - true division (same as divide)
np.floor_divide(x, y)  # x // y - floor division
np.mod(x, y)           # x % y - modulo/remainder
np.remainder(x, y)     # x % y - remainder (same as mod)
np.divmod(x, y)        # (x // y, x % y) - quotient and remainder
np.power(x, y)         # x ** y - element-wise power
np.float_power(x, y)   # x ** y - power with float result

# UNARY ARITHMETIC
np.negative(x)         # -x - element-wise negation
np.positive(x)         # +x - element-wise positive (no-op)
np.absolute(x)         # |x| - absolute value
np.abs(x)              # |x| - absolute value (alias)
np.fabs(x)             # |x| - floating-point absolute value
np.sign(x)             # sign of x (-1, 0, or 1)
np.conj(x)             # complex conjugate
np.conjugate(x)        # complex conjugate (same as conj)
np.reciprocal(x)       # 1/x - reciprocal
np.square(x)           # x**2 - square
np.sqrt(x)             # √x - square root
np.cbrt(x)             # ∛x - cube root

# EXPONENTIAL AND LOGARITHMIC UFUNCS
np.exp(x)              # e^x - exponential
np.exp2(x)             # 2^x - base-2 exponential
np.expm1(x)            # e^x - 1 - accurate for small x
np.log(x)              # ln(x) - natural logarithm
np.log2(x)             # log₂(x) - base-2 logarithm
np.log10(x)            # log₁₀(x) - base-10 logarithm
np.log1p(x)            # ln(1 + x) - accurate for small x
np.logaddexp(x, y)     # ln(e^x + e^y) - log of sum of exponentials
np.logaddexp2(x, y)    # log₂(2^x + 2^y) - base-2 version

# TRIGONOMETRIC UFUNCS
np.sin(x)              # sine
np.cos(x)              # cosine
np.tan(x)              # tangent
np.arcsin(x)           # arcsine (inverse sine)
np.arccos(x)           # arccosine (inverse cosine)
np.arctan(x)           # arctangent (inverse tangent)
np.arctan2(y, x)       # arctangent of y/x with correct quadrant
np.deg2rad(x)          # degrees to radians
np.rad2deg(x)          # radians to degrees
np.degrees(x)          # radians to degrees (same as rad2deg)
np.radians(x)          # degrees to radians (same as deg2rad)

# HYPERBOLIC UFUNCS
np.sinh(x)             # hyperbolic sine
np.cosh(x)             # hyperbolic cosine
np.tanh(x)             # hyperbolic tangent
np.arcsinh(x)          # inverse hyperbolic sine
np.arccosh(x)          # inverse hyperbolic cosine
np.arctanh(x)          # inverse hyperbolic tangent

# COMPARISON UFUNCS
np.equal(x, y)         # x == y - element-wise equality
np.not_equal(x, y)     # x != y - element-wise inequality
np.less(x, y)          # x < y - element-wise less than
np.less_equal(x, y)    # x <= y - element-wise less than or equal
np.greater(x, y)       # x > y - element-wise greater than
np.greater_equal(x, y) # x >= y - element-wise greater than or equal
np.maximum(x, y)       # element-wise maximum
np.minimum(x, y)       # element-wise minimum
np.fmax(x, y)          # element-wise maximum (ignores NaN)
np.fmin(x, y)          # element-wise minimum (ignores NaN)

# LOGICAL UFUNCS
np.logical_and(x, y)   # x & y - element-wise logical AND
np.logical_or(x, y)    # x | y - element-wise logical OR
np.logical_xor(x, y)   # x ^ y - element-wise logical XOR
np.logical_not(x)      # ~x - element-wise logical NOT

# BITWISE UFUNCS
np.bitwise_and(x, y)   # x & y - bitwise AND
np.bitwise_or(x, y)    # x | y - bitwise OR
np.bitwise_xor(x, y)   # x ^ y - bitwise XOR
np.bitwise_not(x)      # ~x - bitwise NOT (invert)
np.invert(x)           # ~x - bitwise NOT (same as bitwise_not)
np.left_shift(x, y)    # x << y - left shift
np.right_shift(x, y)   # x >> y - right shift

# FLOATING POINT UFUNCS
np.isfinite(x)         # test for finite values
np.isinf(x)            # test for infinity
np.isnan(x)            # test for NaN (Not a Number)
np.isnat(x)            # test for NaT (Not a Time) for datetime
np.signbit(x)          # test for sign bit
np.copysign(x, y)      # copy sign of y to magnitude of x
np.nextafter(x, y)     # next representable value after x towards y
np.spacing(x)          # distance to nearest adjacent number
np.modf(x)             # fractional and integer parts
np.ldexp(x, y)         # x * 2^y
np.frexp(x)            # mantissa and exponent of x

# ROUNDING UFUNCS
np.rint(x)             # round to nearest integer
np.floor(x)            # floor (largest integer <= x)
np.ceil(x)             # ceiling (smallest integer >= x)
np.trunc(x)            # truncate to integer (towards zero)
np.fix(x)              # round towards zero (same as trunc)
np.around(x, decimals) # round to given number of decimals
np.round_(x, decimals) # round to given number of decimals (same as around)

# COMPLEX NUMBER UFUNCS
np.real(x)             # real part of complex number
np.imag(x)             # imaginary part of complex number
np.angle(x)            # angle (argument) of complex number
np.conj(x)             # complex conjugate
np.conjugate(x)        # complex conjugate (same as conj)

# ERROR FUNCTION UFUNCS
np.erf(x)              # error function
np.erfc(x)             # complementary error function
np.erfcx(x)            # scaled complementary error function
np.erfinv(x)           # inverse error function
np.erfcinv(x)          # inverse complementary error function

# GAMMA FUNCTION UFUNCS
np.gamma(x)            # gamma function
np.gammaln(x)          # natural logarithm of gamma function
np.loggamma(x)         # natural logarithm of gamma function (same as gammaln)
np.digamma(x)          # digamma function (derivative of loggamma)
np.polygamma(n, x)     # nth derivative of digamma function

# BESSEL FUNCTION UFUNCS (if available)
# np.j0(x)             # Bessel function of the first kind of order 0
# np.j1(x)             # Bessel function of the first kind of order 1
# np.y0(x)             # Bessel function of the second kind of order 0
# np.y1(x)             # Bessel function of the second kind of order 1

# MISCELLANEOUS UFUNCS
np.heaviside(x, y)     # Heaviside step function
np.gcd(x, y)           # greatest common divisor
np.lcm(x, y)           # least common multiple

# UFUNC METHODS AND ATTRIBUTES

# REDUCTION METHODS
arr = np.array([1, 2, 3, 4, 5])
np.add.reduce(arr)     # Sum all elements: 1+2+3+4+5 = 15
np.multiply.reduce(arr) # Product of all elements: 1*2*3*4*5 = 120
np.maximum.reduce(arr) # Maximum element: 5
np.minimum.reduce(arr) # Minimum element: 1

# ACCUMULATION METHODS
np.add.accumulate(arr)      # Cumulative sum: [1, 3, 6, 10, 15]
np.multiply.accumulate(arr) # Cumulative product: [1, 2, 6, 24, 120]

# OUTER METHODS
arr1 = np.array([1, 2, 3])
arr2 = np.array([10, 20])
np.add.outer(arr1, arr2)    # Outer addition: [[11, 21], [12, 22], [13, 23]]
np.multiply.outer(arr1, arr2) # Outer multiplication: [[10, 20], [20, 40], [30, 60]]

# REDUCEAT METHOD
indices = [0, 2, 4]
np.add.reduceat(arr, indices) # Reduce at specified indices

# AT METHOD (IN-PLACE OPERATIONS)
arr = np.array([1, 2, 3, 4, 5])
np.add.at(arr, [0, 2, 4], 10)  # Add 10 to elements at indices 0, 2, 4

# UFUNC ATTRIBUTES
print(np.add.nin)          # Number of inputs: 2
print(np.add.nout)         # Number of outputs: 1
print(np.add.nargs)        # Total number of arguments: 3
print(np.add.ntypes)       # Number of supported type combinations
print(np.add.types)        # List of supported type signatures
print(np.add.identity)     # Identity element for reduction: 0
print(np.add.signature)    # Signature string (if available)

# UFUNC KEYWORD ARGUMENTS
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
out = np.empty(3)          # Pre-allocated output array
np.add(x, y, out=out)      # Store result in 'out' array

# WHERE PARAMETER
condition = np.array([True, False, True])
np.add(x, y, where=condition)  # Only perform operation where condition is True

# CASTING PARAMETER
np.add(x, y, casting='safe')   # Control type casting behavior

# DTYPE PARAMETER
np.add(x, y, dtype=np.float64) # Specify output data type

# EXAMPLES OF USAGE:

# Basic arithmetic operations
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

addition = np.add(a, b)        # [6, 8, 10, 12]
multiplication = np.multiply(a, b)  # [5, 12, 21, 32]
power = np.power(a, 2)         # [1, 4, 9, 16]

# Trigonometric functions
angles = np.array([0, np.pi/4, np.pi/2, np.pi])
sines = np.sin(angles)         # [0, 0.707, 1, 0]
cosines = np.cos(angles)       # [1, 0.707, 0, -1]

# Comparison operations
x = np.array([1, 5, 3, 8, 2])
y = np.array([2, 4, 3, 7, 9])
greater = np.greater(x, y)     # [False, True, False, True, False]
maximum = np.maximum(x, y)     # [2, 5, 3, 8, 9]

# Logical operations
bool_arr1 = np.array([True, False, True, False])
bool_arr2 = np.array([True, True, False, False])
logical_and = np.logical_and(bool_arr1, bool_arr2)  # [True, False, False, False]

# Reduction operations
numbers = np.array([1, 2, 3, 4, 5])
sum_all = np.add.reduce(numbers)      # 15
product_all = np.multiply.reduce(numbers)  # 120
cumulative_sum = np.add.accumulate(numbers)  # [1, 3, 6, 10, 15]

# Floating point tests
values = np.array([1.0, np.inf, np.nan, -np.inf])
finite_mask = np.isfinite(values)    # [True, False, False, False]
inf_mask = np.isinf(values)          # [False, True, False, True]
nan_mask = np.isnan(values)          # [False, False, True, False]

# Complex number operations
complex_nums = np.array([1+2j, 3+4j, 5+6j])
real_parts = np.real(complex_nums)   # [1, 3, 5]
imag_parts = np.imag(complex_nums)   # [2, 4, 6]
magnitudes = np.abs(complex_nums)    # [2.236, 5, 7.81]

# Broadcasting with ufuncs
scalar = 10
array = np.array([1, 2, 3, 4])
result = np.add(array, scalar)       # [11, 12, 13, 14] - broadcasting

# Using ufuncs with different shapes
matrix = np.array([[1, 2], [3, 4]])
vector = np.array([10, 20])
broadcasted = np.add(matrix, vector) # [[11, 22], [13, 24]]