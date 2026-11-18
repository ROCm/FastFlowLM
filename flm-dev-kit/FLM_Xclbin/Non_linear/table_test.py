# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import math
from matplotlib.pyplot import sca
import numpy as np




def sigmoid(x: float) -> float:
    """Computes the standard sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))



def sigmoid_multiple_scalar(x: float, scalar:float) -> float:
    """Computes the standard sigmoid function."""
    x_o = x
    x = x*scalar
    x =  1.0 / (1.0 + math.exp(-x))
    x *= x_o
    return x
def silu(x: float) -> float:
    """Computes the Sigmoid Linear Unit (SiLU/Swish) function."""
    return x * sigmoid(x)

def silu_derivative(x: float) -> float:
    """Computes the derivative of the SiLU function."""
    sig = sigmoid(x)
    return sig + x * sig * (1.0 - sig)

def softplus(x:float):
    """Computes the Softplus function."""
    return math.log(1.0 + math.exp(x))
    
def gelu(x: float) -> float:
    """Computes the Gaussian Error Linear Unit (GELU) function."""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
def gelu_derivative(x: float) -> float:
    """Computes the derivative of the GELU function."""
    # PDF of standard normal distribution
    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x**2)
    # CDF of standard normal distribution
    cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return cdf + x * pdf
# --- Tunable Hardware Parameters ---
LUT_ELEMS = 64 #TODO: change me

STEP_BITS = -2 #TODO: change me
BIAS = LUT_ELEMS//2
SAMPLES_PER_SEGMENT = 2000

assert (LUT_ELEMS//2) %16 == 0

# --- 1. Calculate Optimal Coefficients ---
coefficients = [0.0] * (  2*LUT_ELEMS)
# The floating point scaling factor
scaling_factor = 2.0**(-STEP_BITS)

print(f"Generating dense LUT with correct sub-interval mapping... (Scaling Factor: {scaling_factor})")
"""
# Goal: Reverse index = floor(input * scaling_factor) + BIAS
# to find the input range for each index 'i'.
# for i in range(LUT_ELEMS):
    
    # # --- Reverse the formula to solve for the input range ---
    # #   floor(input * scaling_factor) = i - BIAS
    # # This holds true when:
    # #   (i - BIAS) <= input * scaling_factor < (i - BIAS + 1)
    # # Dividing by scaling_factor gives the input range:
    # j = i - BIAS
    # x_start = j / scaling_factor
    # x_end = (j + 1) / scaling_factor

    # # Perform linear regression on the GELU function over this small sub-interval.
    # x_points = np.linspace(x_start, x_end, SAMPLES_PER_SEGMENT, endpoint=False)
    # y_points = np.array([gelu(x) for x in x_points])
    
    # slope, offset = np.polyfit(x_points, y_points, 1)
    
    # coefficients[i*2] = slope
    # coefficients[i*2 + 1] = offset
"""
for i in range(LUT_ELEMS):
    # Reverse the hardware formula to find the starting point of the input range
    # j = i - BIAS
    # x_start = j / scaling_factor

    # # Calculate the function value and derivative AT THE START of the interval
    # y_start = silu(x_start)
    # slope = silu_derivative(x_start) # The slope of the tangent line
    
    # # Calculate the offset from the point-slope form: y = m*x + b => b = y - m*x
    # offset = y_start - slope * x_start
    
    # coefficients[i*2] = slope
    # coefficients[i*2 + 1] = offset   
    j = i - BIAS
    x_start = j / scaling_factor
    x_end = (j + 1) / scaling_factor

    x_points = np.linspace(x_start, x_end, SAMPLES_PER_SEGMENT, endpoint=False)
    y_points = np.array([softplus(x) for x in x_points])
    
    slope, offset = np.polyfit(x_points, y_points, 1)
    
    coefficients[i*2] = slope
    coefficients[i*2 + 1] = offset        

# --- 2. Format Coefficients for Hardware Memory Layout ---
final_lut = []
# The repeating block is now 4 pairs (8 floats).
# With 64 total pairs, we have 64/4 = 16 unique blocks.
num_unique_4_pair_blocks = LUT_ELEMS // 4

print(f"Formatting into {num_unique_4_pair_blocks} unique blocks of 4 pairs, duplicated...")

USE_GELU = True # for sigmoid
USE_SIGMOD = False # TODO: change me# cornercase
coefficients[0] = 0
coefficients[1] = 0
if USE_GELU:
    coefficients[-2] = 1
    coefficients[-1] = 0
elif USE_SIGMOD:
    coefficients[-2] = 0
    coefficients[-1] = 1
for i in range(num_unique_4_pair_blocks):
    # Get the block of 4 pairs (8 floats)
    start_index = i * 8
    block = coefficients[start_index : start_index + 8]
    
    # Append the block twice to match the required memory pattern
    final_lut.extend(block)
    final_lut.extend(block)
    
# assert len(final_lut) == 256
# --- 3. Print the Final Table in C++ Array Format ---
print("\n// Generated SiLU LUT (Corrected Formatting)")
print(f"float chess_storage(% chess_alignof(v32int8)) silu_lut_ab[{len(final_lut)}] = "+"{")

for i in range(0, len(final_lut), 4):
    line_values = [f"{val:.8f}f" for val in final_lut[i:i+4]]
    print(f"    {', '.join(line_values)},")

print("};")