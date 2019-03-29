"""
Name:           Carlos Meza
Date:           02/25/19
"""

import numpy as np
import matplotlib.pyplot as plt

# Collect data from csv file
data = np.genfromtxt('data-filtering.csv', delimiter=',') 
data = np.array(data, dtype=float)

""" CONSTANTS """
sampling_rate = 2000
L = 21
M = L - 1
# Array of ones to be replaced with added filtered numbers
fr = np.ones(L)
# Set x for time interval for wave application
x = np.arange(0, 1, 1/sampling_rate)
# Set size multiplier of wave of first 4hz signal ( in Hertz == Hz )
H1 = np.cos(2 * np.pi * 4 * x)
# Set size multiplier of wave of second 330hz signal ( in Hertz == Hz )
H2 = np.cos(2 * np.pi * 330 * x)
# Arrange time for wave which would be our x
x = np.arange(0, sampling_rate, sampling_rate/len(data))


""" PART A (Design a lowpass filter) """
# Function for the low Pass Filter
def low_pass(n):
    # Given values
    cut_freq = 50
    freq = cut_freq / sampling_rate
    # Determine filter weights
    if(n != (M/2)):
        filter_weight = np.sin(2 * np.pi * freq * (n - M/2)) / (np.pi * (n - M/2))
    else:
        filter_weight = 2 * freq
    # Return weight
    return filter_weight

# Populate frequency array
for i in range(L):
    fr[i] = H1[i] * low_pass(i)
# Convolve applied filter coefficients with original data
y = np.convolve(data, fr)


""" PART B (Apply the lowpass filter) """
plt.figure(0)
# Plot original source
plt.subplot(3, 1, 1)
plt.title("Original Source")
plt.plot(x, data)
plt.xlabel("seconds")
# Plot 4hz signal
plt.subplot(3, 1, 2)
plt.title("4 Hz signal")
plt.plot(x, H1)
plt.xlabel("seconds")
# Arange x to same length as y for graphing
x = np.arange(0, sampling_rate, sampling_rate/len(y))
# Plot after application of low pass filter
plt.subplot(3, 1, 3)
plt.title("application of low pass filter")
plt.plot(x, y)
plt.xlabel("seconds")
# Space out graphs to be clear to read
plt.tight_layout()


""" PART C (Design a highpass filter) """
# Function for the High Pass Filter
def high_pass(n):
    # Given Values
    cut_freq = 280
    freq = cut_freq / sampling_rate
    # Determine filter weights
    if(n != ( M/2 )):
       filter_weight = -np.sin(2 * np.pi * freq * (n - M/2)) / (np.pi * (n - M/2))
    else:
        filter_weight = 1 - (2 * freq)
    # Return weight
    return filter_weight

# Set x for time interval for wave application
x = np.arange(0, sampling_rate, sampling_rate/len(data))
# Populate frequency array
for k in range(L):
    fr[k] = H2[k] * high_pass(k)
# Convolve applied filter coefficients with original data
y = np.convolve(data, fr)


""" PART D (Apply highpass filter) """
plt.figure(1)
# Plot original source but only 100 values
plt.subplot(3, 1, 1)
plt.title("Original Source")
plt.plot(x[0:100], data[0:100])
plt.xlabel("seconds")
# Plot 330Hz signal but only 100 values
plt.subplot(3,1,2)
plt.title("330 Hz Signal")
plt.plot(x[0:100],H2[0:100])
plt.xlabel("seconds")
# Arange x to same length as y for graphing
x = np.arange(0, sampling_rate, sampling_rate/len(y))
# Plot the signal after highpass filter
plt.subplot(3,1,3)
plt.title("application of highpass filter")
plt.plot(x[0:100], y[0:100])
plt.xlabel("seconds")
# Space out graphs to be clear to read
plt.tight_layout()
