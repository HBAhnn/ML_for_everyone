import numpy as np

b= np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])


print(b[:, 0:1])

print(b[2:, 1:2])

print(b[3:, 1:])

print(b[ : , 1:])

print(b[3:, :])