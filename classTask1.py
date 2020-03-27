CLASS TASK:
Create two random arrays A and B, and multiply them. Get their result in C and add 1 to every element of C.

\n
import numpy as np
x = np.array([[1, 2], [4, 5]])
y = np.array([[7, 8], [9, 10]])
z = np.multiply(x,y)
​
print("x : \n", x)
print("y : \n", y)
print("\nMultiplication of array : ")
print(z)
z +=1
print("\nAfter adding 1 on each element of resulting array : ")
print(z)