import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from GradientDescent import gradient_descent

A, b = load_diabetes(return_X_y=True)

initial_x = np.zeros(A.shape[1])
x, errors = gradient_descent(A, b, initial_x, 1e-6, 0.01, 1000)

x_star = np.linalg.inv(A.T @ A) @ A.T @ b
minimum_error = np.mean((A @ x_star - b) ** 2)

relative_diff = 100 * abs(minimum_error - errors[-1]) / minimum_error

print(f"The analytic error is {minimum_error}, the algorithm's error is {errors[-1]}")
print(f"The difference between the exprcted error and the actual error is {relative_diff}")

plt.figure(figsize=(8,5))
plt.plot(errors)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.grid(True)
plt.tight_layout()
plt.savefig('figure1.png')
plt.show()