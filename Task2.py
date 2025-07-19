from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt
from GradientDescent import gradient_descent_train_test 
A, b = load_diabetes(return_X_y=True)

A_train, A_test, b_train, b_test = train_test_split(
    A, b, test_size=0.2, random_state=42
)
initial_x = np.zeros(A.shape[1])

x, errors_train, errors_test = gradient_descent_train_test(A_train, A_test, b_train, b_test, initial_x, 1e-6, 0.01, 1000)

print(f"Final Train MSE: {errors_train[-1]:.4f}")
print(f"Final Test  MSE: {errors_test[-1]:.4f}")

plt.figure(figsize=(8,5))
plt.plot(errors_train, label='Train MSE')
plt.plot(errors_test,  label='Test MSE')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.tight_layout()
plt.savefig('figure2.png')
plt.show()