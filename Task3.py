from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt
from GradientDescent import gradient_descent_train_test 

all_train_errors = []
all_test_errors = []

for i in range(10):
    A, b = load_diabetes(return_X_y=True)

    A_train, A_test, b_train, b_test = train_test_split(
        A, b, test_size=0.2, random_state=i
    )
    initial_x = np.zeros(A.shape[1])

    x, train_errors, test_errors = gradient_descent_train_test(A_train, A_test, b_train, b_test, initial_x, 1e-6, 0.01, 1000)

    all_train_errors.append(train_errors)
    all_test_errors.append(test_errors)

    print(f"Final Train MSE iteration {i}: {train_errors[-1]:.4f}")
    print(f"Final Test  MSE iteration {i}: {test_errors[-1]:.4f}")

mean_train = np.mean(all_train_errors, axis=0)
mean_test  = np.mean(all_test_errors, axis=0)
min_train = np.min(all_train_errors, axis=0)
min_test = np.min(all_test_errors, axis=0)

plt.figure(figsize=(8, 5))
plt.plot(mean_train, label='Average Train MSE')
plt.plot(mean_test,  label='Average Test MSE')
plt.plot(min_train, '--', label='Minimum Train MSE', alpha=0.6)
plt.plot(min_test,  '--', label='Minimum Test MSE', alpha=0.6)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figure3.png')
plt.show()