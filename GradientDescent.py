import numpy as np

def gradient_descent(A, b, x, delta, epsilon, max_iter):
    errors = []

    for i in range(max_iter):
        gradient = 2 * A.T @ (A @ x - b)

        error = np.mean((A @ x - b) ** 2)
        errors.append(error)

        if np.linalg.norm(gradient) < delta:
            print(f"Converged in iteration {i}")
            break

        x -= epsilon * gradient

    return x, errors


def gradient_descent_train_test(A_train, A_test, b_train, b_test, x, delta, epsilon, max_iter):
    errors_train = []
    errors_test = []

    for i in range(max_iter):
        gradient = 2 * A_train.T @ (A_train @ x - b_train)

        error_train = np.mean((A_train @ x - b_train) ** 2)
        errors_train.append(error_train)

        error_test = np.mean((A_test @ x - b_test) ** 2)
        errors_test.append(error_test)

        if np.linalg.norm(gradient) < delta:
            print(f"Converged in iteration {i}")
            break

        x -= epsilon * gradient

    return x, errors_train, errors_test