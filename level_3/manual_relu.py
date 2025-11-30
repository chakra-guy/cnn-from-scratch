import numpy as np


def relu_naive(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


if __name__ == "__main__":
    # Example usage:
    x = np.array([[-1, 0, 1], [2, -3, 4]])

    result = relu_naive(x)
    print("Input:\n", x)
    print("\nOutput:\n", result)

    # Verification:
    # ReLU activation sets all negative values to 0 and keeps non-negative values unchanged.
    # Top row: -1 -> 0, 0 -> 0, 1 -> 1
    # Bottom row: 2 -> 2, -3 -> 0, 4 -> 4
    # Result should be [[0 0 1]
    #                   [2 0 4]]
