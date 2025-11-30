import numpy as np

# 4x4 Input
image = np.array([[1, 1, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 2, 3, 4]])


def max_pool2d_naive(img, pool_size=2, stride=2):
    H, W = img.shape

    # Calculate output shape
    # Formula: (Input - Kernel) / Stride + 1
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # 1. Calculate the starting 'r' (row) and 'c' (col) for the slice.
            # If i=0, start=0. If i=1, start=2 (because stride is 2).
            start_r = i * stride
            start_c = j * stride

            # 2. Slice the patch (pool_size x pool_size)
            patch = img[start_r : start_r + pool_size, start_c : start_c + pool_size]

            # 3. Find the max value in the patch and assign to output
            output[i, j] = np.max(patch)

    return output


if __name__ == "__main__":
    result = max_pool2d_naive(image)
    print("Input:\n", image)
    print("\nOutput:\n", result)

    # Verification:
    # Top-Left 2x2 is [[1, 1], [5, 6]]. Max is 6.
    # Top-Right 2x2 is [[2, 4], [7, 8]]. Max is 8.
    # Bottom-Left 2x2 is [[3, 2], [1, 2]]. Max is 3.
    # Bottom-Right 2x2 is [[1, 0], [3, 4]]. Max is 4.
    # Result should be [[6, 8], [3, 4]]
