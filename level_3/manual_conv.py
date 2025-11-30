import numpy as np

# 1. Create a dummy 5x5 "Image" (just simple numbers)
# Think of this as a small patch of a grayscale image
image = np.array(
    [
        [1, 2, 3, 0, 1],
        [0, 1, 2, 3, 0],
        [1, 0, 1, 0, 1],
        [2, 3, 0, 1, 2],
        [0, 1, 2, 3, 0],
    ]
)

# 2. Create a dummy 3x3 "Kernel" (Filter)
# This specific filter is a "Vertical Edge Detector"
# It highlights differences between left and right pixels.
kernel = np.array(
    [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ]
)


def conv2d_naive(img: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """
    Arguments:
        img: (H, W) numpy array
        kern: (kH, kW) numpy array
    Returns:
        output: (OutH, OutW) numpy array
    """
    H, W = img.shape
    kH, kW = kern.shape

    # Based on what we learned in Level 1:
    out_h = H - kH + 1
    out_w = W - kW + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # Extract the patch from the image at position (i, j)
            patch = img[i : i + kH, j : j + kW]
            # Multiply patch element-wise with kernel and sum
            output[i, j] = np.sum(patch * kern)

    return output


if __name__ == "__main__":
    # Test
    result = conv2d_naive(image, kernel)
    print("Input Shape:", image.shape)
    print("Kernel Shape:", kernel.shape)
    print("Output Shape:", result.shape)  # Should be (3, 3)
    print("\nOutput Values:\n", result)

    # Verification Logic
    # Let's manually calculate the very first pixel (top-left) to check your work.
    # Image top-left 3x3:
    # [[1, 2, 3],
    #  [0, 1, 2],
    #  [1, 0, 1]]
    #
    # Kernel:
    # [[1, 0, -1],
    #  [1, 0, -1],
    #  [1, 0, -1]]
    #
    # (1*1 + 2*0 + 3*-1) + (0*1 + 1*0 + 2*-1) + (1*1 + 0*0 + 1*-1)
    # (1 + 0 - 3)        + (0 + 0 - 2)        + (1 + 0 - 1)
    # -2                 + -2                 + 0
    # Total = -4
    #
    # If your code prints -4 for the top-left number, you win.
