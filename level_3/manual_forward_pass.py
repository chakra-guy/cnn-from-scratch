import numpy as np

from manual_relu import relu_naive
from manual_pool import max_pool2d_naive
from manual_conv import conv2d_naive


def run_manual_forward_pass():
    # 1. Input: A single 28x28 MNIST-like image (random noise for now)
    input_img = np.random.randn(28, 28)
    print(f"1. Input Shape:      {input_img.shape}")

    # 2. Define a random 3x3 Kernel
    kernel = np.random.randn(3, 3)

    # 3. Layer 1: Conv2d
    # Expected: 28 - 3 + 1 = 26
    x = conv2d_naive(input_img, kernel)
    print(f"2. After Conv2d:     {x.shape}")

    # 4. Activation: ReLU
    # Shape shouldn't change, but negative numbers should be gone
    x = relu_naive(x)
    print(f"3. After ReLU:       {x.shape} (Min value: {np.min(x):.2f})")

    # 5. Layer 2: MaxPool
    # Expected: (26 - 2)/2 + 1 = 13
    x = max_pool2d_naive(x, pool_size=2, stride=2)
    print(f"4. After MaxPool:    {x.shape}")


if __name__ == "__main__":
    run_manual_forward_pass()
