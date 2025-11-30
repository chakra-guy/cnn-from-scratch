from tinygrad.nn.datasets import mnist
import matplotlib.pyplot as plt

# 1. Load the data
print("Loading data...")
X_train, Y_train, X_test, Y_test = mnist()

# 2. Check the raw shapes
# N=Number of images, C=Channels, H=Height, W=Width
print(f"X_train shape: {X_train.shape}")  # Should be (60000, 1, 28, 28)
print(f"Y_train shape: {Y_train.shape}")  # Should be (60000,)

# 3. Visualize the first 5 examples
fig, axs = plt.subplots(1, 5, figsize=(15, 3))

for i in range(5):
    # We need to grab the image.
    # X_train[i] is a Tensor of shape (1, 28, 28).
    # .numpy() converts it to a format matplotlib understands
    # [0] removes the channel dimension to make it (28, 28)
    img_data = X_train[i].numpy()[0]
    label = Y_train[i].numpy()

    axs[i].imshow(img_data, cmap="gray")
    axs[i].set_title(f"Label: {label}")
    axs[i].axis("off")

plt.tight_layout()
plt.show()
