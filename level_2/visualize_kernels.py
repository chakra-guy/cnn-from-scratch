from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist
import matplotlib.pyplot as plt


# 1. Define the same model (Concept: Consistency)
class Model:
    def __init__(self):
        # We are interested in l1.weight -> Shape (32, 1, 6, 6)
        self.l1 = nn.Conv2d(1, 32, kernel_size=(6, 6))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.l3 = nn.Linear(1024, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2, 2))
        x = self.l2(x).relu().max_pool2d((2, 2))
        x = self.l3(x.flatten(1).dropout(0.5))
        return x


# 2. Quick Train (just to get patterns)
# We don't need high accuracy, just enough updates to form edges.
print("Training briefly to form patterns...")
X_train, Y_train, _, _ = mnist()
model = Model()
optim = nn.optim.Adam(nn.state.get_parameters(model))

# Run 100 steps only
for i in range(500):
    samp = Tensor.randint(128, high=X_train.shape[0])
    # Basic training step
    loss = (
        model(X_train[samp]).sparse_categorical_crossentropy(Y_train[samp]).backward()
    )
    Tensor.training = True
    optim.step()
    optim.zero_grad()

print("Training done. Extracting kernels...")

# 3. Extract the weights
# l1.weight shape is (32, 1, 6, 6) -> (Out_Channels, In_Channels, H, W)
kernels_l1 = model.l1.weight.numpy()
# l2.weight shape is (64, 32, 3, 3) -> (Out_Channels, In_Channels, H, W)
kernels_l2 = model.l2.weight.numpy()

# 4. Visualization - First Layer
fig1, axs1 = plt.subplots(4, 8, figsize=(10, 5))  # 4 rows * 8 cols = 32 filters
fig1.suptitle("The First Layer Eyes (32 Kernels of 6x6 size)")

for i in range(32):
    # Get the i-th kernel. It has shape (1, 6, 6), so we take [0] to get (6, 6)
    k = kernels_l1[i][0]

    # We plot it on the grid
    row = i // 8
    col = i % 8
    ax = axs1[row, col]

    # cmap='gray' isn't great for weights because weights can be negative.
    # We use 'bwr' (Blue-White-Red).
    # Blue = Negative (Inhibitory), Red = Positive (Excitatory), White = Zero (Ignore)
    im = ax.imshow(k, cmap="bwr", vmin=-0.5, vmax=0.5)
    ax.axis("off")

plt.tight_layout()
plt.show()

# 5. Visualization - Second Layer
fig2, axs2 = plt.subplots(8, 8, figsize=(10, 10))  # 8 rows * 8 cols = 64 filters
fig2.suptitle("The Second Layer Eyes (64 Kernels of 3x3 size)")

for i in range(64):
    # Get the i-th kernel. It has shape (32, 3, 3), so we take [0] to get (3, 3)
    # We show the first input channel for each output channel
    k = kernels_l2[i][0]

    # We plot it on the grid
    row = i // 8
    col = i % 8
    ax = axs2[row, col]

    im = ax.imshow(k, cmap="bwr", vmin=-0.5, vmax=0.5)
    ax.axis("off")

plt.tight_layout()
plt.show()
