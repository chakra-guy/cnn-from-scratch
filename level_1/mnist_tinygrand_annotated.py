# type: ignore
from tinygrad import Device
from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn.datasets import mnist


print(Device.DEFAULT)


class Model:
    def __init__(self) -> None:
        # Layer 1: Input is (1, 28, 28)
        # We use 32 filters of size 3x3.
        # Output shape formula: (W - K + 1) -> (28 - 3 + 1) = 26
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3, 3))

        # Layer 2: We take the 32 channels as input. Output 64 channels.
        # We will calculate the spatial dimensions in the forward pass.
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3, 3))

        # Layer 3: The "Classifier"
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x: Tensor) -> Tensor:
        # x shape: (Batch_Size, 1, 28, 28)

        # 1. Conv: (28 - 3 + 1) = 26 -> (Batch, 32, 26, 26)
        # 2. MaxPool(2,2): Halves the size -> (Batch, 32, 13, 13)
        x = self.l1(x).relu().max_pool2d((2, 2))

        # 3. Conv: (13 - 3 + 1) = 11 -> (Batch, 64, 11, 11)
        # 4. MaxPool(2,2): Halves (rounds down) -> 5.5 -> 5 -> (Batch, 64, 5, 5)
        x = self.l2(x).relu().max_pool2d((2, 2))

        # 5. Flatten: We keep the batch, squash the rest.
        # 64 channels * 5 height * 5 width = 1600
        # Shape -> (Batch, 1600)
        x = self.l3(x.flatten(1).dropout(0.5))

        return x


X_train, Y_train, X_test, Y_test = mnist()
print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)
# (60000, 1, 28, 28) dtypes.uchar (60000,) dtypes.uchar


model = Model()
acc = (model(X_test).argmax(axis=1) == Y_test).mean()
# NOTE: tinygrad is lazy, and hasn't actually run anything by this point
print(acc.item())  # ~10% accuracy, as expected from a random model


optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128


@TinyJit
def step() -> Tensor:
    Tensor.training = True  # makes dropout work
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]
    optim.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optim.step()
    return loss


for cycle in range(7000):
    loss = step()
    if cycle % 100 == 0:
        Tensor.training = False
        acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
        print(f"cycle {cycle:4d}, loss {loss.item():.2f}, acc {acc * 100.0:.2f}%")
