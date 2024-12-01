from tinygrad import Tensor, nn, Device
from tinygrad.nn.datasets import mnist
import numpy as np
import os

# Enable GPU if available
os.environ["GPU"] = "1"

# Model Definition
class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu()
        x = x.max_pool2d((2, 2))
        x = self.l2(x).relu()
        x = x.max_pool2d((2, 2))
        x = x.flatten(1)
        x = x.dropout(0.5)
        return self.l3(x)

# Training Setup
def train_model():
    print(f"Using device: {Device.DEFAULT}")

    # Load MNIST dataset
    X_train, Y_train, X_test, Y_test = mnist()
    print(f"Dataset shapes: {X_train.shape}, {Y_train.shape}")

    model = Model()
    optim = nn.optim.Adam(nn.state.get_parameters(model))
    batch_size = 128
    epochs = 10
    batches_per_epoch = 100

    def training_step():
        Tensor.training = True
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        X, Y = X_train[samples], Y_train[samples]

        optim.zero_grad()
        out = model(X)
        loss = out.sparse_categorical_crossentropy(Y)
        loss.backward()
        optim.step()

        # Calculate accuracy
        accuracy = (out.argmax(axis=1) == Y).mean().item()

        return loss.item(), accuracy

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for i in range(batches_per_epoch):
            loss_val, accuracy = training_step()

            if (i + 1) % 10 == 0:
                print(f"Batch {i + 1}/{batches_per_epoch}: Loss = {loss_val:.4f}, Accuracy = {accuracy:.4f}")

        # Test accuracy after each epoch
        Tensor.training = False
        test_outputs = model(X_test)
        test_acc = (test_outputs.argmax(axis=1) == Y_test).mean().item()
        print(f"Test accuracy after epoch {epoch + 1}: {test_acc:.4f}")

if __name__ == '__main__':
    train_model()
