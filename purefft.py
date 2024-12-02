import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import os
import math
from tinygrad import Tensor, nn as tg_nn, Device, dtypes

# Enable GPU if available
os.environ["GPU"] = "1"


def compute_fft_chunk(args):
    start_idx, end_idx, images = args
    ffts = []
    for img in images:
        fft = torch.abs(torch.fft.rfft(img.view(-1))).numpy()
        ffts.append(fft)
    return start_idx, np.array(ffts, dtype=np.float32)


class PrecomputedFFTDataset(Dataset):
    def __init__(self, dataset, cache_dir="fft_cache"):
        self.dataset = dataset
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(cache_dir, f"fft_cache_{len(dataset)}.pt")
        labels_file = os.path.join(cache_dir, f"labels_cache_{len(dataset)}.pt")

        if os.path.exists(cache_file) and os.path.exists(labels_file):
            print("Loading pre-computed FFTs from cache...")
            # Load directly into shared memory
            self.fft_cache = torch.load(cache_file)
            self.labels = torch.load(labels_file)
            # Move to shared memory for efficient access
            self.fft_cache.share_memory_()
            self.labels.share_memory_()
        else:
            print("Pre-computing FFTs using multiple processes...")
            # Prepare data in chunks for multiprocessing
            chunk_size = 1000
            chunks = []
            all_images = []
            all_labels = []

            for idx in range(len(dataset)):
                img, label = dataset[idx]
                all_images.append(img)
                all_labels.append(label)

            for i in range(0, len(dataset), chunk_size):
                end = min(i + chunk_size, len(dataset))
                chunks.append((i, end, all_images[i:end]))

            # Process chunks in parallel
            with ProcessPoolExecutor() as executor:
                results = list(
                    tqdm(
                        executor.map(compute_fft_chunk, chunks),
                        total=len(chunks),
                        desc="Computing FFTs",
                    )
                )

            # Combine results directly into PyTorch tensors
            results.sort(key=lambda x: x[0])
            all_ffts = np.concatenate([r[1] for r in results])
            self.fft_cache = torch.from_numpy(all_ffts)
            self.labels = torch.tensor(
                [label for label in all_labels], dtype=torch.long
            )

            # Move to shared memory
            self.fft_cache.share_memory_()
            self.labels.share_memory_()

            # Save as PyTorch tensors
            torch.save(self.fft_cache, cache_file)
            torch.save(self.labels, labels_file)
            print("FFT pre-computation complete and cached!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # No copy needed as we're using shared memory
        return self.fft_cache[idx], self.labels[idx]


def pure_forward(x, pattern_pool, layer_patterns, class_patterns, n_layers, freq_dim):
    # Forward pass - all pure operations
    curr = x  # [batch, freq]
    batch_size = curr.shape[0]
    
    for i in range(n_layers):
        # Pattern selection - [n_patterns]
        pattern_scores = (pattern_pool * layer_patterns[i]).sum(axis=1)
        pattern_weights = pattern_scores.softmax()
        
        # Expand patterns to match batch - [n_patterns, batch, freq]
        weighted_patterns = (pattern_pool.reshape(-1, 1, freq_dim) * 
                           pattern_weights.reshape(-1, 1, 1))
        
        # Expand curr to match patterns - [1, batch, freq]
        curr_expanded = curr.reshape(1, batch_size, freq_dim)
        
        # Multiply and sum - [batch, freq]
        curr = (curr_expanded * weighted_patterns).sum(axis=0)
    
    # Classification - expand for class dimension
    curr_expanded = curr.reshape(batch_size, 1, freq_dim)
    class_scores = (curr_expanded * class_patterns.reshape(1, 10, -1)).sum(axis=2)
    return class_scores.log_softmax()


class FFTPatternTransformer:
    def __init__(self, input_dim, n_patterns=8, n_layers=2):
        self.freq_dim = input_dim // 2 + 1
        self.n_layers = n_layers

        # Initialize pattern pool directly in frequency domain (real only)
        t = Tensor.arange(self.freq_dim, dtype=dtypes.float32)
        i = Tensor.arange(n_patterns, dtype=dtypes.float32).reshape(-1, 1)
        self.pattern_pool = Tensor.cos(2 * math.pi * i * t / n_patterns) * 0.02

        # Initialize layer patterns in frequency domain
        i = Tensor.arange(n_layers, dtype=dtypes.float32).reshape(-1, 1)
        self.layer_patterns = Tensor.cos(2 * math.pi * i * t / n_layers) * 0.02

        # Class filters in frequency domain
        i = Tensor.arange(10, dtype=dtypes.float32).reshape(-1, 1)
        self.class_patterns = Tensor.cos(2 * math.pi * i * t / 10) * 0.02

    def __call__(self, x):
        return pure_forward(Tensor(x.numpy(), dtype=dtypes.float32), 
                          self.pattern_pool, self.layer_patterns, self.class_patterns,
                          self.n_layers, self.freq_dim)


def train_model(model, train_loader, test_loader, epochs=10):
    print(f"Training on {Device.DEFAULT}")
    optim = tg_nn.optim.Adam([model.pattern_pool, model.layer_patterns, model.class_patterns], lr=0.001)

    for epoch in range(epochs):
        Tensor.training = True
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            optim.zero_grad()
            output = model(data)  # This is now log_softmax output
            target = Tensor(target.numpy(), dtype=dtypes.float32)
            # NLL loss with log_softmax outputs
            loss = output.mul(-1).gather(1, target.reshape(-1, 1)).mean()
            loss.backward()
            optim.step()

            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.numpy():.6f}"
                )

        Tensor.training = False
        correct = 0
        total = 0
        for data, target in test_loader:
            output = model(data)  # log_softmax output
            pred = output.argmax(axis=1)
            correct += (pred.numpy() == target.numpy()).sum()
            total += len(target)

        accuracy = 100.0 * correct / total
        print(
            f"Epoch {epoch}: Accuracy: {correct}/{total} "
            f"({accuracy:.2f}%), Time: {time.time() - start_time:.2f}s"
        )


def main():
    input_dim = 28 * 28
    batch_size = 256
    epochs = 12

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load original datasets
    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    # Wrap datasets with FFT pre-computation
    train_dataset = PrecomputedFFTDataset(train_dataset)
    test_dataset = PrecomputedFFTDataset(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = FFTPatternTransformer(input_dim)
    train_model(model, train_loader, test_loader, epochs)


if __name__ == "__main__":
    main()
