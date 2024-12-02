import numpy as np
import torch
from torch.fft import fft, ifft
import matplotlib.pyplot as plt
import itertools 


def freq_init(shape, scale=1.0):
    """Initialize weights directly in frequency domain with proper conjugate symmetry"""
    n = shape[0]
    freqs = np.zeros(shape, dtype=np.complex64)
    # Only fill first half + middle frequency (nyquist)
    half_n = (n // 2) + 1
    freqs[:half_n] = (
        (np.random.randn(half_n, *shape[1:]) + 1j * np.random.randn(half_n, *shape[1:]))
        * scale
        / np.sqrt(n)
    )
    # Ensure conjugate symmetry for real signal
    freqs[half_n:] = np.conj(freqs[1 : n - half_n + 1][::-1])
    # Make DC component real
    freqs[0] = freqs[0].real
    if n % 2 == 0:  # Make nyquist frequency real if even length
        freqs[n // 2] = freqs[n // 2].real
    return torch.from_numpy(freqs)


def verify_equivalence(seq_len=32, embed_dim=8, batch_size=2, threshold=1e-4):
    print("\nInitializing test with:")
    print(f"Sequence length: {seq_len}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Batch size: {batch_size}\n")

    # Initialize input and weights in frequency domain
    x_f = freq_init((seq_len, batch_size, embed_dim))
    Wq_f = freq_init((seq_len, embed_dim, embed_dim))
    Wk_f = freq_init((seq_len, embed_dim, embed_dim))

    print("Computing spatial domain attention...")
    # Convert to spatial domain for traditional attention
    x = ifft(x_f, dim=0).real  # (seq_len, batch, embed)
    Wq = ifft(Wq_f, dim=0).real  # (seq_len, embed, embed)
    Wk = ifft(Wk_f, dim=0).real  # (seq_len, embed, embed)

    # Traditional attention in spatial domain
    Q = x @ Wq  # (seq_len, batch, embed)
    K = x @ Wk  # (seq_len, batch, embed)
    scores_spatial = Q @ K.transpose(1, 2)  # (seq_len, batch, batch)

    print("Computing frequency domain attention...")
    # Frequency domain computation
    # Compute Q = x @ Wq in frequency domain
    Q_f = (
        torch.sum(x_f[:, :, None, :] * Wq_f[:, None, :, :], dim=-1) / seq_len
    )  # (seq, batch, embed)

    # Compute K = x @ Wk in frequency domain
    K_f = (
        torch.sum(x_f[:, :, None, :] * Wk_f[:, None, :, :], dim=-1) / seq_len
    )  # (seq, batch, embed)

    # Compute Q @ K.T in frequency domain
    scores_freq = (
        torch.sum(Q_f[:, :, None, :] * torch.conj(K_f[:, None, :, :]), dim=-1) / seq_len
    )  # (seq, batch, batch)
    scores_freq = ifft(scores_freq, dim=0).real * seq_len  # Scale back after IFFT

    print("\nShape information:")
    print(f"Spatial scores shape: {scores_spatial.shape}")
    print(f"Frequency scores shape: {scores_freq.shape}")

    print("\nValue ranges:")
    print(
        f"Spatial scores range: [{scores_spatial.min():.6f}, {scores_spatial.max():.6f}]"
    )
    print(f"Frequency scores range: [{scores_freq.min():.6f}, {scores_freq.max():.6f}]")

    # Convert to numpy for comparison and plotting
    scores_spatial = scores_spatial.numpy()
    scores_freq = scores_freq.numpy()

    # Verify equivalence
    max_diff = np.max(np.abs(scores_spatial - scores_freq))
    rel_diff = max_diff / (np.abs(scores_spatial).max() + 1e-8)
    print(f"\nAbsolute maximum difference: {max_diff:.8f}")
    print(f"Relative maximum difference: {rel_diff:.8f}")

    # Plot comparison
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(scores_spatial[0])
    plt.title("Spatial Domain Scores")
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(scores_freq[0])
    plt.title("Frequency Domain Scores")
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(np.abs(scores_spatial[0] - scores_freq[0]))
    plt.title("Absolute Difference")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("attention_equivalence.png")
    plt.close()

    return rel_diff < threshold  # Use relative difference with threshold


def sanity_check_attention_equivalence(
    seq_len=32, embed_dim=8, batch_size=2, seed=42, scale=1.0
):
    np.random.seed(seed)

    # Use your exact freq_init
    x_f = freq_init((seq_len, batch_size, embed_dim), scale)
    Wq_f = freq_init((seq_len, embed_dim, embed_dim), scale)
    Wk_f = freq_init((seq_len, embed_dim, embed_dim), scale)

    # Compute spatial domain values through ifft like your code
    x = ifft(x_f, dim=0).real
    Wq = ifft(Wq_f, dim=0).real
    Wk = ifft(Wk_f, dim=0).real

    # Spatial attention exactly as your code
    Q = x @ Wq
    K = x @ Wk
    scores_spatial = Q @ K.transpose(1, 2)

    # Frequency attention with your exact scaling
    Q_f = torch.sum(x_f[:, :, None, :] * Wq_f[:, None, :, :], dim=-1) / seq_len
    K_f = torch.sum(x_f[:, :, None, :] * Wk_f[:, None, :, :], dim=-1) / seq_len
    scores_freq = (
        torch.sum(Q_f[:, :, None, :] * torch.conj(K_f[:, None, :, :]), dim=-1) / seq_len
    )
    scores_freq = ifft(scores_freq, dim=0).real * seq_len

    # Print same diagnostics as your code
    print("sanity check2")
    print("\nShape information:")
    print(f"Spatial scores shape: {scores_spatial.shape}")
    print(f"Frequency scores shape: {scores_freq.shape}")

    print("\nValue ranges:")
    print(
        f"Spatial scores range: [{scores_spatial.min():.6f}, {scores_spatial.max():.6f}]"
    )
    print(f"Frequency scores range: [{scores_freq.min():.6f}, {scores_freq.max():.6f}]")

    # Use torch operations instead of numpy
    max_diff = torch.max(torch.abs(scores_spatial - scores_freq))
    rel_diff = max_diff / (torch.abs(scores_spatial).max() + 1e-8)
    print(f"\nAbsolute maximum difference: {max_diff:.8f}")
    print(f"Relative maximum difference: {rel_diff:.8f}")

    return scores_spatial, scores_freq, max_diff


def comprehensive_verification(
    seq_lengths=[32, 64, 128, 512],  # Test different sequence lengths
    embed_dims=[8, 16, 32, 64],  # Test different embedding sizes
    batch_sizes=[1, 2, 4, 8],  # Test different batch sizes
    seeds=[42, 123, 456],  # Multiple seeds for robustness
):
    results = {}

    for seq_len, embed_dim, batch_size, seed in itertools.product(
        seq_lengths, embed_dims, batch_sizes, seeds
    ):
        print(f"\nTesting configuration:")
        print(f"Sequence length: {seq_len}")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Batch size: {batch_size}")
        print(f"Seed: {seed}")

        # Run original verification
        scores_spatial, scores_freq, max_diff = sanity_check_attention_equivalence(
            seq_len=seq_len, embed_dim=embed_dim, batch_size=batch_size, seed=seed
        )

        # Store results
        key = (seq_len, embed_dim, batch_size, seed)
        results[key] = {
            "abs_diff": max_diff,
            "rel_diff": max_diff / (torch.abs(scores_spatial).max() + 1e-8),
            "spatial_range": (scores_spatial.min(), scores_spatial.max()),
            "freq_range": (scores_freq.min(), scores_freq.max()),
        }

        # Quick pattern correlation check
        pattern_corr = torch.corrcoef(
            torch.stack([scores_spatial.flatten(), scores_freq.flatten()])
        )[0, 1]
        print(f"Pattern correlation: {pattern_corr:.6f}")

    return results


def analyze_results(results):
    print("\nAnalysis Summary:")
    print("=================")

    # Analyze relative differences
    rel_diffs = [v["rel_diff"] for v in results.values()]
    print(
        f"Relative differences - Mean: {np.mean(rel_diffs):.6f}, Std: {np.std(rel_diffs):.6f}"
    )

    # Analyze scaling patterns
    for (seq_len, embed_dim, batch_size, seed), data in results.items():
        print(f"\nConfig {seq_len}x{embed_dim}x{batch_size} (seed {seed}):")
        spatial_range = data["spatial_range"]
        freq_range = data["freq_range"]
        print(
            f"Scale ratio: {(spatial_range[1]-spatial_range[0])/(freq_range[1]-freq_range[0]):.4f}"
        )


if __name__ == "__main__":
    # First run the original verification
    print("=== RUNNING BASIC VERIFICATION ===")
    np.random.seed(42)
    is_equivalent = verify_equivalence(threshold=1e-4)
    print(f"Basic verification result: {is_equivalent}\n")

    # Run comprehensive verification with diverse parameters
    print("=== RUNNING COMPREHENSIVE VERIFICATION ===")
    results = comprehensive_verification(
        seq_lengths=[32, 64, 256],  # Test a few different sequence lengths
        embed_dims=[8, 32, 128],  # Test different embedding dimensions
        batch_sizes=[2, 8],  # Test different batch sizes
        seeds=[42, 123],  # Test multiple seeds
    )

    # Analyze the results
    print("\n=== ANALYSIS ===")
    analyze_results(results)

    # Save visualization for a particularly interesting case
    print("\n=== GENERATING VISUALIZATION ===")
    verify_equivalence(
        seq_len=256,  # Larger sequence length for better visualization
        embed_dim=32,  # Moderate embedding dimension
        batch_size=8,  # Larger batch size
        threshold=1e-4,
    )
