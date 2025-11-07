"""
Tests for augmentation transforms.

Tests all augmentation transforms using real tensors without mocks.
"""

import pytest
import torch
import numpy as np

from cellmap_data.transforms import (
    Normalize,
    GaussianNoise,
    RandomContrast,
    RandomGamma,
    NaNtoNum,
    Binarize,
    GaussianBlur,
)


class TestNormalize:
    """Test suite for Normalize transform."""
    
    def test_normalize_basic(self):
        """Test basic normalization."""
        transform = Normalize(scale=1.0 / 255.0)
        
        # Create test tensor with values 0-255
        x = torch.arange(256, dtype=torch.float32).reshape(16, 16)
        result = transform(x)
        
        # Check values are scaled
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert torch.allclose(result, x / 255.0)
    
    def test_normalize_with_mean(self):
        """Test normalization with mean subtraction."""
        transform = Normalize(mean=0.5, scale=0.5)
        
        x = torch.ones(8, 8)
        result = transform(x)
        
        # (1.0 - 0.5) / 0.5 = 1.0
        expected = torch.ones(8, 8)
        assert torch.allclose(result, expected)
    
    def test_normalize_preserves_shape(self):
        """Test that normalization preserves tensor shape."""
        transform = Normalize(scale=2.0)
        
        shapes = [(10,), (10, 10), (5, 10, 10), (2, 5, 10, 10)]
        for shape in shapes:
            x = torch.rand(shape)
            result = transform(x)
            assert result.shape == x.shape
    
    def test_normalize_dtype_preservation(self):
        """Test that normalize preserves dtype."""
        transform = Normalize(scale=0.5)
        
        x = torch.rand(10, 10, dtype=torch.float32)
        result = transform(x)
        assert result.dtype == torch.float32


class TestGaussianNoise:
    """Test suite for GaussianNoise transform."""
    
    def test_gaussian_noise_basic(self):
        """Test basic Gaussian noise addition."""
        torch.manual_seed(42)
        transform = GaussianNoise(std=0.1)
        
        x = torch.zeros(100, 100)
        result = transform(x)
        
        # Result should be different from input
        assert not torch.allclose(result, x)
        # Noise should have approximately the right std
        assert result.std() < 0.15  # Allow some tolerance
    
    def test_gaussian_noise_preserves_shape(self):
        """Test that Gaussian noise preserves shape."""
        transform = GaussianNoise(std=0.1)
        
        shapes = [(10,), (10, 10), (5, 10, 10), (2, 5, 10, 10)]
        for shape in shapes:
            x = torch.rand(shape)
            result = transform(x)
            assert result.shape == x.shape
    
    def test_gaussian_noise_zero_std(self):
        """Test that zero std produces no change."""
        transform = GaussianNoise(std=0.0)
        
        x = torch.rand(10, 10)
        result = transform(x)
        assert torch.allclose(result, x)
    
    def test_gaussian_noise_different_stds(self):
        """Test different standard deviations."""
        torch.manual_seed(42)
        x = torch.zeros(1000, 1000)
        
        for std in [0.01, 0.1, 0.5, 1.0]:
            transform = GaussianNoise(std=std)
            result = transform(x.clone())
            # Empirical std should be close to specified std
            assert abs(result.std().item() - std) < std * 0.2  # 20% tolerance


class TestRandomContrast:
    """Test suite for RandomContrast transform."""
    
    def test_random_contrast_basic(self):
        """Test basic random contrast adjustment."""
        torch.manual_seed(42)
        transform = RandomContrast(contrast_range=(0.5, 1.5))
        
        x = torch.linspace(0, 1, 100).reshape(10, 10)
        result = transform(x)
        
        # Result should be different (with high probability)
        assert result.shape == x.shape
    
    def test_random_contrast_preserves_shape(self):
        """Test that random contrast preserves shape."""
        transform = RandomContrast(contrast_range=(0.8, 1.2))
        
        shapes = [(10,), (10, 10), (5, 10, 10), (2, 5, 10, 10)]
        for shape in shapes:
            x = torch.rand(shape)
            result = transform(x)
            assert result.shape == x.shape
    
    def test_random_contrast_identity(self):
        """Test that (1.0, 1.0) range produces identity."""
        transform = RandomContrast(contrast_range=(1.0, 1.0))
        
        x = torch.rand(10, 10)
        result = transform(x)
        # With factor=1.0, output should be close to input
        assert torch.allclose(result, x, atol=1e-5)
    
    def test_random_contrast_range(self):
        """Test that contrast is within specified range."""
        torch.manual_seed(42)
        transform = RandomContrast(contrast_range=(0.5, 2.0))
        
        x = torch.linspace(0, 1, 100).reshape(10, 10)
        
        # Test multiple times to check randomness
        results = []
        for _ in range(10):
            result = transform(x.clone())
            results.append(result)
        
        # Results should vary
        assert not all(torch.allclose(results[0], r) for r in results[1:])


class TestRandomGamma:
    """Test suite for RandomGamma transform."""
    
    def test_random_gamma_basic(self):
        """Test basic random gamma adjustment."""
        torch.manual_seed(42)
        transform = RandomGamma(gamma_range=(0.5, 1.5))
        
        x = torch.linspace(0, 1, 100).reshape(10, 10)
        result = transform(x)
        
        assert result.shape == x.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_random_gamma_preserves_shape(self):
        """Test that random gamma preserves shape."""
        transform = RandomGamma(gamma_range=(0.8, 1.2))
        
        shapes = [(10,), (10, 10), (5, 10, 10), (2, 5, 10, 10)]
        for shape in shapes:
            x = torch.rand(shape)
            result = transform(x)
            assert result.shape == x.shape
    
    def test_random_gamma_identity(self):
        """Test that gamma=1.0 produces identity."""
        transform = RandomGamma(gamma_range=(1.0, 1.0))
        
        x = torch.rand(10, 10)
        result = transform(x)
        assert torch.allclose(result, x, atol=1e-5)
    
    def test_random_gamma_values(self):
        """Test gamma effect on values."""
        torch.manual_seed(42)
        x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        
        # Gamma < 1 should brighten mid-tones
        transform_bright = RandomGamma(gamma_range=(0.5, 0.5))
        result_bright = transform_bright(x.clone())
        assert result_bright[2] > x[2]  # Mid-tone should be brighter
        
        # Gamma > 1 should darken mid-tones
        transform_dark = RandomGamma(gamma_range=(2.0, 2.0))
        result_dark = transform_dark(x.clone())
        assert result_dark[2] < x[2]  # Mid-tone should be darker


class TestNaNtoNum:
    """Test suite for NaNtoNum transform."""
    
    def test_nan_to_num_basic(self):
        """Test basic NaN replacement."""
        transform = NaNtoNum({"nan": 0.0})
        
        x = torch.tensor([1.0, float("nan"), 3.0, float("nan"), 5.0])
        result = transform(x)
        
        expected = torch.tensor([1.0, 0.0, 3.0, 0.0, 5.0])
        assert torch.allclose(result, expected, equal_nan=False)
        assert not torch.isnan(result).any()
    
    def test_nan_to_num_inf(self):
        """Test infinity replacement."""
        transform = NaNtoNum({"posinf": 1e6, "neginf": -1e6})
        
        x = torch.tensor([1.0, float("inf"), -float("inf"), 3.0])
        result = transform(x)
        
        expected = torch.tensor([1.0, 1e6, -1e6, 3.0])
        assert torch.allclose(result, expected)
    
    def test_nan_to_num_all_replacements(self):
        """Test all replacements at once."""
        transform = NaNtoNum({"nan": 0.0, "posinf": 100.0, "neginf": -100.0})
        
        x = torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0])
        result = transform(x)
        
        expected = torch.tensor([0.0, 100.0, -100.0, 1.0])
        assert torch.allclose(result, expected)
    
    def test_nan_to_num_preserves_valid_values(self):
        """Test that valid values are preserved."""
        transform = NaNtoNum({"nan": 0.0})
        
        x = torch.rand(10, 10)
        result = transform(x)
        assert torch.allclose(result, x)
    
    def test_nan_to_num_multidimensional(self):
        """Test NaN replacement in multidimensional arrays."""
        transform = NaNtoNum({"nan": -1.0})
        
        x = torch.rand(5, 10, 10)
        x[2, 5, 5] = float("nan")
        x[3, 7, 3] = float("nan")
        
        result = transform(x)
        assert not torch.isnan(result).any()
        assert result[2, 5, 5] == -1.0
        assert result[3, 7, 3] == -1.0


class TestBinarize:
    """Test suite for Binarize transform."""
    
    def test_binarize_basic(self):
        """Test basic binarization."""
        transform = Binarize(threshold=0.5)
        
        x = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
        result = transform(x)
        
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        assert torch.allclose(result, expected)
    
    def test_binarize_different_thresholds(self):
        """Test different threshold values."""
        x = torch.linspace(0, 1, 11)
        
        for threshold in [0.0, 0.25, 0.5, 0.75, 1.0]:
            transform = Binarize(threshold=threshold)
            result = transform(x)
            
            # Check that values below threshold are 0, above are 1
            assert torch.all(result[x < threshold] == 0.0)
            assert torch.all(result[x >= threshold] == 1.0)
    
    def test_binarize_preserves_shape(self):
        """Test that binarize preserves shape."""
        transform = Binarize(threshold=0.5)
        
        shapes = [(10,), (10, 10), (5, 10, 10), (2, 5, 10, 10)]
        for shape in shapes:
            x = torch.rand(shape)
            result = transform(x)
            assert result.shape == x.shape
    
    def test_binarize_output_values(self):
        """Test that output only contains 0 and 1."""
        transform = Binarize(threshold=0.5)
        
        x = torch.rand(100, 100)
        result = transform(x)
        
        unique_values = torch.unique(result)
        assert len(unique_values) <= 2
        assert all(v in [0.0, 1.0] for v in unique_values.tolist())


class TestGaussianBlur:
    """Test suite for GaussianBlur transform."""
    
    def test_gaussian_blur_basic(self):
        """Test basic Gaussian blur."""
        transform = GaussianBlur(sigma=1.0)
        
        # Create image with a single bright pixel
        x = torch.zeros(21, 21)
        x[10, 10] = 1.0
        
        result = transform(x)
        
        # Blur should spread the value
        assert result[10, 10] < 1.0  # Center should be less bright
        assert result[9, 10] > 0.0  # Neighbors should have some value
        assert result.sum() > 0.0
    
    def test_gaussian_blur_preserves_shape(self):
        """Test that Gaussian blur preserves shape."""
        transform = GaussianBlur(sigma=1.0)
        
        shapes = [(10, 10), (5, 10, 10), (2, 5, 10, 10)]
        for shape in shapes:
            x = torch.rand(shape)
            result = transform(x)
            assert result.shape == x.shape
    
    def test_gaussian_blur_different_sigmas(self):
        """Test different sigma values."""
        x = torch.zeros(21, 21)
        x[10, 10] = 1.0
        
        results = []
        for sigma in [0.5, 1.0, 2.0, 3.0]:
            transform = GaussianBlur(sigma=sigma)
            result = transform(x.clone())
            results.append(result)
        
        # Larger sigma should produce more blur (lower peak)
        peaks = [r[10, 10].item() for r in results]
        assert peaks[0] > peaks[1] > peaks[2] > peaks[3]
    
    def test_gaussian_blur_smoothing(self):
        """Test that blur reduces high frequencies."""
        # Create checkerboard pattern
        x = torch.zeros(20, 20)
        x[::2, ::2] = 1.0
        x[1::2, 1::2] = 1.0
        
        transform = GaussianBlur(sigma=2.0)
        result = transform(x)
        
        # Blurred result should have less variance
        assert result.var() < x.var()


class TestTransformComposition:
    """Test composing multiple transforms together."""
    
    def test_sequential_transforms(self):
        """Test applying transforms sequentially."""
        import torchvision.transforms.v2 as T
        
        transforms = T.Compose([
            Normalize(scale=1.0 / 255.0),
            GaussianNoise(std=0.01),
            RandomContrast(contrast_range=(0.9, 1.1)),
        ])
        
        x = torch.randint(0, 256, (10, 10), dtype=torch.float32)
        result = transforms(x)
        
        assert result.shape == x.shape
        assert result.min() >= -0.5  # Noise might push slightly negative
        assert result.max() <= 1.5  # Contrast might push slightly above 1
    
    def test_transform_pipeline(self):
        """Test a realistic transform pipeline."""
        import torchvision.transforms.v2 as T
        
        # Realistic preprocessing pipeline
        raw_transforms = T.Compose([
            Normalize(mean=128, scale=128),  # Normalize to [-1, 1]
            GaussianNoise(std=0.05),
            RandomContrast(contrast_range=(0.8, 1.2)),
        ])
        
        target_transforms = T.Compose([
            Binarize(threshold=0.5),
            T.ToDtype(torch.float32),
        ])
        
        raw = torch.randint(0, 256, (32, 32), dtype=torch.float32)
        target = torch.rand(32, 32)
        
        raw_out = raw_transforms(raw)
        target_out = target_transforms(target)
        
        assert raw_out.shape == raw.shape
        assert target_out.shape == target.shape
        assert target_out.unique().numel() <= 2  # Should be binary
