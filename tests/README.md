# CellMap-Data Test Suite

Comprehensive test coverage for the cellmap-data library using pytest with real implementations (no mocks).

## Overview

This test suite provides extensive coverage of all core components:

- **test_helpers.py**: Utilities for creating real Zarr/OME-NGFF test data
- **test_cellmap_image.py**: CellMapImage initialization and configuration
- **test_transforms.py**: All augmentation transforms with real tensors
- **test_cellmap_dataset.py**: CellMapDataset configuration and parameters
- **test_dataloader.py**: CellMapDataLoader setup and optimizations
- **test_multidataset_datasplit.py**: Multi-dataset and train/val splits
- **test_dataset_writer.py**: CellMapDatasetWriter for predictions
- **test_empty_image_writer.py**: EmptyImage and ImageWriter utilities
- **test_mutable_sampler.py**: MutableSubsetRandomSampler functionality
- **test_utils.py**: Utility function tests
- **test_integration.py**: End-to-end workflow integration tests
- **test_windows_stress.py**: TensorStore read-limiter unit tests, executor lifecycle, and concurrent-read stress tests

## Running Tests

### Prerequisites

Install the package with test dependencies:

```bash
pip install -e ".[test]"
```

Or install dependencies individually:

```bash
pip install pytest pytest-cov pytest-timeout
pip install torch torchvision tensorstore xarray zarr numpy
pip install pydantic-ome-ngff xarray-ome-ngff xarray-tensorstore
```

### Run All Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=cellmap_data --cov-report=html

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_cellmap_dataset.py -v
```

### Run Specific Test Categories

```bash
# Core component tests
pytest tests/test_cellmap_image.py tests/test_cellmap_dataset.py

# Transform tests
pytest tests/test_transforms.py

# DataLoader tests
pytest tests/test_dataloader.py

# Integration tests
pytest tests/test_integration.py

# Utility tests
pytest tests/test_utils.py tests/test_mutable_sampler.py
```

### Run Tests by Pattern

```bash
# Run all initialization tests
pytest tests/ -k "test_initialization"

# Run all configuration tests
pytest tests/ -k "test.*config"

# Run all integration tests
pytest tests/ -k "integration"
```

## Test Design Principles

### No Mocks - Real Implementations

All tests use real implementations:
- **Real Zarr arrays** with OME-NGFF metadata
- **Real TensorStore** backend for array access
- **Real PyTorch tensors** for data and transforms
- **Real file I/O** using temporary directories

This ensures tests validate actual behavior, not mocked interfaces.

### Test Data Generation

The `test_helpers.py` module provides utilities to create realistic test data:

```python
from tests.test_helpers import create_test_dataset

# Create a complete test dataset
config = create_test_dataset(
    tmp_path,
    raw_shape=(64, 64, 64),
    num_classes=3,
    raw_scale=(8.0, 8.0, 8.0),
)
# Returns paths, shapes, scales, and class names
```

### Fixtures and Reusability

Tests use pytest fixtures for common setups:

```python
@pytest.fixture
def test_dataset(self, tmp_path):
    """Create a test dataset for loader tests."""
    config = create_test_dataset(tmp_path, ...)
    return create_dataset_from_config(config)
```

## Test Coverage

### Core Components

- ✅ **CellMapImage**: Initialization, device selection, transforms, 2D/3D, dtypes
- ✅ **CellMapDataset**: Configuration, arrays, transforms, parameters, `close()` lifecycle
- ✅ **CellMapDataLoader**: Batching, workers, sampling, optimization
- ✅ **CellMapMultiDataset**: Combining datasets, multi-scale
- ✅ **CellMapDataSplit**: Train/val splits, configuration
- ✅ **CellMapDatasetWriter**: Prediction writing, bounds, multiple outputs
- ✅ **EmptyImage/ImageWriter**: Placeholders and writing utilities
- ✅ **MutableSubsetRandomSampler**: Weighted sampling, reproducibility
- ✅ **read_limiter**: Semaphore state, context manager correctness, deadlock safety, stress reads

### Transforms

- ✅ **Normalize**: Scaling, mean subtraction
- ✅ **GaussianNoise**: Noise addition, different std values
- ✅ **RandomContrast**: Contrast adjustment, ranges
- ✅ **RandomGamma**: Gamma correction, ranges
- ✅ **NaNtoNum**: NaN/inf replacement
- ✅ **Binarize**: Thresholding, different values
- ✅ **GaussianBlur**: Blur with different sigmas
- ✅ **Transform Composition**: Sequential application

### Utilities

- ✅ **Array operations**: Shape utilities, 2D detection
- ✅ **Coordinate transforms**: Scaling, translation
- ✅ **Dtype utilities**: Torch/numpy conversion, max values
- ✅ **Sampling utilities**: Weights, balancing
- ✅ **Path utilities**: Path splitting, class extraction

### Integration Tests

- ✅ **Training workflows**: Complete pipelines, transforms
- ✅ **Multi-dataset training**: Combining datasets, loaders
- ✅ **Train/val splits**: Complete workflows
- ✅ **Transform pipelines**: Complex augmentation sequences
- ✅ **Edge cases**: Small datasets, single class, anisotropic, 2D

## Test Organization

```
tests/
├── conftest.py                      # Pytest configuration
├── __init__.py                      # Test package init
├── README.md                        # This file
├── test_helpers.py                  # Test data generation utilities
├── test_cellmap_image.py           # CellMapImage tests
├── test_cellmap_dataset.py         # CellMapDataset tests
├── test_dataloader.py              # CellMapDataLoader tests
├── test_multidataset_datasplit.py  # MultiDataset/DataSplit tests
├── test_dataset_writer.py          # DatasetWriter tests
├── test_empty_image_writer.py      # EmptyImage/ImageWriter tests
├── test_mutable_sampler.py         # MutableSubsetRandomSampler tests
├── test_transforms.py              # Transform tests
├── test_utils.py                   # Utility function tests
├── test_integration.py             # Integration tests
└── test_windows_stress.py          # TensorStore read-limiter & concurrent stress tests
```

## Continuous Integration

Tests are designed to run in CI environments:

- **No GPU required**: Tests use CPU by default (configured in `conftest.py`)
- **Fast execution**: Tests use small datasets for speed
- **Isolated**: Each test uses temporary directories
- **Parallel-safe**: Tests can run in parallel with pytest-xdist

### CI Configuration

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pytest tests/ --cov=cellmap_data --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Extending Tests

### Adding New Test Files

1. Create new file: `tests/test_new_component.py`
2. Import test helpers: `from .test_helpers import create_test_dataset`
3. Use pytest fixtures for setup
4. Follow existing patterns for consistency

### Adding New Test Cases

```python
class TestNewComponent:
    """Test suite for new component."""
    
    @pytest.fixture
    def test_config(self, tmp_path):
        """Create test configuration."""
        return create_test_dataset(tmp_path, ...)
    
    def test_basic_functionality(self, test_config):
        """Test basic functionality."""
        # Use real data from test_config
        component = NewComponent(**test_config)
        assert component is not None
```

## Debugging Tests

### Run Single Test with Output

```bash
pytest tests/test_cellmap_dataset.py::TestCellMapDataset::test_initialization_basic -v -s
```

### Run with Debugger

```bash
pytest tests/test_cellmap_dataset.py --pdb
```

### Check Test Coverage

```bash
pytest tests/ --cov=cellmap_data --cov-report=term-missing
```

### Generate HTML Coverage Report

```bash
pytest tests/ --cov=cellmap_data --cov-report=html
# Open htmlcov/index.html in browser
```

## Known Limitations

### GPU Tests

GPU-specific tests are limited because:
- CI environments typically don't have GPUs
- GPU availability varies across systems
- Tests focus on CPU to ensure broad compatibility

GPU functionality can be tested manually:
```bash
# Run tests with GPU if available
CUDA_VISIBLE_DEVICES=0 pytest tests/
```

### Large-Scale Tests

Tests use small datasets for speed. For large-scale testing:
- Manually test with production-sized data
- Use integration tests with larger configurations
- Monitor memory usage and performance

### Windows Crash Regression Tests

`test_windows_stress.py::TestConcurrentGetitem::test_windows_high_concurrency_no_crash` is
skipped on non-Windows platforms (via `@pytest.mark.skipif`). To run it on Windows CI:

```yaml
# GitHub Actions — add a Windows runner
runs-on: windows-latest
steps:
  - run: pytest tests/test_windows_stress.py -v
```

A native TensorStore abort caused by concurrent reads will appear as a **non-zero process exit
code** rather than a Python exception; pytest will report the job as failed, which is the
correct CI signal.

The cross-platform deadlock and semaphore tests (`TestReadLimiterUnit`,
`TestExecutorLifecycle`, serial and multi-worker `TestConcurrentGetitem` tests) run on all
platforms and are included in the normal `pytest tests/` run.

## Contributing

When adding tests:

1. **Use real implementations** - no mocks unless absolutely necessary
2. **Use test helpers** - leverage existing test data generation
3. **Add docstrings** - explain what each test validates
4. **Keep tests fast** - use minimal datasets
5. **Test edge cases** - include boundary conditions
6. **Follow patterns** - maintain consistency with existing tests

## Questions or Issues

If you have questions about the tests or find issues:

1. Check this README for guidance
2. Look at existing tests for patterns
3. Review test helper utilities
4. Open an issue with specific questions
