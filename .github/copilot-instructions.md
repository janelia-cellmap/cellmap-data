# CellMap-Data Copilot Instructions

## Project Overview

CellMap-Data is a PyTorch-based data loading library for large-scale 2D/3D biological imaging datasets. It specializes in efficient streaming of OME-NGFF/Zarr data for deep learning training on cell microscopy data.

## Architecture Patterns

### Core Components (src/cellmap_data/)
- **CellMapDataset**: Main PyTorch Dataset subclass handling data loading and transformations
- **CellMapDataLoader**: Custom iterator replacing PyTorch DataLoader with optimized streaming
- **CellMapImage**: Handles individual image loading with TensorStore backend and spatial transformations
- **CellMapMultiDataset**: Combines multiple datasets for multi-scale or multi-modal training
- **CellMapDataSplit**: Manages train/validation splits across datasets

### Data Flow Architecture
1. **Image Loading**: TensorStore → XArray → Spatial transforms → Value transforms → PyTorch tensors
2. **Batching**: Custom iterator with memory-optimized streaming and CUDA stream management
3. **Transform Pipeline**: 
   - **Spatial transforms MUST come first**: mirror/rotate/transpose operations modify array geometry
   - **Value transforms applied after**: normalize/augment work on final spatial layout
   - **Critical ordering**: Spatial transforms affect coordinate systems; applying value transforms first breaks spatial consistency

### Key Dependencies & Integration Points
- **TensorStore**: Primary backend for efficient array access (not Zarr directly)
- **XArray**: Handles labeled multi-dimensional arrays with OME-NGFF metadata
- **PyDantic**: Used for OME-NGFF specification validation and settings
- **Custom Samplers**: MutableSubsetRandomSampler for weighted sampling strategies

## Development Conventions

### Testing Patterns
- Use `DummyDataset` classes for unit tests (see `tests/test_dataloader.py`)
- Mock arrays with specific shapes for memory calculation tests
- Force CPU for CI with `torch.backends.mps.is_available = lambda: False` in conftest.py
- GPU tests in separate files (`test_*_gpu.py`) with device detection

### Code Organization
- Main classes follow pattern: `__init__` → validation → source setup → data loading methods
- Transform functions in `transforms/` with separate `augment/` and `targets/` modules  
- Utility functions in `utils/` for dtype handling, array operations, and visualization

### Configuration Patterns
- Array specifications use dict format: `{"shape": (64,64,64), "scale": (8,8,8)}`
- Spatial transforms: `{"mirror": {"axes": {"x": 0.5}}, "rotate": {"axes": {"z": [-30,30]}}}`
- Support both single callable and per-target transform mappings
- Environment variables for performance tuning (e.g., `MIN_BATCH_MEMORY_FOR_STREAMS_MB`)

## Critical Implementation Details

### Memory Management
- Custom memory calculation in CellMapDataLoader based on array shapes and dtypes
- Automatic CUDA stream optimization when batch memory > threshold
- Use `non_blocking=True` for GPU transfers with proper stream synchronization

### Biological Data Specifics
- Default axis order is "zyx" (depth-height-width) not "xyz"
- Handle mutually exclusive class relationships via `class_relation_dict`
- **Multi-scale Data Processing**:
  - Uses OME-NGFF multiscale pyramids with automatic resolution matching
  - `target_scale` in CellMapImage automatically selects appropriate pyramid level
  - Different datasets can have different native resolutions - automatically harmonized
  - Example: raw data at 4nm/voxel, labels at 8nm/voxel → automatically resampled to match
- NaN handling for empty/invalid regions with configurable `empty_value`
- Class pixel counts normalized by resolution for proper weighted sampling across scales

### Performance Optimizations
- ThreadPoolExecutor for concurrent data loading
- Weighted sampling based on class pixel counts normalized by resolution
- Mutable samplers that can be refreshed without recreating loaders
- TensorStore context management for connection pooling

## Common Pitfalls

- Don't assume standard PyTorch DataLoader behavior - this uses custom iteration
- **Transform ordering is critical**: Spatial transforms must be applied before value transforms in the pipeline
  - Spatial transforms (mirror, rotate, transpose) change array coordinate systems
  - Value transforms (normalize, augment) work on pixel values in final spatial layout
  - Wrong order breaks spatial consistency and coordinate mapping
- Device transfers require explicit `.to(device)` calls on both dataset and loader
- Array shapes in config dicts are (z,y,x) not (x,y,z) for 3D data
- Use `force_has_data=True` when debugging empty array issues
- Multi-scale datasets: ensure `target_scale` matches across input and target arrays