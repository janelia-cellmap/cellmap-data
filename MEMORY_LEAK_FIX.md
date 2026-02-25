# Memory Leak Fix - Summary

## Problem

Training loops with CellMap data were experiencing severe memory leaks, with memory consumption growing from 0 to nearly 500GB over ~20 minutes despite:
- Batch size: ~350MB
- Configuration: `num_workers=11`, `persistent_workers=False`, `prefetch_factor=1`, `CELLMAP_TENSORSTORE_CACHE_BYTES=1`

## Root Cause

The issue was in `CellMapImage.array` property:

1. **Cached Property Accumulation**: The `array` property was decorated with `@cached_property`, meaning the xarray.DataArray was cached indefinitely in `__dict__`

2. **xarray Operations Create Intermediates**: During data retrieval, methods like:
   - `self.array.interp()` (for interpolation/upsampling)
   - `self.array.reindex()` (for padding)
   - `self.array.sel()` (for selection)
   
   These operations create new xarray DataArray objects that accumulate in memory

3. **No Cleanup**: The cached array and intermediate arrays were never freed, leading to unbounded memory growth across training iterations

## Solution

Added explicit cache clearing in `CellMapImage.__getitem__()`:

```python
def __getitem__(self, center: Mapping[str, float]) -> torch.Tensor:
    # ... retrieve and transform data ...
    
    # Clear cached array property to prevent memory accumulation
    self._clear_array_cache()
    
    return data
```

The `_clear_array_cache()` method removes the cached xarray wrapper from `__dict__`:

```python
def _clear_array_cache(self) -> None:
    """
    Clear the cached array property to free memory.
    
    Note: This only clears the Python-level xarray wrapper. 
    The underlying TensorStore connection and chunk cache 
    (managed by self.context) are preserved.
    """
    if "array" in self.__dict__:
        del self.__dict__["array"]
```

## Why This Works

1. **Prevents Accumulation**: By clearing the cache after each `__getitem__` call, we ensure xarray intermediate objects can be garbage collected

2. **Preserves Performance**: The TensorStore chunk cache (configured via `tensorstore_cache_bytes`) is managed by `self.context` and persists independently. We only clear the lightweight xarray wrapper, not the actual data cache

3. **Minimal Overhead**: Reopening the array on next access is fast because:
   - TensorStore maintains connections via the context
   - The chunk cache is unaffected
   - We're just recreating a thin Python wrapper

## Changes Made

1. **src/cellmap_data/image.py**:
   - Modified `__getitem__()` to call `_clear_array_cache()` after data retrieval
   - Added `_clear_array_cache()` method to explicitly remove cached array
   - Updated `array` property docstring to explain cache management
   - Added detailed documentation about TensorStore cache preservation

2. **tests/test_memory_management.py** (new file):
   - Tests that array cache is cleared after `__getitem__`
   - Tests that cache can be repopulated after clearing
   - Simulates training loop with multiple iterations
   - Tests cache clearing with interpolation, transforms, etc.

## Impact

- **Memory**: Bounded memory usage - array wrappers are garbage collected after each iteration
- **Performance**: Minimal impact - TensorStore chunk cache still provides performance benefits
- **Compatibility**: No breaking changes - existing code continues to work
- **Safety**: Fixes critical memory leak in long-running training loops

## Testing

The fix includes comprehensive tests:
- Cache clearing behavior
- Repopulation after clearing
- Simulated training loops
- Interaction with transforms and interpolation

To run tests:
```bash
pytest tests/test_memory_management.py -v
```

## Related Configuration

The fix works in conjunction with existing memory management features:
- `tensorstore_cache_bytes`: Bounds TensorStore's chunk cache
- `CELLMAP_TENSORSTORE_CACHE_BYTES`: Environment variable for cache size
- `persistent_workers`: Worker process lifecycle management

## Future Considerations

This fix addresses the immediate memory leak. Future optimizations could include:
- Monitoring memory usage metrics during training
- Adaptive cache clearing strategies
- Profile-guided cache retention for specific use cases
