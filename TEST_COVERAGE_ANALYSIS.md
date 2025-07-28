# Test Coverage Gap Analysis
## Week 4 Day 3-4 → Performance Optimization Transition

### Current Status Summary
- **Total Test Coverage**: 54% (1,262 of 2,743 statements missed)
- **Test Status**: 19 failed, 227 passed
- **Critical Issue**: Many test failures blocking performance optimization prep

### Priority Test Coverage Gaps

#### 1. Core Module Coverage Issues (Immediate Fix Required)

**Dataset Module (38% coverage)**
- **Missing**: Core initialization paths, parameter validation
- **Failing Tests**: Parameter mocking issues, ErrorMessages attribute errors
- **Impact**: Blocks dataset functionality testing

**Dataset Writer Module (36% coverage)**  
- **Missing**: Target bounds validation, image writer initialization
- **Failing Tests**: KeyError 'target' in target_bounds lookup
- **Root Cause**: Test setup using wrong key mapping (target_bounds keys don't match target_arrays keys)

**Image Writer Module (78% coverage)**
- **Missing**: Offset calculation edge cases  
- **Failing Tests**: List indices accessed with string keys
- **Root Cause**: Bounding box format mismatch (expecting dict, getting list)

#### 2. High-Impact Low Coverage Areas

**View Utilities (18% coverage)**
- **Files**: `src/cellmap_data/utils/view.py` 
- **Missing**: 206 of 252 statements uncovered
- **Impact**: Visualization and debugging capabilities

**Figure Utilities (9% coverage)**
- **Files**: `src/cellmap_data/utils/figs.py`
- **Missing**: 104 of 114 statements uncovered  
- **Impact**: Data analysis and presentation tools

**Metadata Utilities (44% coverage)**
- **Files**: `src/cellmap_data/utils/metadata.py`
- **Missing**: 29 of 52 statements uncovered
- **Impact**: Data provenance and configuration management

#### 3. Mock Configuration Issues (Immediate Fix Required)

**Problem Pattern**: Tests using `Mock(spec=Class)` cannot access magic methods
```python
# Failing pattern:
mock_image = Mock(spec=CellMapImage)  
mock_image.__getitem__.return_value = ...  # ❌ Fails

# Fix pattern:
mock_image = MagicMock(spec=CellMapImage)  # ✅ Works
```

**Parameter Validation Issues**: Missing ErrorMessages methods
```python
# Failing:
ErrorMessages.format_required_parameter()  # ❌ AttributeError

# Need to verify/fix:
# src/cellmap_data/utils/error_messages.py implementation
```

### Action Plan for Performance Optimization Readiness

#### Phase 1: Fix Critical Test Failures (Day 1)
1. **Fix Mock Configuration**
   - Replace `Mock(spec=...)` with `MagicMock(spec=...)` for magic method access
   - Fix ErrorMessages attribute errors in error handling utilities

2. **Fix Dataset Writer Target Bounds**
   - Correct key mapping between target_arrays and target_bounds in tests
   - Ensure bounding_box format consistency (dict vs list)

3. **Fix Image Writer Offset Calculation**
   - Verify bounding_box parameter format in ImageWriter initialization
   - Fix axis iteration over bounding_box structure

#### Phase 2: Address Coverage Gaps (Day 2)
1. **Prioritize Core Functionality**
   - Dataset initialization edge cases
   - DataLoader memory calculation accuracy
   - Parameter validation paths

2. **Add Utility Coverage** 
   - View utilities for debugging support
   - Metadata utilities for configuration management
   - Figure utilities for analysis tools

#### Phase 3: Performance Test Setup (Day 3)
1. **Benchmark Infrastructure**
   - Memory usage testing
   - I/O performance measurement  
   - Transform pipeline benchmarking

2. **Edge Case Robustness**
   - Large dataset handling
   - Memory-constrained environments
   - Concurrent access patterns

### Test Quality Improvements

#### Current Test Issues
1. **Over-mocking**: Tests mock too much, missing integration issues
2. **Parameter Setup**: Complex parameter configurations not properly tested
3. **Error Paths**: Exception handling paths under-tested

#### Improvement Strategies
1. **Integration Tests**: More end-to-end scenarios with real data flows
2. **Property-Based Testing**: Generate test cases for edge conditions
3. **Performance Regression Tests**: Ensure optimizations don't break functionality

### Coverage Targets for Performance Phase

**Minimum Thresholds for Optimization Work**:
- Core modules (dataset.py, dataloader.py): >80% coverage
- Transform modules: >90% coverage (already achieved)
- Error handling: >95% coverage (already achieved)
- Critical utilities: >70% coverage

**Current vs Target**:
- dataset.py: 38% → 80% (need +42%)
- dataset_writer.py: 36% → 70% (need +34%)  
- dataloader.py: 66% → 80% (need +14%)
- image.py: 61% → 70% (need +9%)

### Next Steps
1. **Fix all failing tests** to establish stable test baseline
2. **Implement missing core functionality tests** for critical paths
3. **Add performance measurement infrastructure** once stability achieved
4. **Begin performance optimization** with comprehensive test coverage

**Success Criteria**: All tests passing + >70% coverage on core modules before performance work begins.
