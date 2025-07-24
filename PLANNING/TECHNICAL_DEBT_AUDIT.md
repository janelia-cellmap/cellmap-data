# Technical Debt Audit - Phase 1 Week 1

**Audit Date**: 2025-07-24  
**Auditor**: System Analysis  
**Scope**: Complete codebase scan for TODO/FIXME/HACK/Warning patterns  

## Summary Statistics

- **Total Issues Found**: 24
- **P0 Critical**: 3 issues
- **P1 High**: 8 issues
- **P2 Medium**: 9 issues  
- **P3 Low**: 4 issues

## P0 Critical Issues (🔴) - Immediate Action Required

### 1. Coordinate Transformation Hack - CRITICAL
**File**: `src/cellmap_data/dataset.py:487`  
**File**: `src/cellmap_data/dataset_writer.py:296`  
```python
# TODO: This is a hacky temprorary fix. Need to figure out why this is happening
```
**Impact**: Data corruption risk, potential incorrect training data  
**Priority**: CRITICAL - Fix in Days 3-5  
**Estimated Effort**: 2 days  
**Root Cause**: Coordinate transformation system not properly understood/implemented  

### 2. NaN Handling Hack - CRITICAL
**File**: `src/cellmap_data/transforms/augment/random_contrast.py:40`  
```python
# Hack to avoid NaNs
torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0, out=result)
```
**Impact**: Potential training instability, hidden numerical issues  
**Priority**: CRITICAL - Fix in Days 3-5  
**Estimated Effort**: 1 day  
**Root Cause**: Improper numerical stability handling in contrast transforms  

### 3. Missing Test Coverage - CRITICAL
**File**: `src/cellmap_data/dataset.py:634`  
**File**: `src/cellmap_data/dataset_writer.py:386`  
```python
# TODO: ADD TEST
```
**Impact**: Core functionality untested, regression risk  
**Priority**: CRITICAL  
**Estimated Effort**: 1 day each  

## P1 High Priority Issues (🟡) - Week 2 Target

### 4. API Breaking Parameter Names - HIGH
**Files**: Multiple locations  
```python
raw_path: str,  # TODO: Switch "raw_path" to "input_path"
```
**Locations**:
- `src/cellmap_data/dataset.py:32, 46`
- `src/cellmap_data/dataset_writer.py:27`

**Impact**: API consistency, user confusion  
**Priority**: HIGH - Week 2 Days 1-3  
**Estimated Effort**: 2 days (includes deprecation warnings)  

### 5. Data Type Assumptions - HIGH
**File**: `src/cellmap_data/image.py:35-36`  
```python
target_scale: Sequence[float],  # TODO: make work with dict
target_voxel_shape: Sequence[int],  # TODO: make work with dict
```
**Impact**: API flexibility, type safety  
**Priority**: HIGH  
**Estimated Effort**: 1.5 days  

### 6. Robustness Issues - HIGH
**Files**: Multiple locations  
```python
# TODO: make more robust
```
**Locations**:
- `src/cellmap_data/dataset.py:625`
- `src/cellmap_data/dataset_writer.py:377`

**Impact**: Error handling, production stability  
**Priority**: HIGH  
**Estimated Effort**: 1 day each  

### 7. Implementation Review Needed - HIGH
**File**: `src/cellmap_data/multidataset.py:122`  
```python
# TODO: review this implementation
```
**Impact**: Correctness of class weight calculations  
**Priority**: HIGH  
**Estimated Effort**: 1 day  

### 8. Grayscale Assumptions - HIGH
**File**: `src/cellmap_data/dataset_writer.py:300`  
```python
# TODO: Assumes 1 channel (i.e. grayscale)
```
**Impact**: Multi-channel data support limitation  
**Priority**: HIGH  
**Estimated Effort**: 1 day  

## P2 Medium Priority Issues (🟡) - Week 3-4 Target

### 9. Coordinate Transformation Architecture - MEDIUM
**File**: `src/cellmap_data/dataset.py:606`  
```python
# TODO: Should do as many coordinate transformations as possible at the dataset level
# (duplicate reference frame images should have the same coordinate transformations)
# --> do this per array, perhaps with CellMapArray object
```
**Impact**: Performance optimization, architecture improvement  
**Priority**: MEDIUM  
**Estimated Effort**: 3 days (major refactoring)  

### 10. Array Size Configuration - MEDIUM
**File**: `src/cellmap_data/datasplit.py:192`  
```python
# TODO: probably want larger arrays for validation
```
**Impact**: Validation effectiveness  
**Priority**: MEDIUM  
**Estimated Effort**: 0.5 days  

### 11-18. Warning/UserWarning Patterns - MEDIUM
Multiple instances of improper warning usage:

**Locations**:
- `src/cellmap_data/utils/view.py:278-279`
- `src/cellmap_data/datasplit.py:254` 
- `src/cellmap_data/image_writer.py:121-122`
- `src/cellmap_data/multidataset.py` (multiple)
- `src/cellmap_data/image.py` (multiple)

**Pattern**:
```python
Warning(e)  # Should be warnings.warn()
UserWarning("message")  # Should be warnings.warn(message, UserWarning)
```

**Impact**: Improper error signaling, debugging difficulty  
**Priority**: MEDIUM  
**Estimated Effort**: 2 days total (standardize warning patterns)  

## P3 Low Priority Issues (🟢) - Future Enhancement

### 19. Visualization Enhancement - LOW
**File**: `src/cellmap_data/utils/figs.py:254`  
```python
# TODO: Get list of figs for the batches, instead of one fig per class
```
**Impact**: User experience improvement  
**Priority**: LOW  
**Estimated Effort**: 1 day  

## Action Plan Summary

### Week 1 Days 3-5: P0 Critical Items
1. **Fix coordinate transformation hack** (2 days)
   - Investigate root cause in dataset.py:487 and dataset_writer.py:296
   - Implement proper coordinate transformation logic
   - Add comprehensive unit tests

2. **Fix NaN handling hack** (1 day)
   - Replace torch.nan_to_num hack with proper numerical stability
   - Add edge case tests for extreme contrast values
   - Validate against existing datasets

3. **Add missing test coverage** (1 day)
   - Add tests for untested methods in dataset.py:634 and dataset_writer.py:386

### Week 2: P1 High Priority Items
1. **API parameter standardization** (2 days)
   - Create deprecation path for raw_path → input_path
   - Update documentation and examples
   - Add migration warnings

2. **Type system improvements** (1.5 days)
   - Make target_scale and target_voxel_shape work with dict types
   - Improve type annotations

3. **Robustness improvements** (2 days)
   - Review and strengthen error handling in flagged methods
   - Add input validation

### Week 3-4: P2 Medium Priority Items
1. **Warning system standardization** (2 days)
   - Replace improper Warning() calls with warnings.warn()
   - Standardize warning categories and messages

2. **Architecture improvements** (3 days)
   - Begin coordinate transformation refactoring
   - Plan CellMapArray object introduction

## Risk Assessment

### High Risk
- Coordinate transformation hack: Could affect all training data
- NaN handling hack: Potential silent model degradation

### Medium Risk  
- Missing test coverage: Regression potential
- API inconsistencies: User confusion, adoption barriers

### Low Risk
- Warning patterns: Development/debugging inconvenience
- Documentation TODOs: User experience impact

## Dependencies and Blockers

### External Dependencies
- None identified for P0/P1 items

### Internal Dependencies
- Coordinate transformation fix may require CellMapImage class refactoring
- API changes require coordination with documentation team

## Success Metrics

### Week 1 Success Criteria
- [ ] All P0 critical hacks resolved
- [ ] No TODO/FIXME items remaining in core data loading path
- [ ] Test coverage added for critical untested methods
- [ ] All changes have corresponding tests

### Week 2 Success Criteria  
- [ ] API parameter names standardized with deprecation warnings
- [ ] Type system improvements completed
- [ ] Robustness improvements implemented
- [ ] All P1 items resolved or have clear implementation plans

## Notes

- This audit focused on explicit TODO/FIXME/HACK markers
- Additional technical debt may exist in code patterns and architecture
- Coordinate transformation issues appear to be systemic and may require broader refactoring
- Warning system needs comprehensive standardization beyond just the flagged instances
