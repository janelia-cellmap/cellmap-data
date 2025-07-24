# Phase 1 Execution Plan: Foundation Stabilization

## Overview

This document provides a concrete, step-by-step execution plan for Phase 1 of the CellMap-Data refactoring project. Phase 1 focuses on foundation stabilization through technical debt resolution, consistency improvements, and establishing refactoring infrastructure.

**Duration**: 8 weeks  
**Priority**: Critical (🔴)  
**Success Criteria**: Eliminate all TODO/FIXME items, establish consistent patterns, improve immediate maintainability  

## Week 1-2: Critical Technical Debt Resolution

### Week 1: TODO/FIXME Inventory and Prioritization

#### Day 1-2: Complete Technical Debt Audit ✅ COMPLETED

- [x] **Scan entire codebase for TODO/FIXME items**

  ```bash
  grep -r "TODO\|FIXME\|HACK" src/ --include="*.py" > technical_debt_inventory.txt
  grep -r "Warning\|bare except" src/ --include="*.py" >> technical_debt_inventory.txt
  ```
  **Results**: 24 total issues identified and documented

- [x] **Categorize by priority and complexity**:
  - **P0 (Critical)**: 3 issues - API breaking changes, security issues, data corruption risks
  - **P1 (High)**: 8 issues - Performance issues, resource leaks, incorrect behavior
  - **P2 (Medium)**: 9 issues - Code quality, maintainability
  - **P3 (Low)**: 4 issues - Documentation, minor improvements

- [x] **Create comprehensive audit document**: `TECHNICAL_DEBT_AUDIT.md`
  - Detailed breakdown of all 24 issues
  - Risk assessment and impact analysis
  - Prioritized action plan with time estimates
  - Success criteria and metrics

#### Day 3-5: Resolve P0 Critical Items ✅ COMPLETED
- [x] **Fix `src/cellmap_data/dataset_writer.py` critical issues**:
  ```python
  # TODO: This is a hacky temprorary fix. Need to figure out why this is happening
  # Line ~487: Review and fix coordinate transformation hack
  ```
  - **Action**: ✅ Investigated root cause of coordinate transformation issue
  - **Fix**: ✅ Added `_validate_index_bounds()` and `_safe_unravel_index()` methods
  - **Test**: ✅ Added comprehensive unit tests in `test_p0_fixes_focused.py`
  - **Timeline**: 2 days ✅ COMPLETED

- [x] **Fix `src/cellmap_data/transforms/augment/random_contrast.py` NaN hack**:
  ```python
  # Hack to avoid NaNs
  torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0, out=result)
  ```
  - **Action**: ✅ Implemented proper numerical stability handling
  - **Fix**: ✅ Added input validation and proper error handling in `forward()` method
  - **Test**: ✅ Added edge case tests for extreme contrast values and NaN/Inf detection
  - **Timeline**: 1 day ✅ COMPLETED

**✅ P0 FIXES COMPLETION**: All 3 critical issues resolved with:
- 14 comprehensive unit tests (100% pass rate)
- Full integration with existing test suite (91/91 tests pass)
- Complete removal of hack code from codebase
- Proper exception handling using `CellMapIndexError` and `CoordinateTransformError`
- Detailed completion report in `P0_FIXES_COMPLETION_REPORT.md`

### Week 2: P1 High Priority Items

#### Day 1-3: Parameter Name Standardization
- [ ] **Create API migration plan for breaking changes**:
  ```python
  # dataset.py, dataset_writer.py
  # TODO: Switch "raw_path" to "input_path"
  ```
  - **Action**: 
    1. Add deprecation warnings for `raw_path`
    2. Support both `raw_path` and `input_path` temporarily
    3. Update all internal usage to `input_path`
    4. Update documentation and examples
  - **Timeline**: 2 days

- [ ] **Standardize class relationship parameter names**:
  ```python
  # Inconsistent: class_relation_dict vs class_relationships
  ```
  - **Action**: Choose `class_relationships` as standard, add deprecation path
  - **Timeline**: 1 day

#### Day 4-5: Error Handling Infrastructure
- [ ] **Create centralized exception hierarchy**:
  ```python
  # Create src/cellmap_data/exceptions.py
  class CellMapDataError(Exception):
      """Base exception for CellMap-Data operations."""
      pass
  
  class DataLoadingError(CellMapDataError):
      """Errors during data loading operations."""
      pass
  
  class ValidationError(CellMapDataError):
      """Data validation errors."""
      pass
  
  class ConfigurationError(CellMapDataError):
      """Configuration and parameter errors."""
      pass
  ```
- [ ] **Update imports in `__init__.py`**
- [ ] **Add to documentation**

## Week 3-4: Error Handling Standardization

### Week 3: Implement Consistent Error Patterns

#### Day 1-2: Replace Bare Except Clauses
- [ ] **Fix `src/cellmap_data/utils/view.py`**:
  ```python
  # Current problematic pattern:
  try:
      array = array_future.result()
  except ValueError as e:
      Warning(e)  # Replace with proper logging
      UserWarning("Falling back to zarr3 driver")
  ```
  - **Action**: Replace with specific exception handling and proper logging
  - **Pattern**:
    ```python
    try:
        array = array_future.result()
    except ValueError as e:
        logger.warning(f"Failed to open with default driver: {e}")
        logger.info("Attempting fallback to zarr3 driver")
        # Fallback logic
    except Exception as e:
        raise DataLoadingError(f"Failed to load data from {path}: {e}") from e
    ```

#### Day 3-5: Standardize Error Messages
- [ ] **Create error message templates**:
  ```python
  # src/cellmap_data/messages.py
  ERROR_TEMPLATES = {
      'missing_array_shape': "Array info for '{array_name}' must include 'shape' parameter",
      'file_exists': "File already exists at {path}. Set overwrite=True to overwrite",
      'invalid_scale': "Invalid scale configuration for {array_name}: {details}",
  }
  ```
- [ ] **Apply consistent formatting across modules**
- [ ] **Update all assertion messages to use f-strings**

### Week 4: Logging Standardization

#### Day 1-3: Centralize Logging Configuration
- [ ] **Create `src/cellmap_data/logging_config.py`**:
  ```python
  import logging
  import sys
  from typing import Optional
  
  def setup_logging(
      level: str = "INFO",
      format_string: Optional[str] = None,
      handlers: Optional[list] = None
  ) -> logging.Logger:
      """Configure centralized logging for CellMap-Data."""
      if format_string is None:
          format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
      
      logger = logging.getLogger("cellmap_data")
      logger.setLevel(getattr(logging, level.upper()))
      
      if not logger.handlers:
          handler = logging.StreamHandler(sys.stdout)
          handler.setFormatter(logging.Formatter(format_string))
          logger.addHandler(handler)
      
      return logger
  ```

#### Day 4-5: Remove Hardcoded Logger Configuration
- [ ] **Remove all `logger.setLevel()` calls from modules**:
  ```python
  # Remove from all files:
  logger.setLevel(logging.INFO)  # Delete these lines
  ```
- [ ] **Update all modules to use centralized logger**:
  ```python
  # Standard pattern for all modules:
  import logging
  logger = logging.getLogger(__name__)
  ```

## Week 5-6: Code Quality Improvements

### Week 5: Import Organization and Unused Code

#### Day 1-2: Import Standardization
- [ ] **Organize imports in all modules using consistent groupings**:
  ```python
  # Standard library imports
  import json
  import logging
  import os
  from typing import Any, Optional, Sequence
  
  # Third-party imports
  import numpy as np
  import torch
  import tensorstore
  
  # Local imports
  from .exceptions import DataLoadingError
  from .utils import torch_max_value
  ```
- [ ] **Remove unused imports**:
  ```bash
  # Use autoflake to identify unused imports
  autoflake --check --imports=cellmap_data src/
  ```

#### Day 3-5: Dead Code Elimination
- [ ] **Identify and remove unused functions/classes**:
  ```bash
  # Use vulture to find dead code
  vulture src/ --min-confidence 60
  ```
- [ ] **Remove commented-out code blocks**
- [ ] **Consolidate duplicate validation logic**

### Week 6: Documentation Infrastructure

#### Day 1-3: Docstring Standardization
- [ ] **Create docstring templates**:
  ```python
  # Template for functions
  def example_function(param1: int, param2: str = "default") -> bool:
      """Brief description of what the function does.
      
      Parameters
      ----------
      param1 : int
          Description of param1.
      param2 : str, optional
          Description of param2 (default is "default").
      
      Returns
      -------
      bool
          Description of return value.
      
      Raises
      ------
      DataLoadingError
          When data loading fails.
      
      Examples
      --------
      >>> result = example_function(42, "test")
      True
      """
  ```

#### Day 4-5: Module Documentation
- [ ] **Add module-level docstrings**:
  ```python
  # At top of each module
  """Module for handling CellMap dataset operations.
  
  This module provides classes and functions for loading, processing,
  and managing CellMap biological imaging datasets for machine learning
  training workflows.
  """
  ```
- [ ] **Update `__init__.py` with comprehensive module documentation**

## Week 7-8: Testing and Validation

### Week 7: Test Infrastructure Improvements

#### Day 1-3: Test Organization
- [ ] **Reorganize test files by functionality**:
  ```
  tests/
  ├── unit/
  │   ├── test_dataset.py
  │   ├── test_transforms.py
  │   └── test_utils.py
  ├── integration/
  │   ├── test_dataloader_integration.py
  │   └── test_end_to_end.py
  └── conftest.py (shared fixtures)
  ```

#### Day 4-5: Error Path Testing
- [ ] **Add tests for all new exception types**:
  ```python
  def test_data_loading_error_handling():
      """Test that DataLoadingError is raised appropriately."""
      with pytest.raises(DataLoadingError, match="Failed to load data"):
          # Test error condition
  ```

### Week 8: Validation and Integration

#### Day 1-3: Regression Testing
- [ ] **Run full test suite and ensure no regressions**
- [ ] **Performance benchmarking to ensure no degradation**
- [ ] **Memory leak testing for ThreadPoolExecutor changes**

#### Day 4-5: Documentation Updates
- [ ] **Update API documentation for any breaking changes**
- [ ] **Update README examples with new parameter names**
- [ ] **Create migration guide for users**

## Quality Gates and Success Criteria

### Code Quality Metrics
- [ ] **Zero TODO/FIXME items remaining**
- [ ] **All modules have consistent import organization**
- [ ] **All public functions have proper docstrings**
- [ ] **No bare except clauses**
- [ ] **Consistent error handling patterns**

### Testing Requirements
- [ ] **All existing tests pass**
- [ ] **New tests for error handling paths**
- [ ] **Test coverage maintained or improved**
- [ ] **Performance benchmarks within 5% of baseline**

### Documentation Standards
- [ ] **All modules have module-level docstrings**
- [ ] **API documentation updated**
- [ ] **Migration guide created**
- [ ] **Code review checklist updated**

## Risk Mitigation

### Breaking Changes
- **Risk**: API changes break existing user code
- **Mitigation**: Implement deprecation warnings, maintain backward compatibility for 1 release cycle

### Performance Regression
- **Risk**: Refactoring introduces performance issues
- **Mitigation**: Continuous benchmarking, performance test suite

### Test Coverage Loss
- **Risk**: Refactoring breaks existing tests
- **Mitigation**: Run tests frequently, fix tests before code changes

## Deliverables

1. **Clean Codebase**: Zero technical debt items (TODO/FIXME)
2. **Consistent Patterns**: Standardized error handling, logging, imports
3. **Exception Hierarchy**: Centralized error handling infrastructure
4. **Documentation**: Complete module and API documentation
5. **Test Suite**: Enhanced testing with error path coverage
6. **Migration Guide**: Documentation for any breaking changes

## Next Phase Preparation

At the end of Phase 1, prepare for Phase 2 (Architecture Refactoring) by:
- [ ] **Identifying monolithic classes for refactoring** (CellMapDataset, CellMapImage)
- [ ] **Creating interface specifications** for extracted components
- [ ] **Planning device management separation**
- [ ] **Designing coordinate transformation utilities**

This foundation work in Phase 1 will enable more significant architectural changes in subsequent phases while maintaining code stability and user experience.
