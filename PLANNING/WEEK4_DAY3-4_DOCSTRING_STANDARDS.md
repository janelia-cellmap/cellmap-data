# Week 4 Day 3-4: Docstring Standardization Implementation

## 🎯 Objective
Standardize all docstrings across the CellMap-Data codebase to use NumPy-style format as specified in CONTRIBUTING.md, ensuring consistent, professional documentation that supports API documentation generation.

## 📊 Current State Analysis

### Issues Identified
1. **Mixed Format Usage**:
   - Most docstrings use Google-style (`Args:`, `Returns:`)
   - CONTRIBUTING.md specifies NumPy-style (`Parameters`, `-------`)
   - Some functions have minimal one-line docstrings
   - Inconsistent parameter documentation depth

2. **Specific Examples**:
   - `dataset.py` CellMapDataset.__init__: Google-style with incomplete structure
   - `dataloader.py` methods: Brief one-liners without parameter docs
   - `transforms/normalize.py`: Good Google-style example but wrong format
   - `image.py` CellMapImage.__init__: Incomplete parameter documentation

## 🎨 Standardization Template

### NumPy-Style Docstring Template
```python
def example_function(param1: str, param2: int = 0, param3: Optional[bool] = None) -> bool:
    """Brief description of function purpose.
    
    Longer description with usage examples and important notes about behavior,
    edge cases, or implementation details.
    
    Parameters
    ----------
    param1 : str
        Description of parameter with type and purpose.
    param2 : int, optional
        Description with default value info, by default 0.
    param3 : bool, optional
        Description of optional parameter, by default None.
        
    Returns
    -------
    bool
        Description of return value and what it represents.
        
    Raises
    ------
    ValueError
        When invalid input provided or constraint violated.
    DataLoadingError
        When data cannot be loaded from specified path.
        
    Examples
    --------
    >>> result = example_function("test", 5)
    >>> print(result)
    True
    
    >>> # Example with optional parameter
    >>> result = example_function("data", param3=True)
    >>> print(result)
    False
    
    Notes
    -----
    Important implementation details, performance considerations,
    or relationships to other functions.
    
    See Also
    --------
    related_function : Related functionality
    another_class : Alternative approach
    """
```

### Class Docstring Template
```python
class ExampleClass:
    """Brief description of class purpose and responsibility.
    
    Longer description explaining the class design, main use cases,
    and how it fits into the broader system architecture.
    
    Parameters
    ----------
    init_param : str
        Description of initialization parameter.
    optional_param : int, optional
        Optional initialization parameter, by default 1.
        
    Attributes
    ----------
    public_attr : str
        Description of important public attributes.
    computed_property : dict
        Description of computed or derived attributes.
        
    Methods
    -------
    main_method()
        Brief description of primary method.
    helper_method(param)
        Brief description of helper method.
        
    Examples
    --------
    >>> obj = ExampleClass("input")
    >>> result = obj.main_method()
    >>> print(result)
    
    Notes
    -----
    Important usage patterns, threading considerations,
    or performance characteristics.
    """
```

## 🔧 Implementation Plan

### Phase 1: Core Classes (Day 3 Morning)
**Priority: Critical public APIs**

1. **CellMapDataset** (`dataset.py` lines 50-90)
   - Current: Google-style with extensive Args section
   - Issues: Inconsistent formatting, missing Examples/Notes
   - Action: Convert to NumPy-style, add Examples section

2. **CellMapImage** (`image.py` lines 45-55)
   - Current: Incomplete parameter documentation
   - Issues: Missing parameter descriptions, no return docs
   - Action: Complete parameter docs, add usage examples

3. **CellMapDataLoader** (`dataloader.py`)
   - Current: Brief one-liners
   - Issues: No parameter documentation for complex methods
   - Action: Add comprehensive parameter documentation

### Phase 2: Transform Modules (Day 3 Afternoon)  
**Priority: Recently modified in Week 3**

1. **Transform Classes** (`transforms/augment/*.py`)
   - Current: Good Google-style structure (normalize.py example)
   - Issues: Wrong format, need conversion
   - Action: Convert to NumPy-style, preserve good examples

2. **Utility Functions** (`utils/*.py`)
   - Current: Mixed quality
   - Issues: Some missing documentation entirely
   - Action: Add missing docs, standardize existing

### Phase 3: Secondary Modules (Day 4 Morning)
**Priority: Internal APIs and helpers**

1. **DatasetWriter** (`dataset_writer.py`)
2. **MultiDataset** (`multidataset.py`)  
3. **Validation utilities** (`validation/*.py`)
4. **Device utilities** (`device/*.py`)

### Phase 4: Verification (Day 4 Afternoon)
**Priority: Quality assurance**

1. **Automated Checking**:
   - Create docstring format checker script
   - Verify all public methods have complete docs
   - Check parameter-signature matching

2. **Documentation Generation**:
   - Test Sphinx API doc generation
   - Verify all examples execute correctly
   - Check for broken references

## 📝 Quality Standards

### Required Elements
- **Brief Description**: One-line summary of purpose
- **Extended Description**: Usage context and important behavior notes
- **Parameters**: All parameters with types and descriptions
- **Returns**: Return type and meaning (if applicable)
- **Raises**: All exceptions that can be raised
- **Examples**: At least one working usage example for public APIs
- **Notes**: Important implementation details (when relevant)

### Parameter Documentation Rules
1. **Type specification**: Use actual type hints from signature
2. **Optional parameters**: Always include "optional" and default value
3. **Complex types**: Explain structure for dictionaries/mappings
4. **Deprecated parameters**: Mark clearly with deprecation info

### Examples Requirements
1. **Executable**: All examples must run without error
2. **Realistic**: Use realistic parameter values
3. **Multiple scenarios**: Show common use cases and edge cases
4. **Output included**: Show expected output with `>>>`

## 🧪 Validation Process

### Automated Checks
```python
# Create docstring_checker.py
def check_docstring_format(func_or_class):
    """Check if docstring follows NumPy format standards."""
    # Implementation details...
    
def verify_parameter_coverage(func_or_class):
    """Verify all parameters are documented."""
    # Implementation details...
    
def test_examples_execute(docstring):
    """Test that docstring examples actually work."""  
    # Implementation details...
```

### Manual Review Checklist
- [ ] All public APIs have complete docstrings
- [ ] Parameter types match function signatures
- [ ] Examples execute successfully
- [ ] Sphinx generation works without warnings
- [ ] Cross-references resolve correctly

## 📊 Success Metrics

### Quantitative Goals
- **100%** of public methods have NumPy-style docstrings
- **90%** of public methods have working examples
- **Zero** Sphinx warnings during documentation generation
- **All** parameters documented with types and descriptions

### Qualitative Goals
- Consistent professional appearance across all modules
- Clear guidance for new contributors
- Enhanced API discoverability
- Improved developer experience

## 🚀 Implementation Priority

### High Priority (Must Complete)
1. CellMapDataset, CellMapImage, CellMapDataLoader
2. Transform base classes and common augmentations
3. Error handling utilities (recent Week 2 additions)

### Medium Priority (Should Complete)  
1. MultiDataset and validation utilities
2. Device and logging configuration utilities
3. Internal helper functions with complex interfaces

### Low Priority (Nice to Have)
1. Private/internal methods with simple interfaces
2. Test utilities and development helpers
3. Legacy or deprecated functionality

---

**Next Steps**: Begin implementation with Phase 1 high-priority classes, focusing on CellMapDataset as the primary public API.
