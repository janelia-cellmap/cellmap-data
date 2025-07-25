# CellMap-Data Refactoring Project Proposal

## Executive Summary

This document outlines a comprehensive multiphase project to address critical architectural issues, technical debt, and code quality concerns identified in the CellMap-Data codebase review. The project aims to transform the current B- (75/100) codebase into a production-ready, maintainable system through systematic refactoring, debt reduction, and quality improvements.

**Timeline: 8-12 months across 4 phases**
**Estimated Effort: 120-160 person-days**

## Current State Assessment

### Critical Issues Identified

1. **Monolithic Classes**: `CellMapDataset` (941 lines) and `CellMapImage` (537 lines) violate single responsibility principle
2. **Technical Debt**: 20+ TODO/FIXME items indicate incomplete implementation
3. **Inconsistent Error Handling**: Mixed patterns across modules
4. **Documentation Gaps**: Inconsistent docstring styles and missing module documentation
5. **Code Duplication**: Device handling, coordinate transformations, validation logic repeated
6. **Architecture Concerns**: Property-heavy interfaces, complex state management

### Quality Metrics (Baseline)
- **Overall Score**: B- (75/100)
- **Lines of Code**: ~8,000 LOC
- **Test Coverage**: 10 test files, ~1,572 test LOC
- **Technical Debt**: 20+ TODO items
- **Duplicate Code**: ~15% estimated

## Phase 1: Foundation Stabilization (Weeks 1-8)

### Objectives
- Resolve critical technical debt
- Establish consistent patterns
- Create refactoring infrastructure
- Improve immediate maintainability

### 1.1 Technical Debt Resolution (Weeks 1-4)
**Priority: Critical (🔴)**

#### TODO/FIXME Cleanup
- **Target**: Complete all 20+ TODO/FIXME items
- **Focus Areas**:
  ```python
  # dataset.py
  - Switch "raw_path" to "input_path" (API breaking change)
  - Fix hacky temporary fixes in coordinate transformations
  - Add missing tests for edge cases
  - Make coordinate transformations more robust
  
  # transforms/augment/random_contrast.py
  - Replace NaN hack with proper handling
  - Implement robust numerical stability
  
  # utils/view.py
  - Fix bare except clauses
  - Replace Warning() with proper logging
  ```

#### Error Handling Standardization
- **Create centralized exception hierarchy**:
  ```python
  # cellmap_data/exceptions.py
  class CellMapDataError(Exception):
      """Base exception for CellMap-Data"""
  
  class DataLoadingError(CellMapDataError):
      """Errors during data loading operations"""
  
  class ValidationError(CellMapDataError):
      """Data validation errors"""
      
  class ConfigurationError(CellMapDataError):
      """Configuration and setup errors"""
  ```

- **Implement consistent error handling patterns**:
  ```python
  # Standard pattern for all modules
  try:
      result = risky_operation()
  except SpecificError as e:
      logger.error(f"Operation failed: {e}")
      raise DataLoadingError(f"Failed to load data: {e}") from e
  ```

#### Logging Standardization
- **Centralize logging configuration**:
  ```python
  # cellmap_data/logging_config.py
  def setup_logging(level=logging.INFO):
      """Configure consistent logging across all modules"""
  ```
- **Standardize logging levels and messages**
- **Remove hardcoded `logger.setLevel()` calls**

### 1.2 Dependency Management (Week 5)
**Priority: Critical (🔴)**

#### Version Constraints
```toml
# pyproject.toml updates
dependencies = [
    "torch>=2.0.0,<3.0.0",
    "numpy>=1.24.0,<2.0.0",
    "tensorstore>=0.1.45,<1.0.0",
    "pydantic_ome_ngff>=0.5.0,<1.0.0",
    "matplotlib>=3.7.0,<4.0.0",
]
```

#### Security Review
- **Path validation utilities**:
  ```python
  # cellmap_data/utils/security.py
  def validate_path(path: str) -> str:
      """Validate and sanitize file paths"""
      # Prevent path traversal attacks
      # Validate file extensions
      # Check permissions
  ```

### 1.3 Documentation Framework (Weeks 6-8)
**Priority: High (🟡)**

#### Docstring Standardization
- **Establish Google-style docstring template**:
  ```python
  def example_function(param1: str, param2: int = 0) -> bool:
      """Brief description of function purpose.
      
      Detailed description with usage examples and important notes.
      
      Args:
          param1: Description of parameter
          param2: Description with default value info
          
      Returns:
          Description of return value
          
      Raises:
          ValueError: When invalid input provided
          DataLoadingError: When data cannot be loaded
          
      Example:
          >>> result = example_function("test", 5)
          >>> print(result)
          True
      """
  ```

#### Module Documentation
- Add comprehensive module-level docstrings
- Create API reference documentation
- Establish documentation testing framework

### Phase 1 Deliverables
- [x] All TODO/FIXME items resolved
- [x] Consistent error handling implemented
- [x] Centralized logging configuration
- [ ] Dependency version constraints added
- [ ] Security validation utilities
- [ ] Standardized documentation format
- [x] Updated test suite (no regressions)

**Success Metrics**:
- 0 TODO/FIXME items remaining
- 95%+ consistent error handling patterns
- All dependencies version-constrained
- 100% modules with proper docstrings

## Phase 2: Architecture Refactoring (Weeks 9-20)

### Objectives
- Break down monolithic classes
- Extract common functionality
- Implement clean architecture patterns
- Eliminate code duplication

### 2.1 Monolithic Class Decomposition (Weeks 9-16)
**Priority: Critical (🔴)**

#### CellMapDataset Refactoring (941 lines → ~300 lines per class)

**Extract Device Management**:
```python
# cellmap_data/device/device_manager.py
class DeviceManager:
    """Centralized device management and GPU optimization"""
    
    def __init__(self):
        self.device = self._auto_detect_device()
        self.cuda_streams = self._setup_cuda_streams()
    
    def _auto_detect_device(self) -> torch.device:
        """Auto-detect best available device"""
    
    def to_device(self, tensor, non_blocking=True):
        """Move tensor to managed device"""
    
    def get_cuda_stream(self, key: str):
        """Get CUDA stream for parallel operations"""
```

**Extract Coordinate Transformations**:
```python
# cellmap_data/transforms/coordinate_transforms.py
class CoordinateTransformer:
    """Handle all coordinate transformation operations"""
    
    def __init__(self, axis_order: str = "zyx"):
        self.axis_order = axis_order
    
    def apply_spatial_transforms(self, coords, transforms):
        """Apply spatial transformations to coordinates"""
    
    def rotate_coords(self, coords, angles):
        """Apply rotation transformations"""
    
    def validate_coordinates(self, coords, bounds):
        """Validate coordinate bounds"""
```

**Extract Data Loading Logic**:
```python
# cellmap_data/loaders/data_loader.py
class DataLoader:
    """Core data loading functionality"""
    
    def __init__(self, sources, device_manager, coord_transformer):
        self.sources = sources
        self.device_manager = device_manager
        self.coord_transformer = coord_transformer
    
    def load_batch(self, centers, transforms=None):
        """Load data batch with optional transformations"""
```

**Extract Validation Logic**:
```python
# cellmap_data/validation/dataset_validator.py
class DatasetValidator:
    """Validate dataset configuration and data integrity"""
    
    def validate_array_config(self, arrays):
        """Validate array configuration"""
    
    def validate_data_sources(self, sources):
        """Validate data source accessibility"""
    
    def check_data_integrity(self, dataset):
        """Check for data corruption or missing files"""
```

**Refactored CellMapDataset**:
```python
class CellMapDataset(Dataset):
    """Streamlined dataset class focused on PyTorch Dataset interface"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.device_manager = DeviceManager()
        self.coord_transformer = CoordinateTransformer(config.axis_order)
        self.data_loader = DataLoader(
            self._create_sources(), 
            self.device_manager, 
            self.coord_transformer
        )
        self.validator = DatasetValidator()
        self.validator.validate_configuration(config)
    
    def __getitem__(self, idx) -> dict:
        """Simplified getitem focusing on core functionality"""
        # ~50 lines instead of 200+
    
    def __len__(self) -> int:
        """Dataset length"""
    
    # Minimal property interface (5-7 properties instead of 20+)
```

#### CellMapImage Refactoring (537 lines → ~200 lines per class)

**Extract Metadata Management**:
```python
# cellmap_data/metadata/image_metadata.py
class ImageMetadataManager:
    """Handle image metadata and multiscale information"""
    
    def __init__(self, path: str):
        self.path = path
        self._load_metadata()
    
    def get_multiscale_info(self):
        """Get multiscale dataset information"""
    
    def find_optimal_scale_level(self, target_scale):
        """Find best scale level for target resolution"""
    
    def get_coordinate_transforms(self):
        """Get coordinate transformation metadata"""
```

**Extract Array Operations**:
```python
# cellmap_data/arrays/array_ops.py
class ArrayOperations:
    """Handle array access and manipulation operations"""
    
    def __init__(self, metadata_manager, device_manager):
        self.metadata = metadata_manager
        self.device_manager = device_manager
    
    def read_array_region(self, coords, interpolation="nearest"):
        """Read array data for specified region"""
    
    def apply_padding(self, array, target_shape, pad_value):
        """Apply padding to match target shape"""
```

**Refactored CellMapImage**:
```python
class CellMapImage:
    """Streamlined image class focused on data access"""
    
    def __init__(self, config: ImageConfig):
        self.config = config
        self.metadata = ImageMetadataManager(config.path)
        self.device_manager = DeviceManager()
        self.array_ops = ArrayOperations(self.metadata, self.device_manager)
    
    def __getitem__(self, center) -> torch.Tensor:
        """Simplified data access"""
        # ~30 lines instead of 100+
    
    # Minimal property interface (5-7 properties instead of 15+)
```

### 2.2 Code Deduplication (Weeks 17-18)
**Priority: Important (🟡)**

#### Common Utilities Extraction
```python
# cellmap_data/utils/common.py
def validate_array_info(array_info: dict) -> dict:
    """Common array information validation"""

def calculate_bounding_box(sources: list) -> dict:
    """Common bounding box calculation"""

def handle_path_parsing(path: str) -> tuple:
    """Common path handling logic"""
```

#### Device Handling Consolidation
- All device logic moved to `DeviceManager`
- Consistent `to()` method implementation
- Centralized CUDA stream management

#### Configuration Management
```python
# cellmap_data/config/
class DatasetConfig(BaseModel):
    """Pydantic model for dataset configuration"""
    raw_path: str
    target_path: str
    classes: List[str]
    input_arrays: Dict[str, ArrayConfig]
    target_arrays: Dict[str, ArrayConfig]
    
    @validator('raw_path')
    def validate_raw_path(cls, v):
        return validate_path(v)
```

### 2.3 Interface Standardization (Weeks 19-20)
**Priority: Important (🟡)**

#### Method-based APIs
- Replace property-heavy interfaces with explicit methods
- Implement consistent parameter validation
- Add comprehensive type hints

#### Naming Convention Standardization
- `raw_path` → `input_path` (breaking change)
- `class_relation_dict` → `class_relationships`
- Consistent parameter naming across all classes

### Phase 2 Deliverables
- [ ] CellMapDataset reduced from 941 to ~300 lines
- [ ] CellMapImage reduced from 537 to ~200 lines
- [ ] 5 new utility classes extracted
- [ ] Code duplication reduced by 80%+
- [ ] Consistent interfaces across all classes
- [ ] Breaking changes documented and migration guide provided

**Success Metrics**:
- Average class size < 400 lines
- Code duplication < 5%
- 100% type hint coverage
- All classes follow single responsibility principle

## Phase 3: Quality Enhancement (Weeks 21-28)

### Objectives
- Comprehensive testing strategy
- Performance validation
- Integration testing
- Code quality enforcement

### 3.1 Testing Infrastructure Expansion (Weeks 21-24)
**Priority: Important (🟡)**

#### Test Coverage Enhancement
```python
# tests/integration/
class TestEndToEndWorkflows:
    """Test complete data loading workflows"""
    
    def test_training_pipeline_integration(self):
        """Test full training data pipeline"""
    
    def test_inference_pipeline_integration(self):
        """Test full inference pipeline"""
    
    def test_multi_dataset_training(self):
        """Test multi-dataset training workflow"""

# tests/performance/
class TestPerformanceBenchmarks:
    """Validate performance claims and regressions"""
    
    def test_cuda_stream_performance(self):
        """Validate CUDA stream performance improvements"""
    
    def test_memory_efficiency(self):
        """Validate memory usage claims"""
    
    def test_threadpool_performance(self):
        """Validate ThreadPoolExecutor improvements"""
```

#### Error Path Testing
```python
# tests/error_handling/
class TestErrorScenarios:
    """Test error handling and recovery"""
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted data files"""
    
    def test_missing_file_handling(self):
        """Test handling of missing data files"""
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure"""
```

#### Docstring Example Testing
```python
# tests/documentation/
class TestDocstringExamples:
    """Ensure all docstring examples work correctly"""
    
    @pytest.mark.parametrize("module", get_all_modules())
    def test_docstring_examples(self, module):
        """Test that all docstring examples execute correctly"""
```

### 3.2 Performance Validation (Weeks 25-26)
**Priority: Important (🟡)**

#### Benchmark Framework
```python
# benchmarks/
class PerformanceBenchmarks:
    """Comprehensive performance benchmarking"""
    
    def benchmark_data_loading_speed(self):
        """Benchmark data loading performance"""
    
    def benchmark_memory_usage(self):
        """Benchmark memory consumption"""
    
    def benchmark_gpu_utilization(self):
        """Benchmark GPU resource utilization"""
```

#### Performance Regression Testing
- Automated performance regression detection
- Memory usage monitoring
- GPU utilization optimization validation

### 3.3 Code Quality Enforcement (Weeks 27-28)
**Priority: Enhancement (🟢)**

#### Static Analysis Integration
```bash
# CI/CD pipeline additions
- black --check src/  # Code formatting
- ruff check src/     # Linting
- mypy src/           # Type checking
- bandit -r src/      # Security analysis
```

#### Code Quality Metrics
```python
# scripts/quality_metrics.py
def calculate_code_quality_score():
    """Calculate comprehensive code quality score"""
    metrics = {
        'complexity': calculate_cyclomatic_complexity(),
        'duplication': calculate_code_duplication(),
        'coverage': get_test_coverage(),
        'documentation': calculate_doc_coverage(),
        'type_hints': calculate_type_coverage()
    }
    return weighted_average(metrics)
```

### Phase 3 Deliverables
- [ ] Test coverage increased to 90%+
- [ ] Integration test suite (20+ tests)
- [ ] Performance benchmark suite
- [ ] Automated code quality checks
- [ ] Documentation testing framework
- [ ] Error handling test coverage 85%+

**Success Metrics**:
- Test coverage > 90%
- Performance within 5% of baseline
- Code quality score > 85/100
- 0 critical security issues

## Phase 4: Production Readiness (Weeks 29-36)

### Objectives
- Finalize API stability
- Complete documentation
- Release preparation
- Long-term maintainability

### 4.1 API Stabilization (Weeks 29-32)
**Priority: Critical (🔴)**

#### API Compatibility Layer
```python
# cellmap_data/compat.py
class LegacyDatasetAdapter:
    """Backward compatibility for v1.x API"""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Legacy API deprecated, use new configuration-based API",
            DeprecationWarning
        )
        # Map old parameters to new configuration
```

#### Breaking Changes Management
- Comprehensive migration guide
- Automated migration scripts where possible
- Clear deprecation timeline (6 months)

#### Version Strategy
```python
# Version 2.0.0 (SemVer)
# - Major version bump for breaking changes
# - Clear migration path from 1.x
# - Long-term support plan
```

### 4.2 Documentation Completion (Weeks 33-34)
**Priority: Important (🟡)**

#### Comprehensive Documentation
```markdown
# docs/user-guide/
- getting-started.md
- configuration.md
- advanced-usage.md
- performance-optimization.md
- troubleshooting.md

# docs/developer-guide/
- architecture-overview.md
- extending-functionality.md
- testing-guidelines.md
- contributing.md

# docs/api-reference/
- Auto-generated from docstrings
- Interactive examples
- Performance notes
```

#### Tutorial Development
```python
# tutorials/
- 01_basic_usage.ipynb
- 02_multi_dataset_training.ipynb
- 03_custom_transforms.ipynb
- 04_performance_optimization.ipynb
- 05_troubleshooting.ipynb
```

### 4.3 Release Engineering (Weeks 35-36)
**Priority: Important (🟡)**

#### CI/CD Pipeline Enhancement
```yaml
# .github/workflows/release.yml
- Automated testing across Python versions
- Performance regression testing
- Documentation building and deployment
- Automated security scanning
- Package publishing automation
```

#### Quality Gates
```python
# Release criteria checklist
RELEASE_CRITERIA = {
    'test_coverage': 90,
    'performance_regression': 0,
    'security_issues': 0,
    'documentation_coverage': 95,
    'api_stability_score': 100
}
```

### Phase 4 Deliverables
- [ ] API compatibility layer implemented
- [ ] Comprehensive migration guide
- [ ] Complete documentation suite
- [ ] Tutorial notebooks (5+)
- [ ] Automated release pipeline
- [ ] Performance regression testing
- [ ] Security scanning integration

**Success Metrics**:
- API stability score 100%
- Documentation coverage 95%+
- User migration success rate > 80%
- Release automation 100% functional

## Success Metrics and Monitoring

### Overall Quality Improvement Targets

| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|--------|----------|---------|---------|---------|---------|
| Overall Code Quality | B- (75/100) | B (78/100) | B+ (82/100) | A- (87/100) | A (90/100) |
| Average Class Size | 450 lines | 400 lines | 250 lines | 200 lines | 180 lines |
| Code Duplication | ~15% | 12% | 6% | 4% | <3% |
| Test Coverage | 70% | 75% | 85% | 90% | 92% |
| TODO/FIXME Items | 20+ | 0 | 0 | 0 | 0 |
| Documentation Coverage | 60% | 75% | 85% | 95% | 98% |

### Monitoring Framework
```python
# scripts/quality_monitoring.py
def monitor_quality_metrics():
    """Continuous monitoring of code quality metrics"""
    metrics = {
        'complexity': measure_complexity(),
        'duplication': measure_duplication(),
        'coverage': get_coverage_report(),
        'performance': run_benchmarks(),
        'security': run_security_scan()
    }
    
    # Alert on regressions
    # Generate quality reports
    # Track improvements over time
```

## Risk Assessment and Mitigation

### High-Risk Areas
1. **Breaking Changes**: API changes may impact existing users
   - **Mitigation**: Comprehensive compatibility layer and migration tools
   
2. **Performance Regression**: Refactoring may impact performance
   - **Mitigation**: Continuous benchmarking and performance testing
   
3. **Complex Refactoring**: Large classes may be difficult to decompose
   - **Mitigation**: Incremental refactoring with extensive testing

### Medium-Risk Areas
1. **Test Coverage Gaps**: New code may lack sufficient testing
   - **Mitigation**: Test-first development approach
   
2. **Documentation Debt**: New APIs need comprehensive documentation
   - **Mitigation**: Documentation-first development approach

## Resource Requirements

### Personnel
- **Lead Developer**: 1 FTE for full project duration
- **Testing Engineer**: 0.5 FTE for Phases 2-3
- **Technical Writer**: 0.25 FTE for Phase 4
- **Code Reviewer**: 0.25 FTE throughout project

### Infrastructure
- **CI/CD Resources**: Extended build times for comprehensive testing
- **Performance Testing**: GPU resources for benchmarking
- **Documentation Hosting**: Enhanced documentation infrastructure

## Return on Investment

### Immediate Benefits (Phase 1)
- Reduced maintenance burden (20+ TODO items resolved)
- Improved developer confidence (consistent error handling)
- Better debugging experience (centralized logging)

### Medium-term Benefits (Phases 2-3)
- Faster feature development (modular architecture)
- Reduced bug rates (better testing, cleaner code)
- Easier onboarding (better documentation)

### Long-term Benefits (Phase 4+)
- Reduced technical debt accumulation
- Higher code quality standards
- Improved project reputation and adoption
- Easier maintenance and evolution

## Conclusion

This comprehensive refactoring project addresses the critical architectural and quality issues identified in the CellMap-Data codebase. Through systematic, phased improvements, the project will transform the current B- (75/100) codebase into a production-ready A (90/100) system.

The investment in this refactoring project will pay dividends through:
- **Reduced maintenance costs** (cleaner, more modular code)
- **Faster feature development** (better architecture)
- **Improved reliability** (comprehensive testing)
- **Enhanced user experience** (better documentation, more stable APIs)
- **Increased adoption** (higher quality, more trustworthy codebase)

**Recommended Approach**: Begin with Phase 1 (Foundation Stabilization) to address critical technical debt, then proceed systematically through each phase. Each phase builds upon the previous one and delivers immediate value while working toward the ultimate goal of production-ready code quality.
