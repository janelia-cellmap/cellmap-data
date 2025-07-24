# CellMap-Data Refactoring Master Plan

**Project Goal**: Transform CellMap-Data from B- (75/100) to production-ready codebase  
**Approach**: 8-week Phase 1 focusing on foundation stabilization  
**Status**: Week 1 Complete ✅ | Week 2 In Progress 🔄

---

## 📊 Executive Summary

### Week 1 Results ✅ COMPLETE
- **All P0 critical issues resolved** - 3/3 eliminated data corruption risks
- **Comprehensive testing** - 14 new tests, 91/91 existing tests passing
- **Foundation established** - Exception hierarchy and error handling patterns
- **Zero regressions** - All fixes maintain backward compatibility

### Next Focus: Week 2 P1 Issues 🔄
- **Parameter standardization** - API consistency improvements
- **Error handling** - Replace bare except clauses and improve messages
- **Code quality** - Remove remaining TODO/FIXME items

---

## 📋 Technical Debt Status

### Priority Breakdown (24 Total Issues)
- **P0 Critical**: 3/3 resolved ✅ (Data corruption, security)
- **P1 High**: 0/8 resolved 🔄 (Performance, incorrect behavior) 
- **P2 Medium**: 0/9 resolved ⏳ (Code quality, maintainability)
- **P3 Low**: 0/4 resolved ⏳ (Documentation, minor improvements)

### P0 Issues Resolved ✅

#### 1. Coordinate Transformation Bounds Hack
- **Files**: `dataset.py`, `dataset_writer.py`
- **Issue**: Silent data corruption from index out-of-bounds hack
- **Fix**: Added `_validate_index_bounds()` and `_safe_unravel_index()` methods
- **Testing**: 3 comprehensive unit tests covering edge cases
- **Impact**: Eliminated silent data corruption risk

#### 2. RandomContrast NaN Handling Hack  
- **File**: `transforms/augment/random_contrast.py`
- **Issue**: Silent NaN propagation using `torch.nan_to_num()` hack
- **Fix**: Input validation with explicit NaN/Inf detection and proper exceptions
- **Testing**: 8 unit tests covering numerical stability and edge cases
- **Impact**: Prevented training instability from silent NaN propagation

#### 3. Exception Hierarchy Infrastructure
- **File**: `exceptions.py`
- **Implementation**: Complete hierarchy with specialized exceptions
- **Classes**: `CellMapIndexError`, `CoordinateTransformError`, `DataLoadingError`, etc.
- **Impact**: Consistent error handling patterns across codebase

---

## 🚀 Week 2 Action Plan (Current Focus)

### Priority 1: Parameter Standardization (Days 1-3)
- **`raw_path` → `input_path` migration** in `dataset.py`, `dataset_writer.py`
  - Add deprecation warnings for backward compatibility
  - Support both parameters temporarily
  - Update internal usage to new parameter name
  
- **Class relationship standardization**
  - `class_relation_dict` → `class_relationships`
  - Add deprecation path for existing code

### Priority 2: Error Handling Improvements (Days 4-5)  
- **Fix bare except clauses** in `utils/view.py`
- **Standardize error messages** with consistent templates
- **Improve assertion messages** using f-strings

---

## 📈 Success Metrics

### Week 1 Achievements ✅
- **Critical risk elimination**: 3/3 P0 issues resolved
- **Test coverage**: 14 new focused tests + 91 existing tests passing
- **Code quality**: 100% hack code removal from critical paths
- **Documentation**: Comprehensive technical reports and execution plans

### Week 2 Targets 🎯
- **API consistency**: Parameter naming standardized
- **Error handling**: All bare except clauses replaced
- **Technical debt**: P1 issues reduced to 4/8 or fewer
- **Test stability**: Maintain 100% pass rate

---

## 📁 Key Documents

### Planning & Execution
- **This document** - Master plan and status tracker
- `PHASE1_EXECUTION_PLAN.md` - Detailed week-by-week execution timeline
- `TECHNICAL_DEBT_AUDIT.md` - Complete inventory of 24 issues with priority analysis

### Technical Reports  
- `P0_FIXES_COMPLETION_REPORT.md` - Detailed technical analysis of critical fixes
- `CODE_REVIEW.md` - Comprehensive codebase analysis and recommendations

### Development Support
- `CONTRIBUTING.md` - Enhanced developer guidelines with refactoring patterns
- `tests/test_p0_fixes_focused.py` - Validation suite for critical fixes

---

## 🎯 Long-term Vision

### Phase 1 (Weeks 1-8): Foundation Stabilization
- Week 1 ✅: P0 critical issues resolved
- Week 2 🔄: P1 high priority items (parameter standardization, error handling)
- Week 3-4: Code quality and consistency improvements
- Week 5-6: Documentation and testing completeness
- Week 7-8: Performance optimization and final validation

### Success Criteria
- **Zero TODO/FIXME** items in critical paths
- **Consistent patterns** across all modules  
- **100% test coverage** for critical functionality
- **Production-ready stability** and error handling

The project is on track with strong foundational work completed and clear priorities established for continued success.
