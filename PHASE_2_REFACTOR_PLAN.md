# Phase 2 Major Refactor Plan: Simplification & Performance Optimization

## Refactor Goals
- Reduce package complexity
- Enable targeted performance optimizations
- Improve maintainability and testability
- Ensure compatibility with external frameworks (Accelerate, PyTorch Lightning)

## Refactor Proposal (Summary)
1. **Modularize Core Components**
   - Split monolithic classes into focused modules: io, device, prefetch, validation, framework, logging, dataloader
2. **Centralize Device and Memory Management**
   - Create a DeviceManager class for all device/memory logic
3. **Unified DataLoader Interface**
   - Design a single extensible DataLoader with plugin support
4. **Simplify Configuration and Validation**
   - Move config validation to a dedicated module, use schema-based validation
5. **Remove Redundant Logic**
   - Eliminate duplicate device/memory code, rely on DeviceManager
6. **Enhance Testability and Maintainability**
   - Refactor tests to use shared fixtures/mocks, add integration tests
7. **Document and Automate Migration**
   - Provide migration scripts and guides

## Separable Tasks & Prioritization

### Priority 1: Foundation & Modularization
- [ ] Task 1.1: Create new module structure in `src/cellmap_data/`
- [ ] Task 1.2: Move IO logic to `io.py`, device logic to `device.py`, etc.
- [ ] Task 1.3: Stub out new DeviceManager class

### Priority 2: Device & Memory Management
- [ ] Task 2.1: Implement DeviceManager with transfer/memory pool API
- [ ] Task 2.2: Refactor DataLoader/device logic to use DeviceManager

### Priority 3: DataLoader & Plugin System
- [ ] Task 3.1: Design unified DataLoader interface
- [ ] Task 3.2: Implement plugin hooks for prefetch, augmentation, device transfer

### Priority 4: Validation & Configuration
- [ ] Task 4.1: Move config validation to `validation.py`, add schema-based checks
- [ ] Task 4.2: CLI tool for config validation/migration

### Priority 5: Testing & Migration
- [ ] Task 5.1: Refactor tests for new modules, add shared fixtures/mocks
- [ ] Task 5.2: Add integration tests for framework compatibility
- [ ] Task 5.3: Provide migration scripts/guides

## Next Steps
- Begin with Priority 1: Create new module structure and stub DeviceManager
