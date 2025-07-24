# Contributing to CellMap-Data

We welcome contributions to CellMap-Data! This guide will help you get started with contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style and Standards](#code-style-and-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Expectations](#documentation-expectations)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Development Workflow](#development-workflow)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Basic understanding of PyTorch and biological imaging data

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/your-username/cellmap-data.git
   cd cellmap-data
   ```

3. Add the upstream remote:

   ```bash
   git remote add upstream https://github.com/janelia-cellmap/cellmap-data.git
   ```

## Development Setup

### Environment Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:

   ```bash
   pip install -e .[dev]
   ```

3. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

### Project Structure

```text
cellmap-data/
├── src/cellmap_data/          # Main package source
│   ├── transforms/            # Data transformation modules
│   ├── utils/                 # Utility functions
│   ├── validation/            # Data validation
│   └── device/                # Device management
├── tests/                     # Test suite
├── docs/                      # Documentation
├── .github/                   # GitHub workflows and templates
└── pyproject.toml             # Project configuration
```

## Code Style and Standards

We use several tools to maintain code quality:

### Code Formatting

- **Black**: Automatic Python code formatting
- **Ruff**: Fast Python linter for code quality

Run formatting tools:

```bash
black src tests
ruff check src tests --fix
```

### Code Quality Standards

- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters and return values
- Write clear, descriptive variable and function names
- Keep functions focused and reasonably sized (< 50 lines when possible)
- Use docstrings for all public functions, classes, and modules

### Docstring Style

We use NumPy-style docstrings:

```python
def process_data(data: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Process input data with optional scaling.
    
    Parameters
    ----------
    data : np.ndarray
        Input array to process.
    scale : float, optional
        Scaling factor to apply, by default 1.0.
        
    Returns
    -------
    np.ndarray
        Processed data array.
        
    Raises
    ------
    ValueError
        If data is empty or scale is negative.
    """
    if data.size == 0:
        raise ValueError("Data array cannot be empty")
    return data * scale
```

## Testing Requirements

### Test Structure

- Tests are located in the `tests/` directory
- Test files should be named `test_*.py`
- Use descriptive test function names: `test_feature_under_specific_condition`

### Writing Tests

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Verify performance optimizations work
4. **Coverage**: Aim for >90% test coverage

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cellmap_data --cov-report=html

# Run specific test file
pytest tests/test_dataset.py

# Run tests matching pattern
pytest -k "transform"
```

### Test Guidelines

- Each test should be independent and idempotent
- Use fixtures for common test setup
- Mock external dependencies (file systems, networks)
- Include edge cases and error conditions
- Test performance-critical code paths

Example test structure:
```python
import pytest
from cellmap_data import CellMapDataset

class TestCellMapDataset:
    def test_dataset_creation_success(self):
        """Test successful dataset creation with valid parameters."""
        # Test implementation
        
    def test_dataset_creation_invalid_path(self):
        """Test dataset creation fails with invalid path."""
        with pytest.raises(FileNotFoundError):
            # Test implementation
```

## Documentation Expectations

### Code Documentation

- All public APIs must have comprehensive docstrings
- Include parameter types, return types, and descriptions
- Document exceptions that may be raised
- Provide usage examples for complex functions

### User Documentation

- Update README.md for new features
- Add examples to demonstrate usage
- Update API documentation in `docs/`
- Include performance considerations where relevant

### Building Documentation

```bash
cd docs
make html
```

## Pull Request Process

### Before Creating a PR

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow code style guidelines
   - Add/update tests
   - Update documentation

4. **Run quality checks**:
   ```bash
   # Format code
   black src tests
   ruff check src tests --fix
   
   # Run tests
   pytest --cov=cellmap_data
   
   # Build docs
   cd docs && make html
   ```

### PR Submission

1. **Commit Guidelines**:
   - Use clear, descriptive commit messages
   - Consider using conventional commits: `feat:`, `fix:`, `docs:`, etc.
   - Keep commits focused on single changes

2. **Push and Create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```
   
3. **PR Description**:
   - Clearly describe what the PR does
   - Reference related issues
   - Include testing notes
   - Add screenshots for UI changes

### PR Template

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## Issue Guidelines

### Reporting Bugs

Include:

- Python version
- CellMap-Data version
- Operating system
- Minimal code example
- Error messages/stack traces
- Expected vs actual behavior

### Feature Requests

Include:

- Clear description of the feature
- Use cases and motivation
- Proposed API (if applicable)
- Willingness to implement

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation updates
- `good first issue`: Good for newcomers
- `help wanted`: Community help needed

## Development Workflow

### Branch Strategy

- `main`: Stable release branch
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical bug fixes

### Typical Workflow

1. Create feature branch from `main`
2. Develop and test locally
3. Submit PR to `main`
4. Code review and CI checks
5. Merge after approval

### Performance Considerations

- Profile performance-critical code
- Consider memory usage for large datasets
- Test with realistic data sizes
- Document performance characteristics

## Release Process

Releases are automated through GitHub Actions:

1. **Version Tagging**: Automatic date-based versioning on main branch
2. **PyPI Publishing**: Automatic publication to PyPI
3. **Release Notes**: Auto-generated from commit messages

### Manual Release Steps (if needed)

```bash
# Update version
hatch version patch  # or minor, major

# Build distribution
hatch build

# Upload to PyPI
twine upload dist/*
```

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Maintain professionalism in all interactions

### Getting Help

- 📖 [Documentation](https://janelia-cellmap.github.io/cellmap-data/)
- 🐛 [Issue Tracker](https://github.com/janelia-cellmap/cellmap-data/issues)
- 💬 [Discussions](https://github.com/janelia-cellmap/cellmap-data/discussions)
- 📧 [Email Support](mailto:rhoadesj@hhmi.org)

### Recognition

Contributors are recognized in:

- Release notes
- Repository contributors list
- Documentation acknowledgments

## Questions?

If you have questions about contributing, please:

1. Check existing documentation and issues
2. Start a discussion on GitHub
3. Reach out to maintainers

Thank you for contributing to CellMap-Data! 🎉
