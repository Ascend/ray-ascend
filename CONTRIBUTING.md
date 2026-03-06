# Contributing to ray-ascend

Thank you for your interest in contributing to ray-ascend! This document provides
guidelines and instructions for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
1. Clone your fork locally:
   ```bash
   git clone https://gitcode.com/YOUR_USERNAME/ray-ascend.git
   cd ray-ascend
   ```
1. Add the upstream repository:
   ```bash
   git remote add upstream https://gitcode.com/Ascend/ray-ascend.git
   ```

## Development Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

1. Install the package in development mode:

   ```bash
   pip install -e .
   ```

1. Install pre-commit hooks:

   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Making Changes

1. Create a new branch for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

1. Make your changes and commit them:

   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

1. Push your changes to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

1. Open a pull request on GitHub

## Code Style

This project uses the following tools to maintain code quality:

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Pre-commit hooks will automatically run these tools before each commit. You can also run
them manually:

```bash
# Format code
black .

# Sort imports
isort .

# Run linter
flake8 .

# Type checking
mypy .
```

## Testing

Before submitting a pull request, please ensure that:

1. All existing tests pass
1. New features include appropriate tests
1. Code coverage is maintained or improved

Run tests with:

```bash
pytest
```

## Pull Request Guidelines

- Keep pull requests focused on a single feature or bug fix
- Write clear, descriptive commit messages
- Update documentation as needed
- Ensure all tests pass
- Follow the existing code style
- Add tests for new features

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Assume good intentions

## Questions?

If you have questions about contributing, feel free to:

- Open an issue for discussion
- Reach out to the maintainers

Thank you for contributing to ray-ascend!
