# Developer Guide

> _Last upated: 03/09/2026_

## Target

This guide is intended for developers who wish to read, modify or expand the source code
of ray-ascend. It provides a comprehensive walkthrough from setting up your development
environment to submitting a pull request.

## Quick Start Checklist

1. ✅ [Environment Setup](#preparation) - Install dependencies and set up your
   development environment
1. ✅ [Clone and Install](#clone--install) - Get ray-ascend repository and install it
1. ✅ [Understand Code Structure](#code-structure) - Familiarize yourself with the
   project layout
1. ✅ [Follow Coding Standards](#coding-standards-and-submission) - Write code according
   to project conventions
1. ✅ [Write and Test](#build-and-test) - Implement features and run tests
1. ✅ [Submit Contribution](#submit-your-contribution) - Sign CLA and create a pull
   request

## Preparation

Before starting development, you need to set up your environment. Refer to
[Setup Guide](setup.md#install-cann) for detailed instructions:

### Key Steps:

- **Install CANN (Optional)**: Required only if you have NPU devices and want to use NPU
  tensor for transmission.
- **Choose Installation Type**:
  - **Basic Installation**: `pip install -e .`
  - **With YR Support**: `pip install -e ".[yr]"` (includes YuanRong direct tensor
    transport)
  - **Full Installation**: `pip install -e ".[all]"` (all features for development and
    testing)

See [Setup Guide](setup.md#install-ray-ascend-without-yr) for complete installation
instructions.

## Clone & Install

```bash
# Clone the repository
git clone https://github.com/Ascend/ray-ascend.git
cd ray-ascend

# Install from source (development mode)
pip install -e .

# Or for full development with all features
pip install -e ".[all]"
```

## Code Structure

The ray-ascend project is organized as follows:

```
ray_ascend/
├── collective/              # HCCL collective communication
└── direct_transport/        # YuanRong direct tensor transport


tests/
├── collective/              # Tests for collective communication
└── direct_transport/        # Tests for tensor transport

```

**Main Components:**

- **collective/**: HCCL-based collective communication group implementation
- **direct_transport/**: YuanRong direct tensor transport implementation
- **tests/**: Comprehensive test suite using pytest

## Coding Standards and Submission

Before committing your code:

1. **Setup pre-commit hooks** to automatically check code quality:

   ```bash
   pip install pre-commit
   pre-commit install
   git config user.name "Your Name"
   git config user.email "your.email@example.com"
   ```

1. **Follow project coding standards** - see
   [Contributing Guide](contributing.md#code-style) for detailed style conventions

1. **Commit with signature**:

   ```bash
   git commit -s  # -s flag adds sign-off, triggering pre-commit hooks
   ```

For complete information on code style, pre-commit setup, and contribution guidelines,
refer to [Contributing Guide](contributing.md#instructions-for-contribution).

## Build and Test

Run tests using pytest to ensure your changes don't break existing functionality:

```bash
# Install all development dependencies
pip install -e ".[all]"

# Run all tests
pytest -v
```

All tests must pass before submitting a PR. For more testing details and options, see
[Contributing Guide](contributing.md#run-tests).

## Submit Your Contribution

### Prerequisites

Before submitting a PR, you must:

#### 1. Sign the Ascend CLA (Contributor License Agreement)

_Required for first-time contributors only_

Visit:
[Ascend CLA Sign Portal](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1)

For details, see [Sign Ascend CLA](contributing.md#sign-ascend-cla)

⚠️ **Important**: The email address used to sign the CLA must match your git commit
email address.

#### 2. Verify Your Setup

```bash
# Confirm git configuration
git config user.name
git config user.email

# These should match your CLA signature
```

### Submission Steps

1. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

1. **Make Your Changes**

   - Write code following all coding standards above
   - Add or update tests as necessary
   - Update documentation if needed

1. **Test Everything**

   ```bash
   pytest -v
   ```

   All tests must pass.

1. **Commit with Signature**

   ```bash
   git commit -s -m "Clear and descriptive commit message"
   ```

1. **Push and Create PR**

   ```bash
   git push origin feature/your-feature-name
   ```

   Then create a pull request on GitHub.

### PR Review Checklist

Before submitting, ensure:

- ✅ Code follows all style conventions
- ✅ All tests pass
- ✅ New features have corresponding tests
- ✅ Documentation is updated
- ✅ Commits are signed (`-s` flag)
- ✅ CLA is signed (first-time contributors)
- ✅ Commit email matches CLA email

## Documentation

Documentation updates should be made to the relevant `.md` files in the `docs/`
directory. Use clear, concise language with code examples where appropriate.

## Additional Resources

- [Setup Guide](setup.md) - Detailed environment setup instructions
- [Contributing Guide](contributing.md) - Contribution workflow and guidelines
- [Main README](../../README.md) - Project overview and getting started

## Getting Help

If you encounter any issues:

1. Check the documentation above
1. Review existing issues and PRs
1. Open a new issue on GitHub with detailed information
