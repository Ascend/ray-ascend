# Developer Guide

> _Last updated: 03/09/2026_

## Target

This guide is designed for developers looking to read, modify, or extend the ray-ascend
source code. It provides a comprehensive walkthrough from setting up your development
environment to submitting a pull request.

## Quick Start Checklist

- ✅ [Environment Setup](#preparation) - Install dependencies and set up your development
  environment
- ✅ [Clone and Build](#clone-and-build) - Get the ray-ascend repository and install it
- ✅ [Understand Code Structure](#code-structure) - Familiarize yourself with the project
  layout
- ✅ [Follow Coding Standards](#coding-standards-and-submission) - Write code according
  to project conventions
- ✅ [Build and Test](#build-and-test) - Implement features and run tests
- ✅ [Submit Contribution](#submit-your-contribution) - Sign the CLA and create a pull
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

See [Setup Guide](setup.md#install-minimum-ray-ascend-hccl-only) for complete
installation instructions.

## Clone and Build

```bash
# Clone the repository
git clone https://github.com/Ascend/ray-ascend.git
cd ray-ascend

# Build from source (editable installation)
pip install -e .

# Or build with all features
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
refer to [Contributing Guide](contributing.md#contribution-guidelines).

## Build and Test

Run tests using pytest to ensure your changes don't break existing functionality:

```bash
# Install all development dependencies
pip install -e ".[all]"

# Run all tests
pytest -v
```

All tests must pass before submitting a PR. For more testing details and options, see
[Contributing Guide](contributing.md#running-tests).

## Submit Your Contribution

### Prerequisites

Before submitting a Pull Request, you must:

#### 1. Sign the Ascend CLA (Contributor License Agreement)

*Required for first-time contributors only*

Visit the
[Ascend CLA Sign Portal](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1).

For details, see [Sign the Ascend CLA](contributing.md#sign-the-ascend-cla).

⚠️ **Important**: The email address used to sign the CLA must match your Git commit
email address.

#### 2. Verify Your Setup

```bash
# Confirm Git configuration
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

Before submitting, please ensure:

- ✅ Code follows all style conventions
- ✅ All tests pass
- ✅ New features have corresponding tests
- ✅ Documentation is updated
- ✅ Commits are signed (`-s` flag)
- ✅ CLA is signed (first-time contributors)
- ✅ Commit email matches CLA email

## Performance Testing

For instructions on running performance benchmarks for YuanRong Direct Transport, see
[Performance Testing Guide](performance_testing.md).

## Documentation

Documentation updates should be made to the relevant `.md` files in the `docs/`
directory. Use clear, concise language with code examples where appropriate.

## Additional Resources

- [Setup Guide](setup.md) - Detailed environment setup instructions
- [Contributing Guide](contributing.md) - Contribution workflow and guidelines

## Getting Help

If you encounter any issues:

1. Check the documentation above
1. Review existing issues and PRs
1. Open a new issue on GitHub with detailed information
