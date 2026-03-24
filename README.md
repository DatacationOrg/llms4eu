# llms4eu

> **Keep this README up-to-date!** As the project evolves, update this file to reflect the current state: new features, changed commands, updated dependencies, and any other information that helps teammates (and your future self) get started quickly.

## Overview

_Replace this section with a short description of what this project does and why it exists._

## Getting Started

### Prerequisites

**[uv](https://github.com/astral-sh/uv)** – Python package and project manager

```bash
# macOS
brew install uv

# Ubuntu / Debian
sudo snap install --classic uv

# Windows
winget install astral-sh.uv
```

### Installation

```bash
# Clone the repository
git clone git@github.com:DatacationOrg/lllms4eu.git
cd lllms4eu

# Install dependencies
uv sync
```

### Running the Application

```bash
uv run python llms4eu/main.py
```

### Running Tests

```bash
uv run pytest
```

### Code Quality

Pre-commit hooks are configured to enforce code quality standards. Install them to automatically run checks before each commit:

```bash
uvx prek install
```

Run checks manually on all files:

```bash
uvx prek run --all-files
```

> **Tip:** To commit even when pre-commit checks fail, use `git commit --no-verify`.

## Project Structure

```shell
lllms4eu/
├── .github/
│   └── workflows/              # CI/CD pipelines (pytest, pre-commit, deploy)
├── .vscode/                    # VSCode workspace settings and debug configs
├── data/                       # Data directories (not committed to git)
│   ├── raw/                    # Original, immutable source data
│   ├── interim/                # Intermediate transformed data
│   ├── processed/              # Final, analysis-ready datasets
│   └── external/               # Data from third-party sources
├── deployment/
│   └── terraform/              # Infrastructure-as-code for Azure resources
├── notebooks/                  # Jupyter notebooks for exploration and analysis
├── tests/                      # Test suite
├── llms4eu/          # Main Python package
│   ├── __init__.py
│   └── main.py                 # Application entry point
├── .env                        # Local environment variables (not committed)
├── .gitignore
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── .python-version             # Python version pin
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml              # Project metadata and dependencies
└── README.md
```

## Docker

```bash
# Build and run the container
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop and remove containers
docker-compose down
```

## Deployment

This project contains [Terraform](https://developer.hashicorp.com/terraform) configurations to provision resources in Azure:

```bash
cd deployment/terraform
terraform init
terraform apply -var-file='.tfvars'
```

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes and ensure tests pass: `uv run pytest`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request targeting the `dev` branch

## License

_Add your license information here._
