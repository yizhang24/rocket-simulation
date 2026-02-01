# Simulation analysis for rocket team

## Quickstart
### Prerequisites
- Python 3.13 or higher (lower versions may work but YMMV)
- [uv](https://docs.astral.sh/uv/) for package management
- OpenRocket (if you want to simulate your own flight data)
### Installation

```bash
# Clone the repository
git clone https://github.com/yizhang24/rocket-simulation
cd rocket-simulation

# Install dependencies and create/activate virtual environment
uv sync --all-extras --dev

# Analyze a CSV file
uv run python -m scripts.analyze openrocket.csv
