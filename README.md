# Tabula

This library aims to apply traditional reinforcement learning methods to play games!

## Environment

We've included three simple games: `boat`, `grid world` and `geo search`.

## Solvers

We've included three slovers using traditional reinforcement learning algorithms:

- Dynamic Programming
- Monte Carlo
- Temporal Difference

## Metrics

We provide visualization for optimal policy and gameplay under optimal policy, along with plot for mean episodic rewards / value functions for visualizing convergence.

- convergence_plot.png
- gameplay.gif
- policy_visualization.png

## Installation

Download latest release and install via `pip install` or other package manager of choice.

## Examples

Using **Temporal Difference** to solve `grid world`
```bash
python examples/example_usage.py --env grid_world --algo td --save_metrics --verbose
```

Examine the metrics in `output/` folder

## Build

Contributions are welcome! Build this project by installing requirements and build from source!

```bash
git clone https://github.com/kanghengliu/Tabula.git
pip install requirements.txt
python -m build
```

