# Social Network Opinion Dynamics with Bots

A computational model simulating opinion formation and disease spread in social networks with both human users and automated bots.

## Overview

This simulation models how opinions evolve in a social network where:
- **Humans** update their beliefs based on posts from neighbors (Bounded Confidence Model)
- **Bots** systematically upvote content below a threshold to influence visibility
- **Feed algorithm** shows users the most upvoted posts from their network
- **Disease dynamics** spread based on opinion-influenced protective behaviors

## Key Features

- **Vectorized computation** using NumPy for performance
- **Bounded Confidence Model** for human opinion updates
- **Bot influence** through strategic upvoting of divisive content
- **Realistic social media dynamics** with posts, upvotes, and personalized feeds
- **Multiple simulation replicas** for statistical analysis
- **Rich visualization suite** for analyzing results

## Model Components

### Network Structure
- Watts-Strogatz small-world network for humans
- Bots connected to random human subsets
- Precomputed neighbor lists for efficiency

### Opinion Dynamics
- Humans post content reflecting their current opinion (with noise)
- Feed algorithm ranks posts by upvotes
- Humans read top posts from neighbors and update beliefs if within confidence bound
- Bots automatically upvote posts below threshold

### Disease Dynamics (Optional)
- SIR model where protective behavior depends on opinion
- More polarized individuals take fewer precautions
- Disease transmission through network connections

## Installation

```bash
pip install igraph numpy numba matplotlib pickle
```

## Quick Start

```python
from network_generator import simulations, save_results
from observables_plot import plot_multiple_replicas

# Define parameters
param = {
    "n_humans": 450,
    "n_bots": 50, 
    "N_steps": 1000,
    "n_of_replicas": 5,
    "nei": 6,  # initial neighbors in Watts-Strogatz
    "p": 0.05,  # rewiring probability
    "mu": 0.075,  # opinion update rate
    "epsilon": 0.3,  # confidence bound
    "bot_threshold": -0.5,  # bots upvote posts below this
    "communication_error": 0.2,  # noise in posts
    "post_history": 10,  # posts remembered
    "feed_size": 5,  # posts shown per round
    "beta0" : 0.0125*4,
    "recovery_rate" : 0.025*4,
    "I0": 1  # initial infected
}

# Run simulation
results = simulations(param)

# Plot results
fig, axes, final_infection = plot_multiple_replicas(results, param)

# Save for later analysis  
save_results(results, param)
```

## File Structure

- `network_generator.py` - Main simulation engine
- `observables_plot.py` - Visualization functions
- `results/` - Saved simulation outputs

## Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n_humans` | Number of human users | 50-500 |
| `n_bots` | Number of bots | 0-100 |
| `mu` | Opinion update strength | 0.05-0.2 |
| `epsilon` | Confidence bound | 0.1-0.5 |
| `bot_threshold` | Bot upvoting threshold | -1.0 to 0.0 |
| `feed_size` | Posts per user per round | 1-10 |

## Visualization Options

- `plot_multiple_replicas()` - Overview of all replicas
- `plot_trajectories_2x4()` - Individual user trajectories  
- `plot_opinion_space_simple()` - Phase space trajectories
- `plot_multiple_replicas_3x3()` - Detailed multi-replica view

## Research Applications

This model can investigate:
- **Bot influence** on opinion polarization
- **Feed algorithm effects** on information spread
- **Public health messaging** effectiveness
- **Echo chamber formation** in social media
- **Intervention strategies** for misinformation

## Performance Notes

- Uses JIT compilation (Numba) for critical loops
- Vectorized operations for array processing
- Precomputed neighbor lookups for efficiency
- Scales to networks of 1000+ nodes

## Citation

If you use this code in research, please cite:

```
[Your citation format here]
```

## License

[Your chosen license]