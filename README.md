# Social Media Opinion Dynamics and Disease Spread

A simulation framework studying how social media algorithms and bot networks influence opinion polarization and disease transmission in networked populations.

## Key Findings

🔍 **Feed Algorithm Effects**: Popularity-based ranking stabilizes extreme opinions, while random feeds promote opinion diversity

🤖 **Bot Paradox**: Anti-mask bots create short-term polarization but paradoxically reduce disease spread through behavioral clustering

📊 **Opinion-Behavior Coupling**: Individual opinions directly influence mask-wearing behavior, creating feedback loops between social dynamics and epidemiological outcomes

## Model Overview

The simulation combines three interconnected dynamics:

- **Network Structure**: Watts-Strogatz small-world networks with humans and bots
- **Opinion Dynamics**: Bounded confidence model with social media post sharing
- **Disease Dynamics**: SIR model where transmission rates depend on mask-wearing behavior

## Quick Start

```python
from simulation import run_and_save_simulation

# Basic parameter set
param = {
    "n_humans": 100,
    "n_bots": 20, 
    "nei": 4,
    "p": 0.1,
    "N_steps": 5000,
    "n_of_replicas": 10,
    "feed_algorithm": "popularity",  # or "random"
    "beta0": 0.1,
    "recovery_rate": 0.05,
    "bot_threshold": -0.5
}

# Run simulation
results, analysis = run_and_save_simulation(param)
print(f"Final opinion: {analysis['mean_final_opinion']:.3f}")
```

## Core Components

### Network (`network.py`)
- Creates small-world networks with bot connections
- Supports flexible opinion initialization (uniform, gaussian, bimodal)
- Precomputes neighbor relationships for performance

### Opinion Dynamics (`opinion_dynamic.py`)
- Bounded confidence model with communication noise
- Social media feed algorithms (popularity vs random)
- Post generation, reading, and upvoting mechanics
- Bots contribution, upvoting posts below threshold

### Disease Dynamics (`disease_dynamics.py`) 
- SIR model with opinion-dependent transmission rates
- Discrete behavioral groups for computational efficiency
- Vectorized infection/recovery calculations

### Feed Algorithms (`feed_algorithms.py`)
- **Popularity**: Global ranking by upvotes, personalized by network
- **Random**: Uniform sampling from neighbor posts
- **Similarity**: Content-based filtering (extensible)

## Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n_humans` | Population size | 50-1000 |
| `n_bots` | Number of bots | 0-64 |
| `feed_algorithm` | "popularity", "random" | - |
| `bot_threshold` | Bot upvoting threshold | -0.8 to 0.8 |
| `epsilon` | Confidence bound | 0.3 |
| `mu` | Opinion update rate | 0.15 |

## Experimental Design

The framework supports systematic parameter sweeps:

```python
# Example: Bot influence study
bot_counts = [0, 10, 20, 50]
algorithms = ["popularity", "random"]

for n_bots in bot_counts:
    for algo in algorithms:
        param["n_bots"] = n_bots
        param["feed_algorithm"] = algo
        results = run_and_save_simulation(param)
        # Analyze polarization vs disease spread
```

## Output Data

Results include both aggregate time series and individual trajectories:

- **Epidemiological**: S/I/R counts, infection rates
- **Opinion**: Mean, variance, distribution bins
- **Individual**: Full opinion trajectories for first 5 replicas
- **Network**: Static structure and dynamic post interactions

## Performance Features

- **Vectorized Operations**: NumPy-based calculations for 10x+ speedup
- **JIT Compilation**: Numba acceleration for critical functions  
- **Memory Efficiency**: Selective trajectory saving, compressed data types
- **Scalability**: Handles 1000+ agents, 10K+ time steps

## Dependencies

```
numpy >= 1.20
igraph >= 0.9
numba >= 0.56
pickle (standard library)
```

## Citation

If you use this code, please cite:

```
[Author et al.] "Social Media Algorithms and Disease Dynamics: 
The Paradoxical Effects of Bot Networks on Opinion Polarization 
and Epidemic Spread" [Conference/Journal] (2025)
```

## License

MIT License - see LICENSE file for details.

## Contact

fabio.sartori@kit.edu

---

*This research explores the complex interplay between information dynamics and public health in the digital age.*