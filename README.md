# BIKE SHARING STATIONS LOCATION OPTIMIZATION 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16993699.svg)](https://doi.org/10.5281/zenodo.16993699)

## Overview

This project implements an optimization framework for determining optimal locations of BSS stations. It combines spatial data analysis, graph theory, and metaheuristic optimization algorithms to solve the complex facility location problem while considering multiple urban factors. To prove its capabilities, it is applied to the city of Barcelona.

## Features

- **Multi-criteria decision-making framework** with modular scoring approach for BSS optimization
- **Comprehensive data processing pipeline** integrating multiple urban data sources
- **Advanced optimization algorithms** with spatial awareness and elevation modeling
- **Interactive visualization suite** including maps and analysis tools
- **Docker support** for reproducible environments


## Project Structure

```
├── data/                         # Data storage
│   ├── raw/                      # Raw data files (INE, OSM, Barcelona)
│   └── processed/                # Processed datasets
│
├── src/                         # Source code
│   ├── input/                    # Input data processing
│   ├── data_loader.py            # Data loading utilities
│   ├── create_input_dataset/     # Dataset preparation│ 
│   ├── assign_to_nodes/          # Node assignment logic  
│   ├── optimization/             # Optimization set up, validation and experiments
│   └── results_exploration/      # Results analysis and visualization tools
│
├── notebooks/                     # Jupyter notebooks for analysis
│
├── visualizations/                # Generated visualizations
│
├── requirements.txt               # Python dependencies
├── Dockerfile                    # Docker configuration
└── paths.py                      # Path configuration
```

## Key Components

### Data Sources & Spatial Analysis
- **Multi-source integration**: INE census data, OpenStreetMap infrastructure, Barcelona Open Data
- **Network modeling**: Graph-based analysis using NetworkX and OSMnx
- **Geographic processing**: Geopandas and Shapely for spatial operations
- **Elevation modeling**: Terrain-aware routing with slope-adjusted calculations

### Optimization Algorithms
- **Genetic Algorithm**: Population-based optimization with tournament selection, greedy crossover, and configurable mutation rates. Validated against MILP solutions achieving near-optimal results (≤1% gap)
- **Simulated Annealing**: Single-solution optimization with temperature-based search
- **Multi-objective approach**: Balances coverage, accessibility, and socioeconomic equity

## Installation

### Prerequisites

- **Python 3.11+** (required for local installation)
- **Docker** (recommended for reproducible environments)

### Option 1: Docker Installation (Recommended)

The project includes a complete Docker setup for reproducible environments:

1. **Clone the repository:**
```bash
git clone <repository-url>
cd bss-station-optimization
```

2. **Build the Docker image:**
```bash
docker build -t stations_optim .
```

3. **Run the container:**
```bash
docker run -it \
  -v $(pwd)/:/home/ \
  --user $(id -u):$(id -g) \
  stations_optim:latest
```

**Docker Features:**
- Python 3.11-slim base image
- Pre-installed JupyterLab and IPython kernel
- All system dependencies included
- **Volume mounting**: The `-v $(pwd)/:/home/` flag mounts your current directory to `/home/` inside the container, ensuring:
  - All your code, data, and results persist between container runs
  - Changes made inside the container are immediately visible on your host machine
  - You can edit files on your host and see changes in the container (and vice versa)
  - Data processing results are saved to your local filesystem

### Option 2: VS Code Dev Container

For VS Code users, the project includes a `.devcontainer` configuration:

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the project in VS Code
3. When prompted, click "Reopen in Container"
4. VS Code will automatically build and start the container with:
   - Python extension
   - Jupyter extension
   - Pre-configured workspace settings



## Usage

The project is organized into four main pipelines. Each one has its own detailed documentation:

#### 1. **Input Dataset Creation** (`src/create_input_dataset/`)
Downloads, processes and saves multi-source data from INE, OSM, and OpenDataBarcelona:
- **Socioeconomic data**: Population by age/sex, education, unemployment, income, vehicle ownership
- **Public transport infrastructure**: Metro, tram, and bus entrances, stops and lines from OSM
- **Urban structure**: Bike lanes and Points of Interest categorized by user input
- **Topography**: Elevation data and slope calculations from OpenTopoData

📖 **For detailed usage**: See [`src/create_input_dataset/README.md`](src/create_input_dataset/README.md)

#### 2. **Node Attribute Assignment** (`src/assign_to_nodes/`)
Enrichment of the network nodes by spatially aggregating the data obtained in the first pipeline.
- **Buffer-based assignment**: configurable radius buffers around network nodes to define their spatial context.
- **Multi-layer integration**: Combines infrastructure, urbanism, and socioeconomic data
- **Attribute normalization**: Multiple normalization methods (minmax, log, boxcox, robust, zscore)

📖 **For detailed usage**: See [`src/assign_to_nodes/README.md`](src/assign_to_nodes/README.md)

#### 3. **Optimization Engine** (`src/optimization/`)
Metaheuristic algorithms based on the result of the second pipeline:
- **GA**: Selection, elitism, crossover, and mutation strategies
- **Elevation-aware routing**: Realistic travel times using slope-adjusted travel distances
- **Graph metrics integration**: Network analysis for BSS accessibility and inter-station proximity 
- **Experimental pipeline**: from GA hyperparameter tuning to scenario analysis.

📖 **For detailed usage**: See [`src/optimization/README.md`](src/optimization/README.md)

#### 4. **Results Exploration** (`src/results_exploration/`)
Analysis and visualization tools for optimization results:
- **Four-panel comparison maps**: Administrative divisions, income, infrastructure, and POIs
- **Interactive mapping**: Folium-based maps with multiple basemap options
- **Utility functions**: Grid-based analysis, node plotting, and radar charts

📖 **For detailed usage**: See [`src/results_exploration/README.md`](src/results_exploration/README.md)



### Notebooks

The repository includes Jupyter notebooks for demonstration and exploration, with step-by-step pipeline examples, interactive parameter tuning, and rich visualizations. 

The visualizations of the paper where generated in these notebooks.

## Output

The system generates:

- **Optimized station locations**
- **Spatial analysis results** including visualizations and different empircal metrics
- **Interactive maps** for exploration and presentation



## Barcelona Case Study

The framework has been extensively tested on Barcelona's Bicing system, demonstrating:

- **Real-world scale**: 18,700+ network nodes, 38,000+ edges covering 101 km²
- **Complex urban context**: 10 districts, 73 neighborhoods with marked socioeconomic contrasts
- **Topographic challenges**: Coastal flatlands to 450m mountain slopes requiring elevation-aware routing
- **Multi-modal integration**: 260+ km bike lanes, 8 metro lines, 6 tram lines, 100+ bus routes

The Barcelona implementation showcases the framework's ability to handle real-world complexity while maintaining computational efficiency and solution quality.

## License

This software is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
You are free to use, modify, and share this code for non-commercial purposes, provided that proper credit is given to the original author.
