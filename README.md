# SPLINTER

### "The linear **SPL**ine **IN**terannual **T**rend **E**valuato**R**"

SPLINTER is a generalized additive model (GAM) framework with outlier detection and removal for time series analysis of environmental monitoring data for chemical contaminants.

The SPLINTER analyzes time series data using:

* GAM decomposition
* Change point detection
* Outlier identification and removal

Outputs can be saved automatically to a timestamped results directory.

## Repository Contents

* [`requirements.txt`](requirements.txt) – Python package dependencies for pip installation
* [`splinter.yml`](splinter.yml) – Conda environment file
* [`SPLINTER_UserGuidelines.docx`](SPLINTER_UserGuidelines.docx) – detailed user instructions
* [`SPLINTER_V012.ipynb`](SPLINTER_V012.ipynb) – notebook for running SPLINTER on user datasets, including an example analysis of [`PAH`](PAH_PCB)
* [`SPLINTER_V012_batch.py`](SPLINTER_V012_batch.ipynb) – batch-mode execution script for processing multiple datasets
* [`splinter_function.py`](splinter_function.py) – main SPLINTER functions

## Guide for Users

### Environment and Dependencies

We recommend using Visual Studio Code as the IDE for running SPLINTER. To set up the model environment and install all necessary libraries and packages, you can use either of the following approaches:

- **Option 1: pip**

  For users with an existing Python environment:

  ```shell
  pip install -r requirements.txt
  ```

  - Tested with Python 3.13
  - Required packages are listed in [`requirements.txt`](requirements.txt)

- **Option 2: Conda**

  Recommended for managing multiple project environments.

  Create a new environment from the provided YML file:

  ```shell
  conda env create -f splinter.yml
  ```

  - Environment file: [`splinter.yml`](splinter.yml)

Both options allow users to run the Jupyter Notebook files interactively within Visual Studio Code.

## Documentation

Detailed instructions for each execution step are provided in:

* [`SPLINTER_UserGuidelines.docx`](SPLINTER_UserGuidelines.docx)

## Authors and Contributors

SPLINTER was developed and is maintained by [Matthew MacLeod](https://www.su.se/english/profiles/m/mmacl), Stockholm University, with contributions from:

* Xiaoyu Zhang ([`@xy2gh`](https://github.com/xy2gh)), Stockholm University
* Silke Cornelissen ([`@silkenc`](https://github.com/silkenc)), HAS Green Academy

The code was developed as a resource for the Arctic Monitoring and Assessment Programme (AMAP).
