# Deep Reinforcement Learning for Inverse Inorganic Materials Design
This repository contains the code and data for the paper *Deep Reinforcement Learning for Inverse Inorganic Materials Design* by Karpovich et al.

# Installation Instructions
- Clone this repository and navigate to it. 
- Create the conda environment for the PGN tasks. `conda env create --name PGN_env --file requirements_PGN.txt`
- For DQN tasks, follow the instructions in the repo linked under `DQN` (https://github.com/eltonpan/RL_materials_generation)
- Create the conda environment for the DING tasks. `conda env create --name DING_env --file requirements_DING.txt`
- Create the conda environment 
- Switch to the new environment, depending on which notebook you are running. `conda activate <env_name>`
- Add the environment to jupyter and activate it `python -m ipykernel install --name <env_name>`

# Data
The full datasets used in the paper are available online. Data must be downloaded to an appropriate `data` folder before and preprocessed before any of the notebooks can be run. The data used in this work is from the following papers:
- Kononova, O., Huo, H., He, T., Rong Z., Botari, T., Sun, W., Tshitoyan, V. and Ceder, G. Text-mined dataset of inorganic materials synthesis recipes. Sci Data 6, 203 (2019). (https://doi.org/10.1038/s41597-019-0224-1)
  - Github link to dataset: (https://github.com/CederGroupHub/text-mined-synthesis_public)
- Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., ... & Persson, K. A. Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. APL materials, 1, 1 (2013).
  - Link to Materials Project: (https://next-gen.materialsproject.org/)

# Usage
Each folder pertains to a particular task (synthesis route classification or synthesis condition prediction) containing the associated Jupyter notebooks and python code.
- The `PGN` folder contains the necessary code for the Policy Gradient Network (PGN) training and evaluation tasks.
- The `DQN` folder contains the necessary code for the Deep-Q Network (DQN) training and evaluation tasks, also linked in a separate repo (https://github.com/eltonpan/RL_materials_generation).
- The `DING` folder contains the necessary code for the Deep Inorganic Material Generator (DING) training and evaluation tasks.

# Cite
If you use or adapt this code in your work please cite as:
```
TBD
```

# Disclaimer
This is research code shared without support or guarantee of quality. Please report any issues found by opening an issue in this repository. 
