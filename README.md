# CPSC-485-Final-Project

This repository details the EM training algorithm used to learn temporally-abstract world models inspired by the [OPOSM](https://proceedings.mlr.press/v202/freed23a.html) paper.

[Notion Page](https://determined-bathroom-72a.notion.site/William-Huang-CPSC-4850-Final-Project-2ca099a878b58035a9a4f8b0a526ec3e?source=copy_link)

<img width="482" height="480" alt="Screenshot 2025-12-17 at 2 48 18 PM" src="https://github.com/user-attachments/assets/ffff71f7-d47a-4ea3-b9bd-24b5a02e97d8" />


## Quick Start
First, start and activate a Conda environment.

```
conda create --name your_env_name
conda activate your_env_name
```

Then, install the necessary packages.

```
pip install -r requirements.txt
```

Ensure that you have downloaded the AntMaze Medium-Diverse dataset from Minari. 
```
minari download antmaze-medium-diverse-v1
```

Also, it is recommended to utilize Weights and Biases (W&B) to track your experiments. To do so, first create an account and API key. Then, run 

```
wandb login
```

When prompted for the API key, paste it in.

Then, train the model by running model.py (training on 1 NVIDIA A100 GPU with the AntMaze dataset over 250 epochs requires around 7 hours).

```
python training.py
```

Once training has concluded, run the experiments in plot_models.ipynb. For details on the experiments that were run, please see the Notion page.
