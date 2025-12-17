# CPSC-485-Final-Project

This repository details the EM training algorithm used to learn temporally-abstract world models. 

## Quick Start
Ensure that you have downloaded the AntMaze Medium-Diverse dataset from Minari. 
```
minari download antmaze-medium-diverse-v1
```

Then, train the model (training on 1 GPU with the AntMaze dataset over 250 epochs requires around 7 hours).

```
python training.py
```

Once training has concluded, run the experiments in plot_models.ipynb.
