# Physical-Social Forces (PSF) Model for Representing Physical and Social Events

## Download data and saved results
Download training/testing data, saved model parameters and testing results from [here](https://www.tshu.io/HeiderSimmel/GenCoord-DataModel.zip). Put the folders `checkpoints` and `data` in the root directory.

## Our approach
### Training soical potentials

1. Training collision
```
python -m experiments.train_physical --dataset collision
``` 

2. Training spring
```
python -m experiments.train_spring --dataset spring
```

### Training soical potentials.

1. Training goal 'leaving' for the first agent:
```
python -m experiments.train_social --dataset leaving --trainable-entities 0
```

2. Training goal 'blocking' for the second agent:
```
python -m experiments.train_social --dataset blocking --trainable-entities 1
```

### Intention inference
```
python -m experiments.test_social
```

### Physical violation inference
```
python -m experiments.test_physical --dataset collision
python -m experiments.test_physical --dataset spring
```


### Ablation: using all variables
For this ablated model, run the commands as above, but use `experiments.train_all_coord_physical` and `experiments.train_all_coord_social` for training, and use`experiments.test_all_coord_physical` and `experiments.test_all_coord_social` for testing.


## Baseline (LSTM)

### Training for physical events
```
python -m experiments.train_physical_dnn --dataset collision
python -m experiments.train_physical_dnn --dataset spring
```

### Training for social events
```
python -m experiments.train_social_dnn --dataset leaving --agent-id 0
python -m experiments.train_social_dnn --dataset blocking --agent-id 1
```

### Physical violation inference
```
python -m experiments.test_phyiscal_dnn --dataset collision
python -m experiments.test_physical_dnn --dataset spring
```

### Intention inference
```
python -m experiments.test_social_dnn
```

## Human response and model prediction comparison

The Jupyter notebooks under `scripts` folder contain the codes for ploting human response and model predictions, training HH/HO/OO classiers, and computing the correlation between human responses and model predictions for our full model (`analysis_visualization_model_human.ipynb`), the ablated model using all variables (`analysis_visualization_all_coord_baseline.ipynb`), the DNN baseline (`analysis_visualization_DNN_baseline.ipynb`), and the baseline using only the degree of animacy as input (`analysis_visualization_freq_baseline.ipynb`).








