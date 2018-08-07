# Dependency
- pip3 install tensorflow
- install [openrec](https://github.com/ylongqi/openrec)

# Dataset 
We used the [MovieLens 20M](http://files.grouplens.org/datasets/movielens/ml-20m-README.htm) dataset for our experiments. Follow [this notebook](/notebook/movielens_dataset_parse.ipynb) to parse the dataset.

# Experiments
Please refer to our paper for more details about the experiments. Here we focus on explaining how to reproduce the results.

### 1. hyperparameters selection using a validation set
Under the project folder, run `./scripts/time_validation.sh $RECOMMENDER` to conduct hyperparemeter selection for the recommender.  `$RECOMMENDER` is one of the three: "CML", "BPR", "PMF". Log files will be saved into the `./movielens_validation_logs/` folder. 

### 2. generate model configurations
After the validation logs are generated, follow the **model configuration** section in [this notebook](/notebook/movielens_experiments.ipynb#config) to generate model configurations.

### 3. training and testing
Under the project folder, run `./scripts/time_test.sh $RECOMMENDER $EVALUATOR` to evaluate performance on test set. `$EVALUATOR` we used include "Recall" (Hit Ratio) and "NDCG" (Normalized Discounted Cumulative Gain). Test logs will be saved into the `./movielens_test_logs/` folder. 

### 4. evaluations 
For details, follow the **experiments** sections in [this notebook](/notebook/movielens_experiments.ipynb#experiments) 

# Reference


