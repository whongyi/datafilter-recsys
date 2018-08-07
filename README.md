# Dependency
- pip3 install tensorflow
- install [openrec](https://github.com/ylongqi/openrec)

# Dataset 
We used the [MovieLens 20M](http://files.grouplens.org/datasets/movielens/ml-20m-README.html) dataset for our experiments. Follow [this notebook](/notebook/movielens_dataset_parse.ipynb) to parse the dataset.

# Experiments
Please refer to our paper for more details about the experiments. Here we focus on explaining how to reproduce the results.

### Hyperparameters selection using a validation set
Under the project folder, run `./scripts/time_validation.sh $RECOMMENDER` to conduct hyperparemeter selection for the recommender.  `$RECOMMENDER` is one of the three: "CML", "BPR", "PMF". Log files will be saved into the `./movielens_validation_logs/` folder. 

### Generate model configurations
After the validation logs are generated, follow the [Model configuration](/notebook/movielens_experiments.ipynb#Model-configuration) section to generate model configurations.

### Training and testing
Under the project folder, run `./scripts/time_test.sh $RECOMMENDER $EVALUATOR` to evaluate performance on test set. `$EVALUATOR` we used include "Recall" (Hit Ratio) and "NDCG" (Normalized Discounted Cumulative Gain). Test logs will be saved into the `./movielens_test_logs/` folder. 

### Model evaluations 
Follow the [Experiments](/notebook/movielens_experiments.ipynb#Experiments) sections to generate figures for the experiments.

# Reference
Hongyi Wen, Longqi Yang, Michael Sobolev, and Deborah Estrin. 2018. Exploring Recommendations Under User-Controlled Data Filtering. In Twelfth ACM Conference on Recommender Systems (RecSys ’18), October 2–7, 2018, Vancouver, BC, Canada.

