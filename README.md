# Dependency
We used [OpenRec](http://openrec.ai/), an open-source and modular library for neural network-inspired recommendation algorithms, to conduct our experiements in the paper. Please refer to the [repo](https://github.com/ylongqi/openrec) for installation details.

# Dataset 
The [MovieLens 20M](http://files.grouplens.org/datasets/movielens/ml-20m-README.html) dataset was used as a testbed to evaluate how user-controlled data filtering could affect recommendation performance. Follow [this notebook](/notebook/movielens_dataset_parse.ipynb) to preprocess the dataset.

# Experiments
Please refer to our paper for more details about the experiments and the findings. Here we focus on explaining how to reproduce the results.

### Hyperparameters selection
Under the project folder, run `./scripts/time_validation.sh $RECOMMENDER` to conduct hyperparemeter selection for the recommender.  `$RECOMMENDER` is one of the three: "CML", "BPR", "PMF" (you can extend it to other recommenders as well). Log files will be saved into the `./movielens_validation_logs/` folder.

### Model configurations
After the validation logs are generated, follow the [Model configuration](/notebook/movielens_experiments.ipynb#Model-configuration) section to generate model configurations for testing.

### Evaluation
Under the project folder, run `./scripts/time_test.sh $RECOMMENDER $EVALUATOR` to evaluate recommendation performance on test set. `$EVALUATOR` is one of "Recall" (Hit Ratio) and "NDCG" (Normalized Discounted Cumulative Gain). Test logs will be saved into the `./movielens_test_logs/` folder.

### Results
Follow the [Experiments](/notebook/movielens_experiments.ipynb#Experiments) sections to generate figures that illustrate the experimental results.

# Reference
Hongyi Wen, Longqi Yang, Michael Sobolev, and Deborah Estrin. 2018. Exploring Recommendations Under User-Controlled Data Filtering. In Twelfth ACM Conference on Recommender Systems (RecSys ’18), October 2–7, 2018, Vancouver, BC, Canada.

