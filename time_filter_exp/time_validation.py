import os
import sys
from openrec import ImplicitModelTrainer
from openrec.utils import ImplicitDataset
from openrec.recommenders import PMF, CML, BPR
from openrec.utils.evaluators import Recall, NDCG
from openrec.utils.samplers import PairwiseSampler,PointwiseSampler
import logging 
import datetime
import csv
import tensorflow as tf
import numpy as np

batch_size = 1000
test_batch_size = 100
num_itr = 1e4+1
display_itr = 1000
LOG_TYPE  = 'validation'
LOGGING = True 

def load_data(dataset, hist_len):
    if dataset == 'movielens':
        raw_data = np.load("./dataset/user_data_truncated_%s.npy" % hist_len)
        print ('raw movielens dataset loaded')
        return raw_data
    else:
        print ("No dataset loaded...")
        return

def run_exp(model_name=None, raw_data=None, user_per=1.0, keep_days=1, l2_reg=0.001, test_date=None, outdir=None):

    # parse dataset into incremental training and testing set    
    data = raw_data
    max_user = len(np.unique(data["user_id"]))
    max_item = len(np.unique(data["item_id"]))
    print ("max_user:{}, max_item:{}".format(max_user, max_item))

    test_date = datetime.datetime.strptime(test_date, "%Y-%m-%d").date()
    print ("test date:%s" % test_date)
    train_data = data[data["timestamp"] < test_date]
    
    np.random.seed(10)
     
    # filter training data, for selected users keep only the most recent n days of data
    print ("filter user percentage:%f" % user_per)
    print ("ratings before filter:%d" % len(train_data)) 
    user_list = np.unique(train_data["user_id"])
    filter_user = np.random.choice(user_list, int(len(user_list)*user_per), replace=False)
    mask = (np.isin(train_data["user_id"], filter_user)) & (train_data["timestamp"] < (test_date-datetime.timedelta(days=keep_days)))
    train_data = train_data[~mask] 
    print ("ratings after filter:%d" % len(train_data)) 
    
    # random select one item for each user for validation
    user_list = np.unique(train_data["user_id"])
    val_index = [np.where(train_data["user_id"] == uid)[0][0] for uid in user_list] # leave out the most recent rating for validation
    val_data = train_data[val_index]
    train_data = np.delete(train_data, val_index)
    print ("trian data: %d, validation data %d" % (len(train_data), len(val_data)))

    train_dataset = ImplicitDataset(train_data, max_user, max_item, name='Train')
    val_dataset = ImplicitDataset(val_data, max_user, max_item, name='Val')

    num_process = 8
    dim_embed = 50
    if model_name == 'PMF':
        model = PMF(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(), dim_embed=dim_embed, opt='Adam', l2_reg=l2_reg)
        sampler = PointwiseSampler(batch_size=batch_size, dataset=train_dataset, pos_ratio=0.5, num_process=num_process)
    elif model_name == 'CML':
        model = CML(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(),
                    dim_embed=dim_embed, opt='Adam', l2_reg=l2_reg)
        sampler = PairwiseSampler(batch_size=batch_size, dataset=train_dataset, num_process=num_process)
    elif model_name == 'BPR':
        model = BPR(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(),
                    dim_embed=dim_embed, opt='Adam', l2_reg=l2_reg)
        sampler = PairwiseSampler(batch_size=batch_size, dataset=train_dataset, num_process=num_process)
    else:
        print ("Wrong model assigned")
        return 
    
    recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ndcg_evaluator  = NDCG(ndcg_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size, train_dataset=train_dataset, model=model, sampler=sampler, item_serving_size=1)
    model_trainer.train(num_itr=num_itr, display_itr=display_itr, eval_datasets=[val_dataset], evaluators=[recall_evaluator, ndcg_evaluator], num_negatives=200)

if __name__ == "__main__":
    print (sys.argv)
    dataset = sys.argv[1]
    model_name = sys.argv[2]
    l2_reg = float(sys.argv[3])
    test_date = sys.argv[4] 
    hist_len = sys.argv[5]

    if len(sys.argv) >= 8:
        user_per = float(sys.argv[6])
        keep_days = int(sys.argv[7])
    else:
        user_per = 0.0
        keep_days = 0

    raw_data = load_data(dataset, hist_len) 
    # logging
    outdir = None
    if LOGGING:
        outdir = "{}_{}_logs/{}_{}_{}_{}_{}/".format(dataset, LOG_TYPE, model_name, user_per, keep_days, l2_reg, hist_len)
        os.popen("mkdir -p %s" % outdir).read()
        log = open(outdir+ test_date + "_training.log", "w")
        sys.stdout = log
        
    run_exp(model_name=model_name, raw_data=raw_data, user_per=user_per, keep_days=keep_days, l2_reg=l2_reg, test_date=test_date, outdir=outdir)
