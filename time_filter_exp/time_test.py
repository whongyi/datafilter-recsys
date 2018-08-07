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
LOG_TYPE = 'test'
LOGGING = True 

def load_data(dataset, hist_len):
    if dataset == 'movielens':
        raw_data = np.load("./dataset/user_data_truncated_%s.npy" % hist_len)
        print ('raw movielens dataset loaded')
        return raw_data
    else:
        print ("No dataset loaded...")
        return

def run_test_exp(model_name=None, evaluator=None, raw_data=None, user_per=1.0, keep_days=1, l2_reg=0.001, test_date=None, outdir=None, num_itr=1e4+1):

    # parse dataset into incremental training and testing set    
    data = raw_data
    max_user = len(np.unique(data["user_id"]))
    max_item = len(np.unique(data["item_id"]))
    print ("max_user:{}, max_item:{}".format(max_user, max_item))

    test_date = datetime.datetime.strptime(test_date, "%Y-%m-%d").date()
    print ("test date:%s" % test_date)
    train_data = data[data["timestamp"] < test_date]
    test_data = data[(data["timestamp"] >= test_date) & (data["timestamp"] < (test_date+datetime.timedelta(days=7)))]
    np.random.seed(10)
    test_data = np.asarray([np.random.choice(test_data[test_data["user_id"] == uid], 1)[0] for uid in np.unique(test_data["user_id"])])   
    
    # filter training data, for selected users keep only the latest n days of data
    print ("filter user percentage:%f" % user_per)
    print ("ratings before filter:%d" % len(train_data)) 
    user_list = np.unique(train_data["user_id"])
    filter_user = np.random.choice(user_list, int(len(user_list)*user_per), replace=False)
    filter_mask = (np.isin(train_data["user_id"], filter_user)) & (train_data["timestamp"] < (test_date-datetime.timedelta(days=keep_days)))

    # output filtered data and test data
    if LOGGING:
        np.save(outdir+"filtered_data.npy", train_data[filter_mask])
        np.save(outdir+"train_data.npy", train_data[~filter_mask])
        np.save(outdir+"test_data.npy", test_data)

    train_data = train_data[~filter_mask] 
    print ("ratings after filter:%d" % len(train_data)) 
    
    train_dataset = ImplicitDataset(train_data, max_user, max_item, name='Train')
    test_dataset = ImplicitDataset(test_data, max_user, max_item, name='Test')

    num_process = 8
    dim_embed = 50
    if model_name == 'pmf':
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
    
    if evaluator == 'Recall':
        test_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    elif evaluator == 'NDCG':
        test_evaluator  = NDCG(ndcg_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    else:
        print ("Wrong evaluator assisgned")
        return

    model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size, train_dataset=train_dataset, model=model, sampler=sampler, item_serving_size=1,eval_save_prefix=outdir)
    model_trainer.train(num_itr=num_itr+1, display_itr=num_itr, eval_datasets=[test_dataset], evaluators=[test_evaluator], num_negatives=200)


if __name__ == "__main__":
    print (sys.argv)
    dataset = sys.argv[1] 
    model_name = sys.argv[2]
    user_per = float(sys.argv[3])
    keep_days = int(sys.argv[4])
    l2_reg = float(sys.argv[5])
    num_itr = int(sys.argv[6])
    hist_len = sys.argv[7]
    test_date = sys.argv[8] 
    evaluator = sys.argv[9]
    
    raw_data = load_data(dataset, hist_len)
    # logging
    outdir = None
    if LOGGING:
        outdir = "{}_{}_logs/{}_{}_{}_{}_{}/{}/".format(dataset, LOG_TYPE, model_name, evaluator, user_per, keep_days, hist_len, test_date)
        os.popen("mkdir -p %s" % outdir).read()
        log = open(outdir+ "training.log", "w")
        sys.stdout = log
    
    parameters = {"model": model_name, 'evaluator': evaluator, "l2_reg": l2_reg, "num_itr":num_itr, "user_per":user_per, "keep_days":keep_days, "dataset": dataset}
    print (parameters) 

    run_test_exp(model_name=model_name, evaluator=evaluator, raw_data=raw_data, user_per=user_per, keep_days=keep_days, l2_reg=l2_reg, test_date=test_date, outdir=outdir, num_itr=num_itr)
