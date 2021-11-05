import os
import argparse
import pandas as pd
import numpy as np
from evaluation import Evaluation
from preprocess import convert_data 
from train import *
from recommend import predict_rate

## Add params
parser = argparse.ArgumentParser()
parser.add_argument("--date", default='2018-12-31', help="Recommendation date")
parser.add_argument("--eval_duration", default='1m', type=str, help="one month or 7 days")
parser.add_argument("--dim", default=128, type=int, help="feature emb. dimensions")
parser.add_argument("--epoch", default=1, type=int, help="epoch num")
parser.add_argument("--user_ft", help="Use user features", action='store_true')
parser.add_argument("--item_ft", help="Use item features", action='store_true')
args = parser.parse_args()
#today = args.date
duration = args.eval_duration
dim = args.dim
epoch = args.epoch

dates = ['2019-06-30', '2019-05-31', '2019-04-30', '2019-03-31', '2019-02-28', '2019-01-31', '2018-12-31']
for d in dates:
    today = d
    ## Load data
    print("Loading Data...")
    w103_df = pd.read_csv('/tmp2/jeding/Esun_online_data/w_103_' + today + '.csv')
    w106_df = pd.read_csv('data/w106_df_filter_' + today + '.csv')

    ## Intersection of w103 & w106 wrt wm_prod_code
    _filter = w106_df.wm_prod_code.isin(w103_df['wm_prod_code'].tolist())
    w106_df_filter = w106_df[_filter]
    _selected_col = ['wm_prod_code','invest_type']
    w106_df_filter = w106_df_filter[_selected_col]

    ## data preprocess
    ratings, fund, user_n, item_n, user_dict, fund_dict = convert_data(w103_df, w106_df_filter)

    ## training
    model = model_training(user_n, item_n, ratings, fund, epoch)
    model.eval()

    # model = torch.load('AMCF_model.pt')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model.eval()

    ## predict/recommend
    # all the users & items
    user_list, item_list= ratings['uid'].unique().tolist(), fund['fid'].unique().tolist()
    print("Start Recommendation...")
    pred = predict_rate(user_list, item_list, model, fund, user_dict, fund_dict)

    ## evaluation
    evaluation_path = '/tmp2/jeding/Esun_online_data/evaluation_data/evaluation_df_' + today + '.csv'
    print("Start Evaluation...")
    evaluation = Evaluation(today, evaluation_path, pred)
    score = evaluation.results()
    print(f'Today: {today} Mean Precision: {score}\n')

    ## Save results
    with open('amcf_results.txt', 'a') as f_out:
        f_out.write(f'{today} {score}\n')