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
parser.add_argument("--epoch", default=20, type=int, help="epoch num")
parser.add_argument("--user_ft", help="Use user features", action='store_true')
parser.add_argument("--item_ft", help="Use item features", action='store_true')
args = parser.parse_args()
today = args.date
duration = args.eval_duration
dim = args.dim
epoch = args.epoch

## Load data
print("Loading Data...")
w103_df = pd.read_csv('data/w_103_' + args.date + '.csv')
w106_df = pd.read_csv('data/w106_df_filter_' + args.date + '.csv')

## Intersection of w103 & w106 wrt wm_prod_code
_filter = w106_df.wm_prod_code.isin(w103_df['wm_prod_code'].tolist())
w106_df_filter = w106_df[_filter]
_selected_col = ['wm_prod_code','invest_type']
w106_df_filter = w106_df_filter[_selected_col]

## data preprocess
ratings, fund, user_n, item_n = convert_data(w103_df, w106_df_filter)

## training
# model = model_training(user_n, item_n, ratings, fund)

model = torch.load('AMCF_model.pt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

## predict/recommend
# all the users & items
user_list, item_list= ratings['uid'].unique().tolist(), fund['fid'].unique().tolist()
all_ratings = predict_rate(user_list, item_list, model, fund)
print(all_ratings.shape, all_ratings)

## evaluation
evaluation = Evaluation('', evaluation_path, pred)
score = evaluation.results()
#print(f'Mean Precision: {score}\n')
print("===",score)