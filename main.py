import os
import argparse
import pandas as pd
import numpy as np
from evaluation import Evaluation
from mlaas_tools.config_build import config_set
from db_connection.utils import get_conn
from utils import recommendation_all, load_w103, load_w106, load_cust_pop, create_all_feature_pairs, build_feature_tuples
from preprocess import convert_data 
from train import *

## Configure env
if not os.path.isfile('config.ini'):
    config_set()
## Add params
parser = argparse.ArgumentParser()
parser.add_argument("--date", default='2019-06-30', help="Recommendation date")
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

## Load db connection
rawdata_conn = get_conn('edu')
## Load data
print("Loading Data...")
w103_df = load_w103(today, rawdata_conn)
w106_df = load_w106(rawdata_conn)

## Intersection of w103 & w106 wrt wm_prod_code
_filter = w106_df.wm_prod_code.isin(w103_df['wm_prod_code'].tolist())
w106_df_filter = w106_df[_filter]
_selected_col = ['wm_prod_code','invest_type']
w106_df_filter = w106_df_filter[_selected_col]

## data preprocess
ratings, fund, user_n, item_n = convert_data(w103_df, w106_df_filter)

## training
model = model_training(user_n, item_n, ratings, fund)
print(model.state_dict())