import os
import argparse
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from local_evaluation import Evaluation
from utils import recommendation_all, create_all_feature_pairs, build_feature_tuples
from Transform import Transform
import pickle

## Add param
parser = argparse.ArgumentParser()
parser.add_argument("--train", help="training data path")
parser.add_argument("--evaluation", help="evaluation data path")
parser.add_argument("--item_ft", default='', type=str, help="w106 item static feature data path")
parser.add_argument("--user_ft", default='', type=str, help="cm_customer_m user static feature data path")
parser.add_argument("--dim", default=128, type=int, help="feature emb. dimensions")
parser.add_argument("--epoch", default=20, type=int, help="epoch num")
parser.add_argument("--decay_factor", default=0.9, type=float, help="decay rate")
parser.add_argument("--decay_window", default=1, type=int, help="decay window size")
parser.add_argument("--date", default='2019-06-30', type=str, help="recommendation date")
args = parser.parse_args()
dim = args.dim
epoch = args.epoch
end_dt, decay_factor, decay_window = args.date, args.decay_factor, args.decay_window
## Get data path
# path = '/tmp2/ctyeh/fund-recommendation/lightfm/data'# os.getcwd()
path = '/home/jeding/Esun_data_11_19_train_1M/'

train_path =os.path.join(path, args.train)
# evaluation_path =os.path.join(path, args.evaluation)
evaluation_path =os.path.join('/tmp2/jeding/Esun_online_data/evaluation_data/', args.evaluation)
if args.item_ft != '':
    item_feature_path =os.path.join(path, 'data', args.item_ft)
if args.user_ft != '':
    user_feature_path =os.path.join(path, 'data', args.user_ft)
## Read data
w103_df = pd.read_csv(train_path)
if args.item_ft != '':
    w106_df = pd.read_csv(item_feature_path)
if args.user_ft != '':
    cm_customer_m_df = pd.read_csv(user_feature_path)
## Transform data
#transform = Transform(w103_df, end_dt, decay_factor, decay_window)
#w103_df = transform.transformation()
## Intersection of w103 & w106 wrt wm_prod_code
if args.item_ft != '':
    _filter = w106_df.wm_prod_code.isin(w103_df['wm_prod_code'].tolist())
    w106_df_filter = w106_df[_filter]
    _selected_col = ['wm_prod_code','mkt_rbot_ctg_ic','prod_detail_type_code','prod_ccy','prod_risk_code','can_rcmd_ind']
    w106_df_filter = w106_df_filter[_selected_col]
## Intersection of w103 & cm_customer_m wrt cust_no
if args.user_ft != '':
    _filter = cm_customer_m_df.cust_no.isin(w103_df['cust_no'].tolist())
    cust_df_filter = cm_customer_m_df[_filter]
    _selected_col = ['cust_no', 'etl_dt', 'age', 'gender_code', 'cust_vintage', 'income_range_code']
    cust_df_selected = cust_df_filter[_selected_col].groupby('cust_no').tail(1)
## Create features
user_fts, item_fts = None, None
if args.user_ft != '':
    user_fts = create_all_feature_pairs(cust_df_selected)
if args.item_ft != '':
    item_fts = create_all_feature_pairs(w106_df_filter)
## Fit the dataset
dataset1 = Dataset()
dataset1.fit(
        w103_df['cust_no'].unique(), # all the users
        w103_df['wm_prod_code'].unique(), # all the items
        user_features = user_fts,
        item_features = item_fts
)
## Build features
user_features, item_features = None, None
if args.user_ft != '':
    user_tuples = build_feature_tuples(cust_df_selected)
    user_features = dataset1.build_user_features(user_tuples, normalize= False)
if args.item_ft != '':
    item_tuples = build_feature_tuples(w106_df_filter)
    item_features = dataset1.build_item_features(item_tuples, normalize= False)
## Build interactions
# (interactions, weights) = dataset1.build_interactions([(x[1], x[2], x[8]) for x in w103_df.values ])
(interactions, weights) = dataset1.build_interactions([(x[1], x[2], x[4]) for x in w103_df.values ])
## Get Id mappings 
user_id_map, user_feature_map, item_id_map, item_feature_map = dataset1.mapping()
pickle.dump([user_id_map, item_id_map], open('latent_representations_1m/'+ end_dt +'_id_map.pkl', 'wb'))

## Train model
model = LightFM(no_components=dim, loss='warp')
model.fit(interactions, # spase matrix representing whether user u and item i interacted
      user_features= user_features, # we have built the sparse matrix above
      item_features= item_features, # we have built the sparse matrix above
      sample_weight= weights, # spase matrix representing how much value to give to user u and item i inetraction: i.e ratings
      epochs=epoch)

## Get item & user embedding
pickle.dump(model.get_item_representations(), open('latent_representations_1m/'+ end_dt +'_item_latents.pkl', 'wb'))
pickle.dump(model.get_user_representations(), open('latent_representations_1m/'+ end_dt +'_user_latents.pkl', 'wb'))

## Get user list for predictions
user_list = w103_df['cust_no'].unique().tolist()
## Recommend each user's top5 funds
pred = recommendation_all(model, interactions, user_list, user_id_map, item_id_map, user_features, item_features)
## Evaluate each user's precision@5
evaluation = Evaluation('', evaluation_path, pred)
score = evaluation.results()
print(f'Today: {end_dt} Mean Precision: {score}\n')
