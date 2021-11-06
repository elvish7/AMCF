import pandas as pd
import os
import numpy as np
from sklearn import preprocessing

# cfda4
def convert_data(w103, w106):
    """
    convert original dataset to AMCF format
    Input: 
        w103_df, w106_df
    Output:
        rating, item aspect features(fund)
    """

    total_amt = w103.groupby('cust_no').sum()['txn_amt'].to_dict()
    ratings = w103[['cust_no', 'wm_prod_code', 'txn_amt', 'txn_dt']].dropna()
    ratings['txn_dt'] = pd.to_datetime(ratings['txn_dt'], format="%Y-%m-%d")
    # 計算交易額占比
    # ratings['txn_amt'] = [int((amt/total_amt[i])*10)+1 for i, amt in zip(ratings['cust_no'], ratings['txn_amt'])]
    ratings['txn_amt'] = [amt/total_amt[i] for i, amt in zip(ratings['cust_no'], ratings['txn_amt'])]
    ratings['txn_amt'] =pd.cut(ratings.txn_amt, bins=5, labels=np.arange(5), right=False).astype(int)+1

    # encode to index
    le1 = preprocessing.LabelEncoder()
    ratings['cust_no'] = le1.fit_transform(ratings['cust_no'])
    user_dict = dict(zip(le1.transform(le1.classes_), le1.classes_))

    le2 = preprocessing.LabelEncoder()
    ratings['wm_prod_code'] = le2.fit_transform(ratings['wm_prod_code'])
    fund_dict = dict(zip(le2.transform(le2.classes_), le2.classes_))
    fund_label_id = dict(zip(le2.classes_, le2.transform(le2.classes_)))

    ratings.rename({'cust_no':'uid', 'wm_prod_code':'fid', 'txn_amt':'rating', 'txn_dt':'timestamp'}, axis=1, inplace=True)
    ratings = ratings.sort_values(by=['uid'], axis=0).reset_index(drop=True)

    fund = w106.join(pd.get_dummies(w106.invest_type)).drop('invest_type', axis=1)
    fund['wm_prod_code'] = [fund_label_id[i] for i in fund['wm_prod_code']]
    fund.rename({'wm_prod_code':'fid'}, axis=1, inplace=True)
    fund = fund.sort_values(by=['fid'], axis=0).reset_index(drop=True)
    
    user_n, item_n = len(user_dict), len(fund_dict)
    
    return ratings, fund, user_n, item_n, user_dict, fund_dict
