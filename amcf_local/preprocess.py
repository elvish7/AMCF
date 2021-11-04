import pandas as pd
import os
from sklearn import preprocessing

# cfda4
def convert_data(source_dir, target_dir):
    """
    convert original dataset to AMCF format
    """
    # set input and output directories
    source_data = source_dir + '/w103.csv'
    source_fund = source_dir + '/w106.csv'
    target_data = target_dir + '/ratings'
    target_fund = target_dir + '/funds'

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    # from source data to target data
    w103 = pd.read_csv(source_data, index_col=0, 
                        names=['cust_no', 'wm_prod_code', 'txn_dt', 'txn_amt', 'dta_src', 'deduct_cnt', 'etl_dt'])

    total_amt = w103.groupby('cust_no').sum()['txn_amt'].to_dict()
    ratings = w103[['cust_no', 'wm_prod_code', 'txn_amt', 'txn_dt']].dropna()
    ratings['txn_dt'] = pd.to_datetime(ratings['txn_dt'], format="%Y-%m-%d")
    # 計算交易額占比
    ratings['txn_amt'] = [int((amt/total_amt[i])*10)+1 for i, amt in zip(ratings['cust_no'], ratings['txn_amt'])]

    # encode to index
    le1 = preprocessing.LabelEncoder()
    ratings['cust_no'] = le1.fit_transform(ratings['cust_no'])
    user_dict = dict(zip(le1.classes_, le1.transform(le1.classes_)))

    le2 = preprocessing.LabelEncoder()
    ratings['wm_prod_code'] = le2.fit_transform(ratings['wm_prod_code'])
    fund_dict = dict(zip(le2.classes_, le2.transform(le2.classes_)))

    ratings.rename({'cust_no':'uid', 'wm_prod_code':'fid', 'txn_amt':'rating', 'txn_dt':'timestamp'}, axis=1, inplace=True)
    ratings = ratings.sort_values(by=['uid'], axis=0).reset_index(drop=True)
    ratings.to_csv(target_data, index=False)

    # from source fund to target fund
    w106 = pd.read_csv('/tmp2/jeding/Esun_fund_data/1000000筆/witwo106data.csv', index_col=0)[['invest_type', 'wm_prod_code']]
    w106['wm_prod_code'] = [fund_dict[i] if i in fund_dict else 'N' for i in w106['wm_prod_code']]
    fund = w106[w106['wm_prod_code'] != 'N']   
    fund = fund.join(pd.get_dummies(fund.invest_type)).drop('invest_type', axis=1)
    fund.rename({'wm_prod_code':'fid'}, axis=1, inplace=True)
    fund = fund.sort_values(by=['fid'], axis=0).reset_index(drop=True)
    fund.to_csv(target_fund, index=False)
    

if __name__ == "__main__":
    convert_data('/tmp2/cytsao/esun_fund/edu_framework/data/CF_table_0912', 'data_amcf')
