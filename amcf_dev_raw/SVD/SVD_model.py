import numpy as np
import pandas as pd
import torch

w103_df = pd.read_csv('/tmp2/jeding/Esun_online_data/w_103_2019-06-30.csv')

total_amt = w103_df.groupby('cust_no').sum()['txn_amt'].to_dict()
ratings = w103_df[['cust_no', 'wm_prod_code', 'txn_amt', 'txn_dt']].dropna()
ratings['txn_dt'] = pd.to_datetime(ratings['txn_dt'], format="%Y-%m-%d").astype(int) / 10**9
# deal with duplicate funds brought by the same user
ratings = ratings.groupby(['cust_no', 'wm_prod_code'], as_index=False).agg({'txn_amt': 'sum', 'txn_dt': 'mean'})
# 計算交易額占比
ratings['txn_amt'] = [amt/total_amt[i] for i, amt in zip(ratings['cust_no'], ratings['txn_amt'])]
ratings['txn_amt'] = pd.cut(ratings.txn_amt, bins=11, labels=np.arange(1, 12), right=False).astype(int)  # best

print(ratings.head())
data_train = ratings.pivot(index = 'cust_no', columns= 'wm_prod_code', values = 'txn_amt')
print(data_train)
matrax = data_train.fillna(0)
matrix = np.array(matrax)
data = torch.tensor(matrix)
u, s, v = torch.svd(data)

s_topk = torch.topk(s, 100)
idx_list = s_topk[1].tolist()

u_topk = torch.index_select(u, 1, s_topk[1].squeeze())
v_topk = torch.index_select(v, 1, s_topk[1].squeeze())

matrix_data = torch.mm(u_topk, v_topk.t())
print(matrix_data[0][1])
