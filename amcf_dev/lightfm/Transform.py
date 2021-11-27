import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import relativedelta

class Transform:
    def __init__(self, df, end_dt, decay_factor, decay_window):
        self.end = datetime(*[int(item) for item in end_dt.split('-')])
        self.factor = decay_factor
        self.window = decay_window
        self.w103 = df

    def diff_months(self, start, end):
        return (end.year - start.year) * 12 + (end.month  - start.month )
    
    def scale(self, start_dt):
        diff = self.diff_months(start_dt, self.end)
        return self.factor ** (diff//self.window)
    
    def weight_amount(self, txn_dt, txn_amt, deduct_cnt):
        return pd.DataFrame([amt*self.scale(dt)*cnt if cnt > 0 else amt*self.scale(dt)*1 for dt, amt, cnt in zip(txn_dt, txn_amt, deduct_cnt)])
     
    def transformation(self):
        w103_df = self.w103
        w103_df = w103_df.assign(weighted_txn_amt = lambda x: self.weight_amount(pd.to_datetime(x['txn_dt']), x['txn_amt'],x['deduct_cnt']))
        #w103_df = w103_df.groupby(["cust_no","wm_prod_code"]).apply(lambda x: pd.Series({'weighted_amt':x["weighted_txn_amt"].sum()})).reset_index()
        #w103_df = w103_df.groupby(["cust_no","wm_prod_code"]).first().reset_index()
        return w103_df
