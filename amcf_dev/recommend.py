import torch
import numpy as np
from tqdm import tqdm 
from utils_amcf import item_to_genre

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict_rate(user_list, item_list, model, data_fund, top_n=5):
    # user_n x item_n
    for u in tqdm(user_list):
        # pred = {}
        users = torch.tensor([u]*len(item_list), dtype=torch.long).to(device)
        items = torch.tensor(item_list, dtype=torch.long).to(device)

        # get genre information from item id
        item_asp = item_to_genre(items.cpu(), data_fund).values
        item_asp = torch.Tensor(item_asp).to(device)

        outputs, cos_sim, pref = model(users, items, item_asp) # ratings, _, _
        user_rate = np.array([outputs.cpu().detach().numpy()])

        # top n funds_id
        top_n_fund = user_rate[0].argsort()[-top_n:]
        if u == user_list[0]:
            all_topn_fundid = np.array([top_n_fund])
        else:
            all_topn_fundid = np.append(all_topn_fundid, np.array([top_n_fund]), axis=0)

    return all_topn_fundid