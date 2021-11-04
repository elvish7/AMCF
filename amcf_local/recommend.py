import torch
from utils import get_data, item_to_genre
import numpy as np

# load model
model = torch.load('AMCF_model.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

user_num = 10
item_num = 1000

all_ratings = np.empty((user_num,item_num), int)
for i in range(user_num):
    users = torch.tensor([i]*item_num, dtype=torch.long).to(device)
    items = torch.tensor(list(range(item_num)), dtype=torch.long).to(device)

    # get genre information from item id
    item_asp = item_to_genre(items.cpu()).values
    item_asp = torch.Tensor(item_asp).to(device)

    outputs, cos_sim, pref = model(users, items, item_asp) # ratings, _, _
    user_rate = np.array([outputs.cpu().detach().numpy()])
    all_ratings = np.append(all_ratings, user_rate, axis=0)

print(all_ratings.shape)