import pickle
dates = ['2019-06-30', '2019-05-31', '2019-04-30', '2019-03-31', '2019-02-28', '2019-01-31', '2018-12-31']
for d in dates:
    item_repts = pickle.load(open('latent_representations/'+ d +'_item_latents.pkl', 'rb'))
    user_repts = pickle.load(open('latent_representations/'+ d +'_user_latents.pkl', 'rb'))
    print(item_repts[1].shape, user_repts[1].shape)