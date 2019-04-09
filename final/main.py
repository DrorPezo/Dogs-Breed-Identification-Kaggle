
"""
# For SIPL lab computers add:

from os.path import dirname
import sys
sys.path.append(dirname("/opt/anaconda/lib/python3.6/site-packages/"))
"""

from finetuning import FineTuning
import torch.optim as optim

ft = FineTuning('~/Documents/data/', 'resnet', 120, 128, 'sort/')
params = ft.__get_params_to_update__()
optimizer = optim.Adam(params, 1e-3)
ft.__set_optim__(optimizer)
ft.train(15)
ft.kaggle_csv('test1/', 'results.csv')
ft.save_model('/home/dore/Downloads/dog-breed-identification/')
