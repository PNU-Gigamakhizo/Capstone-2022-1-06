from pydoc import classname
import torch
from differnet.utils import * #load_datasets, make_dataloaders
import differnet.config as c
from differnet.localization import *
from differnet.model import DifferNet

def train(train_loader, test_loader):
    #모델 로드
    model = DifferNet()
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    model.to(c.device)
    model.eval()
    
    export_gradient_maps(model, test_loader, optimizer, -1)
