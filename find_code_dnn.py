while(1):
    cudidx=input('Enter cuda device index: ')   
    sure=input('Confirm device cuda:{} ? '.format(cudidx))
    if(sure=='y'):
        break     
import numpy as np
import copy
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from code_construct import Hardmard_code, OneHot_code


from models_construct import MLP

device="cuda:{}".format(cudidx)

class_num = 200
code_len = 2048

lam_list = [0, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1]
opt_code_list = []

enc = nn.Sequential(
    nn.Linear(class_num, 4096),
    torch.nn.BatchNorm1d(4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    torch.nn.BatchNorm1d(4096),
    nn.ReLU(),
    nn.Linear(4096, code_len),
    nn.Tanh()
    ).train().to(device)

num_epochs = 30000
test_num = 50



for i in range(len(lam_list)):
    dist_min_best = 0
    loss_best = 10000
    for j in range(test_num):
        enc = nn.Sequential(
            nn.Linear(class_num, 4096),
            torch.nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            torch.nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, code_len),
            nn.Tanh()
            ).train().to(device)
        lam = lam_list[i]
        optimizer = torch.optim.SGD(enc.parameters(), lr = 0.001, momentum=0.9)
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        for ep in range(num_epochs):
            onehot_inputs = torch.eye(class_num).to(device)
            enc_outputs = enc(onehot_inputs)
            optimizer.zero_grad()
            cov = enc_outputs@enc_outputs.T
            loss_orth = torch.sum(torch.triu(cov, diagonal=1)**2)/code_len/class_num/(class_num-1) - 0.01*torch.sum((torch.diag(cov))**2)/class_num/code_len  
            loss = -torch.sum(F.pdist(enc_outputs, p=2))/class_num/(class_num-1)/code_len + lam*loss_orth
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            codes = torch.sign(enc_outputs)
            dist_min = torch.min(F.pdist(codes, p=2)**2)
            dist_mean = torch.mean(F.pdist(codes, p=2)**2)
            cov_curr = codes @ codes.T
            loss_orth_curr = torch.sum(torch.triu(cov_curr, diagonal=1)**2)/code_len/class_num/(class_num-1) - 0.01*torch.sum((torch.diag(cov_curr))**2)/class_num/code_len  
            loss_curr = -torch.sum(F.pdist(codes, p=2))/class_num/(class_num-1)/code_len + lam*loss_orth_curr
            corr = torch.sum(torch.triu(codes@torch.transpose(codes, 1,0), diagonal=1)**2)/code_len/class_num/(class_num-1)
            print("curr | lam: {}, dist_min: {}, dist_mean: {}, corr: {}, loss: {}".format(lam, dist_min.item(), dist_mean.item(), corr.item(), loss_curr.item()))

            # if dist_min_best < dist_min:
            #     codes_best = codes
            #     dist_min_best = dist_min
            #     dist_mean_best = dist_mean
            #     corr_best = corr

            if loss_best > loss_curr:
                codes_best = codes
                dist_min_best = dist_min
                dist_mean_best = dist_mean
                loss_best = loss_curr
                corr_best = corr
            print("best | lam: {}, dist_min: {}, dist_mean: {}, corr: {}, loss: {}".format(lam, dist_min_best.item(), dist_mean_best.item(), corr_best.item(), loss_best.item()))


    opt_code_list.append(codes_best.detach().cpu().numpy())


np.save('./optcodes_dnn_best50_{}class_codelen{}'.format(class_num, code_len), opt_code_list)
