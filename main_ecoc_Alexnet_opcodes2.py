while(1):
    cudidx=input('Enter cuda device index: ')   
    sure=input('Confirm device cuda:{} ? '.format(cudidx))
    if(sure=='y'):
        break
memo = input('memo:')        
import numpy as np
import time
import os
import copy
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dataset_construct import cifar100_dataset, cifar10_dataset, mnist_dataset, cifar10_dataset_alexnet

from models_construct import MLP, VGG16, resnet18, resnet50
from models_noBN import resnet18_noBN, resnet50_noBN, VGG16_noBN, AlexNet
from utils import date_and_time, eval_on_dataset_ecoc, apply_noise_by_layer_gaussian_fixed, apply_noise_by_layer_gaussian_classifier_alexnet, apply_noise_by_layer_gaussian_total_norm, apply_noise_by_layer_lognormal, hinge_loss
from code_construct import Hardmard_code, OneHot_code, Hardmard_code_list



# loss_type = 'mse'
loss_type = 'bce'
# loss_type = 'hinge'

task_name = 'Alexnet_optcodes/'
file_dir = './printfiles/' + task_name + loss_type + '/'
if not os.path.exists(file_dir):
    os.makedirs(file_dir)

curr_time = date_and_time()
file = open(file_dir+curr_time+'.txt', 'w')
file.write(memo+'\n')


aug_transform = [transforms.RandomResizedCrop(64, [0.2,1.0]), transforms.RandomHorizontalFlip()]
batch_size=512
num_epochs = 100
device="cuda:{}".format(cudidx)

if loss_type == 'mse':
    act = None
    is_bin = True
    criterion=F.mse_loss

elif loss_type == 'bce':
    act = torch.sigmoid
    is_bin = True
    criterion=F.binary_cross_entropy_with_logits
else:
    act = None
    is_bin = False
    criterion=hinge_loss


train_dataset, train_dataset_aug, test_dataset = cifar10_dataset_alexnet(aug_transform)
# train_dataset, train_dataset_aug, test_dataset = mnist_dataset(aug_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
train_loader_aug = DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader= DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

opt_codes = np.load('./optcodes_dnn_best50_10class_codelen16.npy', allow_pickle=True)
lam_list = [0, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1]
# lam_list = [0, 0.002, 0.005, 0.008, 0.1]
# lam_list = [1, 3, 5, 10]
par_list_gaussian = [2.5, 3, 3.5, 4] # normal
par_list_gaussian = [0.01, 0.02, 0.03, 0.04] # normal

test_num = 20

acc_lam_list_gaussian = np.zeros([len(lam_list), test_num, len(par_list_gaussian)+1])


for p in range(len(opt_codes)):
    for q in range(test_num):
        lam = lam_list[p]
        codebook = torch.tensor((opt_codes[p]+1)/2).to(device)
        code_len = codebook.size()[1]
        model = AlexNet(code_len).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.2, momentum=0.9)
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)


        model_dir = './saved_models/'+task_name + loss_type + '/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for epoc in range(num_epochs):    
            model.train()                            
            for batch_idx, (img, lab) in enumerate(train_loader_aug):
                
                img, lab = img.to(device), lab.to(device)
                tar = codebook[lab]
                optimizer.zero_grad()
                pred = model(img)
                loss = criterion(pred, tar)                             
                loss.backward()
                optimizer.step()

            scheduler.step()
            test_acc = eval_on_dataset_ecoc(model, device, codebook, act = act, testloader = test_loader)

        print("lam: {}, q: {}, epoch: {}, test_acc: {}".format(lam, q, epoc, test_acc))
        file.write("lam: {}, q: {}, epoch: {}, test_acc: {}\n".format(lam, q, epoc, test_acc))

        torch.save(model.state_dict(), model_dir+ 'lam_{}_'.format(lam)+curr_time)

        clean_acc =  test_acc

     
        acc_list_gaussian = [clean_acc]
        num_test = 200
        model_noisy = copy.deepcopy(model).eval()
        for i in range(len(par_list_gaussian)):
            tmp_acc_list_gaussian = []
            for j in range(num_test):
                model_noisy = copy.deepcopy(model).eval()
                model_noisy = apply_noise_by_layer_gaussian_fixed(model_noisy, par_list_gaussian[i], device)
                test_acc = eval_on_dataset_ecoc(model_noisy, device, codebook, act = act, testloader = test_loader)
                tmp_acc_list_gaussian.append(test_acc)

            tmp_acc = np.mean(tmp_acc_list_gaussian)
            acc_list_gaussian.append(tmp_acc)

            # print('par: {}, acc: {}'.format(par_list[i], tmp_acc))
            # file.write('par: {}, acc: {}\n'.format(par_list[i], tmp_acc))

        acc_drop_gaussian = clean_acc - np.array(acc_list_gaussian)
        print('Gaussian acc list: {}'.format(acc_list_gaussian))
        file.write('Gaussian acc_list: {} \n'.format(acc_list_gaussian))
        print('Gaussian acc drop: {}'.format(acc_drop_gaussian))
        file.write('Gaussian acc drop: {} \n'.format(acc_drop_gaussian))



        acc_lam_list_gaussian[p][q] = np.array(acc_list_gaussian)



npy_dir = './npy_files/' + task_name + loss_type + '/'
if not os.path.exists(npy_dir):
    os.makedirs(npy_dir)


np.save(npy_dir+'acc_list_gaussian_'+curr_time, acc_lam_list_gaussian)
