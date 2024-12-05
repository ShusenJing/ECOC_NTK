while(1):
    cudidx=input('Enter cuda device index: ')   
    sure=input('Confirm device cuda:{} ? '.format(cudidx))
    if(sure=='y'):
        break
memo = input('memo:')   
        
import numpy as np
import os
import copy
import importlib

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from scipy.linalg import hadamard
from scipy.spatial import distance


# importlib.reload(ipynb.fs.full.models)
# importlib.reload(ipynb.fs.full.sampling)
# importlib.reload(ipynb.fs.full.InfoNCE_loss)
# importlib.reload(ipynb.fs.full.cust_func)

from models_construct import MLP, VGG16, resnet18, resnet50
from utils import date_and_time, eval_on_dataset_ecoc, apply_noise_by_layer_gaussian_fixed
from code_construct import Hardmard_code, OneHot_code, Hardmard_code_list

# torch.cuda.empty_cache()
class MyDataset(Dataset):
    def __init__(self, data, transform=None):

        self.data = data
        self.transform=transform


    def __getitem__(self, index):
        x, y = self.data[index]
        
        if(self.transform==None):
            
            return torch.tensor(x), torch.tensor(y)
        else:
            
            return self.transform(x), torch.tensor(y)


    def __len__(self):
        return len(self.data)

def tiny_image_net_dataset(aug_transform=None):
    
    
    mean=[0.4914,0.4822,0.4465]
    std=[0.2023,0.1994,0.2010]
    
    normalize_transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    transform=transforms.Compose([aug_transform, normalize_transform])
    
    
    train_data=np.load('./data/tiny-imagenet-200/train_data.npy',allow_pickle=True)
    test_data=np.load('./data/tiny-imagenet-200/test_data.npy',allow_pickle=True)
    
    train_datasets_final=MyDataset(train_data, transform)
    val_datasets_final=MyDataset(test_data, normalize_transform)
     
    return train_datasets_final, val_datasets_final

class ECOCDataset(Dataset):
    def __init__(self, data, codebook, transform=None):

        self.data = data
        self.transform=transform
        self.codebook=codebook
        
    def __getitem__(self, index):
        x, y = self.data[index]
        if(self.transform==None):
            return torch.tensor(x), torch.tensor(y), torch.tensor(self.codebook[y])
        else:      
            return self.transform(x), torch.tensor(y), torch.tensor(self.codebook[y])

    def __len__(self):
        return len(self.data)



loss_type = 'bce'   

task_name = 'resnet50_hadamard2/'
file_dir = './printfiles/' + task_name + loss_type + '/'
if not os.path.exists(file_dir):
    os.makedirs(file_dir)

curr_time = date_and_time()
file = open(file_dir+curr_time+'.txt', 'w')
file.write(memo+'\n')

num_class=200  


aug_transform=transforms.Compose([transforms.RandomCrop(64, padding=4), 
                                  transforms.RandomAffine(0,[0.1,0.1]),
                                  transforms.RandomHorizontalFlip()])

train_datasets, test_datasets =  tiny_image_net_dataset(aug_transform)

batch_size=64
num_epochs=150
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

train_loader=DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader=DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=8)

opt_codes = Hardmard_code_list(num_class=num_class, code_len=2048, is_sigmoid=True)[-1]
par_list = [1, 2, 3, 4] # normal
par_list = [0.001, 0.003, 0.005, 0.008] # normal

test_num = 5

acc_lam_list = np.zeros([1, test_num, len(par_list)+1])

for p in range(1):
    for q in range(test_num):
        lam = 0
        # codebook = torch.tensor((opt_codes[p]+1)/2).to(device)
        codebook = torch.tensor(opt_codes).to(device)
        code_len = codebook.size()[1]
        # model = MLP(784, code_len).to(device)
        model = resnet50(dim_output=code_len).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.2)


        model_dir = './saved_models/'+task_name + loss_type + '/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for epoc in range(num_epochs):    
            model.train()                            
            for idx, (images_batch, labels_batch) in enumerate(train_loader):
                
                img, lab = images_batch.to(device), labels_batch.to(device)
                tar = codebook[lab]
                optimizer.zero_grad()
                pred = model(img)
                loss = criterion(pred, tar)                             
                loss.backward()
                optimizer.step()

            scheduler.step()
            test_acc = eval_on_dataset_ecoc(model, device, codebook, act = act, testloader = test_loader)
            # print('epochs: {}, test_acc: {}'.format(epoc,np.round(test_acc, decimals = 3)))

        print("lam: {}, q: {}, epoch: {}, test_acc: {}".format(lam, q, epoc, test_acc))
        file.write("lam: {}, q: {}, epoch: {}, test_acc: {}\n".format(lam, q, epoc, test_acc))

        torch.save(model.state_dict(), model_dir+ 'lam_{}_'.format(lam)+curr_time)

        clean_acc =  test_acc
        # model.load_state_dict(torch.load("/home/shusen/ECOC_NTK/saved_models/resnet50_random/bce/lam_0_04-13-2024-19:26:21"))
        # model.eval()
        # print(model.state_dict())

     
        acc_list = [clean_acc]
        num_test = 200
        model_noisy = copy.deepcopy(model).eval()
        for i in range(len(par_list)):
            tmp_acc_list = []
            for j in range(num_test):
                model_noisy = copy.deepcopy(model).eval()
                model_noisy = apply_noise_by_layer_gaussian_fixed(model_noisy, par_list[i], device)
                test_acc = eval_on_dataset_ecoc(model_noisy, device, codebook, act = act, testloader = test_loader)
                tmp_acc_list.append(test_acc)
            tmp_acc = np.mean(tmp_acc_list)
            acc_list.append(tmp_acc)
            # print('par: {}, acc: {}'.format(par_list[i], tmp_acc))
            # file.write('par: {}, acc: {}\n'.format(par_list[i], tmp_acc))

        acc_drop = clean_acc - np.array(acc_list)
        print('acc list: {}'.format(acc_list))
        file.write('acc_list: {} \n'.format(acc_list))
        print('acc drop: {}'.format(acc_drop))
        file.write('acc drop: {} \n'.format(acc_drop))


        acc_lam_list[p][q] = np.array(acc_list)



npy_dir = './npy_files/' + task_name + loss_type + '/'
if not os.path.exists(npy_dir):
    os.makedirs(npy_dir)


np.save(npy_dir+'acc_list_'+curr_time, acc_lam_list)
