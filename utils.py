import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import time
from datetime import datetime

def date_and_time():
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)
    readable_time = dt_object.strftime('%m-%d-%Y-%H:%M:%S')
    return readable_time

def decode_batch(batch, codebook):
    # Expand dimensions for broadcasting

    if len(batch.size()) == 1:
        batch = batch.unsqueeze(0)
    batch_expanded = batch.unsqueeze(1)  # Shape: [batch_size, 1, code_length]
    codebook_expanded = codebook.unsqueeze(0)  # Shape: [1, num_codewords, code_length]

    # Calculate Euclidean distances
    distances = torch.norm(batch_expanded - codebook_expanded, dim=2)  # Shape: [batch_size, num_codewords]

    # Find the indices of the closest codewords
    closest_indices = torch.argmin(distances, dim=1)

    return closest_indices

@torch.no_grad()
def eval_on_dataset(model, device, criterion=torch.nn.CrossEntropyLoss(), test_dataset=None, testloader=None):

    criterion=criterion.to(device)
    model1=model.to(device).eval()

    if (test_dataset==None and testloader==None):
        raise ValueError("Need either dataset or loader")

    total, correct = 0.0, 0.0
    loss0=0

    if (testloader == None):
        batch_size=128
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        
        if (batch_idx==0):
            batch_size=len(labels)

        outputs= model1(images)
        loss = criterion(outputs, labels)
        loss0 += loss.item()*len(labels)/batch_size/len(testloader)
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss0



@torch.no_grad()
def eval_on_dataset_ecoc(model, device, codebook, act = None, test_dataset=None, testloader=None):

    if act == None:
        act = torch.nn.Identity().to(device)
    codebook = codebook.to(device)

    if (test_dataset==None and testloader==None):
        raise ValueError("Need either dataset or loader")
    
    model = model.to(device).eval()
    total, correct = 0.0, 0.0

    if (testloader == None):
        batch_size=128
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        
        if (batch_idx==0):
            batch_size=len(labels)

        outputs= act(model(images))
        pred_labels = decode_batch(outputs, codebook)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy


@torch.no_grad()
def flip_bits(model, flip_prob, device):

    for name, tensor in model.named_parameters():
        dtype = tensor.dtype
        # Convert tensor to byte representation
        tensor = tensor.to(torch.uint8)
        # Calculate total number of bits
        num_bits = tensor.numel() * 8
        # Calculate expected number of flipped bits
        num_flips = int(num_bits * flip_prob)
        # Create mask for bits to flip
        mask = torch.zeros_like(tensor, dtype=torch.uint8).to(device)
        # Randomly choose which bits to flip
        indices = torch.randint(0, num_bits, (num_flips,)).to(device)
        # Convert bit indices to byte and bit index
        byte_indices = indices // 8
        bit_indices = indices % 8
        update_values = (1 << bit_indices).to(torch.uint8)
        # Set the corresponding bits in mask
        mask.view(-1).scatter_add_(0, byte_indices, update_values)
        # XOR to flip bits
        tensor ^= mask
        # Convert back to original data type
        tensor.to(dtype)

    return model


@torch.no_grad()
def apply_noise_by_layer_gaussian(model, par, device):

    par = torch.tensor(par).to(device)
    for name, param in model.named_parameters():
        num = param.numel()
        eng = torch.sum(param**2)/num
        # eng = torch.max(param)*0.6
        noise = par*torch.sqrt(eng)*torch.randn(param.size()).to(device)
        param.add_(noise)
    return model

@torch.no_grad()
def apply_noise_by_layer_gaussian_fixed(model, par, device):

    par = torch.tensor(par).to(device)
    for name, param in model.named_parameters():
        num = param.numel()
        # eng = torch.max(param)*0.6
        noise = par*torch.randn(param.size()).to(device)
        param.add_(noise)
    return model

@torch.no_grad()
def apply_noise_by_layer_gaussian_remove_small_value(model, par, device):

    par = torch.tensor(par).to(device)
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            # print(torch.sum(param.data.abs() < 1 * param.data.abs().mean()) / param.numel())
            param.data[param.data.abs() < 1 * param.data.abs().mean()] = 0
        num = param.numel()
        eng = torch.sum(param**2)/num
        # eng = torch.max(param)*0.6
        noise = par*torch.sqrt(eng)*torch.randn(param.size()).to(device)
        if 'classifier' not in name:
            noise[param.data == 0] = 0
            # noise *= 0.5
        print(torch.count_nonzero(noise) / noise.numel())
        param.add_(noise)
        
    return model

@torch.no_grad()
def apply_noise_by_layer_gaussian_total_norm(model, par, device):

    par = torch.tensor(par).to(device)
    total_num = 0
    total_eng = 0
    for name, param in model.named_parameters():
        if 'bias' in name:
            continue
        num = param.numel()
        eng = torch.sum(param ** 2)
        total_eng += eng
        total_num += num
    eng = total_eng / total_num

    for name, param in model.named_parameters():
        if 'bias' in name:
            continue
        noise = par*torch.sqrt(eng)*torch.randn(param.size()).to(device)
        param.add_(noise)
    return model

@torch.no_grad()
def apply_noise_by_layer_gaussian_classifier(model, par, device):

    par = torch.tensor(par).to(device)
    for name, param in model.named_parameters():
        if 'bias' in name or 'fc' not in name:
            continue
        num = param.numel()
        eng = torch.sum(param**2)/num
        # eng = torch.max(param)*0.6
        noise = par*torch.sqrt(eng)*torch.randn(param.size()).to(device)
        param.add_(noise)
    return model

@torch.no_grad()
def apply_noise_by_layer_gaussian_classifier_alexnet(model, par, device):

    par = torch.tensor(par).to(device)
    for name, param in model.named_parameters():
        if 'bias' in name or 'classifier' not in name:
            continue
        num = param.numel()
        eng = torch.sum(param**2)/num
        # eng = torch.max(param)*0.6
        noise = par*torch.sqrt(eng)*torch.randn(param.size()).to(device)
        param.add_(noise)
    return model

@torch.no_grad()
def apply_noise_by_layer_lognormal(model, par, device):
    par = torch.tensor(par).to(device)
    for name, param in model.named_parameters():
        noise = torch.exp(torch.sqrt(par)*torch.randn(param.size()).to(device))
        param.mul_(noise)
    return model


def hinge_loss(outputs, targets):
    hinge_loss_val = 1 - targets * outputs
    hinge_loss_val = torch.clamp(hinge_loss_val, min=0)
    return hinge_loss_val.mean()





