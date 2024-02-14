import torch
import torch.nn as nn
import numpy as np
from model.Unet import Unet
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from dataset.dataset_Unet import dataload
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
from model.Simple_FCN import F_CN
from scipy import ndimage
from model.EfficientNetV2 import efficientnetv2_s
from model.resnet import resnet18, resnet34, resnet50, resnet101
from model.ALEXNET import vgg16
from torch.nn import functional as F
import random
from scipy import ndimage
from scipy import optimize
from scipy.stats import norm
import time




N_D_DATA_TRAIN = 'G:/OneDrive/2022/N2N-1D/data/train'

def add_Gau_peaks(spectrum, a, b, c):#a=10,b=40,c=0.5
    spectrum = normalization(spectrum)
    lenth = len(spectrum)
    y_tot = [0] * lenth
    for g in range(random.randint(1, a)):
        a1 = random.randint(0, lenth)
        b1 = random.randint(1, lenth // b)
        keys_ = range(lenth)
        c1 = random.uniform(0, c)
        gauss = norm(loc=a1, scale=b1)
        y = gauss.pdf(keys_)
        y = normalization(y)
        y = y * c1
        y_tot = y_tot + y
    # y_tot = normalization(y_tot)#考虑是否需要额外归一化，当c=1时
    spectrum = spectrum + y_tot
    spectrum = normalization(spectrum)
    spectrum = np.array(spectrum)
    return spectrum

def read(filename):
    file = open(filename, encoding='utf-8')
    data_lines = file.readlines()
    file.close
    orign_keys = []
    orign_values = []
    for data_line in data_lines:
        pair = data_line.split()
        key = float(pair[0])
        value = float(pair[1])
        orign_keys.append(key)
        orign_values.append(value)
    return orign_keys, orign_values

def write(filename, files, values):
    file = open(filename, 'w')
    for k, v in zip(files, values):
        file.write(str(k) + " " + str(v) + "\n")
    file.close()

def normalization(data):
    #0-1 normalization
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
    ## max=1 normalization
    # data = data/np.max(data)
    # return data

def data_process(spectrum_name):
    global x1
    global x2
    global cycler
    keys, spectrum = read(spectrum_name)
    spectrum_true_raw = spectrum
    x1 = np.mean(spectrum[:len(spectrum) // 2])
    x2 = np.mean(spectrum[len(spectrum) // 2:])
    spectrum = list(spectrum)
    start = spectrum[0]
    end = spectrum[len(spectrum) - 1]
    cycler = 5
    for i in range(cycler):
        spectrum.insert(0, start)
        spectrum.append(end)
    return keys, spectrum, spectrum_true_raw

def train(model, device, optimizer, epoch, spectrum_raw):
    model.train()
    sum_loss = 0
    sum_num = 0
    GS_peak_intensity = 0.45
    for i in range(100):
        sum_num = sum_num+1
        data = add_Gau_peaks(spectrum_raw, 10, 40, GS_peak_intensity)
        data = torch.as_tensor(data, dtype=torch.float32)
        data = data.reshape(1, data.shape[0])
        data = data.reshape(1, data.shape[0], data.shape[1])#data_type: batch_size x embedding_size x text_len
        data = data.permute(1, 0, 2)
        target = add_Gau_peaks(spectrum_raw, 10, 40, GS_peak_intensity)
        target = torch.as_tensor(target, dtype=torch.float32)
        target = target.reshape(1, target.shape[0])
        target = target.reshape(1, target.shape[0], target.shape[1])  # data_type: batch_size x embedding_size x text_len
        target = target.permute(1, 0, 2)
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
    # ave_loss = sum_loss / sum_num
    # print('epoch:{},loss:{}, enhance_num:{}'.format(epoch, ave_loss, sum_num))


def test(model, device, keys, spectrum, epoch, filename):
    global spectrum_true_raw
    global cycler
    model.eval()
    spectrum = Variable(spectrum).to(device)
    output = model(spectrum)
    loss = criterion(output, spectrum)
    print_loss = loss.data.item()
    pl_output = np.array(output.cpu().detach().numpy()[0, 0, :])
    pl_spectrum = np.array(spectrum.cpu().detach().numpy()[0, 0, :])
    for i in range(cycler):
        pl_output = np.delete(pl_output, 0, axis=None)
        pl_output = np.delete(pl_output, len(pl_output) - 1, axis=None)
    for i in range(cycler):
        pl_spectrum = np.delete(pl_spectrum, 0, axis=None)
        pl_spectrum = np.delete(pl_spectrum, len(pl_spectrum) - 1, axis=None)
    spectrum = pl_spectrum/max(pl_spectrum)
    output = pl_output/max(pl_output)
    y1 = np.mean(output[:len(spectrum)//2])
    y2 = np.mean(output[len(spectrum)//2:])
    # if np.std(spectrum) < np.std(output):
    #     print('big noise, it would be denoised by PEER')
    #     peer_results = PEER(spectrum_true_raw, 50)
    #     write(filename + 'raw_nonsignal.txt', keys, spectrum_true_raw)
    #     write(filename + 'byPEER.txt', keys, peer_results)
    # elif y1-y2 == 0:
    #     print('overfitting, it would be denoised by PEER')
    #     peer_results = PEER(spectrum_true_raw, 50)
    #     write(filename + 'raw_nonsignal.txt', keys, spectrum_true_raw)
    #     write(filename + 'byPEER.txt', keys, peer_results)
    # else:
    a = (y1-y2)/(x1-x2)
    b = (x1*y2-x2*y1)/(y1-y2)
    output_corrected = (output/a)-b
    write(filename + '_denoised_corrected.txt', keys, output_corrected)

# Setting Global Parameters
modellr = 1e-3
EPOCHS = 40
BATCH_SIZE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()#loss_function

#Loss
def adjust_learning_rate(optimizer, epoch):
    modellrnew = modellr * (0.1 ** (epoch // 10))
    # print(modellrnew)
    # print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew

# loading data 
top_dir = N_D_DATA_TRAIN
files = os.listdir(top_dir)
for filename in files:
    begin_time = time.perf_counter()
    model = F_CN()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=modellr, weight_decay=1)
    file = os.path.join(top_dir, filename)
    file = file.replace('\\', '/')
    print('Begin:'+file)
    keys, spectrum, spectrum_true_raw = data_process(file)
    # plt.plot(spectrum)
    # plt.show()
    spectrum_raw = spectrum
    spectrum_raw = normalization(spectrum_raw)
    spectrum = normalization(spectrum) * 2
    # plt.plot(spectrum)
    # plt.show()
    spectrum = torch.tensor(spectrum)
    spectrum = spectrum.reshape(1, spectrum.shape[0])
    spectrum = spectrum.reshape(1, spectrum.shape[0], spectrum.shape[1])  # data_type: batch_size x embedding_size x text_len
    spectrum = torch.as_tensor(spectrum, dtype=torch.float32)
    spectrum = spectrum.permute(1, 0, 2)
    filename, ext = os.path.splitext(filename)
    for epoch in range(1, EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        train(model, DEVICE, optimizer, epoch, spectrum_raw)
    test(model, DEVICE, keys, spectrum, epoch, filename)
    end_time = time.perf_counter()
    print(f"time consumed: {end_time - begin_time:0.4f} seconds")
    del model






