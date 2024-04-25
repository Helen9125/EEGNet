#import copy
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from torch.autograd import Variable
from torch.nn import functional as F

#from EEGNet_file import EEGNet
from models.EEGNet import EEGNet
from models.EEGNet import Depthwisw_separable_EEGNet
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)

        return data, label

    def __len__(self):
        return self.data.shape[0]

def plot_accuracy(train_acc_list, val_acc_list, num_epochs):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_acc_list) + 1), train_acc_list, linewidth=2.0, color='royalblue', label='Training accuracy')
    plt.plot(range(1, len(val_acc_list) + 1), val_acc_list, linewidth=2.0, color='orange', label='Validation accuracy')

    epochs = list(range(0, num_epochs, 10))

    plt.xticks(epochs, fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Accuracy Over Epochs', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)

    print('saving acc picture')
    plt.savefig('./result/acc_test.jpg')


def plot_loss(train_loss_list, val_loss_list, num_epochs):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, linewidth=2.0, color='royalblue', label='Training loss')
    plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, linewidth=2.0, color='orange', label='Validation loss')

    epochs = list(range(0, num_epochs, 10))

    plt.xticks(epochs, fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss Over Epochs', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)

    print('saving acc picture')
    plt.savefig('./result/loss_test.jpg')


def train(model, train_loader, val_loader, criterion, optimizer, args):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    val_acc_list = []
    avg_loss_list = []
    val_loss_list = []
    
    for epoch in range(1, args.num_epochs+1):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(tqdm(train_loader), 0):
                inputs, labels = data
            #for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                #outputs = torch.round(outputs).squeeze()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                #_, pred = torch.max(outputs.data, 1)
                #avg_acc += pred.eq(labels).cpu().sum().item()
                pred = torch.round(outputs).squeeze()#.cpu().detach().numpy()
                avg_acc += pred.eq(labels).cpu().sum().item()
                #avg_acc += (torch.tensor(pred) == labels.cpu().numpy()).astype(int).sum().item()

            avg_loss /= len(train_loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(train_loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            print(f'Epoch: {epoch}')
            print(f'Loss: {avg_loss}')
            print(f'Training Acc. (%): {avg_acc:3.2f}%')

        val_acc, val_loss = val(model, val_loader)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = model.state_dict()
        print(f'Validation Acc. (%): {val_acc:3.2f}%')
        
        model.train()

    print(f'Saving at epoch {epoch + 1}')
    torch.save(best_wts, './weights/best_test.pt')
    return avg_acc_list, avg_loss_list, val_acc_list, val_loss_list

def val(model, loader):
    avg_acc = 0.0
    avg_loss = 0.0
    
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)            
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            avg_loss += loss.item()
            pred = torch.round(outputs).squeeze()
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_loss /= len(loader.dataset)
        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc, avg_loss

def test(model, loader):
    avg_acc = 0.0
    
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc



    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=20)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()  
    train_val_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)

    training_proportion = 0.7
    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=1 - training_proportion, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = Depthwisw_separable_EEGNet(activation_func='elu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0006)

    model.to(device)
    criterion.to(device)

    train_acc_list, train_loss_list, test_acc_list, test_loss_list = train(model, train_loader, val_loader, criterion, optimizer, args)
    test_acc, test_loss = val(model, test_loader)

    acc_df = pd.DataFrame({'train acc':  train_acc_list, 'test acc': test_acc_list})
    loss_df = pd.DataFrame({'train loss':  train_loss_list, 'test acc': test_loss_list})

    print('saving data to csv')
    loss_df.to_csv('./result/c_loss_test.csv')
    acc_df.to_csv('./result/c_acc_test.csv')

    plot_accuracy(train_acc_list, test_acc_list, args.num_epochs)
    plot_loss(train_loss_list, test_loss_list, args.num_epochs)
    print('test accuracy:', test_acc)
