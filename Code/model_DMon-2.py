import statistics

import pandas as pd
from sklearn.preprocessing import label_binarize
import tqdm
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score

from utils.data_handle import resoleTXT

use_gdc = True
import numpy as np
import torch

from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import DataLoader
import time
from torch_geometric.nn import GCNConv, DenseGraphConv, DMoNPooling
from math import ceil
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch.nn import Linear
import torch.nn.functional as F


class InputDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index].input, self.dataset.iloc[index].matrix

    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)


device = 'cuda'
model_path = "data/checkpoint5.16topk.pt"
max_nodes = 100000


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes


class Dat():
    num_features = 836
    num_classes = 3


class CNNModel(torch.nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=356, out_channels=256, kernel_size=2)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2)
        self.conv3 = torch.nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2)
        # self.conv4 = torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2)
        # self.conv5 = torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2)
        self.drop = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.type(torch.float32)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 64 * 5)
        x = self.drop(x)

        return x


class Net(torch.nn.Module):
    # def __init__(self, in_channels, out_channels, hidden_channels=128):
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        num_nodes = ceil(0.5 * avg_num_nodes)
        self.pool1 = DMoNPooling([hidden_channels, hidden_channels], num_nodes)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = DMoNPooling([hidden_channels, hidden_channels], num_nodes)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.lin3 = Linear(448, 128)

        self.cnn = CNNModel()

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        _, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        x = self.conv2(x, adj).relu()

        _, x, adj, sp2, o2, c2 = self.pool2(x, adj)

        x = self.conv3(x, adj)

        x = x.mean(dim=1)

        # cnn_out = self.cnn(matrix)
        # output = torch.cat((x, cnn_out), 1)
        # x = self.lin3(output)

        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), sp1 + sp2 + o1 + o2 + c1 + c2


def handleMatrix(matrix):
    bath_size = matrix[0][0].shape[0]
    padding1 = []
    for i in range(bath_size):
        padding1.append(-1)
    new_m = []
    for t in matrix:
        a = torch.stack(t, 0).tolist()
        if len(a) == 305:
            for i in range(51):
                a.append(padding1)

        new_m.append(torch.Tensor(a))

    matrix = torch.stack(new_m, 1)
    matrix = matrix.permute(2, 0, 1)
    return matrix


def train(model, train_loader):
    model.train()
    loss_all = 0
    # print(len(train_loader))
    for data, matrix in train_loader:
        data = data.to(device)
        matrix = handleMatrix(matrix).to(device)
        optimizer.zero_grad()
        out, tot_loss = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y.view(-1)) + tot_loss
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / train_dataset / len(train_loader)


def test(model, loader, number):
    # print(len(loader))
    model.eval()
    correct = 0
    loss_all = 0

    prob_all = []
    label_all = []
    for data, matrix in loader:
        data = data.to(device)
        matrix = handleMatrix(matrix).to(device)
        pred, tot_loss = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(pred, data.y.view(-1)) + tot_loss
        loss_all += data.y.size(0) * float(loss)
        correct += int(pred.max(dim=1)[1].eq(data.y.view(-1)).sum())
    return loss_all / number / len(loader), correct / number


def test_for_all(model, loader, number):
    # print(len(loader))
    model.eval()
    correct = 0
    loss_all = 0

    prob_all = []
    label_all = []
    for data, matrix in loader:
        data = data.to(device)
        matrix = handleMatrix(matrix).to(device)
        pred, tot_loss = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(pred, data.y.view(-1)) + tot_loss
        loss_all += data.y.size(0) * float(loss)
        correct += int(pred.max(dim=1)[1].eq(data.y.view(-1)).sum())

        predict_value = np.argmax(pred.cpu().detach().numpy(), axis=1)
        real_value = data.y.cpu().detach().numpy()
        prob_all.extend(predict_value)  # 求每一行的最大值索引
        label_all.extend(real_value)

    # F1-Score
    F1 = f1_score(list(label_all), list(prob_all), average='macro')
    # AUC
    labels = [0, 1, 2]
    label_all_for_auc = label_binarize(label_all, classes=labels)
    prob_all_for_auc = label_binarize(prob_all, classes=labels)
    AUC = roc_auc_score(label_all_for_auc, prob_all_for_auc, average='macro', multi_class='ovo')
    # MCC
    MCC = matthews_corrcoef(list(label_all), list(prob_all))
    return loss_all / number / len(loader), correct / number, F1, AUC, MCC


if __name__ == "__main__":
    dataset = Dat()
    avg_num_nodes = 1000

    result_file = 'result_DMon_control_data1-two-class.txt'
    pkl_file = "D:\Program\Projects\源代码表征学习\devign-master\ourTry\\used_data" \
               "\input_data_with_matrix_1.5_control_data_type1_2724.pkl"
    input_dataset = pd.read_pickle(pkl_file)
    patch = 0
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5)
    mid = [i for i in range(50, 150)]
    input_dataset = input_dataset.drop(mid)
    for fold, (train_idx, val_idx) in enumerate(skf.split(input_dataset.input, input_dataset.target)):
        mid = []
        time.sleep(3)
        print(len(val_idx))
        patch += 1

        print('Train: %s | Val: %s' % (train_idx, val_idx), '\n')

        val_set = input_dataset.iloc[val_idx]
        train_set = input_dataset.iloc[train_idx]

        count = pd.DataFrame(val_set['target'].value_counts())
        print("Target count:", count)

        input_val_loader = InputDataset(val_set).get_loader(10, shuffle=True)

        valid_dataset = len(val_idx)
        train_dataset = 200 - valid_dataset

        progress_file = open(result_file, 'a')

        input_train_loader = InputDataset(train_set).get_loader(10, shuffle=True)

        input_model = Net(dataset.num_features, dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(input_model.parameters(), lr=0.0003)

        criterion = torch.nn.CrossEntropyLoss()
        output_list = []
        for epoch in range(1, 201):
            time.sleep(1)
            train_loss = train(input_model, input_train_loader)
            _, train_acc = test(input_model, input_train_loader, train_dataset)
            val_loss, val_acc, val_f1, val_auc, val_mcc = \
                test_for_all(input_model, input_val_loader, valid_dataset)
            print(
                f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                f'F1: {val_f1:.4f}, AUC:{val_auc:.4f}, MCC:{val_mcc:.4f}')
            output_list.append(
                f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                f'F1: {val_f1:.4f}, AUC:{val_auc:.4f}, MCC:{val_mcc:.4f}\n')

        for i in output_list[:len(output_list)]:
            progress_file.write(i)
        progress_file.close()
    resoleTXT(result_file)
