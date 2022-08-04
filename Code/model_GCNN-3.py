import statistics

import pandas as pd
from sklearn.preprocessing import label_binarize
import tqdm
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score

from utils.data_handle import resoleTXT

use_gdc = True
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import DataLoader


class Dat():
    num_node_features = 836
    num_classes = 3


class InputDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index].input

    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)


import time

device = 'cuda'
model_path = "data/checkpoint5.16topk.pt"

max_nodes = 100000


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes


from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


def train(model, train_loader):
    loss_all = 0
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        loss_all += loss
    return loss_all / train_dataset


def test(model, loader, number):
    model.eval()

    loss_all = 0
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth label
        loss = criterion(out, data.y)  # Compute the loss.
        loss_all += data.y.size(0) * float(loss)
    return loss_all / len(loader.dataset), correct / len(loader.dataset)  # Derive ratio of correct predictions.


def test_for_all(model, loader, number):
    model.eval()

    prob_all = []
    label_all = []
    loss_all = 0
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth label
        loss = criterion(out, data.y)  # Compute the loss.
        loss_all += data.y.size(0) * float(loss)

        real_value = data.y.cpu().detach().numpy()
        prob_all.extend(pred)  # 求每一行的最大值索引
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

    return loss_all / number, correct / len(loader.dataset), F1, AUC, MCC


if __name__ == "__main__":
    dataset = Dat()
    avg_num_nodes = 1000

    result_file = 'result_GCNN_control_data1.txt'
    pkl_file = "..\Dataset\Packaged Pkl\input_data_with_matrix_1.5_control_data_type1_2724.pkl"
    input_dataset = pd.read_pickle(pkl_file)
    patch = 0
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(skf.split(input_dataset.input, input_dataset.target)):
        mid = []
        time.sleep(2)
        print(len(val_idx))
        patch += 1

        print('Train: %s | Val: %s' % (train_idx, val_idx), '\n')

        val_set = input_dataset.iloc[val_idx]
        train_set = input_dataset.iloc[train_idx]
        count = pd.DataFrame(val_set['target'].value_counts())  # count the appearance for each label
        print("Target count:", count)

        input_val_loader = InputDataset(val_set).get_loader(10, shuffle=True)

        valid_dataset = len(val_idx)
        train_dataset = 200 - valid_dataset

        progress_file = open(result_file, 'a')

        input_train_loader = InputDataset(train_set).get_loader(10, shuffle=True)

        input_model = GCN(hidden_channels=128)
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
