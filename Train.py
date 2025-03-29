import argparse

import torch
import torch.nn as nn
import warnings
from sklearn import metrics

warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Dataset, DataLoader
import pickle
from torch_geometric.nn import EdgeConv, CGConv, HeteroConv, Linear, SAGEConv, GATConv, global_max_pool, MLP, \
    AttentiveFP, global_mean_pool, BatchNorm, HANConv, HGTConv, GCN, HEATConv
from torch_scatter import scatter_mean, scatter_sum, scatter_max
import scipy

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')  # if torch.cuda.is_available() else 'cpu'


class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, edge_dim, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.edge_mlp = MLP(channel_list=[128 + 8, 64, 16], dropout=0.1)
        self.lin_mpl = Linear(in_channels=16, out_channels=16)
        self.edge_lin = Linear(in_channels=1, out_channels=8)  # MLP([10,128,64,32])#

        self.conv_1 = HeteroConv(
            {
                edge_type: SAGEConv([-1, -1], hidden_channels)
                for edge_type in metadata[1]
            }
        )
        self.conv_2 = HeteroConv(
            {
                edge_type: SAGEConv([-1, -1], hidden_channels)
                for edge_type in metadata[1]
            }
        )
        self.conv_3 = HeteroConv(
            {
                edge_type: SAGEConv([-1, -1], hidden_channels)
                for edge_type in metadata[1]
            }
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        # for conv in self.convs:
        #     x_dict = conv(x_dict, edge_index_dict)
        #     x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        x1_dict = self.conv_1(x_dict, edge_index_dict)
        x1_dict = {key: F.leaky_relu(x) for key, x in x1_dict.items()}

        x2_dict = self.conv_2(x1_dict, edge_index_dict)
        x2_dict = {key: F.leaky_relu(x) for key, x in x2_dict.items()}

        x3_dict = self.conv_3(x2_dict, edge_index_dict)
        x3_dict = {key: F.leaky_relu(x) for key, x in x3_dict.items()}

        x_dict['ligand'] = x1_dict['ligand'] + x2_dict['ligand'] + x3_dict['ligand']
        x_dict['protein'] = x1_dict['protein'] + x2_dict['protein'] + x3_dict['protein']

        src, dst = edge_index_dict[('ligand', 'to', 'protein')]
        edge_repr = torch.cat([x_dict['ligand'][src], x_dict['protein'][dst]], dim=-1)

        d_pl = self.edge_lin(edge_attr_dict[('ligand', 'to', 'protein')])
        edge_repr = torch.cat((edge_repr, d_pl), dim=1)
        m_pl = self.edge_mlp(edge_repr)
        edge_batch = batch_dict['ligand'][src]

        w_pl = torch.tanh(self.lin_mpl(m_pl))
        m_w = w_pl * m_pl
        m_w = scatter_sum(m_w, edge_batch, dim=0)

        m_max, _ = scatter_max(m_pl, edge_batch, dim=0)
        m_out = torch.cat((m_w, m_max), dim=1)

        return m_out


class BIPLnet(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        # self.printed_shapes = True
        # self.heterognn = HeteroGNN(metadata, edge_dim=10, hidden_channels=64, out_channels=8, num_layers=3)
        self.ligandgnn = AttentiveFP(in_channels=18, hidden_channels=64, out_channels=16, edge_dim=12, num_timesteps=3,
                                     num_layers=3)
        self.proteingnn = AttentiveFP(in_channels=18, hidden_channels=64, out_channels=16, edge_dim=12, num_timesteps=3,
                                      num_layers=3)
        self.protein_seq_mlp = MLP(channel_list=[2560, 1280, 640, 16], dropout=0.1)
        self.ligand_seq_mlp = MLP(channel_list=[384, 192, 16], dropout=0.1)
        self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=4)
        self.out_mlp = MLP(channel_list=[64, 32, 1], dropout=0.1)

    def forward(self, data):
        g_l = data[0]
        g_p = data[1]
        # g_pl = data[2]
        pro_seq = data[3]
        ligand_seq = data[4]

        l = self.ligandgnn(x=g_l.x, edge_index=g_l.edge_index, edge_attr=g_l.edge_attr, batch=g_l.batch)
        p = self.proteingnn(x=g_p.x, edge_index=g_p.edge_index, edge_attr=g_p.edge_attr, batch=g_p.batch)
        # complex = self.heterognn(g_pl.x_dict, g_pl.edge_index_dict, g_pl.edge_attr_dict, g_pl.batch_dict)
        p_seq = self.protein_seq_mlp(pro_seq)
        l_seq = self.ligand_seq_mlp(ligand_seq)

        # 将四个向量拼接在一起，形状为 (batch_size, 4, 16)
        x = torch.stack([p, l, p_seq, l_seq], dim=1)
        # x = torch.stack([p, l, l_seq], dim=1)
        # x = torch.stack([l, p_seq, l_seq], dim=1)
        # x = torch.stack([p, l, p_seq], dim=1)
        # x = torch.stack([p, p_seq, l_seq], dim=1)

        # 需要转置为 (4, batch_size, 16)
        x = x.transpose(0, 1)

        # 注意力层的输出会是 (4, batch_size, 16)
        attn_output, _ = self.attention(x, x, x)

        # 将输出转回 (batch_size, 4, 16)
        attn_output = attn_output.transpose(0, 1)

        # 将四个 16 维的向量拼接成 (batch_size, 64)
        # emb = x.flatten(start_dim=1)
        emb = attn_output.flatten(start_dim=1)

        y_hat = self.out_mlp(emb)
        return torch.squeeze(y_hat)


class PLBA_Dataset(Dataset):
    def __init__(self, *args):
        if (args[0] == "file"):
            filepath = args[1]
            f = open(filepath, 'rb')
            self.G_list = pickle.load(f)
            self.len = len(self.G_list)
        elif (args[0] == 'list'):
            self.G_list = args[1]
            self.len = len(args[1])

    def __getitem__(self, index):
        G = self.G_list[index]
        return G[0], G[1], G[2], G[3], G[5]

    def __len__(self):
        return self.len

    def k_fold(self, train_idx, val_idx):
        train_list = [self.G_list[i] for i in train_idx]
        val_list = [self.G_list[i] for i in val_idx]
        return train_list, val_list

    def merge(self, data):
        self.G_list += data
        return self.G_list


def set_gpu(data, device):
    data_gpu = []
    for g in data:
        data_gpu.append(g.to(device))
    return data_gpu


def metrics_reg(targets, predicts):
    # 计算 MAE (平均绝对误差)
    mae = metrics.mean_absolute_error(y_true=targets, y_pred=predicts)

    # 计算 RMSE (均方根误差)
    rmse = metrics.mean_squared_error(y_true=targets, y_pred=predicts, squared=False)

    # 计算 Pearson 相关系数
    pearson = scipy.stats.mstats.pearsonr(targets, predicts)[0]

    # 计算 Spearman 相关系数
    spearman = scipy.stats.spearmanr(targets, predicts)[0]

    # 计算 R² (决定系数)
    r2 = metrics.r2_score(y_true=targets, y_pred=predicts)

    return [mae, rmse, pearson, spearman, r2]


def my_val(model, val_loader, device):
    p_affinity = []
    y_affinity = []

    model.eval()
    for data in val_loader:
        with torch.no_grad():
            data = set_gpu(data, device)
            predict = model(data)

            p_affinity.extend(predict.cpu().tolist())
            y_affinity.extend(data[0].y.cpu().tolist())

    affinity_err = metrics_reg(targets=y_affinity, predicts=p_affinity)

    return affinity_err


def my_train(train_loader, val_loader, test_set, metadata, kf_filepath):
    print('start training')

    model = BIPLnet(metadata=metadata).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    model.train()

    loss_list = []
    best_mae = float('inf')
    best_rmse = float('inf')
    for epoch in range(100):
        loss_epoch = 0
        n = 0
        for data in train_loader:
            data = set_gpu(data, device)
            optimizer.zero_grad()
            out = model(data)

            loss = F.mse_loss(out, data[0].y)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()
            n += 1
        loss_list.append(loss_epoch / n)
        print(f"Epoch : {epoch}, loss: {loss_epoch / n:.3f}")

        val_err = my_val(model, val_loader, device)
        val_mae = val_err[0]
        val_rmse = val_err[1]
        if val_rmse < best_rmse and val_mae < best_mae:
            print('            ******************Save Model********************')
            print(f"             Epoch : {epoch}, Val RMSE: {val_rmse:.3f}, MAE: {val_mae:.3f}")
            torch.save(model.state_dict(), kf_filepath + 'best_model.pt')
            best_mae = val_mae
            best_rmse = val_rmse
            affinity_err = my_test(test_set, metadata, kf_filepath + 'best_model.pt')

            test_mae, test_rmse, test_pearson, test_spearman, test_r2 = affinity_err
            print(
                f"             Epoch : {epoch}, Test RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}, Pearson: {test_pearson:.3f}, Spearman: {test_spearman:.3f}, R²: {test_r2:.3f}")

            with open(kf_filepath + "/log.txt", mode="a") as f_log:
                str_log = (
                    f"Epoch: {epoch:<5} | "
                    f"Test RMSE: {test_rmse:.3f} | "
                    f"Test MAE: {test_mae:.3f} | "
                    f"Test Pearson: {test_pearson:.3f} | "
                    f"Test Spearman: {test_spearman:.3f} | "
                    f"Test R²: {test_r2:.3f}\n"
                )
                f_log.write(str_log)

    plt.plot(loss_list)
    plt.ylabel('Loss')
    plt.xlabel("time")
    plt.savefig(kf_filepath + '/loss.png')
    # plt.show()


def my_test(test_set, metadata, model_file):
    p_affinity = []
    y_affinity = []

    m_state_dict = torch.load(model_file)
    best_model = BIPLnet(metadata=metadata).to(device)
    best_model.load_state_dict(m_state_dict)
    best_model.eval()
    test_loder = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=0)

    for i, data in enumerate(test_loder, 0):
        with torch.no_grad():
            data = set_gpu(data, device)
            predict = best_model(data)
            p_affinity.extend(predict.cpu().tolist())
            y_affinity.extend(data[0].y.cpu().tolist())

    affinity_err = metrics_reg(targets=y_affinity, predicts=p_affinity)

    return affinity_err


if __name__ == '__main__':
    """ Please use the Process.py file to preprocess the raw data and set up the training, validation, and test sets """

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process PDBbind data and set split parameter.")
    parser.add_argument(
        '--split',
        type=str,
        default='identity30',
        help="Specify the split type (e.g., identity30, identity60, scaffold.). Default is 'identity30'."
    )

    # Parse arguments
    args = parser.parse_args()
    split = args.split

    print("loading data")
    train_set = PLBA_Dataset('file', f'./Dataset/Processed/{split}/train.pkl')
    val_set = PLBA_Dataset('file', f'./Dataset/Processed/{split}/valid.pkl')
    test_set = PLBA_Dataset('file', f'./Dataset/Processed/{split}/test.pkl')

    train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, batch_size=128, shuffle=True, pin_memory=True)

    metadata = train_set[0][2].metadata()

    filepath = f'./output/{split}/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    my_train(train_loader, val_loader, test_set, metadata, filepath)
