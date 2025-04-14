import argparse

import pandas as pd
import torch
import warnings

warnings.filterwarnings('ignore')
import os
from torch_geometric.data import Dataset, DataLoader
from Train import BIPLnet, set_gpu

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')  # if torch.cuda.is_available() else 'cpu'


class PLBA_Dataset_new(Dataset):
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
        return G[0], G[1], G[2], G[3], G[4], G[5]

    def __len__(self):
        return self.len

    def k_fold(self, train_idx, val_idx):
        train_list = [self.G_list[i] for i in train_idx]
        val_list = [self.G_list[i] for i in val_idx]
        return train_list, val_list

    def merge(self, data):
        self.G_list += data
        return self.G_list


def my_test(test_set, metadata, model_file):
    p_affinity = []
    y_affinity = []
    protein_ids = []

    m_state_dict = torch.load(model_file)
    best_model = BIPLnet(metadata=metadata).to(device)
    best_model.load_state_dict(m_state_dict)
    best_model.eval()
    test_loder = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=0)

    for i, data_raw in enumerate(test_loder, 0):
        with torch.no_grad():
            data = data_raw.iloc[[0, 1, 2, 3, 5]]
            data = set_gpu(data, device)
            predict = best_model(data)
            p_affinity.extend(predict.cpu().tolist())
            y_affinity.extend(data[0].y.cpu().tolist())
            protein_ids.extend(data_raw[4])

    # 打印每个蛋白质ID及其对应的预测和真实亲和力
    print(f"len = {len(protein_ids)}")

    # 创建 DataFrame
    df = pd.DataFrame({
        'Protein ID': protein_ids,
        'Predicted Affinity': [f"{value:.2f}" for value in p_affinity],  # 保留两位小数
        'True Affinity': [f"{value:.2f}" for value in y_affinity]  # 保留两位小数
    })

    # 保存为 CSV 文件
    df.to_csv(filepath + '/test_result.csv', index=False)


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

    print("loading test data")
    test_set = PLBA_Dataset_new('file', f'./Dataset/Processed/test.pkl')

    metadata = test_set[0][2].metadata()

    filepath = f'./output/{split}/'

    my_test(test_set, metadata, filepath + 'best_model.pt')
