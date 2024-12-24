import pandas as pd
import torch
import warnings

warnings.filterwarnings('ignore')
import os
from torch_geometric.data import Dataset, DataLoader
from train import BIPLnet, PLBA_Dataset, set_gpu

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')  # if torch.cuda.is_available() else 'cpu'


def my_test(test_set, metadata, model_file):
    p_affinity = []
    y_affinity = []
    protein_ids = []

    m_state_dict = torch.load(model_file)
    best_model = BIPLnet(metadata=metadata).to(device)
    best_model.load_state_dict(m_state_dict)
    best_model.eval()
    test_loder = DataLoader(dataset=test_set, batch_size=128, shuffle=True, num_workers=0)

    for i, data_raw in enumerate(test_loder, 0):
        with torch.no_grad():
            data = data_raw[0:4]
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
    """ Please use the process.py file to preprocess the raw data and set up the training, validation, and test sets """

    split = 'identity30'
    print("loading test data")
    test_set = PLBA_Dataset('file', f'./data/{split}/test.pkl')

    metadata = test_set[0][2].metadata()

    filepath = f'./output/{split}/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    my_test(test_set, metadata, filepath + 'best_model.pt')
