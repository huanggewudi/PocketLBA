import argparse
import json

import torch
import openbabel
from openbabel import pybel
import warnings

warnings.filterwarnings('ignore')
import os
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import contains_isolated_nodes, tree_decomposition
from scipy.spatial import distance_matrix
import torch_geometric.transforms as T
import pickle
from Feature_extraction import Featurizer, CusBondFeaturizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline
import re
from prody import *
import networkx as nx
import numpy as np
from rdkit import Chem

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0')  # if torch.cuda.is_available() else 'cpu'


def info_3D(a, b, c):
    ab = b - a
    ac = c - a
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


def info_3D_cal(edge, ligand, h_num):
    node1_idx = edge[0]
    node2_idx = edge[1]
    atom1 = ligand.atoms[node1_idx]
    atom2 = ligand.atoms[node2_idx]

    neighbour1 = []
    neighbour2 = []
    for neighbour_atom in openbabel.OBAtomAtomIter(atom1.OBAtom):
        if neighbour_atom.GetAtomicNum() != 1:
            neighbour1.append(neighbour_atom.GetIdx() - h_num[neighbour_atom.GetIdx()] - 1)

    for neighbour_atom in openbabel.OBAtomAtomIter(atom2.OBAtom):
        if neighbour_atom.GetAtomicNum() != 1:
            neighbour2.append(neighbour_atom.GetIdx() - h_num[neighbour_atom.GetIdx()] - 1)

    neighbour1.remove(node2_idx)
    neighbour2.remove(node1_idx)
    neighbour1.extend(neighbour2)

    angel_list = []
    area_list = []
    distence_list = []

    if len(neighbour1) == 0 and len(neighbour2) == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for node3_idx in neighbour1:
        node1_coord = np.array(ligand.atoms[node1_idx].coords)
        node2_coord = np.array(ligand.atoms[node2_idx].coords)
        node3_coord = np.array(ligand.atoms[node3_idx].coords)

        angel, area, distence = info_3D(node1_coord, node2_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    for node3_idx in neighbour2:
        node1_coord = np.array(ligand.atoms[node1_idx].coords)
        node2_coord = np.array(ligand.atoms[node2_idx].coords)
        node3_coord = np.array(ligand.atoms[node3_idx].coords)
        angel, area, distence = info_3D(node2_coord, node1_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    return [np.max(angel_list) * 0.01, np.sum(angel_list) * 0.01, np.mean(angel_list) * 0.01,
            np.max(area_list), np.sum(area_list), np.mean(area_list),
            np.max(distence_list) * 0.1, np.sum(distence_list) * 0.1, np.mean(distence_list) * 0.1]


def get_complex_edge_fea(edge_list, coord_list):
    net = nx.Graph()
    net.add_weighted_edges_from(edge_list)
    edges_fea = []
    for edge in edge_list:
        edge_fea = []

        edge_fea.append(edge[2])
        edges_fea.append(edge_fea)

    return edges_fea


def read_ligand(filepath, ligand_lm):
    featurizer = Featurizer(save_molecule_codes=False)
    ligand = next(pybel.readfile("mol2", filepath))
    ligand_coord, atom_fea, h_num = featurizer.get_features(ligand)

    mol = Chem.MolFromMol2File(filepath)
    smiles = Chem.MolToSmiles(mol)

    embedding = ligand_lm(smiles)
    ligand_seq_emb = torch.tensor(embedding[0])
    ligand_seq_emb = torch.mean(ligand_seq_emb, dim=0)

    return ligand_coord, atom_fea, ligand, h_num, ligand_seq_emb


def read_protein(filepath, prot_lm):
    featurizer = Featurizer(save_molecule_codes=False)
    protein_pocket = next(pybel.readfile("pdb", filepath))
    # 蛋白质的坐标（pocket_coord）、原子特征（atom_fea）以及氢键数量（h_num）
    pocket_coord, atom_fea, h_num = featurizer.get_features(protein_pocket)

    aa_codes = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
        'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W'}

    seq = ''
    # protein_filepath = filepath
    protein_filepath = filepath.replace('_pocket', '')
    for line in open(protein_filepath):
        if line[0:6] == "SEQRES":
            columns = line.split()
            for resname in columns[4:]:
                if resname in aa_codes:
                    seq = seq + aa_codes[resname] + ' '
    sequences_Example = re.sub(r"[UZOB]", "X", seq)

    embedding = prot_lm(sequences_Example)
    pro_seq_emb = torch.tensor(embedding[0])
    pro_seq_emb = torch.mean(pro_seq_emb, dim=0)

    return pocket_coord, atom_fea, protein_pocket, h_num, pro_seq_emb


def mult_graph(lig_file_name, pocket_file_name, id, score, prot_lm, ligand_lm):
    lig_coord, lig_atom_fea, mol, h_num_lig, ligand_seq_emb = read_ligand(lig_file_name, ligand_lm)
    pocket_coord, pocket_atom_fea, protein, h_num_pro, pro_seq_emb = read_protein(pocket_file_name, prot_lm)  # ,pro_seq

    if (mol is not None) and (protein is not None):
        # print("Both not None.")
        G_l = ligand_graph(lig_atom_fea, mol, h_num_lig, score)
        G_p = protein_graph(pocket_atom_fea, protein, h_num_pro, score)
        G_inter = inter_graph(lig_coord, pocket_coord, lig_atom_fea, pocket_atom_fea, score)
        G_list = [G_l, G_p, G_inter, pro_seq_emb, id, ligand_seq_emb]
        return G_list
    else:
        if mol is None and protein is None:
            print("Both 'mol' and 'protein' are None.")
        elif mol is None:
            print("'mol' is None.")
        elif protein is None:
            print("'protein' is None.")
        return None


def bond_fea(bond, atom1, atom2):
    is_Aromatic = int(bond.IsAromatic())
    is_inring = int(bond.IsInRing())
    d = atom1.GetDistance(atom2)

    node1_idx = atom1.GetIdx()
    node2_idx = atom2.GetIdx()

    neighbour1 = []
    neighbour2 = []
    for neighbour_atom in openbabel.OBAtomAtomIter(atom1):
        if (neighbour_atom.GetAtomicNum() != 1) and (neighbour_atom.GetIdx() != node2_idx):
            neighbour1.append(neighbour_atom)

    for neighbour_atom in openbabel.OBAtomAtomIter(atom2):
        if (neighbour_atom.GetAtomicNum() != 1) and (neighbour_atom.GetIdx() != node1_idx):
            neighbour2.append(neighbour_atom)

    if len(neighbour1) == 0 and len(neighbour2) == 0:
        return [d, 0, 0, 0, 0, 0, 0, 0, 0, 0, is_Aromatic, is_Aromatic]

    angel_list = []
    area_list = []
    distence_list = []

    node1_coord = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    node2_coord = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])

    for atom3 in neighbour1:
        node3_coord = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])
        angel, area, distence = info_3D(node1_coord, node2_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    for atom3 in neighbour2:
        node3_coord = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])
        angel, area, distence = info_3D(node2_coord, node1_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    return [d,
            np.max(angel_list) * 0.01, np.sum(angel_list) * 0.01, np.mean(angel_list) * 0.01,
            np.max(area_list), np.sum(area_list), np.mean(area_list),
            np.max(distence_list) * 0.1, np.sum(distence_list) * 0.1, np.mean(distence_list) * 0.1,
            is_Aromatic, is_inring]


def edgelist_to_tensor(edge_list):
    row = []
    column = []
    coo = []
    for edge in edge_list:
        row.append(edge[0])
        column.append(edge[1])

    coo.append(row)
    coo.append(column)

    coo = torch.Tensor(coo)
    edge_tensor = torch.tensor(coo, dtype=torch.long)
    return edge_tensor


def atomlist_to_tensor(atom_list):
    new_list = []
    for atom in atom_list:
        new_list.append([atom])
    atom_tensor = torch.Tensor(new_list)
    return atom_tensor


def ligand_graph(lig_atoms_fea, ligand, h_num, score):
    edges = []
    edges_fea = []
    for bond in openbabel.OBMolBondIter(ligand.OBMol):
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetAtomicNum() == 1) or (atom2.GetAtomicNum() == 1):
            continue
        else:
            idx_1 = atom1.GetIdx() - h_num[atom1.GetIdx() - 1] - 1
            idx_2 = atom2.GetIdx() - h_num[atom2.GetIdx() - 1] - 1

            edge_fea = bond_fea(bond, atom1, atom2)
            edge = [idx_1, idx_2]
            edges.append(edge)
            edges_fea.append(edge_fea)

            re_edge = [idx_2, idx_1]
            edges.append(re_edge)
            edges_fea.append(edge_fea)

    edge_attr = torch.tensor(edges_fea, dtype=torch.float32)
    x = torch.tensor(lig_atoms_fea, dtype=torch.float32)
    edge_index = edgelist_to_tensor(edges)
    G_lig = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=torch.tensor(score))

    return G_lig


def protein_graph(pocket_atom_fea, protein, h_num, score):
    edges = []
    edges_fea = []
    for bond in openbabel.OBMolBondIter(protein.OBMol):
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetAtomicNum() == 1) or (atom2.GetAtomicNum() == 1):
            continue
        else:
            idx_1 = atom1.GetIdx() - h_num[atom1.GetIdx() - 1] - 1
            idx_2 = atom2.GetIdx() - h_num[atom2.GetIdx() - 1] - 1

            edge_fea = bond_fea(bond, atom1, atom2)
            edge = [idx_1, idx_2]
            edges.append(edge)
            edges_fea.append(edge_fea)

            re_edge = [idx_2, idx_1]
            edges.append(re_edge)
            edges_fea.append(edge_fea)

    edge_attr = torch.tensor(edges_fea, dtype=torch.float32)
    x = torch.tensor(pocket_atom_fea, dtype=torch.float32)
    edge_index = edgelist_to_tensor(edges)
    G_pocket = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(score))

    return G_pocket


def inter_graph(lig_coord, pocket_coord, lig_atom_fea, pocket_atom_fea, score, cut=5):
    coord_list = []
    for atom in lig_coord:
        coord_list.append(atom)
    for atom in pocket_coord:
        coord_list.append(atom)

    dis = distance_matrix(x=coord_list, y=coord_list)
    lenth = len(coord_list)
    edge_list = []

    edge_list_fea = []
    # Bipartite Graph; i belongs to ligand, j belongs to protein
    for i in range(len(lig_coord)):
        for j in range(len(lig_coord), lenth):
            if dis[i, j] < cut:
                edge_list.append([i, j - len(lig_coord), dis[i, j]])
                edge_list_fea.append([i, j, dis[i, j]])

    data = HeteroData()
    edge_index = edgelist_to_tensor(edge_list)

    data['ligand'].x = torch.tensor(lig_atom_fea, dtype=torch.float32)
    data['ligand'].y = torch.tensor(score)
    data['protein'].x = torch.tensor(pocket_atom_fea, dtype=torch.float32)
    data['ligand', 'protein'].edge_index = edge_index

    complex_edges_fea = get_complex_edge_fea(edge_list_fea, coord_list)
    edge_attr = torch.tensor(complex_edges_fea, dtype=torch.float32)
    data['ligand', 'protein'].edge_attr = edge_attr
    data = T.ToUndirected()(data)

    return data


def get_Resfea(res):
    aa_codes = {
        'ALA': 1, 'CYS': 2, 'ASP': 3, 'GLU': 4,
        'PHE': 5, 'GLY': 6, 'HIS': 7, 'LYS': 8,
        'ILE': 9, 'LEU': 10, 'MET': 11, 'ASN': 12,
        'PRO': 13, 'GLN': 14, 'ARG': 15, 'SER': 16,
        'THR': 17, 'VAL': 18, 'TYR': 19, 'TRP': 0}
    one_hot = np.eye(21)
    if res in aa_codes:
        code = aa_codes[res]
    else:
        code = 20
    fea = one_hot[code]
    return fea


def GetPDBDict(Path):
    with open(Path, 'rb') as f:
        lines = f.read().decode().strip().split('\n')
    res = {}
    for line in lines:
        if "//" in line:
            temp = line.split()
            name, score = temp[0], float(temp[3])
            res[name] = score
    return res


def process_raw_data(dataset_path, processed_file, set_list):
    res = GetPDBDict(Path='./Dataset/Raw/pdbbind/metadata/index/INDEX_general_PL_data.2019')

    G_list = []

    tokenizer = AutoTokenizer.from_pretrained("LLMs/ESM2_t36_3B_UR50D", do_lower_case=False)
    model = AutoModel.from_pretrained("LLMs/ESM2_t36_3B_UR50D")
    prot_lm = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=0)

    ligand_tokenizer = AutoTokenizer.from_pretrained("/mnt/disk/hzy/pyg/LLMs/ChemBERTa-77M-MLM")
    ligand_model = AutoModel.from_pretrained("/mnt/disk/hzy/pyg/LLMs/ChemBERTa-77M-MLM")
    ligand_lm = pipeline('feature-extraction', model=ligand_model, tokenizer=ligand_tokenizer, device=0)

    for item in tqdm(set_list):
        score = res[item]
        lig_file_name = dataset_path + item + '/' + item + '_ligand.mol2'
        pocket_file_name = dataset_path + item + '/' + item + '_pocket.pdb'

        try:
            G = mult_graph(lig_file_name, pocket_file_name, item, score, prot_lm, ligand_lm)
            if G is not None:
                print("Graph not None.")
                G_list.append(G)
        except Exception as e:
            print(f"发生错误: {e}")  # 打印错误信息
            continue  # 继续下一个循环

    print('sample num: ', len(G_list))
    with open(processed_file, 'wb') as f:
        pickle.dump(G_list, f)
    f.close()


import json


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


if __name__ == '__main__':
    # Please put the raw structure data downloaded from the PDBbind official website in directory "raw_data_path"
    # Directory "data_path" is the processed pkl file
    warnings.filterwarnings("ignore")

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

    raw_data_path = './Dataset/Raw/pdbbind/pdb_files/'

    data_path_train = f'./Dataset/Processed/{split}/train.pkl'
    data_path_valid = f'./Dataset/Processed/{split}/valid.pkl'
    data_path_test = f'./Dataset/Processed/{split}/test.pkl'

    # 如果目录不存在，则创建目录
    data_dir = os.path.dirname(data_path_train)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 读取 JSON 文件
    json_file_path = f"./Dataset/Raw/pdbbind/metadata/{split}_split.json"
    json_data = read_json(json_file_path)

    train_list = json_data['train']
    valid_list = json_data['valid']
    test_list = json_data['test']

    if not os.path.exists(data_path_test):
        process_raw_data(raw_data_path, data_path_test, test_list)
    if not os.path.exists(data_path_train):
        process_raw_data(raw_data_path, data_path_train, train_list)
    if not os.path.exists(data_path_valid):
        process_raw_data(raw_data_path, data_path_valid, valid_list)
