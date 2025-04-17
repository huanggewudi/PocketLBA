# **PocketLBA** Code Repository

This is a repository to deposit the code and data for **PocketLBA** model. **PocketLBA** is a protein-ligand binding affinity prediction model that integrates pocket information.

---

The file and folders in the PocketLBA repository:

- **Dataset**: 

  a. **Raw**: Three PDBbind v2019 benchmark  datasets and CASF-2016 dataset.

  b. **Virtual screening**: The overall JAK1 PDB and JAK1 pocket PDB, along with the SMILES strings and binding affinities of 39 drugs.

  c. **PDB_id_list**: The protein list of different dataset split.

- **LLMs**: Protein language model ESM-2 and small molecule language model ChemBERTa.

- **Feature_extraction.py:** Extract features for protein and ligand.

- **Process.py**: Preprocess protein and ligand data, generating graph files.

- **Train.py: **Train the model.

- **Test.py:** Test the model.

- **environment.yml**: Environment configuration.

---



### **Step-by-step Running:**

## 1. Environment Installation

It is recommended to use the conda environment (python 3.7), mainly installing the following dependencies:

- [ ] ​		**pytorch (1.9.0)、torch-geometric (2.0.4)、dgl-cu111 (0.6.1)、cudatoolkit (11.1.74)**

- [ ] ​		**biopython (1.79)、rdkit (2023.3.1)、transformers (4.24.0)、wandb (0.15.4)**

See environment.yml for details.



## 2. Datasets

Download the datasets from the following links:

-  /Datasets/Raw:  https://zenodo.org/records/15220494

  

## 3. LLMs

Download the LLMs from the following links:

- ​		/LLMs /ChemBERTa-77M-MLM:  https://huggingface.co/DeepChem/ChemBERTa-77M-MLM
- ​		/LLMs /ESM2_t36_3B_UR50D:  https://huggingface.co/facebook/esm2_t36_3B_UR50D



## 4. Data Preprocessing

Extracting protein and ligand features. {split} is one of `identity30, identity60, scaffold, casf`

```
python Process.py --split {split} 
```

The generated files are stored in the dataset path：`./Datasets/Processed/{split}/`



## 5.  Training

Train the model.

```
python Train.py --split {split} 
```

The trained model is stored in the path `/output/{split}/best_model.pt`



## 6. Testing

Test the trained model.

```
python Test.py --split {split} 
```

The test results are stored in the path`/output/{split}/test_result.csv`

