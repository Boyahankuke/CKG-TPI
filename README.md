# CKG-TPI
CKG-TPI is a novel graph neural network-based approach for predicting TCR-peptide binding specificity. This method constructs a collaborative knowledge graph (CKG) that integrates peptide and TCR amino acid sequences along with their biological context.

# Dependencies
CKG-TPI is writen in Python based on Pytorch. The required software dependencies are listed below:
```
torch
tqdm
pandas
numpy
scikit-learn
libauc
```

# Data
All the data used in the paper were collected from public databases: VDJdb, McAPS-TCR, and IEDB.

# Usage of CKG-TPI
Data Preparation:
Prepare the train dataset (https://drive.google.com/drive/folders/1aiqrhqks2sAXgXz2HROgQyvxlEsWNGF_?usp=sharing) in <BASE_FOLDER>/data/.
Move data folder into <BASE_FOLDER>/datasets/
Before training with VDJdb, McPAS or IEDB dataset, please move all the files in corresponding <dataset>_train_data (VDJdb_train_data, McPAS_train_data or IEDB_train_data) into <BASE_FOLDER>/datasets/
The preprocessing steps for the original data of each dataset can be found in the "Data preprocessing" section of the supplementary materials of the article.

Training CKG-TPI with VDJdb dataset on Random split strategy:
```
python main_kgat.py --dataset_name "VDJdb" --split_name "Random" --fold 0 --device_id 0 --n_epoch 20
```
Training CKG-TPI with VDJdb dataset on Strict split strategy:
```
python main_kgat.py --dataset_name "VDJdb" --split_name "Strict" --fold 0 --device_id 0 --n_epoch 20
```
Training CKG-TPI with McPAS dataset on Random split strategy:
```
python main_kgat.py --dataset_name "McPAS" --split_name "Random" --fold 0 --device_id 0 --n_epoch 20
```
Training CKG-TPI with McPAS dataset on Strict split strategy:
```
python main_kgat.py --dataset_name "McPAS" --split_name "Strict" --fold 0 --device_id 0 --n_epoch 20
```
Training CKG-TPI with IEDB dataset on Random split strategy:
```
python main_kgat.py --dataset_name "IEDB" --split_name "Random" --fold 0 --device_id 0 --n_epoch 20
```
Training CKG-TPI with IEDB dataset on Strict split strategy:
```
python main_kgat.py --dataset_name "IEDB" --split_name "Strict" --fold 0 --device_id 0 --n_epoch 20
```

The exact parameters used for model training and evaluation are as follows:
![image](https://github.com/user-attachments/assets/f7eb23b8-ad7e-4b8c-8489-712eb717885f)



