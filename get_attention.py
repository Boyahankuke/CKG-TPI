import torch
from model.KGAT import KGAT
from parser.parser_kgat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
import pandas as pd

kg_data = pd.read_csv("./datasets/KG_data/kg_final_VDJdb.txt", sep=' ', names=['h', 'r', 't'], engine='python')
kg_data = kg_data.drop_duplicates()
args = parse_kgat_args()
# load model
kg_data['r'] += 2
n_relations = max(kg_data['r']) + 1
model = KGAT(args, 1938, max(max(kg_data['h']), max(kg_data['t'])) + 1, n_relations)
checkpoint = torch.load("./results/checkpoint/VDJdb/Random/checkpoint_fold_0.pt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
print(model.A_in)
model.eval()
A_in = model.A_in
indices = A_in._indices()
values = A_in._values()
found = False
all_vis_tcrs = [1751, 1751, 1751, 1751, 1751, 15530, 15530, 15530, 15530, 15530, 15530, 731, 731, 731, 731, 731, 7601, 7601, 7601, 7601, 7601, 22496, 22496, 22496, 22496, 22496, 22496, 22496, 22496, 22496, 22496, 6, 6, 6, 6, 6, 6, 6, 6, 3585, 3585, 3585, 3585, 3585, 1116, 1116, 1116, 1116, 1116, 1116, 3157, 3157, 3157, 3157, 3157, 22445, 22445, 22445, 22445, 22445, 22445, 22445, 22445, 22445, 864, 864, 864, 864, 864, 864, 1047, 1047, 1047, 1047, 1047, 3687, 3687, 3687, 3687, 3687, 3687, 3687, 720, 720, 720, 720, 720, 720, 720, 720, 720, 15457, 15457, 15457, 15457, 15457, 15457, 978, 978, 978, 978, 978, 978, 978, 978, 978, 794, 794, 794, 794, 794, 794, 794, 794, 794, 794, 794, 794, 794, 794, 728, 728, 728, 728, 728, 1149, 1149, 1149, 1149, 1149, 22498, 22498, 22498, 22498, 22498, 22498, 22498, 22498, 22498, 22498, 22498, 22498, 2564, 2564, 2564, 2564, 2564, 746, 746, 746, 746, 746, 746, 746, 746, 725, 725, 725, 725, 725, 725, 725, 725, 725, 2279, 2279, 2279, 2279, 2279, 3634, 3634, 3634, 3634, 3634, 5907, 5907, 5907, 5907, 5907, 5907, 5907, 5907, 5907, 948, 948, 948, 948, 948, 948, 740, 740, 740, 740, 740, 740, 1188, 1188, 1188, 1188, 1188, 1188, 1188, 1188, 1188, 1827, 1827, 1827, 1827, 1827, 1037, 1037, 1037, 1037, 1037, 1037, 3588, 3588, 3588, 3588, 3588, 6007, 6007, 6007, 6007, 6007, 6007, 6007, 2978, 2978, 2978, 2978, 2978, 2978, 2978, 779, 779, 779, 779, 779, 779, 855, 855, 855, 855, 855, 6844, 6844, 6844, 6844, 6844, 22453, 22453, 22453, 22453, 22453, 22453, 22453, 22453, 22453, 22743, 22743, 22743, 22743, 22743, 934, 934, 934, 934, 934, 934, 934, 1458, 1458, 1458, 1458, 1458, 1458, 3646, 3646, 3646, 3646, 3646, 6824, 6824, 6824, 6824, 6824, 22495, 22495, 22495, 22495, 22495, 5863, 5863, 5863, 5863, 5863, 802, 802, 802, 802, 802, 802, 802, 802, 19399, 19399, 19399, 19399, 19399, 863, 863, 863, 863, 863, 863, 863, 22497, 22497, 22497, 22497, 22497, 22497, 3978, 3978, 3978, 3978, 3978, 7660, 7660, 7660, 7660, 7660, 1252, 1252, 1252, 1252, 1252, 977, 977, 977, 977, 977, 977, 977, 977, 977, 2343, 2343, 2343, 2343, 2343, 3673, 3673, 3673, 3673, 3673, 3673, 6047, 6047, 6047, 6047, 6047, 4513, 4513, 4513, 4513, 4513, 750, 750, 750, 750, 750, 750, 750, 750, 16263, 16263, 16263, 16263, 16263, 16263, 16263, 16263, 16263, 2065, 2065, 2065, 2065, 2065, 2065, 1235, 1235, 1235, 1235, 1235, 3774, 3774, 3774, 3774, 3774, 742, 742, 742, 742, 742, 18008, 18008, 18008, 18008, 18008, 946, 946, 946, 946, 946, 6680, 6680, 6680, 6680, 6680, 6680, 3650, 3650, 3650, 3650, 3650, 3650, 3650, 3650, 982, 982, 982, 982, 982, 982]
all_vis_epis = [139, 20, 54, 12, 873, 35, 22, 48, 20, 41, 24, 22, 20, 1, 24, 25, 35, 22, 20, 28, 24, 790, 793, 794, 786, 785, 787, 788, 789, 792, 791, 22, 2, 20, 21, 753, 1, 24, 25, 22, 20, 1, 24, 25, 22, 20, 28, 1, 24, 25, 20, 28, 1, 24, 25, 814, 810, 815, 808, 721, 812, 809, 813, 811, 22, 20, 28, 1, 24, 25, 22, 2, 20, 1, 25, 35, 2, 20, 28, 1, 24, 25, 22, 2, 20, 28, 8, 27, 1, 24, 25, 35, 22, 20, 27, 56, 24, 22, 20, 28, 46, 27, 1, 34, 24, 25, 35, 22, 2, 20, 28, 41, 52, 42, 27, 30, 1, 34, 24, 25, 22, 20, 28, 24, 25, 22, 20, 28, 24, 25, 805, 798, 799, 806, 802, 804, 807, 803, 797, 800, 796, 801, 22, 20, 1, 24, 25, 22, 2, 20, 28, 1, 24, 31, 25, 22, 2, 20, 28, 42, 1, 24, 31, 25, 22, 2, 20, 24, 25, 20, 28, 46, 24, 25, 35, 22, 20, 28, 41, 52, 42, 1, 24, 22, 20, 28, 1, 24, 25, 22, 20, 1, 34, 24, 25, 35, 2, 20, 28, 27, 1, 24, 31, 25, 22, 20, 30, 1, 25, 22, 20, 28, 46, 24, 25, 22, 20, 1, 24, 25, 35, 22, 20, 28, 41, 27, 24, 22, 20, 28, 41, 1, 24, 25, 22, 20, 28, 1, 24, 25, 20, 28, 1, 24, 25, 22, 20, 28, 24, 31, 745, 2, 749, 754, 746, 750, 748, 731, 747, 90, 21, 74, 837, 838, 22, 20, 28, 27, 1, 24, 25, 22, 20, 28, 1, 24, 25, 22, 2, 20, 1, 25, 22, 20, 21, 24, 6, 782, 783, 780, 784, 781, 22, 20, 28, 40, 24, 22, 2, 38, 20, 28, 1, 24, 25, 2, 379, 148, 25, 121, 35, 2, 20, 41, 27, 24, 25, 790, 794, 795, 789, 792, 791, 20, 28, 30, 24, 25, 35, 22, 20, 28, 24, 20, 54, 21, 1, 891, 22, 2, 20, 28, 21, 27, 1, 24, 25, 22, 20, 1, 24, 25, 2, 20, 28, 1, 24, 25, 22, 20, 28, 27, 24, 22, 20, 28, 24, 25, 22, 20, 28, 21, 30, 1, 24, 25, 723, 718, 729, 726, 725, 719, 720, 730, 32, 22, 20, 28, 41, 24, 25, 22, 20, 28, 1, 24, 20, 41, 27, 24, 25, 22, 20, 1, 24, 25, 2, 728, 46, 727, 121, 22, 20, 28, 24, 25, 22, 20, 28, 27, 24, 55, 22, 2, 20, 28, 41, 34, 24, 25, 22, 20, 28, 1, 24, 25]

tcr_dict = dict()
for m in range(len(all_vis_tcrs)):
    tcr_dict[str(all_vis_tcrs[m]) + " " + str(all_vis_epis[m])] = 1
# epi 170685 171524
all_res = ["tcr,epi,att_score"]
epi_seqs = np.load("./datasets/KG_data/epitope_seqs_dict.npy", allow_pickle=True).item()
tcr_seqs = np.load("./datasets/KG_data/TCR_seqs_dict.npy", allow_pickle=True).item()
for k in range(indices.size(1)):
    a = indices[0, k].item()
    b = indices[1, k].item()-170684
    if str(a) + " " + str(b) in tcr_dict:
        value = values[k]
        found = True
        all_res.append(tcr_seqs[a] + "," + epi_seqs[b] + "," + str(value.item()))
        print(len(all_res))
f = open("att_scores.txt", 'w')
f.write("\n".join(all_res))
f.close()


