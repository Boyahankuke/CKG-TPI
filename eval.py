from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.utils.multiclass import type_of_target
dataset_name = "IEDB"
split_name = "Random"
preds = []
labels = []
all_auc = []
all_aupr = []
for i in range(5):
    f = open("results/res/" + dataset_name + "/" + split_name + "/preds" + str(i) + ".txt", 'r')
    for line in f:
        line = line.replace("\n", "")
        line_vec = line.split(" ")
        preds.append(float(line_vec[2]))
        labels.append(float(line_vec[3]))
        if float(line_vec[3]) != 1.0 and float(line_vec[3]) != 0.0:
            print(line_vec[3])

    print(type_of_target(preds))
    print(type_of_target(labels))
    AUC_v = roc_auc_score(labels, preds)
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    aupr = auc(recall, precision)
    print(AUC_v)
    print(aupr)
    print("\n")
    all_auc.append(AUC_v)
    all_aupr.append(aupr)
print(sum(all_auc) / len(all_auc))
print(sum(all_aupr) / len(all_aupr))


'''
McPAS
Random
0.7371382179089963
0.3803727997351901
Strict
0.7380469836966224
0.40639111838151765

TEIM AUC=0.735, AUPR=0.4103

VDJdb
Random
0.6874578337559033
0.31350304158289555
Strict
0.6745171195661669
0.31860243804631544

TEIM AUC=0.7339, AUPR=0.3916


IEDB
Strict
0.6541897089521883
0.23460404026754014
Random
0.687096874754709
0.30001797549057985

TEIM AUC=0.7451, AUPR=0.3403
'''
