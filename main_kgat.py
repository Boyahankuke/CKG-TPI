import os
import sys
import random
from time import time
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model.KGAT import KGAT
from parser.parser_kgat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_kgat import DataLoaderKGAT
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def evaluate_AUC(model, device, best_auc, best_aupr, args):
    test_batch_size = 512
    epi_ids = []
    TCR_ids = []
    labels = []
    f = open("./datasets/KG_data/data/" + args.dataset_name + "/" + args.split_name + "/test" + str(args.fold) + "_pos.txt", 'r')
    for line in f:
        line = line.replace("\n", "")
        line_vec = line.split(" ")
        for t in line_vec[1:]:
            epi_ids.append(int(line_vec[0]))
            TCR_ids.append(int(t))
            labels.append(1.0)
    f = open("./datasets/KG_data/data/" + args.dataset_name + "/" + args.split_name + "/test" + str(args.fold) + "_neg.txt", 'r')
    for line in f:
        line = line.replace("\n", "")
        line_vec = line.split(" ")
        for t in line_vec[1:]:
            epi_ids.append(int(line_vec[0]))
            TCR_ids.append(int(t))
            labels.append(0.0)

    model.eval()
    epi_ids_batches = [epi_ids[i: i + test_batch_size] for i in range(0, len(epi_ids), test_batch_size)]
    TCR_ids_batches = [TCR_ids[i: i + test_batch_size] for i in range(0, len(TCR_ids), test_batch_size)]
    labels_batches = [labels[i: i + test_batch_size] for i in range(0, len(labels), test_batch_size)]
    all_probs = []
    all_y = []
    for b in range(len(epi_ids_batches)):
        epi_ids_b = torch.LongTensor(epi_ids_batches[b]).to(device)
        TCR_ids_b = torch.LongTensor(TCR_ids_batches[b]).to(device)
        y_true = labels_batches[b]
        batch_scores_b = model(epi_ids_b, TCR_ids_b, mode='predict_pn')
        probs = torch.sigmoid(batch_scores_b).tolist()
        all_probs += probs
        all_y += y_true
    AUC_v = roc_auc_score(all_y, all_probs)
    precision, recall, thresholds = precision_recall_curve(all_y, all_probs)
    aupr = auc(recall, precision)
    if aupr > best_aupr:
        best_auc = AUC_v
        best_aupr = aupr
        torch.save(model.state_dict(), './results/checkpoint/' + args.dataset_name + "/" + args.split_name + '/checkpoint_fold_{}.pt'.format(str(args.fold)))
        all_preds = []
        for j in range(len(all_y)):
            all_preds.append(str(epi_ids[j]) + " " + str(TCR_ids[j]) + " " + str(all_probs[j]) + " " + str(all_y[j]))
        f = open("results/res/" + args.dataset_name + "/" + args.split_name + "/preds" + str(args.fold) + ".txt", 'w')
        f.write("\n".join(all_preds))
        f.close()

    print("cur AUC: " + str(AUC_v) + " cur AUPR: " + str(aupr) + " best AUC: " + str(best_auc) + " best AUPR: " + str(best_aupr))
    return best_auc, best_aupr


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderKGAT(args, logging)

    # construct model & optimizer
    model = KGAT(args, data.n_epis, data.n_entities, data.n_relations, data.A_in)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_auc = 0
    best_aupr = 0

    # train model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()
        # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_epi, cf_batch_pos_tcr, cf_batch_neg_tcr = data.generate_cf_batch(data.train_epi_dict, data.cf_batch_size, args)
            cf_batch_epi = cf_batch_epi.to(device)
            cf_batch_pos_tcr = cf_batch_pos_tcr.to(device)
            cf_batch_neg_tcr = cf_batch_neg_tcr.to(device)

            cf_batch_loss = model(cf_batch_epi, cf_batch_pos_tcr, cf_batch_neg_tcr, mode='train_cf')

            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                sys.exit()

            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.cf_print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        time3 = time()
        kg_total_loss = 0
        # n_kg_batch = data.n_kg_train // data.kg_batch_size + 1
        # python main_kgat.py --data_name KG_data
        n_kg_batch = 500  # 一轮kg训练循环次数

        for iter in range(1, n_kg_batch + 1):
            time4 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_epis_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='train_kg')

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (iter % args.kg_print_every) == 0:
                logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / iter))
        logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch))

        # update attention
        time5 = time()
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')

        logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time5))
        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        if (epoch % 2) == 0:
            best_auc, best_aupr = evaluate_AUC(model, device, best_auc, best_aupr, args)

def predict(args):
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderKGAT(args, logging)

    # load model
    model = KGAT(args, data.n_epis, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    best_auc = 0
    best_aupr = 0

    evaluate_AUC(model, device, best_auc, best_aupr, args)


if __name__ == '__main__':
    args = parse_kgat_args()
    train(args)
    # predict(args)


