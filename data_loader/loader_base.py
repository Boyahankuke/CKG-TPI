import os
import time
import random
import collections

import torch
import numpy as np
import pandas as pd


class DataLoaderBase(object):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir

        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.train_file = os.path.join(self.data_dir, 'data/' + args.dataset_name + "/" + args.split_name + "/train" + str(args.fold) + "_pos.txt")
        self.test_file = os.path.join(self.data_dir, 'data/' + args.dataset_name + "/" + args.split_name + "/test" + str(args.fold) + "_pos.txt")
        self.kg_file = os.path.join(self.data_dir, "kg_final_" + str(args.dataset_name) + ".txt")

        self.cf_train_data, self.train_epi_dict = self.load_cf(self.train_file)
        self.cf_test_data, self.test_epi_dict = self.load_cf(self.test_file)
        self.statistic_cf()
        self.node_to_type = np.load(os.path.join(self.data_dir, "node_to_type.npy"), allow_pickle=True).item()
        self.type_to_nodes = np.load(os.path.join(self.data_dir, "type_to_nodes.npy"), allow_pickle=True).item()

        self.store_train = True  # 用于直接从train采样数据，而非从dict随机选择表位

        if self.use_pretrain == 1:
            self.load_pretrained_data()

    def load_cf(self, filename):
        epi = []
        tcr = []
        epi_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                epi_id, tcr_ids = inter[0], inter[1:]
                tcr_ids = list(set(tcr_ids))

                for tcr_id in tcr_ids:
                    epi.append(epi_id)
                    tcr.append(tcr_id)
                epi_dict[epi_id] = tcr_ids

        epi = np.array(epi, dtype=np.int32)
        tcr = np.array(tcr, dtype=np.int32)
        return (epi, tcr), epi_dict

    def statistic_cf(self):
        # 1017  38187
        # 324 28889
        # 1938 168023
        # 1958 168029 external
        # self.n_epis = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_tcrs_max = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_epis = 1938  # total epitopes
        self.n_tcrs = 168023  # total tcrs
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])

    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def generate_cf_batch(self, epi_dict, batch_size, args):
        if self.store_train:
            epi_ids_pos = []
            TCR_ids_pos = []
            epi_ids_neg = []
            TCR_ids_neg = []
            all_tcrs = []
            f = open("./datasets/KG_data/data/" + args.dataset_name + "/" + args.split_name + "/train" + str(args.fold) + "_pos.txt", 'r')
            for line in f:
                line = line.replace("\n", "")
                line_vec = line.split(" ")
                for t in line_vec[1:]:
                    epi_ids_pos.append(int(line_vec[0]))
                    TCR_ids_pos.append(int(t))
                    all_tcrs.append(int(t))
            neg_epi_to_tcr = dict()
            f = open("./datasets/KG_data/data/" + args.dataset_name + "/" + args.split_name + "/train" + str(args.fold) + "_neg.txt", 'r')
            for line in f:
                line = line.replace("\n", "")
                line_vec = line.split(" ")
                if line_vec[0] not in neg_epi_to_tcr:
                    neg_epi_to_tcr[line_vec[0]] = []
                for t in line_vec[1:]:
                    epi_ids_neg.append(int(line_vec[0]))
                    TCR_ids_neg.append(int(t))
                    neg_epi_to_tcr[line_vec[0]].append(int(t))
                    all_tcrs.append(int(t))
            self.train_tcrs_pos = np.array(TCR_ids_pos)
            self.train_epi_pos = np.array(epi_ids_pos)
            self.train_tcrs_neg = np.array(TCR_ids_neg)
            self.train_epi_neg = np.array(epi_ids_neg)
            self.store_train = False
            self.neg_epi_to_tcr = neg_epi_to_tcr
            self.all_tcrs = all_tcrs
        batch_pos_id = random.sample([i for i in range(len(self.train_tcrs_pos))], batch_size)
        batch_tcr_pos = torch.LongTensor(self.train_tcrs_pos[batch_pos_id])
        batch_epi_pos = torch.LongTensor(self.train_epi_pos[batch_pos_id])

        e_p = self.train_epi_pos[batch_pos_id].tolist()
        batch_tcr_neg = []
        for e in e_p:
            if e not in self.neg_epi_to_tcr:
                a = random.choice(self.all_tcrs)
            else:
                a = random.choice(self.neg_epi_to_tcr[e])
            batch_tcr_neg.append(a)
        batch_tcr_neg = torch.LongTensor(batch_tcr_neg)

        batch_epi = batch_epi_pos
        batch_pos_tcr = batch_tcr_pos
        batch_neg_tcr = batch_tcr_neg
        return batch_epi, batch_pos_tcr, batch_neg_tcr

    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx, pos_tail):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        cnt = 0
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            # tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            tail = random.choice(self.type_to_nodes[self.node_to_type[pos_tail]])
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
            cnt += 1
            if cnt > 100:
                print(head)
                print(relation)
                print("------")
                tail = random.choice([x for x in range(self.n_epis_entities)])
                sample_neg_tails.append(tail)
        return sample_neg_tails

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = kg_dict.keys()
        exist_node_types = list(self.type_to_nodes.keys())
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            '''
            batch_head = []
            for k in range(batch_size):
                random_type = random.choice(exist_node_types)
                batch_head.append(random.choice(self.type_to_nodes[random_type]))
            '''
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx, pos_tail[0])
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


    def load_pretrained_data(self):
        self.epi_pre_embed = np.load("./datasets/epi_emb.npy")
        self.tcr_pre_embed = np.load("./datasets/TCR_emb.npy")

        assert self.epi_pre_embed.shape[0] == self.n_epis
        assert self.tcr_pre_embed.shape[0] == self.n_tcrs
        assert self.epi_pre_embed.shape[1] == self.args.embed_dim
        assert self.tcr_pre_embed.shape[1] == self.args.embed_dim


