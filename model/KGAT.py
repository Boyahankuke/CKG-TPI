import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from libauc.losses import AUCMLoss
from model.ref_models import BIN_Interaction_Flat


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=40):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        # x is expected to have shape (batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm1(x)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)  # Residual connection
        x = self.norm2(x)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, return_att=False):
        # Cross-attention
        attn_output, att_matrix = self.cross_attention(query, key, value, attn_mask=mask)
        query = query + self.dropout(attn_output)  # Residual connection
        query = self.norm1(query)

        # Feed-forward network
        ffn_output = self.ffn(query)
        query = query + self.dropout(ffn_output)  # Residual connection
        query = self.norm2(query)

        if not return_att:
            return query

        att_soft = F.softmax(torch.sum(att_matrix, dim=1), dim=-1)

        return query, att_soft


class TCRBindingModel(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TCRBindingModel, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim)

        self.self_attention1 = TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
        self.self_attention2 = TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)

        self.cross_attention1 = CrossAttentionBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
        self.cross_attention2 = CrossAttentionBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
        self.cross_attention3 = CrossAttentionBlock(embed_dim, num_heads, ff_hidden_dim, dropout)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, seq1, seq2, seq3, mask1=None, mask2=None):
        # Permute for MultiheadAttention
        seq1 = self.positional_encoding(seq1)
        seq2 = self.positional_encoding(seq2)

        seq1 = seq1.permute(1, 0, 2)
        seq2 = seq2.permute(1, 0, 2)
        seq3 = seq3.permute(1, 0, 2)

        # Apply self-attention
        seq1 = self.self_attention1(seq1, mask1)
        seq2 = self.self_attention2(seq2, mask2)

        # Apply cross-attention between seq1 and seq2
        seq1_cross2 = self.cross_attention1(seq1, seq2, seq2, mask2)
        seq2_cross1 = self.cross_attention2(seq2, seq1, seq1, mask1)

        # Combine outputs from first cross-attention
        combined_seq12 = torch.cat((seq1_cross2, seq2_cross1), dim=0)
        if mask1 != None and mask2 != None:
            mask_1_1 = torch.cat((mask1, mask2), dim=1)
        else:
            mask_1_1 = None
        # combined_seq12 = seq1_cross2 + seq2_cross1

        # Apply cross-attention with the third sequence
        # final_output = self.cross_attention2(seq3, combined_seq12, combined_seq12, mask_1_1)
        final_output, att_soft = self.cross_attention3(combined_seq12, seq3, seq3, mask_1_1, return_att=True)

        # Permute back to original shape
        final_output = final_output.permute(1, 0, 2)

        final_output = self.norm(torch.sum(final_output, dim=1))

        combined_seq12 = seq1_cross2.permute(1, 0, 2)
        # combined_seq12 = combined_seq12[:, 0, :]
        combined_seq12 = self.norm(torch.sum(combined_seq12, dim=1))

        return combined_seq12, final_output, att_soft


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)
epi_seqs = torch.LongTensor(np.load("./datasets/KG_data/epitope_seqs.npy"))
tcr_seqs = torch.LongTensor(np.load("./datasets/KG_data/TCR_seqs.npy"))
epi_seqs_mask = torch.LongTensor(np.load("./datasets/KG_data/epitope_seqs_mask.npy"))
tcr_seqs_mask = torch.LongTensor(np.load("./datasets/KG_data/TCR_seqs_mask.npy"))

class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        else:
            raise NotImplementedError


    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_epis + n_entities, in_dim)
        A_in:            (n_epis + n_entities, n_epis + n_entities), torch.sparse.FloatTensor
        """
        # Equation (3)
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings)
        return embeddings


class KGAT(nn.Module):

    def __init__(self, args, n_epis, n_entities, n_relations, A_in=None):

        super(KGAT, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_epis = n_epis
        self.n_entities = n_entities - n_epis  # entity加入了表位，因此不需要再添加
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.entity_epi_embed = nn.Embedding(self.n_entities + self.n_epis, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        nn.init.xavier_uniform_(self.entity_epi_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_epis + self.n_entities, self.n_epis + self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False
        self.seq_embedding1 = nn.Embedding(21, 960)
        self.seq_embedding2 = nn.Embedding(21, 960)
        self.pref_embedding = nn.Embedding(100, 960)
        self.bind_layer = BIN_Interaction_Flat()
        self.BN = nn.LayerNorm(960)
        self.BN2 = nn.LayerNorm(960 * 2)
        self.pref_M1 = nn.Sequential(
            nn.Linear(960, 960),
            nn.ReLU(),
            nn.Linear(960, 960)
        )
        self.pref_M2 = nn.Sequential(
            nn.Linear(960, 960),
            nn.ReLU(),
            nn.Linear(960, 960)
        )
        self.Prediction = nn.Sequential(
            nn.LayerNorm(960 * 3),
            nn.Linear(960 * 3, 960),
            nn.ReLU(),
            nn.Linear(960, 1)
        )
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")

    def calc_cf_embeddings(self):
        ego_embed = self.entity_epi_embed.weight
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        all_embed = torch.cat(all_embed, dim=1)
        return all_embed

    def calc_cf_loss(self, epi_ids, tcr_pos_ids, tcr_neg_ids):
        """
        epi_ids:       (cf_batch_size)
        tcr_pos_ids:   (cf_batch_size)
        tcr_neg_ids:   (cf_batch_size)
        """
        device = self.device
        epi_seqs_batch = epi_seqs[(epi_ids).cpu()].to(device)
        tcr_pos_seqs_batch = tcr_seqs[tcr_pos_ids.cpu()].to(device)
        tcr_neg_seqs_batch = tcr_seqs[tcr_neg_ids.cpu()].to(device)
        epi_seqs_batch_mask = epi_seqs_mask[(epi_ids).cpu()].to(device)
        tcr_pos_seqs_batch_mask = tcr_seqs_mask[tcr_pos_ids.cpu()].to(device)
        tcr_neg_seqs_batch_mask = tcr_seqs_mask[tcr_neg_ids.cpu()].to(device)
        # epi_seq_emb = self.seq_embedding1(epi_seqs_batch)
        # tcr_pos_seq_emb = self.seq_embedding2(tcr_pos_seqs_batch)
        # tcr_neg_seq_emb = self.seq_embedding2(tcr_neg_seqs_batch)
        pref_idx = torch.LongTensor([[i for i in range(100)] for j in range(epi_seqs_batch.shape[0])]).to(device)
        pref_emb = self.pref_embedding(pref_idx)
        pos_pref = self.bind_layer(tcr_pos_seqs_batch, epi_seqs_batch, tcr_pos_seqs_batch_mask, epi_seqs_batch_mask)
        neg_pref = self.bind_layer(tcr_neg_seqs_batch, epi_seqs_batch, tcr_neg_seqs_batch_mask, epi_seqs_batch_mask)
        all_embed = self.calc_cf_embeddings()
        epi_embed = self.BN(all_embed[epi_ids + self.n_entities])
        tcr_pos_embed = self.BN(all_embed[tcr_pos_ids])
        tcr_neg_embed = self.BN(all_embed[tcr_neg_ids])

        r_mul_h = self.pref_M1(epi_embed)
        r_mul_pos_t = self.pref_M2(tcr_pos_embed)
        r_mul_neg_t = self.pref_M2(tcr_neg_embed)
        y_true = torch.FloatTensor([1 for i in range(tcr_pos_ids.shape[0])] + [0 for i in range(tcr_neg_ids.shape[0])]).to(device)
        pos_score = torch.pow(r_mul_h + pos_pref - r_mul_pos_t, 2)
        neg_score = torch.pow(r_mul_h + neg_pref - r_mul_neg_t, 2)
        score = torch.cat((pos_pref, neg_pref), dim=0)
        pos_emb = torch.cat((epi_embed, tcr_pos_embed), dim=-1)
        neg_emb = torch.cat((epi_embed, tcr_neg_embed), dim=-1)
        emb = torch.cat((pos_emb, neg_emb), dim=0)
        emb = self.BN2(emb)
        score = torch.cat((score, emb), dim=-1)
        score = self.Prediction(score).view(-1)

        num_positive_samples = (y_true == 1).sum()
        num_negative_samples = (y_true == 0).sum()
        weight_factor = num_negative_samples.float() / num_positive_samples.float()
        pos_weight = torch.ones([y_true.size(0)], device=device) * weight_factor
        train_loss = F.binary_cross_entropy_with_logits(score, y_true, pos_weight=pos_weight)
        aucm_module = AUCMLoss(device=self.device)
        aucm_loss = aucm_module(torch.sigmoid(score), y_true)

        pos_score = torch.sum(pos_score, dim=1)
        neg_score = torch.sum(neg_score, dim=1)

        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        loss = train_loss + cf_loss
        return loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                                                # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                                                           # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_epi_embed(h)                                             # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_epi_embed(pos_t)                                     # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_epi_embed(neg_t)                                     # (kg_batch_size, embed_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)                       # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)

        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        kg_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_epi_embed.weight[h_list]
        t_embed = self.entity_epi_embed.weight[t_list]

        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list

    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def calc_score_pn(self, epi_ids, tcr_ids):
        """
        epi_ids:  (n_epis)
        tcr_ids:  (n_tcrs)
        """
        device = self.device
        epi_seqs_batch = epi_seqs[(epi_ids).cpu()].to(device)
        tcr_seqs_batch = tcr_seqs[tcr_ids.cpu()].to(device)
        epi_seqs_batch_mask = epi_seqs_mask[(epi_ids).cpu()].to(device)
        tcr_seqs_batch_mask = tcr_seqs_mask[tcr_ids.cpu()].to(device)
        pref_idx = torch.LongTensor([[i for i in range(100)] for j in range(tcr_ids.shape[0])]).to(device)
        pref_emb = self.pref_embedding(pref_idx)
        pref = self.bind_layer(tcr_seqs_batch, epi_seqs_batch, tcr_seqs_batch_mask, epi_seqs_batch_mask)
        all_embed = self.calc_cf_embeddings()
        epi_embed = self.BN(all_embed[epi_ids+self.n_entities])
        tcr_embed = self.BN(all_embed[tcr_ids])
        tcr_emb = torch.cat((epi_embed, tcr_embed), dim=-1)
        pos_emb = self.BN2(tcr_emb)
        pos_pref = torch.cat((pref, pos_emb), dim=-1)
        score = self.Prediction(pos_pref)
        return score.view(-1)

    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict_pn':
            return self.calc_score_pn(*input)


