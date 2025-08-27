import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from agg import LocalAggregator, GlobalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F

class FindSimilarIntentSess(nn.Module):
    def __init__(self, hidden_size):

        super(FindSimilarIntentSess, self).__init__()
        self.hidden_size = hidden_size  
        self.neighbor_n = 5  
        self.dropout40 = nn.Dropout(0.40)  

    def compute_sim(self, sess_emb):

        fenzi = torch.matmul(sess_emb, sess_emb.permute(1, 0))  
        fenmu_l = torch.sum(sess_emb * sess_emb + 0.000001, 1) 
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)  
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))  
        cos_sim = fenzi / fenmu  
        cos_sim = nn.Softmax(dim=-1)(cos_sim)  
        return cos_sim

    def forward(self, sess_emb):

        k_v = self.neighbor_n  
        cos_sim = self.compute_sim(sess_emb)  
        if cos_sim.size()[0] < k_v:  
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)  
        cos_topk = nn.Softmax(dim=-1)(cos_topk)  
        sess_topk = sess_emb[topk_indice]  

        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.hidden_size)
        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)  
        neighbor_sess = self.dropout40(neighbor_sess)  
        return neighbor_sess


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: (B, C, L + padding)
        return x[:, :, :-self.chomp_size].contiguous()

class TCN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.1):
        super().__init__()
        dilation = 1
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding,
                      dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.net(x)

class MLTCN(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.5):  # 0.5
        super().__init__()

        self.branches = nn.ModuleList([
            TCN(in_channels, out_channels, k, dropout)
            for k in (3, 4, 5)
        ])

        self.q2 = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):

        B, _, L = x.size()
        vs = [branch(x) for branch in self.branches]
        v = torch.stack(vs, dim=1)
        v_t = v.permute(0, 1, 3, 2)
        scores = torch.einsum('c,bmlc->bml', self.q2, v_t)
        alpha = F.softmax(scores, dim=1)  # (B, M, L)
        weighted = (alpha.unsqueeze(-1) * v_t).sum(dim=1)  # (B, L, C)

        return weighted.permute(0, 2, 1)

'''
class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(n_outputs)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(n_outputs)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.bn1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2, self.bn2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.9):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 3 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
'''
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.W_a = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_a = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W_a.size(0))
        self.W_a.data.uniform_(-stdv, stdv)
        self.V_a.data.uniform_(-stdv, stdv)

    def forward(self, h):
        scores = torch.tanh(h @ self.W_a)  
        attention_weights = torch.softmax(scores @ self.V_a, dim=1)  
        context = attention_weights.unsqueeze(2) * h  
        return context.sum(dim=1), attention_weights  


class TimeGRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TimeGRUWithAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attention = SelfAttention(hidden_size)

    def forward(self, x, h_0=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)

        if h_0 is None:
            h_0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)  # (num_layers, batch_size, hidden_size)

        out, h_n = self.gru(x, h_0)  # out shape: (batch_size, seq_len, hidden_size)
        context, attention_weights = self.attention(out)
        all_h_weighted = out * attention_weights.unsqueeze(2)

        return all_h_weighted


batch_size = 100
input_size = 100
hidden_size = 100


class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        self.FindNeighbor = FindSimilarIntentSess(self.dim)

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.nei = nn.Linear(self.dim, self.dim)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden,hidden2, mask, time1, time2):
        mask = mask.float().unsqueeze(-1)
        len = hidden.shape[1]
        modelTCN = MLTCN(100, 100).to(hidden.device)
        hidden1 = hidden2.transpose(1, 2)
        nh = modelTCN(hidden1)
        nh1 = nh.transpose(1, 2)
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.sigmoid(self.glu1(nh1) + self.glu2(hs))
        #nh = torch.sigmoid(self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)
        neighbor_sess = self.FindNeighbor(select)
        select = select + 1.5 * neighbor_sess
        # torch.nn.utils.clip_grad_norm_(modelLSTM.parameters(), max_norm=1.0)
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores

    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        h_local = self.local_agg(h, adj, mask_item)

        # global
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)

        # mean
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)

        # sum
        # sum_item_emb = torch.sum(item_emb, 1)

        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop + 1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)

        # combine
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        output = h_local + h_global

        return output


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, alias_inputs_fw, adj, items, mask, targets, inputs, time1, time2 = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    alias_inputs_fw = trans_to_cuda(alias_inputs_fw).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    time1 = trans_to_cuda(time1).float()
    time2 = trans_to_cuda(time2).float()

    hidden = model(items, adj, mask, inputs)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    get1 = lambda index: hidden[index][alias_inputs_fw[index]]
    seq_hidden_fw = torch.stack([get1(i) for i in torch.arange(len(alias_inputs_fw)).long()])
    return targets, model.compute_scores(seq_hidden,seq_hidden_fw, mask, time1, time2)


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    for data in test_loader:
        targets, scores = forward(model, data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)

    return result