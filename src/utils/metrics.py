import heapq
import numpy as np
import torch
import torchmetrics
from torchmetrics import Accuracy, Metric

def get_real_api_idx(labels):
    """
    :param labels: (batch_size, api_num)
    :return:
    """
    real_idx_list = []
    for label in labels:
        api_idx = []
        label_list = list(label)
        for idx, val in enumerate(label):
            if val == 1:
                api_idx.append(idx)
        real_idx_list.append(api_idx)
    return real_idx_list


def get_top_k_pred_val_and_index(preds, top_k):
    """
    :param preds: (batch_size, api_num)
    :param top_k: the length of the result
    :return:
    """
    pred_val_list = []
    pred_idx_list = []
    for pred in preds:
        pred_list = list(pred)
        max_index = list(map(pred_list.index, heapq.nlargest(top_k, pred_list)))
        pred_idx_list.append(max_index)
        pred_val_list.append([pred[idx] for idx in max_index])
    return pred_val_list, pred_idx_list


def calculate_precision(real_idx_list, pred_top_k_idx_list):
    """
    :param reals: (batch_size, api_num)
    :param preds: (batch_size, api_num)
    :param top_k: the length of the result
    :return:
    """
    batch_size = len(real_idx_list)
    top_k = len(pred_top_k_idx_list[0])
    total_pre = 0.0
    for i in range(batch_size):
        same_api_index = set(real_idx_list[i]) & set(pred_top_k_idx_list[i])
        pre = len(same_api_index) / top_k
        total_pre += pre
    precision = total_pre / batch_size
    return total_pre / batch_size


def calculate_recall(real_idx_list, pred_top_k_idx_list):
    batch_size = len(real_idx_list)
    total_recall = 0.0
    for i in range(batch_size):
        same_api_index = set(real_idx_list[i]) & set(pred_top_k_idx_list[i])
        recall = len(same_api_index) / len(real_idx_list[i])
        total_recall += recall
    return total_recall / batch_size


def calculate_NDCG(real_idx_list, pred_top_k_val_list, pred_top_k_idx_list):
    top_k = len(pred_top_k_idx_list[0])
    batch_size = len(real_idx_list)
    DCG = 0.0
    IDCG = 0.0
    for n in range(batch_size):
        # DCG
        val_index_tuple_list = list(zip(pred_top_k_val_list[n], pred_top_k_idx_list[n]))
        val_index_tuple_desc_sort_list = sorted(val_index_tuple_list, key=lambda x: x[0], reverse=True)
        for i in range(0, len(val_index_tuple_list)):
            rel_i = 1.0 if val_index_tuple_desc_sort_list[i][1] in real_idx_list[n] else 0.0
            DCG += rel_i / np.log2(i + 2)
        # IDCG
        for i in range(min(len(real_idx_list), top_k)):
            IDCG += 1.0 / np.log2(i + 2)
    return DCG / IDCG


def calculate_MAP(reals, preds, top_k):
    real_idx_list = get_real_api_idx(reals)
    pred_top_k_val_list, pred_top_k_idx_list = get_top_k_pred_val_and_index(preds, top_k)
    batch_size = len(real_idx_list)
    AP = 0.0
    for n in range(batch_size):
        val_index_tuple_list = list(zip(pred_top_k_val_list[n], pred_top_k_idx_list[n]))
        val_index_tuple_desc_sort_list = sorted(val_index_tuple_list, key=lambda x: x[0], reverse=True)
        numerator = 0.0
        denominator = 0.0
        for i in range(top_k):
            rel_i = 1.0 if val_index_tuple_desc_sort_list[i][1] in real_idx_list[n] else 0.0
            pred_top_i_index = list(map(lambda x: x[1], val_index_tuple_desc_sort_list))[: i + 1]
            same_index = set(real_idx_list[n]) & set(pred_top_i_index)
            pre_i = len(same_index) / (i + 1)
            numerator += rel_i * pre_i
            denominator += rel_i
        if denominator > 0.0:
            AP += numerator / denominator
    MAP = AP / batch_size
    return MAP


def retrieval_MAP(preds: torch.Tensor, targets: torch.Tensor, top_k: int):
    top_k_preds = []
    top_k_targets = []
    for i in range(preds.size(0)):
        pred = list(preds[i])
        target = list(targets[i])
        top_k_index = list(map(pred.index, heapq.nlargest(top_k, pred)))
        top_k_preds.append([pred[index] for index in top_k_index])
        top_k_targets.append([target[index] for index in top_k_index])
    top_k_preds = torch.tensor(top_k_preds, dtype=torch.float32)
    top_k_targets = torch.tensor(top_k_targets, dtype=torch.int64)
    mAP = torchmetrics.RetrievalMAP()
    return mAP(top_k_preds, top_k_targets, torch.zeros(size=top_k_preds.shape, dtype=torch.int64))


class Precision(Metric):
    r"""Computes precision.

    Args:
        top_k (int): consider only the top k elements for each query.
        propensity_score (List[float]): propensity score of API.s
        dist_sync_on_step (bool): If metric state should synchronize on forward(). (default: :obj:`False`)
    """
    def __init__(self, top_k, propensity_score=None, dist_sync_on_step=False):
        super(Precision, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = top_k
        self.propensity_score = propensity_score
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        top_k, indices = torch.topk(preds, self.top_k, dim=1)
        res = torch.zeros_like(preds).type_as(preds)
        res = res.scatter(1, indices, torch.ones_like(top_k).type_as(top_k))
        score = res * target

        if self.propensity_score is not None:
            score = score / self.propensity_score

        score = score.sum(dim=1)
        score = score / self.top_k
        self.score += score.sum()
        self.total += preds.size(0)

    def compute(self):
        return self.score / self.total


class NormalizedDCG(Metric):
    r"""Computes Normalized Discounted Cumulative Gain.

    Args:
        top_k (int): consider only the top k elements for each query.
        propensity_score (List[float]): propensity score of API.s
        dist_sync_on_step (bool): If metric state should synchronize on forward(). (default: :obj:`False`)
    """
    def __init__(self, top_k, propensity_score=None, dist_sync_on_step=False):
        super(NormalizedDCG, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = top_k
        self.propen_score = propensity_score
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        top_k, indices = torch.topk(preds, self.top_k, dim=1)
        res = torch.zeros_like(preds).type_as(preds)
        res = res.scatter(1, indices, torch.ones_like(top_k).type_as(top_k))
        score = res * target

        log_val = torch.log2(torch.arange(2, self.top_k + 2)).type_as(preds)
        res_log = torch.ones_like(preds).type_as(preds)  # prevent zero
        log_val = log_val.view(1, -1).repeat(preds.size(0), 1)
        res_log = res_log.scatter(1, indices, log_val)

        score = score / res_log
        if self.propen_score is not None:
            score = score / self.propen_score

        score = score.sum(dim=1)
        self.score += score.sum()
        self.total += preds.size(0)

    def compute(self):
        return self.score / self.total


if __name__ == '__main__':
    # reals = [[1, 0, 1, 1, 1], [1, 1, 0, 0, 0]]
    # reals = torch.tensor(reals, dtype=torch.int)
    # preds = [[0.1, 0.3, 0.0043, 0.132, 0.06], [0.466, 0.09, 0.39, 0.00001, 0.07]]
    # preds = torch.tensor(preds, dtype=torch.float32)
    # mAP = torchmetrics.RetrievalMAP()
    # print(mAP(preds, reals, torch.zeros(size=preds.shape, dtype=torch.int64)))
    target = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
    # accuracy = torchmetrics.Accuracy()
    # print(accuracy(preds, target))
