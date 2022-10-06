DEBUG=False

from ast import Num
import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb

if not DEBUG:
    from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

    
    class L2ClearBatchLoss(nn.Module):
        def __init__(self,alter_norm=False):
            super(L2ClearBatchLoss, self).__init__()
            self.alter_norm = alter_norm
            
        def forward(self, proposals, relation_logits, rel_pair_idxs, rel_labels):
            sub_labels_list=[]
            obj_labels_list=[]

            for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
                labels = proposal.get_field("labels")
                sub_labels = labels[rel_pair_idx[:, 0]]
                obj_labels = labels[rel_pair_idx[:, 1]]
                sub_labels_list.append(sub_labels)
                obj_labels_list.append(obj_labels)

            sub_labels=torch.cat(sub_labels_list, dim=0)
            obj_labels=torch.cat(obj_labels_list, dim=0)
            matched_mask = (sub_labels[:, None] == sub_labels[None, :]) & (obj_labels[:, None] == obj_labels[None, :])
            rel_scores = F.softmax(torch.cat(relation_logits, dim=0), dim=1)
            rel_labels = torch.cat(rel_labels, dim=0)
            loss = -calculateL2Norm(matched_mask, rel_scores, rel_labels, alter_norm=self.alter_norm)
            return loss

else:
    def test():
        rel_scores = torch.rand((10, 20))
        matched_mask = torch.BoolTensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        ])
        # [[0],[1,3,8],[2,5],[4,6,7,9]]

        rel_labels=torch.zeros(10, dtype=torch.int64)
        rel_labels[1]=1
        rel_labels[4]=2
        rel_labels[8]=3
        # rel_labels[2]=4
        # rel_labels[5]=4
        rel_labels[7]=5
        rel_labels[0]=1

        ind1 = torch.LongTensor([1,8])
        ind2 = torch.LongTensor([2,5])
        ind3 = torch.LongTensor([4,7])

# def calculateL2Norm(matched_mask, rel_scores, rel_labels,alter_norm=False):
#     matched_mask = matched_mask & (rel_labels[None,:]>0) & (rel_labels[:,None]>0)
#     matched_num = matched_mask.float().sum(1)
#     select_ind = torch.nonzero(matched_num > 1).view(-1)
#     if select_ind.shape[0]==0:
#         return torch.FloatTensor([0.]).to(rel_scores).squeeze()
#     have_seen = []
#     mix_norms = []
#     for ind in select_ind.tolist():
#         if ind in have_seen:
#             continue
#         set_index = torch.nonzero(matched_mask[ind]).view(-1)
#         if not alter_norm:           
#             mix_norm = rel_scores[set_index].pow(2.0).sum(0).sqrt().sum()/len(set_index)
#         else:
#             mix_norm = rel_scores[set_index].norm('nuc')/len(set_index)
#         mix_norms.append(mix_norm)
#         have_seen = have_seen + set_index.tolist()
    
#     return sum(mix_norms)/len(mix_norms)

def calculateL2Norm(matched_mask, rel_scores, rel_labels,alter_norm=False):
#    Nuclear norm may more stable than L_21 in Transformer-based model
    matched_num = matched_mask.float().sum(1)
    select_ind = torch.nonzero(matched_num > 1).view(-1)
    if select_ind.shape[0]==0:
        return torch.FloatTensor([0.]).to(rel_scores).squeeze()
    have_seen = []
    mix_norms = []
    for ind in select_ind.tolist():
        if ind in have_seen:
            continue
        set_index = torch.nonzero(matched_mask[ind]).view(-1)
        if not alter_norm:           
            list_svd,_ = torch.sort(rel_scores[set_index].pow(2.0).sum(0).sqrt(), descending=True)
            nums = min(rel_scores.shape[0],rel_scores.shape[1])
            mix_norm = torch.sum(list_svd[:nums])
#           mix_norm = rel_scores[set_index].pow(2.0).sum(0).sqrt().sum()/len(set_index)
        else:
            mix_norm = rel_scores[set_index].norm('nuc')/len(set_index)
        mix_norms.append(mix_norm)
        have_seen = have_seen + set_index.tolist()
    
    return sum(mix_norms)/len(mix_norms)


if __name__ == "__main__":
    test()