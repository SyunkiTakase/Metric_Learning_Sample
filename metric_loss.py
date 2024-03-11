import torch
import torch.nn.functional as F

def _pairwise_distances(embeddings, squared=False):

    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    distances[distances < 0] = 0

    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 -mask) * torch.sqrt(distances)

    return distances

def _get_anchor_positive_triplet_mask(label):

    indices_equal = torch.eye(label.size(0), device=label.device).bool()
    indices_not_equal = ~indices_equal

    label_equal = label.unsqueeze(0) == label.unsqueeze(1)

    return label_equal & indices_not_equal

def _get_triplet_mask(label):

    indices_equal = torch.eye(label.size(0), device=label.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = label.unsqueeze(0) == label.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_label = ~i_equal_k & i_equal_j

    return valid_label & distinct_indices

def _get_anchor_negative_triplet_mask(label):

    return ~(label.unsqueeze(0) == label.unsqueeze(1))


def contrastive_loss(output, label, margin):
    eps = 1e-9

    pairwise_distances = torch.cdist(output, output, p=2)
    pos_mask = label.unsqueeze(0) == label.unsqueeze(1)
    neg_mask = ~pos_mask

    pos_distances = pairwise_distances[pos_mask]
    neg_distances = pairwise_distances[neg_mask]

    num_pos_pairs = len(pos_distances)
    num_neg_pairs = len(neg_distances)

    if num_pos_pairs == 0 or num_neg_pairs == 0:
        return torch.tensor(0.0)

    loss = 0.5 * (
        (pos_distances.pow(2) / 2).mean() +  
        (F.relu(margin - neg_distances + eps).pow(2) / 2).mean() 
    )

    return loss


def triplet_loss(embeddings, margin, squared=False):

    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    triplet_loss = F.relu(triplet_loss)

    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)

    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss


def batch_hard_triplet_loss(label, embeddings, margin, squared=False):

    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(label).float()

    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(label).float()

    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    return triplet_loss


def batch_all_triplet_loss(label, embeddings, margin, _, __, squared=False):

    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    mask = _get_triplet_mask(label)
    triplet_loss = mask.float() * triplet_loss

    triplet_loss = F.relu(triplet_loss)

    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss 



