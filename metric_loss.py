import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Args:
            margin (float): マージン値
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # 特徴量間のペアワイズ距離の計算
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float() # 同じラベルのマスク（ポジティブ）
        negative_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float() # 異なるラベルのマスク（ネガティブ）

        positive_dist = pairwise_dist * positive_mask
        negative_dist = pairwise_dist * negative_mask

        # 各ペアの数をカウント
        num_pos_pairs = len(positive_dist)
        num_neg_pairs = len(negative_dist)

        if num_pos_pairs == 0 or num_neg_pairs == 0:
            return torch.tensor(0.0)

        # Contrastive Loss
        loss = 0.5 * (
            (positive_dist.pow(2) / 2).mean() +  
            (F.relu(self.margin - negative_dist + 1e-9).pow(2) / 2).mean() 
        )

        return loss

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, hard_triplets=False):
        """
        Args:
            margin (float): マージン値
            use_hard_triplets (bool): Hard Positive/Negativeを使用するかどうか
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.use_hard_triplets = hard_triplets

    def forward(self, embeddings, labels):
        # 特徴量間のペアワイズ距離の計算
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2) # [2 * batch, 2 * batch]

        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float() # 同じラベルのマスク（ポジティブ）
        negative_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float() # 異なるラベルのマスク（ネガティブ）
        # print('Pos mask:', positive_mask)
        # print('Neg mask:', negative_mask)

        if self.use_hard_triplets: # Hard PositiveおよびHard Negativeを選択
            positive_dist = pairwise_dist * positive_mask
            hardest_positive_dist, _ = positive_dist.max(dim=1) # 各アンカーに対する最も遠いポジティブ

            negative_dist = pairwise_dist + (1 - negative_mask) * 1e6 # 無効なネガティブに大きな値を設定
            hardest_negative_dist, _ = negative_dist.min(dim=1) # 各アンカーに対する最も近いネガティブ

            # Triplet Loss
            loss = torch.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            return loss.mean()

        else: # 全ペアを考慮
            positive_dist = pairwise_dist * positive_mask
            # print('Pos Dist:', positive_dist)
            negative_dist = pairwise_dist * negative_mask
            # print('Neg Dist:', negative_dist)

            # Triplet Loss
            triplet_loss = positive_dist.unsqueeze(2) - negative_dist.unsqueeze(1) + self.margin 
            triplet_loss = torch.relu(triplet_loss) # マージンに基づくReLU適用

            valid_triplets = positive_mask.unsqueeze(2) * negative_mask.unsqueeze(1) # 有効なTripletのマスク
            triplet_loss = triplet_loss * valid_triplets
            num_valid_triplets = valid_triplets.sum() + 1e-16 # 有効ペア数（ゼロ除算を防ぐ）
            loss = triplet_loss.sum() / num_valid_triplets
            return loss