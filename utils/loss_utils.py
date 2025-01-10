import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    def __init__(self, perceptual_loss_type, perceptual_loss_layer, perceptual_loss_feature_dim):
        super(PerceptualLoss, self).__init__()
        self.perceptual_loss_type = perceptual_loss_type
        self.perceptual_loss_layer = perceptual_loss_layer
        self.perceptual_loss_feature_dim = perceptual_loss_feature_dim

    def forward(self, x, y):
        pass


class Cosine_loss(nn.Module):
    def __init__(self):
        super(Cosine_loss, self).__init__()
        self.logloss = nn.BCELoss()  # BEC的输入变量的取值范围[0,1]
        
    def forward(self, f_imbed, m_imbed, label):
        d = nn.functional.cosine_similarity(f_imbed, m_imbed)
        loss = self.logloss(d.unsqueeze(1), label)

        return loss