import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
         Input: features: Features of the input sample, size [batch_size, hidden_dim].
            features: features of the input samples, size [batch_size, hidden_dim].
            labels: ground truth labels for each sample, size [batch_size].
            mask: mask for comparison learning, size [batch_size, batch_size], if samples i and j belong to the same label, then mask_{i,j}=1
        Output.
            Loss value

        """
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # About the labels parameter
        if labels is not None and mask is not None:  # labels and masks can't define values at the same time, because if there's a label, then the mask is needed to get it based on the label
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  # If there are no labels and there is no mask, it is unsupervised learning, and the mask is a matrix with diagonal 1, indicating that (i,i) belongs to the same class
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        elif labels is not None:  #  If labels are given, mask is obtained based on labels, mask_{i,j}=1 when labels of two samples i,j are equal
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().cuda()
        else:
            mask = mask.float().cuda()

        anchor_dot_contrast = torch.div(
            F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0),dim=2),
            self.temperature)   # Calculate the cosine# similarity between two samples.
        # for numerical stability

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(mask).cuda() - torch.eye(batch_size).cuda()
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        num_positives_per_row = torch.sum(positives_mask, axis=1)
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]

        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss





