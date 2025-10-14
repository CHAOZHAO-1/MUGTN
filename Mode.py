
import torch
import torch.nn.functional as F
import torch.nn as nn



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div( torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter( torch.ones_like(mask),  1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device),  0  )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def mse_loss(p, alpha, c):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = 1
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C



class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, num_classes=10):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=32, stride=1),  # 8, 20449, 32
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),  # 8, 20448, 31
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=8, stride=1),  # 16, 20441, 24
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),  # 16, 20440, 23
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1),  # 32, 20438, 21
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),  # 32, 20437, 20
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, stride=1),  # 32, 20435, 18
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4)  # 32, 4, 4
        )


    def forward(self, x):

        x = x.unsqueeze(1)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)

        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)


        x = x.view(x.size(0), -1)

        # x = self.layer5(x)

        # x = self.fc(x)

        return x




class ProjectHead(nn.Module):
    def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, feat):
        feat = F.normalize(self.head(feat), dim=1)
        return feat




class MUGTN(nn.Module):

    def __init__(self, num_classes=31):
        super(MUGTN, self).__init__()

        self.sharedNet1 = CNN()
        self.sharedNet2 = CNN()

        self.num_class=num_classes

        self.cls_fc1 = nn.Linear(128, num_classes)
        self.cls_fc2 = nn.Linear(128, num_classes)
        self.cls_fc3 = nn.Linear(256, num_classes)

        self.proj_1 = ProjectHead(input_dim=256, hidden_dim=256, out_dim=256)
        self.proj_2 = ProjectHead(input_dim=256, hidden_dim=256, out_dim=256)

        self.trans_1 = ProjectHead(input_dim=128, hidden_dim=128, out_dim=128)
        self.trans_2 = ProjectHead(input_dim=128, hidden_dim=128, out_dim=128)


        self.contrast = SupConLoss()

    def forward(self, vibration, sound, cls_label, flag=0):


        if flag==0:  ### 场景1

            vibration_feature = self.sharedNet1(vibration)
            sound_feature = self.sharedNet2(sound)

            # 将振动和声音特征拼接
            combined_features = torch.cat((vibration_feature, sound_feature), dim=1)

            # 分类预测
            pred1 = self.cls_fc1(vibration_feature)
            pred2 = self.cls_fc2(sound_feature)

            fake_vib = self.trans_1(sound_feature)
            fake_aud = self.trans_2(vibration_feature)

            trans_loss = F.mse_loss(fake_vib, vibration_feature) + \
                         F.mse_loss(fake_aud, sound_feature)

            # 生成随机匹配的特征
            random_vib_list = []
            random_aud_list = []
            class_list = []

            for class_idx in range(self.num_class):
                class_mask = (cls_label == class_idx)
                vib_class = vibration_feature[class_mask]
                aud_class = sound_feature[class_mask]
                temp_class = cls_label[class_mask]

                if vib_class.size(0) > 0 and aud_class.size(0) > 0:
                    random_vib = vib_class[torch.randperm(vib_class.size(0))]
                    random_aud = aud_class[torch.randperm(aud_class.size(0))]

                    random_vib_list.append(random_vib)
                    random_aud_list.append(random_aud)
                    class_list.append(temp_class)

            random_vib = torch.cat(random_vib_list, dim=0)
            random_aud = torch.cat(random_aud_list, dim=0)
            random_class_label = torch.cat(class_list, dim=0)

            # 生成随机匹配的拼接特征
            random_match = torch.cat((random_vib, random_aud), dim=1)

            # 计算投影后的特征
            normal_proj = self.proj_1(combined_features)
            random_proj = self.proj_2(random_match)

            # 合并正常和随机匹配的投影特征
            emd_proj = torch.cat((normal_proj, random_proj), dim=0)
            label = torch.cat((cls_label, random_class_label), dim=0)

            # 计算对比学习损失
            loss_contrast = self.contrast(emd_proj.unsqueeze(1), label)

            # 计算融合后的分类预测
            pred3 = self.cls_fc3(normal_proj)

            return pred1, pred2, pred3, loss_contrast,trans_loss



        if flag==1:## vib is missing

            sound_feature = self.sharedNet2(sound)
            fake_vib = self.trans_1(sound_feature)

            # 将振动和声音特征拼接
            combined_features = torch.cat((fake_vib, sound_feature), dim=1)

            # 分类预测
            pred1 = self.cls_fc1(fake_vib)
            pred2 = self.cls_fc2(sound_feature)


            # 计算投影后的特征
            normal_proj = self.proj_1(combined_features)


            # 计算融合后的分类预测
            pred3 = self.cls_fc3(normal_proj)

            return pred1, pred2, pred3

        if flag == 2:  ## aud is missing

            vibration_feature = self.sharedNet1(vibration)
            fake_aud = self.trans_2(vibration_feature)


            # 将振动和声音特征拼接
            combined_features = torch.cat((vibration_feature, fake_aud), dim=1)

            # 分类预测
            pred1 = self.cls_fc1(vibration_feature)
            pred2 = self.cls_fc2(fake_aud)


            # 计算投影后的特征
            normal_proj = self.proj_1(combined_features)

            # 计算融合后的分类预测
            pred3 = self.cls_fc3(normal_proj)

            return pred1, pred2, pred3






