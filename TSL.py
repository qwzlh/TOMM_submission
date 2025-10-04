import torch
import torch.nn.functional as F
import random
from torch.distributions.exponential import Exponential
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma

def random_pick_gamma(values, indices, k, n):
    # 设置伽马分布的参数
    alpha = 2.0  # 形状参数
    beta = k / 4  # 比例参数

    gamma_dist = Gamma(torch.tensor([alpha]), torch.tensor([beta]))

    # # 生成权重
    weights = gamma_dist.sample(torch.Size([k])).squeeze()
    weights = torch.where(weights < k, weights, torch.tensor(k - 1e-6))  # 确保权重在 0 到 k 的范围内
    weights = 1 / (1 + weights)  # 转换权重，使接近 k 的索引权重更小
    weights = weights / weights.sum()  # 归一化以形成概率分布

    # 初始化选中元素和原始索引的列表
    selected_values_list = []
    selected_original_indices_list = []

    # 对每一行进行操作
    for i in range(values.shape[0]):
        # 随机选取 n 个元素
        # 生成权重
        # weights = gamma_dist.sample(torch.Size([k])).squeeze()
        # weights = torch.where(weights < k, weights, torch.tensor(k - 1e-6))  # 确保权重在 0 到 k 的范围内
        # weights = 1 / (1 + weights)  # 转换权重，使接近 k 的索引权重更小
        # weights = weights / weights.sum()  # 归一化以形成概率分布

        selected_indices = torch.multinomial(weights, n, replacement=True)

        # 获取选中元素的值和原始索引
        selected_values = values[i, selected_indices]
        selected_original_indices = indices[i, selected_indices]

        # 将选中的元素添加到列表中
        selected_values_list.append(selected_values)
        selected_original_indices_list.append(selected_indices)

    # 将列表转换为张量
    selected_values_tensor = torch.stack(selected_values_list)
    selected_original_indices_tensor = torch.stack(selected_original_indices_list)

    return selected_values_tensor, selected_original_indices_tensor


def random_pick(values, indices, k, n):
    # 创建一个高斯分布，平均值接近0，标准差小
    gaussian_dist = Normal(torch.tensor([0.0]), torch.tensor([k / 10]))

    # 生成权重
    weights = gaussian_dist.sample(torch.Size([k])).squeeze()
    weights = torch.abs(weights)  # 取绝对值，因为高斯分布是对称的
    weights = torch.where(weights < k, weights, torch.tensor(k - 1e-6))  # 确保权重在 0 到 k-1 的范围内
    weights = 1 / (1 + weights)  # 转换权重，使得接近0的索引权重更大
    weights = weights / weights.sum()  # 归一化以形成概率分布

    # 初始化选中元素和原始索引的列表
    selected_values_list = []
    selected_original_indices_list = []

    # 对每一行进行操作
    for i in range(values.shape[0]):
        # 随机选取 n 个元素
        selected_indices = torch.multinomial(weights, n, replacement=False)

        # 获取选中元素的值和原始索引
        selected_values = values[i, selected_indices]
        selected_original_indices = indices[i, selected_indices]

        # 将选中的元素添加到列表中
        selected_values_list.append(selected_values)
        selected_original_indices_list.append(selected_original_indices)

    # 将列表转换为张量
    selected_values_tensor = torch.stack(selected_values_list)
    selected_original_indices_tensor = torch.stack(selected_original_indices_list)

    return selected_values_tensor, selected_original_indices_tensor


def random_topk(tensor, k_max):
    topk_results = torch.zeros(tensor.shape[0], dtype=tensor.dtype, device=tensor.device)
    for i in range(tensor.size(0)):
        k = random.randint(100, k_max)  # 随机生成k，范围在1到125之间
        values, indices = torch.topk(tensor[i], k, largest=False)
        topk_results[i] = values.sum(dim=-1, keepdim=False).item() / k
    loss = topk_results.mean()
    return loss



def targeted_supervised_contrastive_loss_batch(features1, features2, labels1, labels2, tau=0.07, topk=None, cross_TSL=False, randomk=False, rp=False, intro_mask=False):
    """
    Batch processing version of Targeted Supervised Contrastive Loss with dynamic 'k' value.

    :param features: Tensor of shape (batch_size, feature_dim), the feature representations of the samples.
    :param labels: Tensor of shape (batch_size,), the labels of the samples.
    :param tau: Temperature parameter for scaling the logits.
    :param lambda_val: Lambda parameter to balance the two terms in the loss.
    :param k: (Optional) Number of positive samples to consider. If None, use all positive samples.
    :return: Loss value.
    """
    
    # features1 = features1.to(torch.float32)
    # features2 = features2.to(torch.float32)
    features1 = F.normalize(features1, dim=-1)
    features2 = F.normalize(features2, dim=-1)
    batch_size = features1.size(0)

    # Calculate similarities (batch_size x batch_size)
    similarities = torch.mm(features1, features2.T) / tau
    similarities = torch.abs(similarities)
    similarities = torch.exp(similarities)

    # Mask for selecting positive samples
    labels1 = labels1.view(-1, 1)
    labels2 = labels2.view(-1, 1)
    mask = torch.eq(labels1, labels2.T)

    if intro_mask:
        mask_incls = ~mask
        mask_incls.fill_diagonal_(True)
        similarities = similarities * mask_incls.to(similarities.dtype) 
    # print(mask)
    # Exclude self-similarity
    # if not cross_TSL:
    #     mask.fill_diagonal_(0)
    #     mask2 = torch.ones(size=similarities.size(), device=similarities.device, dtype=similarities.dtype)
        # mask2.fill_diagonal_(0)
        # similarities = similarities * mask2

    # Determine k dynamically if not provided
    if topk is None:
        k = mask.sum(dim=1)  # Number of positives for each sample
    else:
        k = torch.tensor([topk] * batch_size).cuda()  # Use the same k for all samples

    # Positive and negative samples
    # pos_samples = mask.to(similarities.dtype) * similarities
    # neg_samples = similarities

    # Select top k positive values for each row
    # topk_vals, _ = torch.topk(pos_samples, k=k.max().item(), largest=True)

    # Log-sum-exp for positive and negative samples
    # lse_neg = neg_samples.sum(dim=-1, keepdim=True)
    # topk_vals = torch.log(topk_vals / lse_neg + 1e-10)
    
    # lse_neg = torch.clamp(lse_neg, min=1e-10)

    # Loss calculation
    topk_vals =  F.normalize(similarities, dim=-1, p=1).log() * mask.to(similarities.dtype)
    if topk is not None:
        topk_vals = - topk_vals
        inf_value = torch.tensor(float('inf'), dtype=topk_vals.dtype)
        topk_vals = torch.where(topk_vals == 0, inf_value, topk_vals)
        if not randomk:
            topk_vals, tmp = torch.topk(topk_vals, k=k.max().item(), largest=False, dim=-1)
            if rp:
                topk_vals, tmp = random_pick_gamma(topk_vals, tmp, topk, 100)
            loss = torch.mean(topk_vals.sum(dim=-1) / k)
        else:
            loss = random_topk(topk_vals, topk)
        return loss



    loss = -torch.mean(topk_vals.sum(dim=-1) / k)

    return loss


def TSL_loss(visual_features, semantic_features, labels_visual, labels_semantic, topk=None, randomk=False, tau=0.07, rp=True):
    visual_features_tmp = visual_features.reshape(5,-1,visual_features.shape[-1])
    visual_features_tmp = visual_features_tmp[:, :5, :].contiguous().reshape(-1, visual_features_tmp.shape[-1])
    labels_visual_tmp = torch.arange(5, dtype=torch.long).cuda().unsqueeze(-1)
    labels_visual_tmp = labels_visual_tmp.repeat(1, 5)
    labels_visual_tmp = labels_visual_tmp.reshape(-1)
    self_contrastive = targeted_supervised_contrastive_loss_batch(visual_features, visual_features, labels_visual, labels_visual, tau=tau, topk=1, intro_mask=False)
    cross_contrastive = targeted_supervised_contrastive_loss_batch(visual_features, semantic_features, labels_visual, labels_semantic, tau=tau, topk=topk, cross_TSL=True, randomk=randomk, rp=rp)
    se_self_contrastive = targeted_supervised_contrastive_loss_batch(semantic_features, semantic_features, labels_semantic, labels_semantic,tau=tau, topk=1, intro_mask=False)
    return self_contrastive, cross_contrastive, se_self_contrastive