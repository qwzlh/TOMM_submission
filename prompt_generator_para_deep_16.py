import torch
import clip
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import torch.nn as nn
import os
import torchvision.transforms as transforms
import random
from clip_model.prompted_model import Visual_prompt, build_model
from PSG import PseudoSampleGenerator
import open_clip
from TSL import TSL_loss
from adapter import Adapter
from torch.cuda.amp import GradScaler, autocast

device = 'cuda'


# 两种prompt分开训练，好像不用

clip_dim_mapping = {"ViT-B/32":512, 'ViT-B/16':512, 'ViT-L/14':768, 'RN50':1024}


def random_aug_semantic(original_tensor, aug_num=1000):
    weights = torch.softmax(torch.rand(original_tensor.shape[1], aug_num) , dim=0).to(original_tensor.device)
    reshaped_tensor = original_tensor.permute(0, 2, 1)
    new_tensor = torch.matmul(reshaped_tensor, weights).permute(0, 2, 1)
    labels = torch.arange(5, dtype=torch.long).unsqueeze(-1)
    labels = labels.repeat(1, aug_num)
    return new_tensor, labels


def gamma_correction(x, gamma):
    minv = torch.min(x)
    x = x - minv

    maxv = torch.max(x)
    x = x / maxv

    x = x**gamma
    x = x * maxv + minv
    return x

def random_aug(x):
    # gamma correction
    if random.random() <= 0.3:
        gamma = random.uniform(1.0, 1.5)
        x = gamma_correction(x, gamma)
    # random erasing with mean value
    mean_v = tuple(x.view(x.size(0), -1).mean(-1))
    re = transforms.RandomErasing(p=0.5, value=mean_v)
    x = re(x)
    # color channel shuffle
    if random.random() <= 0.3:
        l = [0,1,2]
        random.shuffle(l)
        x_c = torch.zeros_like(x)
        x_c[l] = x
        x = x_c
    # horizontal flip or vertical flip
    if random.random() <= 0.5:
        if random.random() <= 0.5:
            x = torch.flip(x, [1])
        else:
            x = torch.flip(x, [2])
    # rotate 90, 180 or 270 degree
    if random.random() <= 0.5:
        degree = [90, 180, 270]
        d = random.choice(degree)
        x = torch.rot90(x, d//90, [1, 2])
    return x


def style_diversity_loss(prompt_feature):
    """
    Computes the style diversity loss.
    """
    # tensor1_expanded = prompt_feature.unsqueeze(2)  # 形状变为 [5, 20, 1, 512]
    # tensor2_expanded = prompt_feature.unsqueeze(1)
    # cosine_sim = F.cosine_similarity(tensor1_expanded, tensor2_expanded, dim=-1)
    tmp = F.normalize(prompt_feature, dim=-1, p=2)
    cosine_sim = torch.einsum('abc, adc -> abd', tmp, tmp)
    mask = torch.ones([prompt_feature.shape[0], prompt_feature.shape[1], prompt_feature.shape[1]], dtype=prompt_feature.dtype, device=prompt_feature.device)  # 先创建一个全为 1 的张量
    for i in range(cosine_sim.shape[-1]):
        mask[:, i, i] = 0 
    cosine_sim = torch.abs(cosine_sim * mask)
    loss = torch.mean(cosine_sim, dim=[-1, -2]).mean()
    return loss

def content_consistency_loss(style_content_feature, content_feature, class_label):
    """
    Computes the content consistency loss.
    """
    # style_content_feature = content_feature
    style_content_feature =  style_content_feature /  style_content_feature.norm(dim=-1, keepdim=True)
    content_feature = content_feature / content_feature.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_style_content_feature = style_content_feature @ content_feature.t()
    logits_content_feature = logits_style_content_feature.t()
    ground_truth = torch.tensor([class_label], dtype=torch.long, device='cuda')
    loss = F.cross_entropy(logits_style_content_feature,ground_truth)
    return loss

def semantic_consistency_loss(style_content_feature, content_feature):
    """
    Computes the content consistency loss.
    """
    n_way = 5
    n_samper_per_class = style_content_feature.shape[0] / n_way
    # style_content_feature = content_feature
    style_content_feature =  style_content_feature /  style_content_feature.norm(dim=-1, keepdim=True)
    content_feature = content_feature / content_feature.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_style_content_feature = style_content_feature @ content_feature.t()
    logits_content_feature = logits_style_content_feature.t()

    # build ground truth
    ground_truth = torch.zeros([int(n_samper_per_class * n_way)])
    for m in range(n_way):
        for n in range(int(n_samper_per_class)):
            ground_truth[m *  int(n_samper_per_class) + n] = m

    ground_truth = ground_truth.long().cuda()
    loss = F.cross_entropy(logits_style_content_feature,ground_truth)
    return loss

def total_prompt_loss(aug_img, aug_prompt, content_feas, semantic_feas, promp_semantic_feas, promp_semantic_labels, lam=1, use_ori_semantic_loss=False, support_labels=None, use_se_self_loss=True):
    """
    Computes the total prompt loss.
    """
    # ground = torch.arange(5, dtype=torch.long, device=semantic_feas.device).unsqueeze(-1).repeat(1, 5).reshape(-1)
    # semantic_feas = semantic_feas.unsqueeze(-2).repeat(1,5,1).reshape(-1, semantic_feas.shape[-1])
    Lstyle = style_diversity_loss(aug_prompt)
    # Lstyle = 2 * Lstyle
    L3 = semantic_consistency_loss(promp_semantic_feas, semantic_feas)

    use_ori_semantic_loss = False
    use_se_self_loss = False
    randomk = False

    if use_ori_semantic_loss:
        # assert False
        L_cross_contrastive = semantic_consistency_loss(aug_img, semantic_feas)
    else:
        # L_self_contrastive, L_cross_contrastive = TSL_loss(aug_img, promp_semantic_feas, support_labels, promp_semantic_labels)
        L_self_contrastive, L_cross_contrastive, L_se_self = TSL_loss(aug_img, promp_semantic_feas, support_labels, promp_semantic_labels, 300, randomk, tau=0.07)

    if use_ori_semantic_loss:
        # assert False
        Lprompt = Lstyle + L_cross_contrastive
    else:
        # Lprompt = Lstyle + L_cross_contrastive + L_self_contrastive
        Lprompt = Lstyle + L_cross_contrastive

        # Lprompt = Lstyle + L_cross_contrastive

    if use_se_self_loss:
        Lprompt = Lprompt + L_se_self
    Lprompt = Lprompt + 3 * L3
    # Lprompt = L_cross_contrastive
  
    return Lprompt, 3 * L3



def prompt_generator(clip_model, prompt_learner_shallow, prompt_learner_deep, support_sample_tmp, n_support, dim=512, use_deep_pt=False, n_way=5, prompt_learner_shallow_still=None, pseudo_img=None, support_img=None, prompt_learner_deep_still=None, dtype=None):
    support_img2 = support_img.reshape(-1, support_img.shape[-3], support_img.shape[-2], support_img.shape[-1])
    pseudo_img2 = pseudo_img.reshape(-1, pseudo_img.shape[-3], pseudo_img.shape[-2], pseudo_img.shape[-1])
    shallow_visual_prompt = prompt_learner_shallow.forward() # N, 77, 512

    shallow_visual_prompt = shallow_visual_prompt.reshape(n_way, -1, shallow_visual_prompt.shape[-2], shallow_visual_prompt.shape[-1])
    shallow_visual_prompt_tmp = shallow_visual_prompt.reshape(-1, shallow_visual_prompt.shape[-2], shallow_visual_prompt.shape[-1])
    shallow_visual_prompt = torch.cat([shallow_visual_prompt[:, :, i, :] for i in range(shallow_visual_prompt.size(2))], dim=2)
    if len(list(shallow_visual_prompt.shape)) == 3:
        shallow_visual_prompt = shallow_visual_prompt.unsqueeze(2)

    if prompt_learner_shallow_still is not None:
        shallow_visual_prompt_still0 = prompt_learner_shallow_still.forward()
        shallow_visual_prompt_still = shallow_visual_prompt_still0.expand(shallow_visual_prompt_tmp.shape[0], -1, -1)
        # shallow_visual_prompt_tmp = torch.cat([shallow_visual_prompt_tmp, shallow_visual_prompt_still], dim=1)
        shallow_visual_prompt_tmp = torch.cat([shallow_visual_prompt_still, shallow_visual_prompt_tmp], dim=1)
    else:
        shallow_visual_prompt_still0 = None

    if prompt_learner_deep_still is not None:    
            _, deep_pt = prompt_learner_deep_still.forward()
    else:
        deep_pt = None

    pseudo_img2 = pseudo_img2.to(dtype) 
    shallow_visual_prompt_tmp = shallow_visual_prompt_tmp.to(dtype) 
    deep_pt = deep_pt.to(dtype) 
    support_img2 = support_img2.to(dtype)
    if shallow_visual_prompt_still0 is not None:
        shallow_visual_prompt_still0 = shallow_visual_prompt_still0.to(dtype)
    ######################
    cc = 1
    x1 = torch.chunk(pseudo_img2, cc, dim=0)
    c1 = torch.chunk(shallow_visual_prompt_tmp, cc, dim=0)
    y2 = []
    for index, i in enumerate(x1): 
        y1 = clip_model.encode_image(x1[index], c1[index], deep_pt=deep_pt)
        y2.append(y1)
    aug_img = torch.cat(y2, dim=0)
    ######################
    aug_img2 = clip_model.encode_image(support_img2, shallow_visual_prompt_still0, deep_pt=deep_pt)
    aug_img = aug_img.reshape(5, -1, aug_img.shape[-1])
    aug_img2 = aug_img2.reshape(5, -1, aug_img2.shape[-1])
    aug_img = torch.cat([aug_img2, aug_img], dim=1).reshape(-1, aug_img.shape[-1])
    # aug_img = aug_img.to(torch.float32)
    # shallow_visual_prompt = shallow_visual_prompt.to(torch.float32)
    
    # aug_prompt_fea = clip_model.encode_image(None, shallow_visual_prompt.reshape(-1, shallow_visual_prompt.shape[-2], shallow_visual_prompt.shape[-1]), deep_pt=deep_pt)
    # aug_prompt_fea = aug_prompt_fea.reshape(n_way, -1, aug_prompt_fea.shape[-1])
    # style_content_feas = text_encoder(style_content_vec, tokenized_prompts_content) # prompt之间的diversity后面再加，想想怎么加
    return aug_img, shallow_visual_prompt.squeeze(-2)


def vp_generator(support_img, clip_model, clip_model_name, layers, num_token_shallow=1, num_token_deep=5, use_deep_pt=True, semantic_features=None, promp_semantic_features=None, promp_semantic_labels=None, use_semantic_adapter=True, num_token_shallow_still=4, dtype=None):  # layers, deep_prompt_tuning insert layers
    support_img = support_img.to(dtype)
    semantic_features3 = semantic_features.unsqueeze(1)
    semantic_features = semantic_features.to(dtype)
    promp_semantic_features = promp_semantic_features.to(dtype)
    promp_semantic_features = promp_semantic_features.reshape(5,-1,promp_semantic_features.shape[-1])
    promp_semantic_features = torch.cat([semantic_features3, promp_semantic_features], dim=1)
    promp_semantic_features, promp_semantic_labels = random_aug_semantic(promp_semantic_features)
    promp_semantic_features = promp_semantic_features.reshape(-1, promp_semantic_features.shape[-1])
    promp_semantic_labels = promp_semantic_labels.reshape(-1).cuda()
    promp_semantic_features_tmp = promp_semantic_features.to(dtype)
    n_way = support_img.shape[0]
    n_support = support_img.shape[1]
    dim = clip_dim_mapping[clip_model_name]

    
    lr = 0.001 # euro:0.001 0.0001
    # class_names = os.listdir('D:\work_code\dg\miro-main\miro-main\domainbed\scripts\PACS\cartoon')
    # if class_tokens is not None:
    #     class_tokens = clip.tokenize(class_names).cuda()
    # with torch.no_grad():
    #     # content_feas = clip_model.encode_text(class_tokens)
    #     support_img_tmp = support_img.reshape(-1, support_img.shape[-3], support_img.shape[-2], support_img.shape[-1])
    #     support_fea = clip_model.encode_image(support_img_tmp, None)
    #     support_fea = support_fea.reshape(support_img.shape[0],support_img.shape[1], support_fea.shape[-1])
    #     content_feas = support_fea.mean(dim=1, keepdim=False)

    train_clip = False
    aug_feas_out_all = []
    prompt_feas_out_all = []
    aug_prompt_feas = []
    # text_encoder.eval()
    use_deep = True
    use_shallow = False
    L = 55 # 60
    times = 4 # 24 for 1shot
    num_generate_sample = 5 * n_support * times  # *4
    num_token_shallow = 1
    data_generator = PseudoSampleGenerator(n_way, n_support, num_generate_sample)
    if use_shallow:
        prompt_learner_shallow_still = Visual_prompt(deep_pt=False, num_layers=layers, num_tokens=num_token_shallow_still, batch_size=1)
        prompt_learner_shallow_still.to(dtype)
        optimizer_shallow_still = optim.Adam(prompt_learner_shallow_still.parameters(), lr=lr) 
        prompt_learner_shallow_still.train()
        prompt_learner_shallow_still.cuda()
    else:
        prompt_learner_shallow_still = None
    if use_deep:
        prompt_learner_deep_still = Visual_prompt(deep_pt=True, num_layers=layers, num_tokens=5, batch_size=1) # 5 token
        prompt_learner_deep_still.to(dtype)
        optimizer_deep_still = optim.Adam(prompt_learner_deep_still.parameters(), lr=lr) 
        prompt_learner_deep_still.train()
        prompt_learner_deep_still.cuda()
    else:
        prompt_learner_deep_still = None

    if train_clip:
        optimizer_clip = optim.Adam(clip_model.parameters(), lr=1e-6) 

    prompt_learner_shallow = Visual_prompt(deep_pt=False, num_layers=layers, num_tokens=num_token_shallow, batch_size=num_generate_sample)
    prompt_learner_shallow.to(dtype)
    semantic_adapter = Adapter(promp_semantic_features.shape[-1])
    semantic_adapter.to(dtype)
      
    prompt_learner_shallow.train()
    prompt_learner_shallow.cuda()
    semantic_adapter.train()
    semantic_adapter.cuda()

    optimizer_shallow = optim.Adam(prompt_learner_shallow.parameters(), lr=lr) 
    optimizer_semantic = optim.Adam(semantic_adapter.parameters(), lr=0.01) #0.5
    # prompt_learner_deep =  Visual_prompt(deep_pt=use_deep_pt, num_layers=layers, num_tokens=num_token_deep)
    # pkl.dump(style_content_feas_out, open('./{}_style_vecs_out'.format(dataset_name), 'wb'))
    support_sample_tmp1 = data_generator.generate2(support_img, times=times) # [way, n_support+num_generate_per_class, 3, 224, 224]
    pseudo_imgs = support_sample_tmp1[:, support_img.shape[1]:, :, : ,:]
    pseudo_imgs = pseudo_imgs.to(dtype)
    # print(pseudo_imgs[0,0,:,:,:])
    support_labels = torch.arange(support_sample_tmp1.size(0)).unsqueeze(1).repeat(1, support_sample_tmp1.size(1))
    support_sample_tmp = support_sample_tmp1.reshape(-1, support_sample_tmp1.shape[-3], support_sample_tmp1.shape[-2], support_sample_tmp1.shape[-1])  # [num_generate_sample,]
    support_labels = support_labels.reshape(-1)
    support_labels = support_labels.cuda()
    support_img2 = support_img.reshape(-1, support_img.shape[-3], support_img.shape[-2], support_img.shape[-1])
    pseudo_img2 = pseudo_imgs.reshape(-1, pseudo_imgs.shape[-3], pseudo_imgs.shape[-2], pseudo_imgs.shape[-1])
    scaler = torch.cuda.amp.GradScaler()
    for j in range(L):
        with torch.cuda.amp.autocast():
            aug_img, aug_prompt_fea = prompt_generator(clip_model, prompt_learner_shallow, None, support_sample_tmp, n_support, dim, use_deep, prompt_learner_shallow_still=prompt_learner_shallow_still,
                                                    pseudo_img=pseudo_imgs, support_img=support_img,  prompt_learner_deep_still= prompt_learner_deep_still, dtype=dtype)
            # style_content_feas = style_content_feas.cuda()
            # content_feas = content_feas.cuda()
            if use_semantic_adapter:
                promp_semantic_features = semantic_adapter(promp_semantic_features_tmp)
                # promp_semantic_features = promp_semantic_features.float()
                # semantic_features2 = semantic_adapter(semantic_features)
                semantic_features2 = semantic_features
            else:
                semantic_features2 = semantic_features

            loss, L_se_self = total_prompt_loss(aug_img, aug_prompt_fea, None, semantic_features2, promp_semantic_features, promp_semantic_labels, support_labels=support_labels) 
        
        # print(loss.detach().item())
        optimizer_shallow.zero_grad()
        # optimizer_deep.zero_grad()
        if train_clip:
            optimizer_clip.zero_grad()
        if use_semantic_adapter:
            optimizer_semantic.zero_grad()
        if use_shallow:
            optimizer_shallow_still.zero_grad()
        if use_deep:
            optimizer_deep_still.zero_grad()
        if use_semantic_adapter:
            scaler.scale(loss).backward(retain_graph=True)
        else:
            scaler.scale(loss).backward()
        # print(prompt_learner.ctx.grad)
        scaler.step(optimizer_shallow)
        if use_shallow:
            scaler.step(optimizer_shallow_still)
        if use_deep:
            scaler.step(optimizer_deep_still)
        if train_clip:
            scaler.step(optimizer_clip)
        # optimizer_deep.step()
        if use_semantic_adapter:
            optimizer_semantic.zero_grad()
            scaler.scale(L_se_self).backward()
            scaler.step(optimizer_semantic)
        scaler.update()
    
    with torch.no_grad():
        shallow_visual_prompt = prompt_learner_shallow.forward()
        shallow_visual_prompt = shallow_visual_prompt.reshape(n_way, -1, shallow_visual_prompt.shape[-2], shallow_visual_prompt.shape[-1])
        shallow_visual_prompt_tmp = shallow_visual_prompt.reshape(-1, shallow_visual_prompt.shape[-2], shallow_visual_prompt.shape[-1])
        
        if use_shallow:
            shallow_visual_prompt_still_tmp = prompt_learner_shallow_still.forward()
            shallow_visual_prompt_still = shallow_visual_prompt_still_tmp.expand(shallow_visual_prompt_tmp.shape[0], -1, -1)
            # shallow_visual_prompt_tmp = torch.cat([shallow_visual_prompt_tmp, shallow_visual_prompt_still], dim=1)
            shallow_visual_prompt_tmp = torch.cat([shallow_visual_prompt_still, shallow_visual_prompt_tmp], dim=1)
        else:
            shallow_visual_prompt_still_tmp = None
            shallow_visual_prompt_still = None
      
        if use_deep:
            _, deep_pt = prompt_learner_deep_still.forward()
        else:
            deep_pt = None
        
       
        # zero_tmp = torch.zeros([n_way, n_support, shallow_visual_prompt.shape[-2], shallow_visual_prompt.shape[-1]], dtype=shallow_visual_prompt.dtype, device=shallow_visual_prompt.device)
        # shallow_visual_prompt_tmp2 = torch.cat([zero_tmp, shallow_visual_prompt], dim=1)
        # shallow_visual_prompt_tmp2 = shallow_visual_prompt_tmp2.contiguous().reshape(-1, shallow_visual_prompt.shape[-2], shallow_visual_prompt.shape[-1])
        # shallow_visual_prompt_tmp2 = torch.cat([shallow_visual_prompt_tmp2, shallow_visual_prompt_still_tmp.expand(shallow_visual_prompt_tmp2.shape[0], -1, -1)], dim=1)

        
        aug_feas = clip_model.encode_image(pseudo_img2, shallow_visual_prompt_tmp, deep_pt=deep_pt)
        aug_feas2 = clip_model.encode_image(support_img2, shallow_visual_prompt_still_tmp, deep_pt=deep_pt)
        aug_feas = aug_feas.reshape(5, -1, aug_feas.shape[-1])
        aug_feas2 = aug_feas2.reshape(5, -1, aug_feas2.shape[-1])
        aug_feas = torch.cat([aug_feas2, aug_feas], dim=1).reshape(-1, aug_feas.shape[-1])
        # aug_feas = clip_model.encode_image(support_sample_tmp, shallow_visual_prompt_still, deep_pt=None)
        aug_feas = aug_feas.reshape(n_way, -1, aug_feas.shape[-1])
        if use_semantic_adapter:
            se_feas = semantic_adapter(promp_semantic_features_tmp)
        else:
            se_feas = promp_semantic_features_tmp
    # support_fea = clip_model.encode_image(support_sample_tmp1.unsqueeze(0), None, deep_pt=deep_pt)
    # return aug_feas.detach().cpu(), support_fea.detach().cpu(), None\
    # return support_sample_tmp.detach().cpu(), shallow_visual_prompt_tmp.contiguous().detach().cpu(), shallow_visual_prompt_still_tmp.detach().cpu(), support_labels.detach().cpu()
    if shallow_visual_prompt_still is not None:
        shallow_visual_prompt_still = shallow_visual_prompt_still.detach().cpu()

    if shallow_visual_prompt_still_tmp is not None:
        shallow_visual_prompt_still_tmp = shallow_visual_prompt_still_tmp.detach().cpu()

    if deep_pt is not None:
        deep_pt = deep_pt.detach().cpu()

    return aug_feas.contiguous().detach().cpu(), shallow_visual_prompt_still, shallow_visual_prompt_still_tmp, support_labels.detach().cpu(), se_feas.detach().cpu(),\
          None,  support_sample_tmp.detach().cpu(), deep_pt, aug_feas2.detach().cpu()


def main(): # for debugging
    clip_model_name = 'ViT-B/32'
    model, preprocess = clip.load('ViT-B/32', 'cpu')
    clip_model, layers = build_model(model.state_dict())
    clip_model.cuda()
    support_img = torch.load('test_pseudo_set.pth')
    support_img = support_img.cuda()
    # support_img = support_img.to(torch.float16))
    vp_generator(support_img=support_img, clip_model=clip_model, clip_model_name=clip_model_name, layers=layers, num_token=5, use_deep_pt=True)
    return


if __name__ == '__main__':
    main()