import os
from xmlrpc.client import FastMarshaller
import numpy as np
from pandas import option_context
import torch
import torch.nn as nn
import torch.optim
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
setup_seed(0)
from options2 import parse_args
import clip
from PSG import PseudoSampleGenerator
from data import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot
from model import ProtoNet
import pickle as pkl
from clip_model.text_pt import TextEncoder, PromptLearner
import pdb
from torchvision.datasets import ImageFolder
from clip_model.prompted_model import Visual_prompt, build_model
from data.datamgr import SimpleDataManager, SetDataManager
import time
from adi import adaptive_instance_normalization as adain
from prompt_generator_para_deep_16 import vp_generator
# from train_classifier_para_deep import train_classifier, ArcMarginProduct
from train_classifier_16 import train_classifier as train_classifier2, ArcMarginProduct
import torch.nn.functional as F
import open_clip

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


params = parse_args()

def cal_zeros(input, cls_dist, labels):
    zero_count = (input == 0).sum(dim=-1)
    for i in range(zero_count.shape[0]):
        count = zero_count[i].item()
        cls = labels[i].item()
        if cls in cls_dist.keys():
            cls_dist[cls] += count
        else:
            cls_dist[cls] = count
    return

def cal_dis2(t1, t2):
 
    # Normalizing the tensors along the first dimension (size 5)
    t1_norm = F.normalize(t1, p=2, dim=-1)
    t2_norm = F.normalize(t2, p=2, dim=1)
    dot_product_matrix = torch.einsum('bij,bkj->bik', t1_norm, t2_norm)
    dot_product_matrix = torch.abs(dot_product_matrix)

    # Finding the index of the maximum similarity in t2 for each element in t1
    max_similarity_value, max_similarity_indices = torch.max(dot_product_matrix, dim=-1)

    return  max_similarity_value, max_similarity_indices


def cal_classes_dis(features_tensor_tmp):
    features_tensor = F.normalize(features_tensor_tmp, p=2, dim=-1)
    intra_class_distances = torch.norm(features_tensor.unsqueeze(2) - features_tensor.unsqueeze(1), dim=3)
    intra_class_distances = intra_class_distances.mean(dim=(-1,-2)).mean()
    class_centers = torch.mean(features_tensor, dim=1)
    inter_class_distances = torch.norm(class_centers.unsqueeze(0) - class_centers.unsqueeze(1), dim=2)
    inter_class_distances = inter_class_distances.mean()
    return intra_class_distances, inter_class_distances

    intra_class_distances.shape, inter_class_distances.shape


def select_semantic_faetures(semantic_feature, semantic_labels, selected_labels):
    sorted_vectors = []
    output_labels = []
    for index in range(selected_labels.shape[0]):
        label = selected_labels[index]
        indices = torch.where(semantic_labels == label)[0]
        selected_vectors = semantic_feature[indices]
        lables_tmp = torch.full(size=[selected_vectors.shape[0]], fill_value=index, dtype=torch.long)
        sorted_vectors.append(selected_vectors)
        output_labels.append(lables_tmp)
    sorted_vectors = torch.cat(sorted_vectors, dim=0)
    output_labels = torch.cat(output_labels, dim=0)
    return sorted_vectors, output_labels



class fc_Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(fc_Classifier, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(dim, n_way)
        # self.relu = nn.functional.relu()

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


def wavelet_DWT_and_styleswap_and_IWT(A_fea, B_fea, xfm, ifm, s_adain):
      # wavelet: DWT
    if s_adain:
        A_fea_out = adain(A_fea, B_fea)
        return A_fea_out
    A_fea_Yl, A_fea_Yh = xfm(A_fea)
    B_fea_Yl, B_fea_Yh = xfm(B_fea)

      # styleswap the A_yl and B_yl
    A_fea_Yl_styleswap = adain(A_fea_Yl, B_fea_Yl)

      # wavelet: IWT1
    A_fea_styleswap = ifm((A_fea_Yl_styleswap, A_fea_Yh))
    return A_fea_styleswap


def load_clip_to_cpu():
    backbone_name = 'ViT-B/32'
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

 
def contrastive(v_fea, s_fea, scale=True):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    image_features = v_fea / v_fea.norm(dim=1, keepdim=True)
    text_features = s_fea / s_fea.norm(dim=1, keepdim=True)
    logit_scale = logit_scale.exp()
    if scale:
        logits_per_image = logit_scale * image_features @ text_features.t()
    else:
        logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    return logits_per_image, logits_per_text


def mixup_module():
    return


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        # self.fc = nn.Sequential(
        #     nn.Linear(c_in, c_in , bias=False),
        #     nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x

def change_lr(optim, a):
    for g in optim.param_groups:
        g['lr'] = a * g['lr']


def finetune(novel_loader, mix_loader, semantic_fea, semantic_fea_source, model, cls_list, prompted_semantic_feature, prompted_semantic_labels, n_pseudo=75, n_way=5, n_support=5, sd=False, model_name=None):
    dtype = list(model.parameters())[0].dtype
    n_pseudo = 0
    iter_num = len(novel_loader)
    acc_all = []
    final_feat_dim = 512
    output_dim = 512
    lr = 0.001
    use_deep_pt = True
    num_token_shallow = 1
    num_token_deep = 5
    use_adapter = True

    # line_selection = 'use_contrustive:{}_useßmixup_source:{}_use_deep_pt:{}_use_ada:{}_a1:{}_a2{}'.format(use_contrustive, use_mixup_source, deep_pt, use_ada, str(a1), str(a2)) 
    # print(line_selection)
    
    print('start')  
    
    in_class_dis = []
    out_class_dis = []

    zero_count = {}

    for ti, (x, x_label) in enumerate(novel_loader):  # x:(5, 20, 3, 224, 224)
        x_label_tmp = x_label[:, 0].squeeze()
        n_query = n_pseudo//n_way
        clip_model, layers = build_model(model.state_dict())
        clip_model.cuda()
        # clip_model.train()
        # layers = 6
        # visual_prompt.train()
        clip_model.float()
        clip_model.train()


        selec_promp_se_feas, select_labels = select_semantic_faetures(prompted_semantic_feature, prompted_semantic_labels, x_label_tmp)
        selec_promp_se_feas = selec_promp_se_feas.float().cuda()
        select_labels = select_labels.cuda()
        
        semantic_classifier = torch.zeros([n_way, final_feat_dim])
        x_label = x_label[:,0].unsqueeze(-1)
        pseudo_label = x_label.repeat(1, n_support + n_query)
        for i in range(pseudo_label.shape[0]):
            id = pseudo_label[i, 0].item()
            semantic_classifier[i, :] = semantic_fea[id].squeeze()
        
        semantic_features = semantic_classifier.float().cuda()

        ###############################################################################################
        # Update model
        x_label = x_label[:,0].unsqueeze(-1)
        pseudo_label = x_label.repeat(1, n_support + n_query)
        cls_list_tmp = []
        for i in range(x_label.shape[0]):
            id = x_label[i, 0].item()
            cls_list_tmp.append(cls_list[id])

        x = x.cuda()
        # Finetune components initialization
        # xs = x[:, :n_support].reshape(-1, *x.size()[2:])  # (25, 3, 224, 224)
        xs = x[:, :n_support]
        # pseudo_q_genrator = PseudoSampleGenerator(n_way, n_support, n_pseudo)
        # opt_vpt = torch.optim.Adam(visual_prompt.parameters(), lr=lr)
        pseudo_set = xs
        # pseudo_set = pseudo_q_genrator.generate(xs)  # (5, n_support+n_query, 3, 224, 224)

        aug_samples, shallow_visual_prompt_tmp, shallow_visual_prompt, support_labels, semantic_feas, shallow_visual_prompt_tmp2, support_sample_tmp, deep_pt, support_feas = vp_generator(pseudo_set, clip_model, model_name, layers=layers, num_token_shallow=num_token_shallow, num_token_deep=num_token_deep,
                                                                    use_deep_pt=use_deep_pt, semantic_features=semantic_features,
                                                                    promp_semantic_features=selec_promp_se_feas, 
                                                                    promp_semantic_labels=select_labels,
                                                                    dtype=dtype)
        ##############
        aug_samples_tmp = aug_samples.reshape(5, -1, aug_samples.shape[-1])
        dis1, dis2 = cal_classes_dis(aug_samples_tmp)
        in_class_dis.append(dis1.item())
        out_class_dis.append(dis2.item())
        print("{}, {}".format(dis1, dis2))
        print("{}, {}".format(sum(in_class_dis) / len(in_class_dis), sum(out_class_dis) / len(out_class_dis)))
        ##############
        semantic_feas = semantic_feas.reshape(5, -1, semantic_feas.shape[-1])
        semantic_feas = torch.cat([semantic_features.unsqueeze(1), semantic_feas.cuda()], dim=1)
        semantic_feas = semantic_feas.to(dtype)
        ##############
        # support_sample_tmp = support_sample_tmp.reshape(5, -1, 3, 224, 224)
        # support_sample_tmp = pseudo_set.repeat(1,5,1,1,1)
        # shallow_visual_prompt_tmp = torch.zeros_like(shallow_visual_prompt_tmp)
        if shallow_visual_prompt is not None:
            shallow_visual_prompt = shallow_visual_prompt.cuda()
        if deep_pt is not None:
            deep_pt = deep_pt.cuda()
        # proto_net, deep_visual_prompt = train_classifier(support_sample_tmp, shallow_visual_prompt_tmp2, clip_model, 'ViT-B/32', use_adapter=use_adapter, input_labels=support_labels)
        proto_net:ArcMarginProduct = train_classifier2(aug_samples, model_name, use_adapter=use_adapter, init_feature=semantic_features, dtype=dtype)


        #########################mixup##################################
        
        # Inference process
        proto_net.eval()
        clip_model.eval()
        torch.cuda.empty_cache()
        n_query = x.size(1) - n_support 
        # model.n_query = n_query
        x_tmp = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x_tmp = x_tmp.to(dtype)
        start = time.time()
        with torch.no_grad():
            with torch.no_grad():
                # if deep_pt:
                #     v_prompt, v_deep_prompt = visual_prompt.forward()
                # else:
                #     v_prompt = visual_prompt.forward()
                #     v_deep_prompt = None
                # deep_visual_prompt = None
                # z_all = clip_model.encode_image(x_tmp, shallow_visual_prompt, deep_pt=deep_pt)
                z_all = clip_model.encode_image(x_tmp, None, deep_pt=deep_pt)
                # z_all = z_all.float()
                t1 = z_all.reshape(5,-1, z_all.shape[-1])
                _, dis2 = cal_dis2(t1, semantic_feas)
                cal_zeros(dis2, zero_count, x_label)
            # z_all = z_all.float()
            z_all = z_all.view(n_way, -1, z_all.shape[-1])

            ###############################
            support_x = z_all[:,:n_support,:]
            support_x = support_x.contiguous().view(-1,support_x.shape[-1]).unsqueeze(0)
            x2 = z_all[:,n_support:, :]
            x2 = x2.contiguous().view(-1, x2.shape[-1]).unsqueeze(0)
            support_y = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()
            support_y = support_y.contiguous().view(-1).unsqueeze(0)
            # model.n_query = n_query
            scores  = proto_net.forward2(x2)

            ##############################

            pred = scores.data.cpu().numpy().argmax(axis = -1)
            # print(pred)
            y2 =  torch.from_numpy(np.repeat(range( n_way ), n_query )).view(-1).numpy()
            acc = np.mean(pred == y2)*100
            if acc > 93:
                c = 0
            acc_all.append(acc)
        del scores
        torch.cuda.empty_cache()
        print('task: {}; acc: {}'.format(ti, acc))
        if ti % 50 == 0:
            print(', '.join(f"{key}: {value}" for key, value in zero_count.items()))
            print('50 mean acc: {}'.format(np.mean(acc_all)))
        end = time.time()
        dura = end - start
        # print(dura)
        # exit()
        
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('Test Acc = %4.2f +- %4.2f%%'%(acc_mean, 1.96*acc_std/np.sqrt(iter_num)))
    line = 'Test Acc = %4.2f +- %4.2f%%'%(acc_mean, 1.96*acc_std/np.sqrt(iter_num))
    return line, None


def func(dataset, shot):
    print(dataset)
    np.random.seed(10)
    s_instance = False
    params.n_shot = shot

    image_size = 224
    iter_num = 1000
    n_query = 16
    n_pseudo = 75
    model_name = 'ViT-B/16'
    print('n_pseudo: ', n_pseudo)

    print('Loading target dataset!')

    # model, preprocess = clip.load(model_name, 'cpu')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='dfn2b')
    model.float()

    semantic_fea = pkl.load(open('./semantic_fea1/{}_ori_prompt_VB{}_2_open.pkl'.format(dataset, model_name.split('/')[1]), 'rb')) # 1是没有class模板
    # semantic_fea_source = pkl.load(open('./semantic_fea/mini_ori_prompt_VB16.pkl', 'rb'))
    
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
    if dataset == 'chest':
        datamgr             =  Chest_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    if dataset == 'ISIC':
        datamgr             =  ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    if dataset == 'euro':
        datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    if dataset == 'crop':
        datamgr             =  CropDisease_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    if dataset == 'ISIC':
        cls_list = ['melanoma',	'melanocytic nevi',	'basal cell carcinoma',	'Actinic Keratoses (Solar Keratoses) and Intraepithelial Carcinoma', 'benign keratosis', 'dermatofibroma', 'Vascular skin lesions']
    if dataset == 'chest':
        cls_list = ["atelectasis", "cardiomegaly", "effusion", "infiltration", "mass", "nodule", "pneumonia", "pneumothorax"]
    
    #---------------------------------------------
    if dataset == 'euro':
        EuroSAT_path = 'D:/work_code/bscd/data/EuroSAT/2750/'
        d = ImageFolder(EuroSAT_path)
        c = d.class_to_idx
        cls_list_tmp = c.keys()
        cls_list_tmp2 = [i.lower() for i in cls_list_tmp]
        cls_list = ['an aerial map photo of a {}'.format(name) for name in cls_list_tmp2]
    if dataset == 'crop':
        cls_list = []
        CropDisease_path = 'D:/work_code/bscd/data/plant-disease' + '/dataset/train/'
        d = ImageFolder(CropDisease_path)
        c = d.class_to_idx
        cls_list_tmp = c.keys()
        for i in cls_list_tmp:
            i = i.lower().replace('___', ' ').replace('__', ' ').replace('_', ' ').replace(',', '')
            cls_list.append(i)
    #--------------------------------------------------
    print(cls_list)
    novel_loader        = datamgr.get_data_loader(preprocess)
    
    #------------laod pronmpted semantic feature--------
    semantic_epoch = 0
    
    prompted_semantic_feature_path = 'D:/work_code/code_zlh(1)/clip_test/text_prompted1/{}_{}_th_prompted_text_feature_{}.pth'.format(dataset, semantic_epoch, 'open_'+model_name.replace('/', '_')) # 正常没有1，1是没有class name没有template
    prompted_semantic_feature, prompted_semantic_labels = torch.load(prompted_semantic_feature_path)
    prompted_semantic_feature = prompted_semantic_feature.cuda()
    prompted_semantic_labels =  prompted_semantic_labels.cuda()

    #---------------------------------------------------
    
    # file_output = open('./segd_output/dataset_{}_shot_{}_model_{}.txt'.format(dataset, shot, model_name.replace('/', '_').replace('-', '_')), 'w')
    final_acc, selection = finetune(novel_loader, None, semantic_fea, None, model, cls_list=cls_list, 
                                    prompted_semantic_feature=prompted_semantic_feature, prompted_semantic_labels=prompted_semantic_labels, 
                                    n_pseudo=n_pseudo, n_way=params.test_n_way, n_support=params.n_shot, sd=False, model_name=model_name)
    # file_output = open('./modules_outputs/dataset:{}_shot:{}_vpbaseline.txt'.format(dataset, selection, s_instance, str(shot)), 'w')
    # file_output.write(final_acc + '\n')
    # file_output.close()
    return

if __name__=='__main__':
    for dataset in ['euro']:
    # dataset = 'crop'
        print(dataset)
        for shot in [5]:
            func(dataset, shot)
