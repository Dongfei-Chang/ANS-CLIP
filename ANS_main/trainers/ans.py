import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict
import scipy.io as sio


from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from .clip_text import clip
from .clip_text.simple_tokenizer import SimpleTokenizer as _Tokenizer
import tqdm
_tokenizer = _Tokenizer()
import numpy as np
import copy
import clip.clip as clip_ori

import sys

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
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


def load_clip_to_cpu_ori():
    backbone_names=['RN50','RN101','ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']
    dims=[1024,1024,512,512,768,768]
    # dims=[1024,1024,512,512,512,512]
    ind=4
    print(backbone_names[ind])
    url = clip_ori._MODELS[backbone_names[ind]]
    model_path = clip_ori._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model,dims[ind]


CUSTOM_TEMPLATES_ori = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of an aircraft {}.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}, a type of car.",
    "Food101": "a photo of a {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

CUSTOM_TEMPLATES_No = {
    "OxfordPets": "this is not a {}, which is not a type of pet.",
    "OxfordFlowers": "this is not a {}, which is a type of flower.",
    "FGVCAircraft": "this is not an aircraft {}.",
    "DescribableTextures": "this is not a {}, which is a type of texture.",
    "EuroSAT": "this is not a centered satellite photo of {}.",
    "StanfordCars": "this is not a {}, which is a type of car.",
    "Food101": "this is not a {}, which is a type of food.",
    "SUN397": "this is not a photo of a {}.",
    "Caltech101": "this is not a photo of a {}.",
    "UCF101": "this is not a photo of a person doing {}.",
    "ImageNet": "this is not a photo of a {}.",
    "ImageNetSketch": "this is not a sketch of a {}.",
    "ImageNetV2": "this is not a photo of a {}.",
    "ImageNetA": "this is not a photo of a {}.",
    "ImageNetR": "this is not a rendered image of a {}.",
}


CUSTOM_TEMPLATES = {
    "OxfordPets": " {}.",
    "OxfordFlowers": " {}.",
    "FGVCAircraft": " {}.",
    "DescribableTextures": " {}.",
    "EuroSAT": " {}.",
    "StanfordCars": " {}.",
    "Food101": " {}.",
    "SUN397": " {}.",
    "Caltech101": " {}.",
    "UCF101": " {}.",
    "ImageNet": "{}.",
    "ImageNetSketch": "{}.",
    "ImageNetV2": "{}.",
    "ImageNetA": "{}.",
    "ImageNetR": "{}.",
}



import pdb

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    
    def forward(self, prompts, promptsNo, prompt_projections, prompt_projectionsNNN,weight, layers, tokenized_prompts, checkFlag,tokenized_promptsNo,flag=False):

        
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        xNo = promptsNo + self.positional_embedding.type(self.dtype)
        xNo = xNo.permute(1, 0, 2)  # NLD -> LND

        if flag:
            x = self.transformer(x)
        else:
            counter=0
            outputs = self.transformer.resblocks([x,xNo,prompt_projections,prompt_projectionsNNN,weight,layers,counter,checkFlag])
 

            x = outputs[0]
            xNo = outputs[1]            

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        #[5,77,512]
        # print("after self-enhance outputs de diyige ", x.shape)

        xNo = xNo.permute(1, 0, 2)  # LND -> NLD
        xNo = self.ln_final(xNo).type(self.dtype)
        # [5,77,512]
        # print("after self-enhance outputs de diyige No", xNo.shape)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        xNo = xNo[torch.arange(xNo.shape[0]), tokenized_promptsNo.argmax(dim=-1)] @ self.text_projection
        # return x
        return x, xNo
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            if hasattr(l, 'weight') and l.weight is not None:
                # 只修改数据，保留参数属性
                l.weight.data = l.weight.data.half()  
            if hasattr(l, 'bias') and l.bias is not None:
                l.bias.data = l.bias.data.half()  

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias", "bias_k", "bias_v"
            ]:
                if hasattr(l, attr):
                    tensor = getattr(l, attr)
                    if tensor is not None:
                        tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if isinstance(attr, torch.Tensor):  # 确保是张量
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)



class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
import json
import os
import traceback
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model,device_txt,device_img):
        super().__init__()
        n_cls = len(classnames)
        # print(f"[PromptLearner.__init__] n_cls = {n_cls}")
        # import traceback, sys
        # print("".join(traceback.format_stack()))
        n_ctx = cfg.TRAINER.COOP.N_CTX
        n_ctx_v = cfg.TRAINER.COOP.N_CTX_V
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.device_txt = device_txt
        self.device_img = device_img
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            print("use given words to initialize context vectors")
            temp = 'a photo of a'
            # tempN = 'a photo of no'

            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

            ctx_vectors_src = embedding[0, 1 : 1 + n_ctx, :]

        else:
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype, device=self.device_txt)
                ctx_vectorsNo = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype, device=self.device_txt)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=self.device_txt)
                ctx_vectorsNo = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=self.device_txt)
            nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(ctx_vectorsNo, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            prompt_prefixNo = " ".join(["X"] * n_ctx)

        
        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial context No: "{prompt_prefixNo}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctxNo = nn.Parameter(ctx_vectorsNo)  # to be optimized

        # print("ctxxxxxxxx",self.ctx)
        # print("ctxxxxxxxxNo",self.ctx)
         
        ctx_vectors_vis = torch.empty(n_ctx_v, 768, dtype=dtype)
        # ctx_vectors_vis = torch.empty(n_ctx_v, 512, dtype=dtype)
        nn.init.normal_(ctx_vectors_vis, std=0.02)
        self.ctx_visual = nn.Parameter(ctx_vectors_vis) 
        self.ctx_visualNo = nn.Parameter(ctx_vectors_vis) 
        #4
        # print("图像的",n_ctx_v)
        # sys.exit()

       
        
        clip_model = load_clip_to_cpu(cfg).to(self.device_txt)
        # clip_model_.cuda()



        # 获取数据集名称
        dataset_name = cfg.DATASET.NAME

        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "..", "scripts", "deepseek_file", f"{dataset_name}_prompt.json")
        json_path = os.path.normpath(json_path)

        assert os.path.exists(json_path), f"描述json文件不存在：{json_path}"

        # 加载 JSON 数据
        with open(json_path, 'r', encoding='utf-8') as f:
            prompts_dict = json.load(f)



        # 替换类别名中的下划线
        classnames = [name.replace("_", " ") for name in classnames]

       # ✅ 正向 prompts 构造
        prompts = []
        for cls in classnames:

            pos_list = prompts_dict.get(cls, {}).get("deepHigh", [])


            if pos_list:
                selected = pos_list[0]  # 使用第一条正向描述
            else:
                selected = f"a photo of a {cls} in {dataset_name}"
                print(f"[警告] 类别 '{cls}' 没有正向描述，使用默认描述：{selected}")
            selected = prompt_prefix + " " + selected  # 拼接
            prompts.append(selected)

        # ✅ 否定 prompts 构造
        promptsNo = []
        for cls in classnames:
            # neg_list = prompts_dict.get(cls, {}).get("negativePin", [])
            neg_list = prompts_dict.get(cls, {}).get("negative", [])
            if neg_list:
                selected = neg_list[0]  # 使用第一条否定描述
            else:
                selected = f"this is not a {cls} in {dataset_name}"
                print(f"[警告] 类别 '{cls}' 没有否定描述，使用默认描述：{selected}")
            selected = prompt_prefixNo + " " + selected  # 拼接
            promptsNo.append(selected)

        

        # ✅ Tokenize（不会再报错）
        tokenized_prompts = clip.tokenize(prompts).to(self.device_txt)
        tokenized_promptsNo = clip.tokenize(promptsNo).to(self.device_txt)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            embeddingNo = clip_model.token_embedding(tokenized_promptsNo).type(dtype)

        print("示例正向prompt:", prompts[:3])
        print("示例否定prompt:", promptsNo[:3])
        # sys.exit()

        with torch.no_grad():
            #####qqqqq
            text_features = clip_model.encode_text(tokenized_prompts)
            text_featuresNo = clip_model.encode_text(tokenized_promptsNo)
            

            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.text_featuresNo = text_featuresNo / text_featuresNo.norm(dim=-1, keepdim=True)
          

        vis_dim = clip_model.visual.output_dim
        self.meta_net_ = PromptAttention()
        self.meta_net_NNN_ = PromptAttention()
        self.prompt_projections = _get_clones(self.meta_net_, 12)
        self.prompt_projectionsNNN = _get_clones(self.meta_net_NNN_, 12)
  
      


        convert_weights(self.prompt_projections)
        convert_weights(self.prompt_projectionsNNN)


        self.visual_meta_net_ = PromptAttention(d_model_visual=768)
        self.visual_meta_netNNN_ = PromptAttention(d_model_visual=768)
        self.visual_projections = _get_clones(self.visual_meta_net_, 12)
        self.visual_projectionsNNN = _get_clones(self.visual_meta_netNNN_, 12)
        convert_weights(self.visual_projections)
        convert_weights(self.visual_projectionsNNN)

        classnames = [name.replace("_", " ") for name in classnames]

      

        

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_prefixNo", embeddingNo[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer("token_suffixNo", embeddingNo[:, 1 + n_ctx :, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.tokenized_promptsNo = tokenized_promptsNo  # torch.Tensor
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.prev_ctx=None
        self.datasetname = dataset_name


        # print(f"[Before] VPT_fc weight shape: {self.VPT_fc.weight.shape}")
        self.VPT_fc = nn.Linear(512, self.n_cls, bias=False)
        # print(f"[After] VPT_fc weight shape: {self.VPT_fc.weight.shape}")

    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)


        prefixNo = self.token_prefixNo
        suffixNo = self.token_suffixNo
        ctxNo = self.ctxNo
        ctxNo = ctxNo.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompt = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        promptNo = torch.cat(
            [
                prefixNo,  # (n_cls, 1, dim)
                ctxNo,
                suffixNo,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        # midfix = x[1:1+n_ctx,:,:]
        imgclass = torch.cat(
            [
                suffix,
                suffixNo,
            ],
            dim = 1,
        )
        
        return prompt, promptNo, self.prompt_projections,self.prompt_projectionsNNN,self.ctx_visual, self.ctx_visualNo, self.visual_projections, self.visual_projectionsNNN, imgclass,self.datasetname


class PromptAttention(nn.Module):
    def __init__(self,d_model_visual=512):
        super().__init__()
        n_head_visual = d_model_visual//64
        self.promt_attention_visual = nn.MultiheadAttention(d_model_visual, n_head_visual)
        self.ln_pa_1_visual = LayerNorm(d_model_visual)
        self.ln_pa_1_token = LayerNorm(d_model_visual)

        self.mlp_pa_visual = nn.Sequential(OrderedDict([
            ("c_fc1", nn.Linear(d_model_visual,256,bias=False)),
            ("gelu1", QuickGELU()),
            ("c_proj", nn.Linear(256, d_model_visual,bias=False))
            #("gelu1", QuickGELU())
        ]))

        self.ln_pa_2_visual = LayerNorm(d_model_visual)

        self.scale1 = nn.Parameter(0.75*torch.ones(1))
        self.scale2 = nn.Parameter(0.75*torch.ones(1))

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, visual_context, token_context):
        #Q：token_context。KV：visual_context,增强的kv
        a,weightLayer = self.promt_attention_visual(self.ln_pa_1_visual(token_context),self.ln_pa_1_visual(visual_context),self.ln_pa_1_visual(visual_context),need_weights=True)
        visual_context = self.scale1.half()*visual_context + (1-self.scale1.half())*a
        visual_context = self.scale2.half()*visual_context + (1-self.scale2.half())*self.mlp_pa_visual(self.ln_pa_2_visual(visual_context))
        
        self.weight_layer = weightLayer
        # print("shape",weightLayer.shape)  # 查看权重矩阵的形状
        # print("shuzhi",weightLayer)        # 打印权重矩阵
        # visual_context = self.scale1.half()*visual_context + (1-self.scale1.half())*self.promt_attention_visual(self.ln_pa_1_visual(token_context),self.ln_pa_1_visual(visual_context),self.ln_pa_1_visual(visual_context),need_weights=False)[0] 
        # visual_context = self.scale2.half()*visual_context + (1-self.scale2.half())*self.mlp_pa_visual(self.ln_pa_2_visual(visual_context))


        return visual_context


import torch


import torch
import torch.nn.functional as F




class Last_Adapter(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(Last_Adapter, self).__init__()
        self.adapterY = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.GELU(),  # GELU 激活函数
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.ReLU(inplace=True),
        )
        self.adapterN = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.SiLU(),  # SiLU 激活函数
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.ReLU(inplace=True),
        )


    def forward(self, features,featuresN):
        features = self.adapterY(features)
        featuresN = self.adapterN(featuresN)
        # x = x + y
        return features,featuresN



from scipy.optimize import linear_sum_assignment
import math
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device_img, device_txt, checkFlag):
        super().__init__()


        self.device_img = device_img
        self.device_txt = device_txt
        self.checkFlag  = checkFlag

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, device_txt, device_img)
        

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts.to(device_txt)
        self.tokenized_promptsNo = self.prompt_learner.tokenized_promptsNo.to(device_txt)
        self.ori_embedding = self.prompt_learner.text_features.to(device_txt)
        self.ori_embeddingNo = self.prompt_learner.text_featuresNo.to(device_txt)

        self.image_encoder = clip_model.visual.to(device_img)
        self.text_encoder = TextEncoder(clip_model).to(device_txt)

       
        self.logit_scale = clip_model.logit_scale.to(device_img) #best-now
        self.dtype = clip_model.dtype

        
        self.domain_sim = -1
        self.domain_sim_src = -1
        self.weight = cfg.TRAINER.COOP.W
        self.weightI = cfg.TRAINER.COOP.WI
        self.weightNo = 4.0
        self.weight_visual = cfg.TRAINER.COOP.W_V
        self.ind_ = 0    
        self.n_ctx = cfg.TRAINER.COOP.N_CTX


        self.adapter_text = Last_Adapter(512, 4).to(clip_model.dtype).to(device_img)
        # self.adapter_text_No = Adapter(512, 4).to(clip_model.dtype).to(device_img)
        self.adapter_image = Last_Adapter(512, 4).to(clip_model.dtype).to(device_img)
        self.text_adapter_m = 0.17
        self.image_adapter_m = 0.2  

        layers=[]
        layer_ =[1,2,3,4,5,6,7,8,9,10,11] # 0
        layers.append(layer_)
        layer_ =[1,2,3,4,5,6,7,8,] # 1
        layers.append(layer_)
        layer_ =[5,6,7,8,9,10,11]  # 2
        layers.append(layer_)
        layer_ =[1,3,5,7,9,11] # 3
        layers.append(layer_)
        layer_ =[2,4,6,8,10] # 4
        layers.append(layer_)
        layer_ =[3,6,9] # 5
        layers.append(layer_)
        layer_ =[4,8,10] # 6
        layers.append(layer_)
        layer_ =[4,8] # 7
        layers.append(layer_)
        layer_ =[4,10]
        layers.append(layer_)
        layer_ =[4,6]
        layers.append(layer_)
        layer_ =[8,10]
        layers.append(layer_)
        layer_ =[2]
        layers.append(layer_)
        layer_ =[6]
        layers.append(layer_)
        layer_ =[1]
        layers.append(layer_)
        layer_ =[3]
        layers.append(layer_)
        layer_ =[5]
        layers.append(layer_)
        layer_ =[7]
        layers.append(layer_)
        layer_ =[9]
        layers.append(layer_)
        layer_ =[10]
        layers.append(layer_)
        layer_ =[11]
        layers.append(layer_)

        self.layers=layers[cfg.TRAINER.COOP.L_IND]

        # 注册 epoch 计数器（不含梯度）
        self.register_buffer('epoch_cnt', torch.tensor(0, dtype=torch.long))




    def forward(self, image, label=None, feat_flag=False, proto=None):
        # 确保image在image设备
        image = image.to(self.device_img)

  
        prompts, promptsNo, prompt_projections,prompt_projectionsNNN, visual_ctx, visual_ctxNo, visual_projections, visual_projectionsNNN, imgclass, datasetname = self.prompt_learner()
        


        visual_ctx = visual_ctx.to(self.device_img)
        visual_ctxNo = visual_ctxNo.to(self.device_img)
        visual_projections = visual_projections.to(self.device_img)
        visual_projectionsNNN = visual_projectionsNNN.to(self.device_img)
        imgclass = imgclass.to(self.device_img)

        with torch.no_grad():
            image_features_fixed = self.image_encoder(image.type(self.dtype))
            self.last_image_features = image_features_fixed.detach().clone()
        image_features_fixed = image_features_fixed / image_features_fixed.norm(dim=-1, keepdim=True)

        if feat_flag:
            return image_features_fixed


        image_features, image_featuresNo = self.image_encoder(image.type(self.dtype), visual_ctx,visual_ctxNo, visual_projections, visual_projectionsNNN, self.layers, self.checkFlag,imgclass)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features + image_features_fixed
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        image_featuresNo = image_featuresNo / image_featuresNo.norm(dim=-1, keepdim=True)


        text_features_old = self.ori_embedding / self.ori_embedding.norm(dim=-1, keepdim=True)
        text_features_oldNo = self.ori_embeddingNo / self.ori_embeddingNo.norm(dim=-1, keepdim=True)

        tokenized_prompts = self.tokenized_prompts.to(self.device_txt)
        tokenized_promptsNo = self.tokenized_promptsNo.to(self.device_txt)

        text_features, text_featuresNo = self.text_encoder(
            prompts, promptsNo, prompt_projections,prompt_projectionsNNN,
            self.n_ctx, self.layers, tokenized_prompts, self.checkFlag,tokenized_promptsNo.detach()
        )

        
        self.last_text_features = text_features.detach().clone()
        self.last_text_features_No = text_featuresNo.detach().clone()

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features + text_features_old
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_featuresNo = text_featuresNo / text_featuresNo.norm(dim=-1, keepdim=True)
        text_featuresNo = text_featuresNo + text_features_oldNo
        text_featuresNo = text_featuresNo / text_featuresNo.norm(dim=-1, keepdim=True)

        text_features = text_features.to(self.device_img)
        text_featuresNo = text_featuresNo.to(self.device_img)


        text_features_adapter,text_featuresNo_adapter = self.adapter_text(text_features,text_featuresNo)
        text_features = (
            self.text_adapter_m * text_features_adapter + (1 - self.text_adapter_m) * text_features
        )
        text_featuresNo = (
            self.text_adapter_m * text_featuresNo_adapter + (1 - self.text_adapter_m) * text_featuresNo
        )

        image_features_adapter,image_featuresN_adapter = self.adapter_image(image_features,image_featuresNo)
        image_features = (
            self.image_adapter_m * image_features_adapter + (1 - self.image_adapter_m) * image_features
        )
        image_featuresNo = (
            self.image_adapter_m * image_featuresN_adapter + (1 - self.image_adapter_m) * image_featuresNo
        )

        

        logits = self.logit_scale.exp() * image_features @ text_features.t()
        logitsNo = self.logit_scale.exp() * image_features @ text_featuresNo.t()
        logitsNo_imageN = self.logit_scale.exp() * image_featuresNo @ text_features.t()


        if self.prompt_learner.training:
            curr_epoch = getattr(self, 'epoch', -1)


            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)

            logits_fixed = 10 * F.linear(
                F.normalize(image_features_fixed, p=2, dim=-1),
                F.normalize(self.prompt_learner.VPT_fc.weight.to(device=self.device_img, dtype=image_features.dtype), p=2, dim=-1)
            )
            
            logits_ = 10 * F.linear(
                F.normalize(image_features, p=2, dim=-1),
                F.normalize(self.prompt_learner.VPT_fc.weight.to(device=self.device_img, dtype=image_features.dtype), p=2, dim=-1)
            )

            ncls = self.prompt_learner.n_cls

            
            loss_ce_v = 0.5 * (
                F.cross_entropy(logits_, label.to(self.device_img)) +
                F.cross_entropy(logits_fixed, label.to(self.device_img))               
            )

            loss_ce = 0.5 * F.cross_entropy(logits, label.to(self.device_img))

            score = cos(text_features, text_features_old.to(self.device_img))

            score = 1.0 - torch.mean(score)
            loss_text = self.weight * score
            loss = loss_ce + loss_text



            visual_dist = F.cosine_similarity(image_features_fixed, image_features, dim=-1)
            visual_dist = 1.0 - torch.mean(visual_dist)
            visual_dist = self.weight_visual * visual_dist
            loss += visual_dist
             

            margin = 0.2
            loss_contrastive  = torch.mean(torch.relu(margin - logits + logitsNo))
            loss_contrastive_imgN  = torch.mean(torch.relu(margin - logits + logitsNo_imageN))

            loss += loss_contrastive
            loss += loss_contrastive_imgN
            loss += loss_ce_v


            return logits, loss

        else:
            return logits


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import torch
@TRAINER_REGISTRY.register()
class ANS(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
    
    def format_m(self, num):
        return f"{num / 1e6:.2f}M"
   
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(classnames)
        self.n_cls = len(classnames)
        # print(f"[build_model] n_cls = {self.n_cls}")
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")


        self.device_img = torch.device(cfg.TRAINER.COOP.DEVICE_IMG)
        self.device_txt = torch.device(cfg.TRAINER.COOP.DEVICE_TXT)
        self.device = self.device_img  # 或者根据你的任务选择
        print("传递img",self.device_img)
        print("传递txt",self.device_txt)

        self.checkFlag = cfg.TRAINER.COOP.CHECK_FLAG
        self.protoFlag = cfg.TRAINER.COOP.PROTO_FLAG

  

        self.model = CustomCLIP(cfg, classnames, clip_model, self.device_img, self.device_txt, self.checkFlag)

       
        

        self.model.prompt_learner.to(self.device_txt)
          # 下面可以打印确认一下设备
        print("===== 下面可以打印确认一下设备=====")
        print("prompt_learner device:", next(self.model.prompt_learner.parameters()).device)
        print("text_encoder device:", next(self.model.text_encoder.parameters()).device)
        print("image_encoder device:", next(self.model.image_encoder.parameters()).device)
        

        self.w = cfg.TRAINER.COOP.W

        print("Turning off gradients in both the image and the text encoder")

        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name and "VPT" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)



    


        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model(
            "adapter_text",
            self.model.adapter_text
        )

        self.register_model(
            "adapter_image",
            self.model.adapter_image
        )
        
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None


        self.proto = None
        if cfg.DATASET.SPLIT == "train" or self.protoFlag:
        # if hasattr(self, 'train_loader_x') and self.train_loader_x is not None:
            # print("[Training] 计算训练集原型 proto...")
            self.proto = self.center_feature()
            self.model.prompt_learner.VPT_fc.weight.data = self.proto
        
        self.seed = cfg.SEED
        

        
    def center_feature(self):
        self.set_model_mode("eval")
        data_loader = self.train_loader_x
        embedding_list=[]
        label_list=[]
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            with torch.no_grad():
                image_feature = self.model(input,feat_flag=True)
            embedding_list.append(image_feature)
            label_list.append(label)
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        class_num = len(torch.unique(label_list))

        proto_list = []
        for class_index in range(class_num):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        proto = torch.stack(proto_list, dim=0)
        proto = proto / proto.norm(dim=-1, keepdim=True)
        return proto
    
    

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        self.model.epoch = self.epoch          # <-- 新增这一行


        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():

                output,loss = self.model(image, label,proto=self.proto)
  
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:

            output,loss = self.model(image, label,proto=self.proto)
            self.model_backward_and_update(loss)
        


        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()


        return loss_summary
    

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)

        return input, label



    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            
            if "token_prefixNo" in state_dict:
                del state_dict["token_prefixNo"]

            if "token_suffixNo" in state_dict:
                del state_dict["token_suffixNo"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]
            

            if "VPT_fc.weight" in state_dict:
                del state_dict["VPT_fc.weight"]
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

