from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict([("-1", nn.AvgPool2d(stride)),
                             ("0",
                              nn.Conv2d(inplanes,
                                        planes * self.expansion,
                                        1,
                                        stride=1,
                                        bias=False)),
                             ("1", nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1],
                      x.shape[2] * x.shape[3]).permute(2, 0,
                                                       1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """
    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 input_resolution=224,
                 width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3,
                               width // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2,
                               width // 2,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2,
                               width,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim,
                                        heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                             (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)),
                         ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False,
                         attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
def debug_tensor_stats(tensor, name="Tensor", approx_zero_thresh=1e-5):
    tensor_fp32 = tensor.to(torch.float32)  # 提高精度查看真实值

    print(f"\n📊 统计信息 - {name}")
    print("-" * 50)
    print(f"设备: {tensor.device}")
    print(f"形状: {tensor.shape}")
    print(f"数据类型: {tensor.dtype}")
    print(f"是否包含 NaN: {torch.isnan(tensor).any().item()}")
    print(f"是否包含 Inf: {torch.isinf(tensor).any().item()}")
    print(f"精确为 0 的个数: {torch.sum(tensor == 0).item()}")
    print(f"近似为 0 的个数 (< {approx_zero_thresh}): {torch.sum(torch.abs(tensor_fp32) < approx_zero_thresh).item()}")
    print(f"最大值: {tensor_fp32.max().item():.6f}")
    print(f"最小值: {tensor_fp32.min().item():.6f}")
    print(f"均值: {tensor_fp32.mean().item():.6f}")
    print(f"标准差: {tensor_fp32.std().item():.6f}")
    print("-" * 50)

def resize_suffix_tensor(suffix, token_data):
    # 获取 token_data 的维度
    B, T, C = token_data.shape
    suffix_B, suffix_T, suffix_C = suffix.shape
    
    # 使用 token_data 的维度来确定目标 batch 和输出维度
    target_batch = B
    output_dim = C
    target_len = T  # 使用 token_data 的 T 作为 target_len
    
    # 取前 target_len 个
    suffix = suffix[:, :target_len, :]
    
    # 扩展 batch 维度
    repeat_factor = (target_batch + suffix_B - 1) // suffix_B
    suffix = suffix.repeat(repeat_factor, 1, 1)[:target_batch]
    
    # 如果输出维度和输入维度不一致，进行线性变换
    if suffix_C != output_dim:
        weight = torch.randn(output_dim, suffix_C, device=suffix.device, dtype=suffix.dtype)
        suffix = F.linear(suffix, weight)
    
    return suffix



class ResidualAttentionBlock_SEP_Visual(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def negate_token_symmetric(self, token_data):
        """
        根据最大绝对值统一确定上限 upper_bound = 10^位数，
        正数用 upper_bound - x，
        负数用 -(upper_bound - |x|) 对称反转。

        这样保证正负数“反转”均匀且语义对称，
        适合用来模拟token_data的“否定语义”。

        Args:
            token_data (Tensor): 输入张量，支持正负数

        Returns:
            Tensor: 反转后的否定语义张量
        """
        with torch.no_grad():
            result = torch.empty_like(token_data)

            max_abs = token_data.abs().max().item()
            digit = int(math.log10(max_abs)) + 1 if max_abs > 0 else 2
            upper_bound = 10 ** digit

            pos_mask = token_data >= 0
            neg_mask = token_data < 0

            # 正数部分：上限减去自身
            result[pos_mask] = upper_bound - token_data[pos_mask]

            # 负数部分：绝对值上限减绝对值再取负号
            result[neg_mask] = -(upper_bound - token_data[neg_mask].abs())

        return result


    def forward(self, inputs):
        # x = self.transformer([x,Visual_Adapters,VPT_ctx.shape[0],layers,counter,imgclass])
        # print("c长度11111",len(inputs))
        #[x,xNo,Visual_Adapters,Visual_AdaptersNo,VPT_ctx.shape[0],VPT_ctxNo.shape[0],layers,counter,imgclass]
        if len(inputs)==10:
            x = inputs[0]
            x_aug = inputs[1]
            compound_prompts_deeper = inputs[2]
            compound_prompts_deeper_aug = inputs[3]
            n_ctx = inputs[4]
            n_ctx_aug = inputs[5]
            layers=inputs[6]
            counter = inputs[7]
            #5,140,512
            datasetname = inputs[8]
            imgclass = inputs[9]
            # print("datasetname",datasetname)

            deviceY = next(compound_prompts_deeper[0].parameters()).device  #1
            if datasetname in {"ImageNet", "SUN397"}:
                deviceNo = "cuda:2"
            else:
                deviceNo = "cuda:0"
            # deviceNo = "cuda:2"
            # print("deviceNo",deviceNo)
            for i, module in enumerate(compound_prompts_deeper_aug):
               compound_prompts_deeper_aug[i] = module.to(deviceNo)
            x_aug = x_aug.to(deviceNo)
            

            flag = 0
            
            # suffix = imgclass['suffix']
            # suffixNo = imgclass['suffixNo']
            #with torch.no_grad():
                    # token_data_R = self.negate_token_symmetric(token_data1)#cuda1
           
            if counter in layers:
                
                #x: [203,16,768]
                #201,32,768...4
                # print("进来注意力的维度",x.shape,n_ctx)
                # print("传递进来的",suffixre.shape,suffixNore.shape,torch.equal(suffixre, suffixNore))
                # # print("传递进来的",suffixre,suffixNore)
                # suffix = imgclass[:, :70, :]
                # suffixNo = imgclass[:, 70:, :]
              
                    
               
                x_power = x.pow(2)
                x_power = x_power.mean(dim=2)
                x_power = x_power.mean(dim=1)
                x_power = x_power[:x.shape[0]-n_ctx]
                value,ind_ = torch.topk(x_power, k=n_ctx, dim=0, largest=True)

                x_power_aug = x_aug.pow(2)
                x_power_aug = x_power_aug.mean(dim=2)
                x_power_aug = x_power_aug.mean(dim=1)
                x_power_aug = x_power_aug[:x_aug.shape[0]-n_ctx_aug]
                value_aug,ind_aug = torch.topk(x_power_aug, k=n_ctx_aug, dim=0, largest=True)
                


                ind = layers.index(counter)

                prefix = x[0:x.shape[0]-n_ctx, :, :]
                token_data = x[ind_,:,:] 
                midfix = x[x.shape[0]-n_ctx:,:,:] 
                token_data = token_data.permute(1,0,2)
                midfix = midfix.permute(1,0,2)

                prefix_aug = x_aug[0:x_aug.shape[0]-n_ctx_aug, :, :]
                token_data_aug = x_aug[ind_aug,:,:] 
                midfix_aug = x_aug[x_aug.shape[0]-n_ctx_aug:,:,:] 
                token_data_aug = token_data_aug.permute(1,0,2)
                midfix_aug = midfix_aug.permute(1,0,2)    #否

                #32,4,768
                # debug_tensor_stats(token_data, name="img 注意力 token_data")
                # debug_tensor_stats(midfix, name="img 注意力 midfix")
                # print("进行注意力计算的维度",token_data.shape,midfix.shape)
                # sys.exit()
                
                token_data1 = compound_prompts_deeper[ind](token_data,midfix)  #0
                token_data_aug1 = compound_prompts_deeper_aug[ind](token_data_aug,midfix_aug)  #0/2
                token_data_aug1 = token_data_aug1.to(deviceY)  #0

                token_data2 = compound_prompts_deeper[ind](token_data_aug1,token_data1)  #0
                token_data1 = token_data1.to(deviceNo)  #0/2
                token_data_aug1 = token_data_aug1.to(deviceNo)  #0/2
                token_data_aug2 = compound_prompts_deeper_aug[ind](token_data1,token_data_aug1)  #0/2
                del token_data1,token_data_aug1


                token_dataF = token_data2.permute(1,0,2)  #0/2
                del token_data2
                token_data_F_aug = token_data_aug2.permute(1,0,2)#0/2
                del token_data_aug2

                x = torch.cat([prefix,token_dataF], dim=0)
                del token_dataF

                # if datasetname == "ImageNet":
                #     torch.cuda.empty_cache()  
                x_aug = torch.cat([prefix_aug,token_data_F_aug], dim=0)
                del token_data_F_aug          
                # token_data = token_data.permute(1,0,2)
                
                # x = torch.cat([prefix,token_data], dim=0)
            x_aug = x_aug.to(deviceY)
            counter += 1
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            if datasetname in {"ImageNet", "SUN397"}:
                    torch.cuda.empty_cache()  

            x_aug = x_aug + self.attention(self.ln_1(x_aug))
            x_aug = x_aug + self.mlp(self.ln_2(x_aug))
            return [x,x_aug, compound_prompts_deeper, compound_prompts_deeper_aug, n_ctx,n_ctx_aug,layers,counter,datasetname,imgclass]  # return again as a list, so that nn.seq can work
        else:
            # x = inputs[0]
            # print("inputs",inputs)
            x = inputs[0]
            x_aug = inputs[1]
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            x_aug = x_aug + self.attention(self.ln_1(x_aug))
            x_aug = x_aug + self.mlp(self.ln_2(x_aug))
            return [x,x_aug]


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock_SEP_Visual(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

import pdb
import sys
import math
class ResidualAttentionBlock_SEP_Text(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attnNo = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # 为 xNo 定义独立一组模块
        self.ln_1_no = LayerNorm(d_model)
        self.ln_2_no = LayerNorm(d_model)
        self.mlp_no = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

         # 标记用于自定义初始化
        self.attnNo._is_custom_no = True
        self.ln_1_no._is_custom_no = True
        self.ln_2_no._is_custom_no = True
        self.mlp_no._is_custom_no = True

        self.attn_mask = attn_mask
        self.attn_mask_no = attn_mask.clone() if attn_mask is not None else None

        # 固定门控alpha（不可学习）：前5层0.26，后7层0.74
        self.cross_alpha = [0.26]*5 + [0.74]*7

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def attention_no(self, x: torch.Tensor):
        self.attn_mask_no = self.attn_mask_no.to(dtype=x.dtype, device=x.device) if self.attn_mask_no is not None else None
        return self.attnNo(x, x, x, need_weights=False, attn_mask=self.attn_mask_no)[0]

    def negate_token_symmetric(self, token_data):
        """
        根据最大绝对值统一确定上限 upper_bound = 10^位数，
        正数用 upper_bound - x，
        负数用 -(upper_bound - |x|) 对称反转。

        这样保证正负数“反转”均匀且语义对称，
        适合用来模拟token_data的“否定语义”。

        Args:
            token_data (Tensor): 输入张量，支持正负数

        Returns:
            Tensor: 反转后的否定语义张量
        """
        with torch.no_grad():
            result = torch.empty_like(token_data)

            max_abs = token_data.abs().max().item()
            digit = int(math.log10(max_abs)) + 1 if max_abs > 0 else 2
            upper_bound = 10 ** digit

            pos_mask = token_data >= 0
            neg_mask = token_data < 0

            # 正数部分：上限减去自身
            result[pos_mask] = upper_bound - token_data[pos_mask]

            # 负数部分：绝对值上限减绝对值再取负号
            result[neg_mask] = -(upper_bound - token_data[neg_mask].abs())

        return result

    def forward(self, inputs):
        #[x,prompt_projections,weight,layers,counter,xNo]
        # (prompts, promptsNo, prompt_projections, prompt_projectionsNNN,self.n_ctx,self.layers,tokenized_prompts, tokenized_promptsNo.detach()) 
        # print("inputs changdu ", len(inputs))
        # pdb.set_trace()
        #[x,xNo,prompt_projections,prompt_projections,weight,layers,counter]####
        # print("inputs changdu sha", inputs.shape)
        # if len(inputs)==6:
        if len(inputs)==8:
        # if len(inputs)==5:
            # pdb.set_trace()
            # print("((((()))))")

            # pdb.set_trace()

            x = inputs[0] #cuda1
            xNo = inputs[1]#cuda0
            compound_prompts_deeper = inputs[2]#cuda1
            compound_prompts_deeperN = inputs[3]#cuda0
            n_ctx = inputs[4]
            layers=inputs[5]
            counter = inputs[6]
            datasetname = inputs[7]
            
            # print("compound_prompts_deeper device:", next(compound_prompts_deeper[0].parameters()).device)
            # print("x",x.device)
            # print("compound_prompts_deeperN device:", next(compound_prompts_deeperN[0].parameters()).device)
            deviceY = next(compound_prompts_deeper[0].parameters()).device  #1
            if datasetname in {"ImageNet", "sun397"}:
                deviceNo = "cuda:2"
            else:
                deviceNo = "cuda:0"
            # deviceNo = "cuda:2"
            for i, module in enumerate(compound_prompts_deeperN):
               compound_prompts_deeperN[i] = module.to(deviceNo)
            xNo = xNo.to(deviceNo)


        
            if counter in layers:
                # print("asasasasasasasasasasasasasasasasasasasasasasasasasasasasasasasas")
                ind = layers.index(counter)
                #1,5,512
                prefix = x[:1, :, :]
                prefixNo = xNo[:1, :, :]

                #70,5,512
                suffix = x[1+n_ctx:, :, :]
                suffixNo = xNo[1+n_ctx:, :, :]
        
                #6,5,512
                token_data = x[1+n_ctx:1+2*n_ctx,:,:] 
                token_dataNo = xNo[1+n_ctx:1+2*n_ctx,:,:] 

                midfix = x[1:1+n_ctx,:,:]
                midfixNo = xNo[1:1+n_ctx,:,:]

                token_data = token_data.permute(1,0,2)
                midfix = midfix.permute(1,0,2)

                token_dataNo = token_dataNo.permute(1,0,2)
                midfixNo = midfixNo.permute(1,0,2)
                #a photo of a Apply Eye Makeup in UCF101, A photo of a person doing Apply Eye Makeup. 
                # A person doing Apply_Eye_Makeup looks like they are holding a makeup brush up to their eye and applying makeup.


                token_data1 = compound_prompts_deeper[ind](token_data, midfix)  #cuda1
                token_dataNo1 = compound_prompts_deeperN[ind](token_dataNo, midfixNo)#cuda0

                # 固定门控交互（多层）
                alpha = self.cross_alpha[ind]
                print(f"[Layer {ind}] cross alpha: ", alpha)
                token_data2 = alpha * compound_prompts_deeper[ind](token_dataNo1, token_data1) + (1 - alpha) * token_data1
                print(f"token_data2 stats: min {token_data2.min().item()}, max {token_data2.max().item()}, mean {token_data2.mean().item()}, nan: {torch.isnan(token_data2).any().item()}")
                token_dataNo2 = alpha * compound_prompts_deeperN[ind](token_data1, token_dataNo1) + (1 - alpha) * token_dataNo1
                print(f"token_dataNo2 stats: min {token_dataNo2.min().item()}, max {token_dataNo2.max().item()}, mean {token_dataNo2.mean().item()}, nan: {torch.isnan(token_dataNo2).any().item()}")

                token_dataF = token_data2.permute(1,0,2)  #1
                del token_data2
                token_dataNoF = token_dataNo2.permute(1,0,2)#0
                del token_dataNo2
                x = torch.cat([prefix,token_dataF,suffix], dim=0)
                print(f"x after cat stats: min {x.min().item()}, max {x.max().item()}, mean {x.mean().item()}, nan: {torch.isnan(x).any().item()}")
                xNo = torch.cat([prefixNo,token_dataNoF,suffixNo], dim=0)
                print(f"xNo after cat stats: min {xNo.min().item()}, max {xNo.max().item()}, mean {xNo.mean().item()}, nan: {torch.isnan(xNo).any().item()}")

                # if datasetname == "ImageNet":
                #     torch.cuda.empty_cache()  

                # print("prefixNo:", prefixNo.device, "token_dataNoF:", token_dataNoF.device, "suffixNo:", suffixNo.device)
                # print("prefix:", prefix.device, "token_dataF:", token_dataF.device, "suffix:", suffix.device)
                # sys.exit()
                # x = torch.cat([prefix,token_dataF], dim=0)
            xNo = xNo.to(deviceY)
            counter += 1
            # print("baocuodian ",len(x))
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            if datasetname in {"ImageNet", "sun397"}:
                    torch.cuda.empty_cache()  

            # xNo = xNo + self.attention_no(self.ln_1_no(xNo))
            # xNo = xNo + self.mlp_no(self.ln_2_no(xNo))
            xNo = xNo + self.attention(self.ln_1(xNo))
            xNo = xNo + self.mlp(self.ln_2(xNo))

            
            # x = x + self.mlp(x)
            # xNo = xNo + self.mlp(xNo)
            
            # return [x, compound_prompts_deeper, n_ctx,layers,counter, xNo]  # return again as a list, so that nn.seq can work
            return [x,xNo,compound_prompts_deeper,compound_prompts_deeperN, n_ctx,layers,counter,datasetname]  # return again as a list, so that nn.seq can work
            # return [x, compound_prompts_deeper, n_ctx,layers,counter, x, compound_prompts_deeper]  # return again as a list, so that nn.seq can work
            # return [x, compound_prompts_deeper, n_ctx,layers,counter]  # return again as a list, so that nn.seq can work
        else:
            # print(inputs)
            # print("dongjiede ", len(inputs))
            # pdb.set_trace()
            x = inputs
            x = x + self.attention(self.ln_1(x)) 
            x = x + self.mlp(self.ln_2(x))
            return x
            # return x, x


class TextTransformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock_SEP_Text(width, heads, attn_mask)
            for _ in range(layers)
        ])
        # 自动初始化自定义模块
        self.init_custom_no_weights()
    def init_custom_no_weights(self):
        def init_fn(m):
            if hasattr(m, '_is_custom_no'):
                for sub in m.modules():
                    if isinstance(sub, nn.Linear):
                        nn.init.xavier_uniform_(sub.weight)
                        if sub.bias is not None:
                            nn.init.zeros_(sub.bias)
                    elif isinstance(sub, nn.LayerNorm):
                        nn.init.ones_(sub.weight)
                        nn.init.zeros_(sub.bias)
        self.apply(init_fn)

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)




class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor,x_aug: torch.Tensor, VPT_ctx=None, VPT_ctx_aug=None, Visual_Adapters=None, Visual_Adapters_aug=None,layers=None,datasetname=None,imgclass=None):
        #image.type(self.dtype), visual_ctx,visual_ctxNo, visual_projections, visual_projectionsNNN, self.layers, imgclass)
        # 传进来的类别信息 torch.Size([5, 140, 512])
        # reshape之前的x torch.Size([32, 768, 14, 14])
        # reshape之后的x torch.Size([32, 768, 196])
        # 拼接后x torch.Size([32, 197, 768])
        # if imgclass is not None:
        #     print("传进来的类别信息",imgclass.shape)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x_aug = self.conv1(x_aug)
        # print("reshape之前的x",x.shape)
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x_aug = x_aug.reshape(x_aug.shape[0], x_aug.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        # print("reshape之后的x",x.shape)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_aug = x_aug.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],dim=1)  # shape = [*, grid ** 2 + 1, width]
        x_aug = torch.cat([
            self.class_embedding.to(x_aug.dtype) + torch.zeros(
                x_aug.shape[0], 1, x_aug.shape[-1], dtype=x_aug.dtype, device=x_aug.device), x_aug
        ],dim=1)  # shape = [*, grid ** 2 + 1, width]
        # print("拼接后x",x.shape)
        x = x + self.positional_embedding.to(x.dtype)
        x_aug = x_aug + self.positional_embedding.to(x_aug.dtype)

        if VPT_ctx is not None and VPT_ctx_aug is not None:

            visual_ctx = VPT_ctx.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([x, visual_ctx], dim=1)

            visual_ctx_aug = VPT_ctx_aug.unsqueeze(0).expand(x_aug.shape[0], -1, -1)
            x_aug = torch.cat([x_aug, visual_ctx_aug], dim=1)


        
        x = self.ln_pre(x)
        x_aug = self.ln_pre(x_aug)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x_aug = x_aug.permute(1, 0, 2)  # NLD -> LND
        

        if Visual_Adapters is not None:
            counter=0
            # print("Visual_Adapters is not None")
            # x,xNo = self.transformer([x,xNo,Visual_Adapters,Visual_AdaptersNo,VPT_ctx.shape[0],VPT_ctxNo.shape[0],layers,counter,datasetname,imgclass])
            output = self.transformer([x,x_aug,Visual_Adapters,Visual_Adapters_aug,VPT_ctx.shape[0],VPT_ctx_aug.shape[0],layers,counter,datasetname,imgclass])
            # print(f"Transformer returned {len(output)} values: {output}") 
            x = output[0]
            x_aug = output[1]
        else:
            # print("Visual_Adapters is None")
            x,x_aug = self.transformer([x,x_aug])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x_aug = x_aug.permute(1, 0, 2)  # LND -> NLD
        
        x = self.ln_post(x[:, 0, :])
        x_aug = self.ln_post(x_aug[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
            x_aug = x_aug @ self.proj

        return x,x_aug


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers,
                                         output_dim=embed_dim,
                                         heads=vision_heads,
                                         input_resolution=image_resolution,
                                         width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(input_resolution=image_resolution,
                                            patch_size=vision_patch_size,
                                            width=vision_width,
                                            layers=vision_layers,
                                            heads=vision_heads,
                                            output_dim=embed_dim)


        self.transformer = TextTransformer(width=transformer_width,
                                       layers=transformer_layers,
                                       heads=transformer_heads,
                                       attn_mask=self.build_attention_mask())

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                    self.visual.layer1, self.visual.layer2, self.visual.layer3,
                    self.visual.layer4
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers)**-0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width)**-0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(
            self.dtype)  # [batch_size, n_ctx, d_model]
        

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
              text.argmax(dim=-1)] @ self.text_projection
        
        # x_no = x

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=-1,
                                                           keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                    *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                    "in_proj_bias", "bias_k", "bias_v"
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1)**0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2] for k in state_dict
                    if k.startswith(f"visual.layer{b}")))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] -
             1)**0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict[
            "visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict
            if k.startswith(f"transformer.resblocks")))

    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width,
                 vision_patch_size, context_length, vocab_size,
                 transformer_width, transformer_heads, transformer_layers)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    # model.load_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)

    return model.eval()
