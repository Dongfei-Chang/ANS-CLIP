import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentAttention(nn.Module):
    """
    Feature Self-reconstruction Module
    """
    def __init__(self, dim, num_heads=8, attention_dropout=0.1):
        super().__init__()
        self.head_dim = dim
        self.num_heads = num_heads
        self.dim = self.head_dim * self.num_heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(self.head_dim, self.dim * 3, bias=False)
        self.agent_proj = nn.Linear(self.head_dim, self.dim, bias=False)

        self.attn_drop = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(self.head_dim, self.head_dim)
        self.proj_drop = nn.Dropout(attention_dropout)


    def forward(self, x, agent):

        B, N, C = x.shape
        B_agent, N_agent, C = agent.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        a = self.agent_proj(agent).reshape(B_agent, N_agent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q_agent_attn = (q @ a.transpose(-2, -1)) * self.scale
        q_agent_attn = q_agent_attn.softmax(dim=-1)
        q_agent_attn = self.attn_drop(q_agent_attn)

        k_agent_attn = (a @ k.transpose(-2, -1)) * self.scale
        k_agent_attn = k_agent_attn.softmax(dim=-1)
        k_agent_attn = self.attn_drop(k_agent_attn)

        value = k_agent_attn @ v
        out = q_agent_attn @ value
        out = out.mean(dim=1)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dropout=0.1, attention_dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = AgentAttention(dim=d_model, num_heads=nhead, attention_dropout=attention_dropout)
        self.pre_norm = nn.LayerNorm(d_model)
        self.linear1  = nn.Linear(d_model, d_model)
        self.linear2  = nn.Linear(d_model, d_model)
        self.norm1    = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = F.gelu


    def forward(self, src: torch.Tensor, agent: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.self_attn(self.pre_norm(src), self.pre_norm(agent))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout1(src2)

        return src



class Transformer(nn.Module):
    def __init__(self,
                 embedding_dim=64,
                 num_layers=1,
                 num_heads=1,
                 attention_dropout=0.,
                 mlp_dropout_rate=0.,
                 *args, **kwargs):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dropout=mlp_dropout_rate,
                                    attention_dropout=attention_dropout)
            for i in range(num_layers)])

        self.norm = nn.LayerNorm(embedding_dim)

        self.apply(self.init_weight)

    def forward(self, x, agent):
        for blk in self.blocks:
            x = blk(x, agent)
        x = self.norm(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



# FSRM Main model
class AgentAttentionModule(nn.Module):
    def __init__(self,
                 embedding_dim=64,
                 *args, **kwargs):
        super(AgentAttentionModule, self).__init__()
        self.transformer = Transformer(embedding_dim=embedding_dim, *args, **kwargs)

    def forward(self, x, agent):
        x = self.transformer(x, agent)
        return x

