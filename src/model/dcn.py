import torch
import torch.nn as nn
from torchfm.layer import FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron


class My_DeepCrossNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.

    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack)
        # return torch.sigmoid(p.squeeze(1))
        return p.squeeze(1)


class My_DeepCrossNetworkModel_withCommentsRanking(nn.Module):
    def __init__(self, field_dims, comments_dims, embed_dim, num_layers, mlp_dims, dropout, text_embeddings, 
                 attention_dim=64, nhead=5):
        super().__init__()

        # 独立嵌入层用于前 -6 列
        self.individual_embedding = FeaturesEmbedding(field_dims, embed_dim)

        # 共享嵌入层用于 -6:-1 列
        self.shared_embedding = FeaturesEmbedding(comments_dims, embed_dim)
        print("comments_dims:", comments_dims, type(comments_dims))

        
        self.embed_dim = embed_dim
        
        self.text_embeddings = text_embeddings[0]
        self.text_embed_dim = self.text_embeddings.size(1)
        self.user_comment_embeddings = text_embeddings[1]

        # 添加降维线性层，将 text_embed_dim 降到 embed_dim
        self.text_dim_reducer = nn.Linear(self.text_embed_dim, embed_dim)
        self.comment_dim_reducer = nn.Linear(self.text_embed_dim, embed_dim)

        # 计算总的嵌入输出维度
        self.embed_output_dim = len(field_dims) * embed_dim + 6 * embed_dim + embed_dim
        self.seq_len = len(field_dims) + 7

        # 初始化 CrossNetwork, MLP 和 Linear 层
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)
        
        # MultiheadAttention 模块，用于额外的评论打分
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.comment_score_linear = nn.Sequential(
            nn.Linear(self.embed_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )
        self.comment_score_linear_ = nn.Sequential(
            nn.Linear(self.embed_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )
        self.softmax = nn.Softmax(dim=1)  # 添加 softmax 层

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        individual_embed_x = self.individual_embedding(x[:, :-6])
        shared_embed_x = self.shared_embedding(x[:, -6:])
        
        # Comment embeddings
        comment_ids = x[:, -6:]
        comment_embeds = self.user_comment_embeddings[comment_ids]
        comment_embeds = self.comment_dim_reducer(comment_embeds)

        # Text embeddings
        text_embed_ids = x[:, -8]
        text_embeds = self.text_embeddings[text_embed_ids]
        text_embeds = self.text_dim_reducer(text_embeds)

        # Combine all embeddings
        embed_x = torch.cat([individual_embed_x, text_embeds.unsqueeze(1), comment_embeds], dim=1)
        embed_x = embed_x.view(-1, self.embed_output_dim)
        
        embed_x = embed_x.view(-1, self.seq_len, self.embed_dim)
        embed_x, _ = self.multihead_attn(embed_x, embed_x, embed_x)
        embed_x = embed_x.contiguous().view(-1, self.embed_output_dim)

        # Cross Network 和 MLP 处理
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack).squeeze(1)  # 原始的预测输出

        comment_scores = self.comment_score_linear(embed_x)  # 输出6个评论的打分
        comment_scores_ = self.comment_score_linear_(embed_x)  # 输出6个评论的打分

        # 通过 softmax 归一化为概率分布
        comment_probs = self.softmax(comment_scores)
        comment_probs_ = self.softmax(comment_scores_)
        self.comment_probs = comment_probs
        self.comment_probs_ = comment_probs_

        return p

    def get_comment_probs(self):
        return self.comment_probs
    
    def get_comment_probs_(self):
        return self.comment_probs_