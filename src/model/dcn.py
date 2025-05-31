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
        print("Forward input x:", x)
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
        self.comments_dims = comments_dims 
        self.field_dims = field_dims
        self.small_feat_indices = [1, 2, 3, 4]  # follow(7), fans(5), friend(6), active(4)
        self.id_feat_indices = [0, 5, 6]         # user_id, video_id, author_id
        
        # 从输入的field_dims计算实际维度
        small_field_dims = [field_dims[i] for i in self.small_feat_indices]
        id_field_dims = [field_dims[i] for i in self.id_feat_indices]
        
        print("小特征维度:", small_field_dims)  # 应为 [7,5,5,6,4] 等小数值
        print("ID特征维度:", id_field_dims)
        
       
        # 独立嵌入层用于前 -6 列
        self.individual_embedding = FeaturesEmbedding(field_dims=small_field_dims, embed_dim=embed_dim)
        
        # 共享嵌入层用于 -6:-1 列
        self.shared_embedding = FeaturesEmbedding(field_dims=id_field_dims, embed_dim=embed_dim)
        self.comment_embedding = FeaturesEmbedding(field_dims=comments_dims, embed_dim=embed_dim)
        
    

        
        self.embed_dim = embed_dim
        
        self.text_embeddings = text_embeddings[0]
        self.text_embed_dim = self.text_embeddings.size(1)
        self.user_comment_embeddings = text_embeddings[1]

        # 添加降维线性层，将 text_embed_dim 降到 embed_dim
        self.text_dim_reducer = nn.Linear(self.text_embed_dim, embed_dim)
        self.comment_dim_reducer = nn.Linear(self.text_embed_dim, embed_dim)

        # 计算总的嵌入输出维度
        self.embed_output_dim = (len(small_field_dims) + len(id_field_dims) + 1 + len(comments_dims)) * embed_dim
        print(f"计算后的embed_output_dim: {self.embed_output_dim}")

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
    # 1. 输入验证
        print("输入特征范围验证:")
        print("user_id:", x[:, 0].min().item(), x[:, 0].max().item())
        print("follow:", x[:, 1].min().item(), x[:, 1].max().item())
        print("video_id:", x[:, 5].min().item(), x[:, 5].max().item())

    # 2. 特征分离
        small_feats = x[:, [1, 2, 3, 4]]  # follow, fans, friend, active [batch, 4]
        id_feats = x[:, [0, 5, 6]]        # user_id, video_id, author_id [batch, 3]
        comment_feats = x[:, -6:]         # 最后6列是评论特征 [batch, 6]

        print("小特征范围:", small_feats.min().item(), small_feats.max().item())
        print("ID特征范围:", id_feats.min().item(), id_feats.max().item())

    # 3. 嵌入处理
        small_emb = self.individual_embedding(small_feats)  # [batch, 4, embed_dim]
        id_emb = self.shared_embedding(id_feats)            # [batch, 3, embed_dim]
        comment_emb = self.comment_embedding(comment_feats) # [batch, 6, embed_dim]

    # 4. 文本和评论嵌入处理
        text_embed_ids = x[:, 5]  # 假设第6列（video_id）是文本嵌入ID
        text_embeds = self.text_embeddings[text_embed_ids]  # [batch, 3584]
        #text_embeds = self.text_dim_reducer(text_embeds)    # [batch, embed_dim]
        reduced_text = self.text_dim_reducer(text_embeds)

        comment_embeds = self.user_comment_embeddings[comment_feats]  # [batch, 6, 3584]
        reduced_comments = self.comment_dim_reducer(comment_embeds)     # [batch, 6, embed_dim]

    # 5. 合并所有嵌入
        embed_x = torch.cat([
        small_emb,                     # [batch, 4, embed_dim]
        id_emb,                        # [batch, 3, embed_dim]
        reduced_text.unsqueeze(1),      # [batch, 1, embed_dim]
        reduced_comments                # [batch, 6, embed_dim]
        ], dim=1)  # [batch, 14, embed_dim]
        
        print(f"合并后的嵌入维度: {embed_x.shape}")
        
    # 6. 调整形状用于注意力
        batch_size = x.size(0)
        embed_x = embed_x.view(batch_size, -1, self.embed_dim)

    # 7. 多头注意力
        attn_output, _ = self.multihead_attn(embed_x, embed_x, embed_x)
        attn_output = attn_output.contiguous().view(batch_size, -1)  # [batch, seq_len*embed_dim]

    # 8. Cross Network和MLP处理
        x_l1 = self.cn(attn_output)
        h_l2 = self.mlp(attn_output)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack).squeeze(1)  # [batch]

    # 9. 评论分数预测
        comment_scores = self.comment_score_linear(attn_output)  # [batch, 6]
        comment_scores_ = self.comment_score_linear_(attn_output)

    # 10. Softmax归一化
        self.comment_probs = self.softmax(comment_scores)     # [batch, 6]
        self.comment_probs_ = self.softmax(comment_scores_)   # [batch, 6]

        return p
    def get_comment_probs(self):
         
        return self.comment_probs

    def get_comment_probs_(self):
         
        return self.comment_probs_