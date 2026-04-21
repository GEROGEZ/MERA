

import torch
import torch.nn as nn

class ContextAwareAttentionFusion(nn.Module):
    def __init__(self, text_dim=512, sev_emb_dim=32, hidden_dim=128, num_classes=4):
        """
        修改点：移除置信度输入，稍微增大 sev_emb_dim 以增强定级信息的权重
        """
        super().__init__()
        
        # 1. 建议定级 Embedding
        self.sev_embedding = nn.Embedding(num_classes, sev_emb_dim)
        
        # 2. 专家特征投影
        # Input Dim = Text(512) + Severity_Emb(32)  <-- 也就是不再加 1 (confidence)
        self.input_dim = text_dim + sev_emb_dim
        
        self.expert_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 3. 自注意力 (核心)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, text_embs, sev_ids):
        """
        Input Signature Changed: Removed `confs`
        text_embs: [Batch, 3, Text_Dim]
        sev_ids:   [Batch, 3]
        """
        
        # Embedding severity IDs
        sev_vecs = self.sev_embedding(sev_ids) # [Batch, 3, Sev_Emb_Dim]
        
        # Concatenate: [Text | Severity]
        # 修改点：不再拼接 confs
        expert_inputs = torch.cat([text_embs, sev_vecs], dim=-1)
        
        # Encode
        expert_encoded = self.expert_encoder(expert_inputs) # [Batch, 3, Hidden]
        
        # Self-Attention Fusion
        # attn_weights 用于可视化
        # 用专家表示构造一个上下文 query（也可以用额外 MLP）
        context = expert_encoded.mean(dim=1, keepdim=True)  # [B,1,H]

        attn_output, attn_weights = self.attention(
            query=context,            # [B,1,H]
            key=expert_encoded,       # [B,3,H]
            value=expert_encoded      # [B,3,H]
        )
        fused_vector = attn_output.squeeze(1)              # [B,H]
        # attn_weights: [B,1,3]（默认 average_attn_weights=True）

        
        # Classification
        logits = self.classifier(fused_vector) # [Batch, Num_Classes]
        
        return logits, attn_weights
    
