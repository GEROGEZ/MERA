# ------------------------- 导入库 -------------------------
import os  # 文件与目录操作
from typing import Dict, Any, Optional, List  # 类型注解
import numpy as np  # 数值运算
import pandas as pd  # Excel 导出
import torch  # PyTorch 框架
from torch.utils.data import Dataset, DataLoader  # 数据集与加载器
from fusion_model import ContextAwareAttentionFusion  # 自定义 Attention 融合模型
from format_logger import logger_instance

# 级别映射（与你前面保持一致）
ID2SEV = {0: "提示", 1: "一般", 2: "严重", 3: "致命"}  # 数字标签到文字标签映射
INVALID_LABEL = "INVALID"  #


def pick_device(prefer: Optional[str] = None) -> str:
    """
    自动选择推理设备
    prefer: "npu" / "cuda" / "cpu" / None
    返回: 可用设备字符串
    """
    if prefer is not None:
        return prefer  # 用户指定设备直接返回

    # NPU
    # 优先尝试 NPU
    try:
        import torch_npu  # noqa: F401
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        return "npu"  # 有些环境没有 is_available 也能跑
    except Exception:
        pass

    # CUDA
    if torch.cuda.is_available():
        return "cuda"
    # 最后回退到 CPU
    return "cpu"


# ------------------------- 自定义 Dataset -------------------------
class FusionDictDataset(Dataset):
    """
    直接用 fusion_data(dict) 做 Dataset，不需要 .pt 文件
    fusion_data 期望字段：
      - text_emb: torch.FloatTensor [N,3,D]
      - sev_id  : torch.LongTensor  [N,3]  (可能含 -1)
      - labels  : torch.LongTensor  [N]    (可选)
      - metadata: list[str]         [N]
    """

    def __init__(self, fusion_data: Dict[str, Any]):
        assert "text_emb" in fusion_data and "sev_id" in fusion_data and "metadata" in fusion_data, \
            "fusion_data 必须至少包含: text_emb, sev_id, metadata"

        self.text_emb = fusion_data["text_emb"]  # 文本嵌入
        self.sev_id_raw = fusion_data["sev_id"]  # 原始专家建议
        self.metadata = fusion_data["metadata"]  # 工单 ID
        self.labels = fusion_data.get("labels", None)  # 真实标签（可选）

        # ------------------- 张量化 -------------------
        if not torch.is_tensor(self.text_emb):
            self.text_emb = torch.tensor(self.text_emb, dtype=torch.float32)
        if not torch.is_tensor(self.sev_id_raw):
            self.sev_id_raw = torch.tensor(self.sev_id_raw, dtype=torch.long)

        if self.labels is not None and (not torch.is_tensor(self.labels)):
            self.labels = torch.tensor(self.labels, dtype=torch.long)

        # 基本形状检查
        if self.text_emb.ndim != 3 or self.text_emb.shape[1] != 3:
            raise ValueError(f"text_emb 形状应为 [N,3,D]，实际为 {tuple(self.text_emb.shape)}")
        if self.sev_id_raw.ndim != 2 or self.sev_id_raw.shape[1] != 3:
            raise ValueError(f"sev_id 形状应为 [N,3]，实际为 {tuple(self.sev_id_raw.shape)}")
        if len(self.metadata) != self.text_emb.shape[0]:
            raise ValueError("metadata 长度与样本数不一致")

    def __len__(self):
        return self.text_emb.shape[0]

    def __getitem__(self, idx):
        """获取单条样本"""
        x = self.text_emb[idx]  # [3,D]
        sev_raw = self.sev_id_raw[idx]  # [3] 可能有 -1

        # ⚠️喂给模型时不能是 -1（embedding index 会报错）
        sev_for_model = torch.where(sev_raw < 0, torch.zeros_like(sev_raw), sev_raw)

        bug_id = self.metadata[idx]
        if self.labels is None:
            label = torch.tensor(-1, dtype=torch.long)  # 无真实标签时填 -1
        else:
            label = self.labels[idx]

        return x, sev_for_model, sev_raw, label, bug_id  # 返回数据元组


def _attn_to_expert_weights(attn: torch.Tensor) -> np.ndarray:
    """
    兼容多种 attention 返回形状，输出每个专家一个权重（长度 3）
    可能的 attn 形状：
      - [B, 3, 3]
      - [B, H, 3, 3]
    返回：np.ndarray shape [B, 3]
    """
    a = attn.detach().cpu()  # 拷贝到 CPU 并脱离计算图
    #logger_instance.info(f'attn_0:{a} {a.ndim} {a.shape} XX attn:{attn}')
    # [B, 1, 3] → [B, 3]
    if a.ndim == 3 and a.shape[1] == 1 and a.shape[2] == 3:
        #logger_instance.warning(f'attn 形状[B, 1, 3] → [B, 3] ')
        return a.squeeze(1).numpy()

    # [B, H, 1, 3] → 对head取均值 → [B, 3]
    if a.ndim == 4 and a.shape[2] == 1 and a.shape[3] == 3:
        #logger_instance.warning(f'[B, H, 1, 3] → 对head取均值 → [B, 3]')
        return a.mean(dim=1).squeeze(1).numpy()

    # 兜底
    #logger_instance.warning(f'未知 attn 形状 {a.shape}，返回均匀权重')
    B = a.shape[0]
    return np.ones((B, 3), dtype=np.float32) / 3.0


# ------------------------- 批量推理并导出 Excel -------------------------
def infer_fusion_to_xlsx(
    fusion_data: Dict[str, Any],
    model_ckpt_path: str,
    out_xlsx_path: str,
    batch_size: int = 32,
    hidden_dim: int = 128,
    num_classes: int = 4,
    device: Optional[str] = None,
    expert_names: Optional[List[str]] = None,
):
    """
    批量推理函数
    输入:
      - fusion_data: 内存数据字典
      - model_ckpt_path: 模型权重路径
      - out_xlsx_path: 输出 Excel 路径
      - batch_size: 推理批量大小
      - hidden_dim: 模型隐藏维度
      - num_classes: 分类数
      - device: 推理设备
      - expert_names: 专家名称列表
    输出:
      - 返回 pandas.DataFrame，包含每条样本的原始标签、专家预测、融合预测及专家权重
    """
    logger_instance.info(f"开始融合推理结果")
    expert_names = expert_names or ["ux_expert", "ss_expert", "hc_expert"]

    device = pick_device(device)  # 选择设备
    #logger_instance.info(f"[Infer] Device: {device}")

    ds = FusionDictDataset(fusion_data)  # Dataset 封装
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)  # DataLoader

    text_dim = int(ds.text_emb.shape[-1])  # 获取文本嵌入维度
    model = ContextAwareAttentionFusion(text_dim=text_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)

    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f"找不到模型权重: {model_ckpt_path}")
    # 加载模型权重
    state = torch.load(model_ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()  # 推理模式

    rows = []
    with torch.no_grad():  # 禁用梯度计算
        for text_embs, sev_for_model, sev_raw, labels, bug_ids in dl:
            text_embs = text_embs.to(device)  # [B,3,D]
            sev_for_model = sev_for_model.to(device)  # [B,3]
            # ------------------- 模型前向 -------------------
            logits, attn = model(text_embs, sev_for_model)  # logits [B,4]
            pred_ids = torch.argmax(logits, dim=1).detach().cpu().numpy()

            # attn -> expert weights [B,3]
            # ------------------- attention -> 专家权重 -------------------
            w = _attn_to_expert_weights(attn)

            sev_raw_np = sev_raw.detach().cpu().numpy()  # [B,3]
            labels_np = labels.detach().cpu().numpy()  # [B]
            bug_ids = list(bug_ids)

            for i in range(len(bug_ids)):
                tid = bug_ids[i]

                # 真实标签（可能没有）
                true_id = int(labels_np[i])
                true_sev = ID2SEV.get(true_id, "N/A") if true_id >= 0 else "N/A"

                # 专家预测（保留 INVALID）
                ux_id, ss_id, hc_id = [int(x) for x in sev_raw_np[i].tolist()]

                def id_to_label(x: int) -> str:
                    return ID2SEV[x] if x in ID2SEV else INVALID_LABEL

                ux_pred = id_to_label(ux_id)
                ss_pred = id_to_label(ss_id)
                hc_pred = id_to_label(hc_id)

                fusion_pred = ID2SEV[int(pred_ids[i])]

                rows.append({
                    "ticket_id": tid,
                    "original_severity": true_sev,

                    f"{expert_names[0]}_pred": ux_pred,
                    f"{expert_names[1]}_pred": ss_pred,
                    f"{expert_names[2]}_pred": hc_pred,

                    "fusion_pred": fusion_pred,

                    f"{expert_names[0]}_weight": float(w[i, 0]),
                    f"{expert_names[1]}_weight": float(w[i, 1]),
                    f"{expert_names[2]}_weight": float(w[i, 2]),
                })
    # ------------------- DataFrame 输出 -------------------
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(out_xlsx_path) or ".", exist_ok=True)
    df.to_excel(out_xlsx_path, index=False)
    logger_instance.info(f"[Saved] {out_xlsx_path}  (N={len(df)})")

    return df  # 返回 DataFrame
