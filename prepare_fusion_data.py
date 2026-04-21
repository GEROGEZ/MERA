# ------------------------- Python 数据结构与类型注解 -------------------------

# dataclass 装饰器，用于简化类定义，自动生成 __init__, __repr__, __eq__ 等方法
# 在 Case 或其他结构化数据对象中经常使用，便于封装专家推理结果

from format_logger import logger_instance
from typing import List, Dict, Any, Tuple
# 提供类型注解功能
# List[T]: 表示列表元素类型
# Dict[K, V]: 字典 key/value 类型
# Any: 任意类型
# Optional[T]: 可为 None 的类型
# Tuple: 元组类型
# 在函数接口、返回值和类属性声明中增强可读性和静态检查能力

# ------------------------- JSON 解析 -------------------------
import json
# Python 内置库，用于解析和生成 JSON
# # 在本项目中用于：
# #   1. 将专家推理输出（JSON）解析为 Python 字典
# #   2. 将结构化结果保存或加载到 JSONL 文件
#
# # ------------------------- 数值计算与矩阵操作 -------------------------
import numpy as np
# NumPy 库，提供高性能数组和矩阵运算
# 在特征工程阶段，将文本向量或定级索引转为数组，方便后续张量处理

# ------------------------- PyTorch 深度学习框架 -------------------------

# PyTorch 核心库，提供：
#   1. 张量操作（Tensor）
#   2. 自动求导（autograd）
#   3. 模型定义与训练
#   4. GPU/NPU 加速
# 注意：在多 GPU/NPU 环境下，张量和模型必须移动到同一设备


# 华为昇腾 NPU 的 PyTorch 扩展库
# 支持在 Ascend NPU 上运行 PyTorch 模型
# 提升大模型推理和训练速度
# 注意：仅在 NPU 环境可用，CPU/GPU 环境需判断或回退

# ------------------------- 进度条显示 -------------------------
from tqdm import tqdm
# 用于可视化循环处理进度
# 在批量处理大量案例时非常有用，便于监控
# 示例：for case in tqdm(cases): ...

# ------------------------- 文本向量化 -------------------------
from langchain_huggingface import HuggingFaceEmbeddings
# LangChain 提供的向量化接口
# 使用预训练 Embedding 模型将文本或专家 reasoning_path 转换为向量
# 用于：
#   1. 构建 FAISS 向量索引
#   2. Attention Fusion 模型输入
#   3. RAG 检索相似历史工单或系统文档

# ------------------------- 案例数据对象 -------------------------
from case import Case
# 自定义数据类，封装单条 Bug 的信息
# 属性通常包括：
#   - ticket_id: 工单编号
#   - description: 问题描述
#   - original_severity: 原始定级
#   - prompt: 给 LLM 的 Prompt
#   - completion: LLM 输出内容
#   - expert: 专家类型 (UX/SS/HC)
#   - predict_severity: 从 LLM 输出解析的预测等级
# 作用：
#   1. 统一存储专家推理结果
#   2. 支持向量化特征构建
#   3. 用于最终 Attention 融合模型训练或推理

# ------------------------- PyTorch 再导入 -------------------------
import torch

# 已经导入 PyTorch，但这里再次导入通常用于避免环境切换问题
# 确保 torch 和 torch_npu 共同作用
# ========== severity 映射 ==========
SEVERITY_MAP = {"提示": 0, "一般": 1, "严重": 2, "致命": 3}

INVALID_PRED = -1  # 不在四级中的预测 -> 记为预测错误（-1）


def _safe_str(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _extract_reasoning_text(c: Case) -> str:
    """
    你原来 embed 的是 reasoning_path。
    现在 Case 里可能只有 completion（可能是 JSON 字符串）或 description。
    优先策略：
      1) completion 是 JSON 且有 reasoning_path -> 用 reasoning_path
      2) completion 不是 JSON -> 用 completion
      3) 都没有 -> 用 description
      4) 仍为空 -> fallback
    """
    comp = _safe_str(c.completion)
    if comp:
        # 尝试按 JSON 解析（很多 LLM 输出会是 JSON）
        try:
            obj = json.loads(comp)
            if isinstance(obj, dict):
                rp = _safe_str(obj.get("reasoning_path"))
                if rp:
                    return rp
        except Exception:
            pass
        return comp

    desc = _safe_str(c.description)
    if desc:
        return desc

    return "无有效推理路径"


def _extract_pred_sev_str(c: Case) -> str:
    """
    优先用 Case.predict_severity；
    如果你有些 case.predict_severity 没填，但 completion JSON 里有 suggested_severity，也支持兜底。
    """
    ps = _safe_str(c.predict_severity)
    if ps:
        return ps

    comp = _safe_str(c.completion)
    if comp:
        try:
            obj = json.loads(comp)
            if isinstance(obj, dict):
                ss = _safe_str(obj.get("suggested_severity"))
                if ss:
                    return ss
        except Exception:
            pass

    return ""  # 让上层 fallback 成 INVALID_PRED


def _index_by_id(cases: List[Case]) -> Dict[str, Case]:
    d = {}
    for c in cases:
        tid = _safe_str(c.ticket_id)
        if not tid:
            continue
        d[tid] = c
    return d


def build_fusion_dataset_from_case_lists(
    ux_cases: List[Case],
    ss_cases: List[Case],
    hc_cases: List[Case],
    embed_model: HuggingFaceEmbeddings,
    require_all_experts: bool = True,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    输入：三个专家的预测 list[Case]
    输出：可直接供下游使用的“向量化后的数据结构”（不写任何文件）
    """
    logger_instance.info(f"开始融合训练数据")
    ux_map = _index_by_id(ux_cases)
    ss_map = _index_by_id(ss_cases)
    hc_map = _index_by_id(hc_cases)

    ux_ids, ss_ids, hc_ids = set(ux_map.keys()), set(ss_map.keys()), set(hc_map.keys())

    if require_all_experts:
        used_ids = sorted(list(ux_ids & ss_ids & hc_ids))
    else:
        used_ids = sorted(list(ux_ids | ss_ids | hc_ids))

    if len(used_ids) == 0:
        return {
            "text_emb": torch.empty(0, 3, 0),
            "sev_id": torch.empty(0, 3, dtype=torch.long),
            "labels": torch.empty(0, dtype=torch.long),
            "metadata": [],
        }

    # 先收集：N 个样本，每个样本 3 个专家文本 + 3 个 sev_id + label
    metadata: List[str] = []
    labels: List[int] = []
    sev_ids: List[List[int]] = []
    texts_flat: List[str] = []  # 按 [id0_ux, id0_ss, id0_hc, id1_ux, ...] 展平，方便批量 embedding

    skipped_no_label = 0
    skipped_missing_expert = 0
    invalid_pred_cnt = 0

    expert_order: Tuple[Tuple[str, Dict[str, Case]], ...] = (
        ("ux_expert", ux_map),
        ("ss_expert", ss_map),
        ("hc_expert", hc_map),
    )

    for tid in used_ids:
        # 取一个来源的 original_severity（一般三份应该一致）
        # 这里优先 ux -> ss -> hc
        c_for_label = ux_map.get(tid) or ss_map.get(tid) or hc_map.get(tid)
        label_str = _safe_str(c_for_label.original_severity) if c_for_label else ""

        if label_str not in SEVERITY_MAP:
            skipped_no_label += 1
            continue

        per_expert_sev: List[int] = []
        per_expert_text: List[str] = []

        ok = True
        for _, mp in expert_order:
            c = mp.get(tid)
            if c is None:
                if require_all_experts:
                    ok = False
                    break
                else:
                    # 不要求三专家齐全时，缺失专家也算“无效推理 + INVALID_PRED”
                    per_expert_text.append("无有效推理路径")
                    per_expert_sev.append(INVALID_PRED)
                    continue

            # 文本
            per_expert_text.append(_extract_reasoning_text(c))

            # 预测等级
            pred_str = _extract_pred_sev_str(c)
            pred_id = SEVERITY_MAP.get(pred_str, INVALID_PRED)
            if pred_id == INVALID_PRED:
                invalid_pred_cnt += 1
            per_expert_sev.append(pred_id)

        if not ok:
            skipped_missing_expert += 1
            continue

        metadata.append(tid)
        labels.append(SEVERITY_MAP[label_str])
        sev_ids.append(per_expert_sev)

        # 展平文本，后面批量 embedding
        texts_flat.extend(per_expert_text)

    if len(metadata) == 0:
        return {
            "text_emb": torch.empty(0, 3, 0),
            "sev_id": torch.empty(0, 3, dtype=torch.long),
            "labels": torch.empty(0, dtype=torch.long),
            "metadata": [],
        }

    # logger_instance.info(
    #     f"[Fusion] used={len(metadata)} | skipped_no_label={skipped_no_label} | skipped_missing_expert={skipped_missing_expert}")
    # logger_instance.info(f"[Fusion] invalid_pred treated as wrong (sev_id=-1): {invalid_pred_cnt}")

    # ========== 批量向量化 ==========
    vecs = []
    for i in tqdm(range(0, len(texts_flat), batch_size), desc="Embedding"):
        chunk = texts_flat[i:i + batch_size]
        chunk_vecs = embed_model.embed_documents(chunk)  # -> list[list[float]]
        vecs.append(np.asarray(chunk_vecs, dtype=np.float32))

    emb = np.vstack(vecs)  # [N*3, D]
    D = emb.shape[1]
    N = len(metadata)

    emb = emb.reshape(N, 3, D)  # [N, 3, D]

    # ========== 转成 torch ==========
    text_emb = torch.tensor(emb, dtype=torch.float32)  # [N,3,D]
    sev_id_t = torch.tensor(sev_ids, dtype=torch.long)  # [N,3]
    labels_t = torch.tensor(labels, dtype=torch.long)  # [N]
    fusion_data =  {
        "text_emb": text_emb,
        "sev_id": sev_id_t,
        "labels": labels_t,
        "metadata": metadata,
    }
    # file =f"./train_data/datasets.json"
    # with open(file, 'w', encoding='utf-8') as fw:
    #     json.dump(fusion_data, fw, ensure_ascii=False, indent=4)
    return {
        "text_emb": text_emb,
        "sev_id": sev_id_t,
        "labels": labels_t,
        "metadata": metadata,
    }


# ========== 一个更“端到端”的封装：你也可以直接传模型路径进去 ==========
def build_embedder(model_path: str, device: str = "npu") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
