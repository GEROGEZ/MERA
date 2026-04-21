
import os  # 文件路径操作
import warnings
warnings.filterwarnings('ignore')
import pandas as pd  # Excel 读写
import json  # 解析 JSON
from typing import List, Dict, Union  # 类型注解
import torch  #
import torch.optim as optim  #
import torch.nn as nn  #
from tqdm import tqdm  #
from vllm import LLM, SamplingParams  # 高效大模型推理

# --- RAG (检索增强生成) 核心库导入 ---
from langchain_huggingface import HuggingFaceEmbeddings  # 用于将文本转化为向量
from langchain_community.vectorstores import FAISS  # Facebook 开发的向量数据库，用于高效检索
from langchain_core.documents import Document  # 规范化文档对象
# import matplotlib.pyplot as plt
# --- 自定义模块导入 (假设在同一工作目录下) ---
from fusion_model import ContextAwareAttentionFusion  # 核心：上下文感知的注意力融合模型
from parse_data import extract_field  # 用于从 LLM 输出的 JSON 字符串中提取字段
from prepare_fusion_data1 import build_fusion_dataset_from_case_lists, build_embedder  # 数据预处理工具

from case import Case  # 基础数据结构类：Bug 案例

from torch.utils.data import Dataset, DataLoader  # PyTorch 数据加载标准库

import argparse

from format_logger import logger_instance

from config import EMBEDDING_MODEL, BATCH_SIZE, DEVICE,VLLM_MODEL_PATH,TENSOR_PARALLEL_SIZE,MODEL_SAVE_DIR1, FORCE_GEN


parser = argparse.ArgumentParser(description='Parameters of Magnitude',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dap", help="Data Path", default="./data/train_dataset_20260408_17.xlsx")
parser.add_argument("--tvec", help="Train Vector Data Path", default="./vector/history_faiss_index")
parser.add_argument("--svec", help="Sys Vector Data Path", default="./vector/pdf_faiss_index")
args = parser.parse_args()


# ==============================================================================
# 1. 专家数据集类 (ExpertDataset)
# 用于将多专家生成的 Embedding 数据封装为 PyTorch 可识别的 Dataset
# ==============================================================================
class ExpertDataset(Dataset):
    """
    该类负责将预处理后的多专家特征（文本嵌入、定级ID）和标签组织起来。
    """

    def __init__(self, data_dict):
        """
        初始化数据集
        Args:
            data_dict (Dict): 包含预先向量化好的张量字典
                - 'text_emb': 形状为 [N, 3, D]，N是样本数，3是三位专家，D是维度
                - 'sev_id': 形状为 [N, 3]，三位专家各自给出的初步定级 ID
                - 'labels': 形状为 [N]，真实的 Bug 严重级别标签
                - 'metadata': 包含工单 ID 等辅助信息
        """
        # 直接获取已经在内存中的 Tensor
        self.text_embs = data_dict['text_emb']  # Shape: [N, 3, D]
        self.sev_ids = data_dict['sev_id']  # Shape: [N, 3]
        self.labels = data_dict['labels']  # Shape: [N]
        self.metadata = data_dict['metadata']  # List[str], 包含 bug_id

    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        根据索引获取单条训练样本
        Returns:
            Tuple: (文本嵌入, 专家预测级别, 真实标签, 工单ID)
        """
        # 直接通过索引切片获取数据
        # text_embs[idx] -> Shape: [3, D]
        # sev_ids[idx]   -> Shape: [3]
        # labels[idx]    -> Scalar
        # metadata[idx]  -> String (bug_id)

        return self.text_embs[idx], self.sev_ids[idx], self.labels[idx], self.metadata[idx]


# ==============================================================================
# 2. VLLM 模型调用封装类 (VllmModel)
# 负责管理本地大语言模型的加载和批量并行推理
# ==============================================================================
class VllmModel:
    """
    针对 vLLM 推理框架封装的调用类，支持 Tensor Parallel (张量并行) 加速。
    """

    def __init__(self, model_path, tensor_parallel_size, temperature=0.3, topp=1.0):
        """
        初始化 LLM 引擎
        Args:
            model_path (str): 本地模型文件夹路径 (如 DeepSeek 系列)
            tensor_parallel_size (int): 使用几张显卡进行并行计算
            temperature (float): 生成多样性控制，定级任务建议较低值以保持稳定
            topp (float): 核采样阈值
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.temperature = temperature
        self.topp = topp
        # 读取模型自带的对话模板 (chat_template)，确保 prompt 格式符合模型预训练要求
        self.chat_template = json.load(
            open(f'{model_path}/tokenizer_config.json', 'r', encoding='utf-8')
        )['chat_template']
        # 配置采样参数：设定最大生成 Token 数，防止死循环或过长输出
        self.sampling_params = SamplingParams(temperature=temperature, top_p=topp, max_tokens=3072)
        # 初始化 vLLM 核心引擎：
        # gpu_memory_utilization 为显存占用率，max_model_len 为最大上下文长度
        self.llm = LLM(model=self.model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.8,
                       max_model_len=4096)

    def call_chat(self, prompts: Union[List[str], str]):
        """
        批量执行对话生成任务
        Args:
            prompts: 单个字符串或字符串列表
        Returns:
            real_outputs: LLM 生成的纯文本答案列表
        """
        # 将输入格式化为标准对话列表结构 [{'role': 'user', 'content': '...'}]
        if isinstance(prompts, str):
            applied_prompts = [
                {
                    'role': 'user',
                    'content': prompts
                }
            ]
        elif isinstance(prompts, List):
            applied_prompts = []
            for prompt in prompts:
                applied_prompts.append(
                    [
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ]
                )
        # 调用 vllm 进行批量推理
        model_outputs = self.llm.chat(applied_prompts, self.sampling_params, chat_template=self.chat_template)
        real_outputs = []
        for model_output in model_outputs:
            prompt = model_output.prompt
            generated_text = model_output.outputs[0].text
            # 提取模型生成的文本内容，舍弃 prompt 部分和 token ID
            real_outputs.append(generated_text)

        return real_outputs


# ==============================================================================
# 3. 全局配置与文件路径定义
# ==============================================================================


# 定义 Few-Shot 样本格式化模板
EXAMPLE = """#####
样例{index}

问题: {description}
定级结果: {risk_level}

"""  ####历史工单定级的数据


# ------------------------------------------------------------------

# 全局变量初始化


# ==============================================================================
# 4. RAG 检索处理器 (RAGHandler)
# 负责管理两个独立的本地 FAISS 数据库并提供相似度搜索功能
# ==============================================================================
class RAGHandler:
    """管理两个独立的 FAISS 索引并提供检索接口"""

    def __init__(self, bug_index_path: str, doc_index_path: str, embedding_model_name: str):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.bug_store = self._load_store(bug_index_path, "历史工单")
        self.doc_store = self._load_store(doc_index_path, "系统文档")

    def _load_store(self, path: str, name: str):
        if os.path.exists(path):
            logger_instance.info(f"正在加载 {name} 向量索引: {path}...")
            # 必须设置 allow_dangerous_deserialization=True
            return FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            raise FileNotFoundError(f"找不到 {name} 向量索引路径: {path}。请先完成 Phase 1 数据的存储。")

    def retrieve_historical_bugs(self, query: str, k: int = 3) -> str:
        """为 HC 专家检索相似工单并格式化输出"""
        results = self.bug_store.similarity_search_with_score(query, k=k)
        context_lines = []
        for doc, _ in results:
            context_lines.append(
                f"[Bug ID: {doc.metadata.get('问题序号', 'N/A')}, "
                f"历史定级: {doc.metadata.get('级别', '未知')}, "
                f"问题描述: {doc.page_content[:100]}...]"  # 截断描述以防过长
            )
        return "历史相似工单记录 (RAG 检索结果):\n" + "\n".join(context_lines)

    def retrieve_system_context(self, query: str, k: int = 2) -> str:
        """为 SS 专家检索增强型文档片段并格式化输出"""
        results = self.doc_store.similarity_search_with_score(query, k=k)
        context_lines = []
        for doc, _ in results:
            context_lines.append(
                f"--- 来源文件: {doc.metadata.get('source_file')}, 页码: {doc.metadata.get('page', 'N/A')} ---\n"
                f"{doc.page_content}"
            )
        return "增强型系统设计知识 (RAG 检索结果):\n" + "\n".join(context_lines)


# ==============================================================================
# 5. 动态 Bug 报告处理系统 (BugReportRAGSystem)
# 包含数据的 Excel 读取、清洗、入库及增量更新逻辑
# ==============================================================================
class BugReportRAGSystem:
    """管理多个 FAISS 索引并提供统一检索接口"""

    def __init__(self, excel_path: str = "/", embedding_model_name: str = EMBEDDING_MODEL):
        """
        初始化 RAG 系统
        """
        self.excel_path = excel_path
        self.embedding_model_name = embedding_model_name
        self.vector_store = None
        self.embeddings = None

        # 初始化向量模型 (部署在 CPU 上以节省显存给 LLM)
        logger_instance.info(f"正在加载 Embedding 模型: {embedding_model_name} ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger_instance.info("模型加载完成。")

    def load_and_preprocess_data(self) -> List[Document]:
        """
        读取 Excel 并转换为 Document 对象
        """
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"找不到文件: {self.excel_path}")

        logger_instance.info("正在读取并清洗数据...")
        df = pd.read_excel(self.excel_path)
        df.dropna(subset=['问题单描述'], inplace=True)

        documents = []
        for _, row in df.iterrows():
            bug_id = str(row['问题序号'])
            description = str(row['问题单描述']).strip()
            severity = str(row['级别']).strip()

            metadata = {
                "bug_id": bug_id,
                "severity": severity
            }
            doc = Document(page_content=description, metadata=metadata)
            documents.append(doc)

        logger_instance.info(f"数据处理完成，共加载 {len(documents)} 条有效工单。")
        return documents

    def build_or_update_vector_store(self, index_path="faiss_index"):
        """
        构建向量数据库，如果已存在则增量更新
        """
        documents = self.load_and_preprocess_data()

        if os.path.exists(index_path):
            logger_instance.info("检测到已有本地索引，加载并更新...")
            self.load_index(index_path)

            # 获取已存在的 bug_id
            existing_ids = {doc.metadata["bug_id"] for doc in self.vector_store.docstore._dict.values()}

            # 筛选新文档
            new_documents = [doc for doc in documents if doc.metadata["bug_id"] not in existing_ids]

            if new_documents:
                logger_instance.info(f"新增 {len(new_documents)} 条工单，追加到索引...")
                self.vector_store.add_documents(new_documents)
                self.save_index(index_path)
            else:
                logger_instance.info("没有新的工单需要更新。")
        else:
            logger_instance.info("本地索引不存在，正在创建新的向量数据库...")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.save_index(index_path)
            logger_instance.info("向量数据库构建完成。")

    def save_index(self, folder_path="faiss_index"):
        """保存索引到本地"""
        if self.vector_store:
            self.vector_store.save_local(folder_path)
            logger_instance.info(f"索引已保存至 {folder_path}")

    def load_index(self, folder_path="faiss_index"):
        """从本地加载索引"""
        if os.path.exists(folder_path):
            self.vector_store = FAISS.load_local(
                folder_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger_instance.info("索引加载成功。")
        else:
            logger_instance.info("未找到本地索引，请先构建。")

    def retrieve_similar_bugs(self, query: str, k: int = 3) -> List[Dict]:
        """检索相似工单"""
        if not self.vector_store:
            raise ValueError("向量数据库未初始化！")

        results = self.vector_store.similarity_search_with_score(query, k=k)

        retrieved_info = []
        for doc, score in results:
            info = {
                "bug_id": doc.metadata.get("bug_id"),
                "description": doc.page_content,
                "severity": doc.metadata.get("severity"),
                "similarity_score": round(score, 4)
            }
            retrieved_info.append(info)

        return retrieved_info


# ==============================================================================
# 6. 专家 Prompt 模板定义
# 为不同领域的专家模型提供不同的上下文和角色定义
# ==============================================================================
OUTPUT_JSON_SCHEMA = """
{{
  "expert_name": "专家名称 (UX/SS/HC)",
  "reasoning_path": "详细的推理过程，必须引用上下文知识",
  "suggested_severity": "致命|严重|一般|提示",
  "confidence_score": 0.0 to 1.0
}}
"""
# UX (用户体验) 专家：关注流程中断和用户感知
EXPERT_UX_TEMPLATE = lambda desc: f"""
你是一位高级用户体验 (UX) 分析师，专注于移动系统 Bug 对**用户使用流程**和**感知**的影响。你的目标是评估该 Bug 导致的**功能受损程度、用户流程中断的严重性**以及**用户感知的负面程度**。忽略任何底层代码或架构信息。请务必输出一个**合法的 JSON 对象**。
请严格根据以下四个等级进行判断：致命、严重、一般、提示。
[输入 Bug 工单]
问题描述: {desc}
[输出要求]
你必须输出一个包含以下键值的 JSON 对象，格式如下：
{OUTPUT_JSON_SCHEMA}
"""

# SS (系统稳定性) 专家：关注崩溃、资源消耗，需引用 RAG 检索的文档上下文
EXPERT_SS_TEMPLATE = lambda desc, doc_context: f"""
你是一位资深的系统稳定性与架构合规分析师，专注于底层代码崩溃、资源泄露、死锁、以及是否违反了**系统设计规范**。你的判断必须高度依赖提供的系统架构知识。请务必输出一个**合法的 JSON 对象**。
请严格根据以下四个等级进行判断：致命、严重、一般、提示。
[输入 Bug 工单与系统背景知识]
问题描述: {desc}
---
{doc_context}
---
[输出要求]
你必须输出一个包含以下键值的 JSON 对象，格式如下：
{OUTPUT_JSON_SCHEMA}
"""

# HC (历史合规) 专家：关注定级一致性，需引用检索到的相似历史样本
EXPERT_HC_TEMPLATE = lambda desc, example_str: f"""
你是一位测试流程合规官和历史数据审计专家，你的职责是确保本次 Bug 定级与**历史已定级工单**保持高度一致性。你的推理过程必须严格引用检索到的历史记录，以避免定级标准漂移。请务必输出一个**合法的 JSON 对象**。
请严格根据以下四个等级进行判断：致命、严重、一般、提示。
[输入 Bug 工单与历史数据]
问题描述: {desc}
---
历史工单
{example_str}
---
[输出要求]
你必须输出一个包含以下键值的 JSON 对象，格式如下：
{OUTPUT_JSON_SCHEMA}
"""

# ==============================================================================
# 7. 核心业务逻辑：专家数据生成 (generate_expert_data)
# ==============================================================================
from dataclasses import asdict


# -------------------------- 核心数据生成逻辑 --------------------------
def dump_cases(casel: list[Case], expert_name):
    if not os.path.exists(f"./data/{expert_name}_cases.json"):
        case_d = {}
    else:
        with open(f"./data/{expert_name}_cases.json", 'r', encoding='utf-8') as fr:
            case_d = json.load(fr)
    for case in casel:
        tid = str(case.ticket_id)
        if tid not in case_d.keys():
            case_d[tid] = asdict(case)
    with open(f"./data/{expert_name}_cases.json", 'w', encoding='utf-8') as fw:
        json.dump(case_d, fw, ensure_ascii=False, indent=4)


def load_cases(expert_name):
    cases={}
    logger_instance.info(f"加载训练集定级数据-{expert_name}")  # 打印加载成功并显示工单条数
    with open(f"./train_data/{expert_name}_cases.json", 'r', encoding='utf-8') as fr:
        cased = json.load(fr)
    for tick, info in cased.items():
        cases[tick] = Case(**info)
    return cases


def generate_expert_data(df: pd.DataFrame, expert_name: str, template_func: callable, rag_handler: RAGHandler,
                         index_path: str, llm_generator=None):
    """
    该函数执行以下流水线：
    1. 根据专家角色检索相关知识 (RAG)
    2. 填充模板构造 Prompts
    3. 调用 VLLM 批量推理
    4. 解析 JSON 输出并封装成 Case 对象列表
    """
    logger_instance.info(f"\n--- 正在为专家: {expert_name} 生成训练数据 ---")

    batch_prompts = []
    batch_description = []

    history_rag_system = BugReportRAGSystem()
    history_rag_system.load_index(folder_path=index_path)
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"构建 {expert_name} 批处理 Prompts"):
        bug_desc = str(row['问题单描述'])

        # 1. 根据专家类型获取 RAG 上下文
        if expert_name == 'UX':
            final_prompt = template_func(bug_desc)
        elif expert_name == 'SS':
            # 实时 RAG 检索：系统文档知识
            doc_context = rag_handler.retrieve_system_context(bug_desc)
            final_prompt = template_func(bug_desc, doc_context)
        elif expert_name == 'HC':
            # 实时 RAG 检索：历史工单记录
            # history_rag_system  = BugReportRAGSystem()
            # history_rag_system.load_index(folder_path=index_path)
            results = history_rag_system.retrieve_similar_bugs(bug_desc, k=5)
            # 构建历史工单数据字符串
            example_str = "".join(EXAMPLE.format(
                index=index + 1,
                description=res['description'],
                risk_level=res['severity']

            )
                                  for index, res in enumerate(results))

            final_prompt = template_func(bug_desc, example_str)
        else:
            continue
        batch_description.append(bug_desc)
        batch_prompts.append(final_prompt)

    # 2. VLLM 批处理推理
    logger_instance.info(f"开始 VLLM 批处理推理，总任务数: {len(batch_description)} {len(batch_prompts)}")
    llm_outputs = []

    # 3. 结构化处理和保存
    infer_data_list: list[Case] = []
    for i in tqdm(range(0, len(batch_prompts), BATCH_SIZE), desc=f"VLLM 批处理 ({expert_name})"):
        batch = batch_prompts[i:i + BATCH_SIZE]
        # 调用用户的 VllmModel 实例
        outputs = llm_generator.call_chat(batch)
        llm_outputs.extend(outputs)
        infer_data_cur_list: list[Case] = []
        for j in range(len(batch)):
            reali = i + j
            # logger_instance.info(f"构建{i} 组 {j} : {BATCH_SIZE} {reali}")
            description = batch_description[reali]
            prompt = batch_prompts[reali]
            original_data = df.iloc[reali].to_dict()
            try:
                # 找到 JSON 块的开始和结束
                json_str = outputs[j].strip().split("```json")[1].split("```")[0].strip()
                case1: Case = Case(
                    ticket_id=original_data['问题序号'],
                    description=description,
                    original_severity=original_data['级别'],
                    prompt=prompt,
                    completion=json_str,
                    expert=expert_name,
                    predict_severity=extract_field(json_str)
                )

            except Exception as e:
                logger_instance.info(
                    f"\n[警告] Bug ID {original_data['问题序号']} JSON 解析失败: {e}. 原始输出: {outputs[:100]}...")
                case1: Case = Case(
                    ticket_id=original_data['问题序号'],
                    description=description,
                    original_severity=original_data['级别'],
                    prompt=prompt,
                    expert=expert_name,
                    predict_severity=extract_field(json_str))
            infer_data_cur_list.append(case1)
        dump_cases(infer_data_cur_list, expert_name)
        infer_data_list.extend(infer_data_cur_list)

    # 4. 保存为 JSONL 格式

    logger_instance.info(f"✅ 专家 {expert_name} 数据集生成完成，共 {len(infer_data_list)} 条")
    return infer_data_list


# ==============================================================================
# 8. 融合模型训练主函数 (train_main)
# 使用 PyTorch 训练最终的 Attention 融合层
# ==============================================================================
def train_main(train_data):
    # 1. 加载封装好的特征数据
    train_dataset = ExpertDataset(train_data)  # 训练集数据处理

    logger_instance.info(f"训练集大小: {len(train_dataset)}")  # 打印形状用于确认

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 2. 初始化模型
    model = ContextAwareAttentionFusion(text_dim=512, hidden_dim=128, num_classes=4).to(DEVICE)  # 模型初始化

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 3. 训练循环
    EPOCHS = 20

    # 定义保存路径的基础名称（可以根据需要修改）
    if not os.path.exists(MODEL_SAVE_DIR1):
        os.makedirs(MODEL_SAVE_DIR1)

    logger_instance.info("\n--- 开始训练融合模型 (无验证模式) ---")
    save_path = ''
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for text_embs, sev_ids, labels, _ in train_loader:  # 训练时忽略 bug_id
            text_embs, sev_ids, labels = text_embs.to(DEVICE), sev_ids.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(text_embs, sev_ids)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        # logger_instance.info(f"")

        # --- 修改点：每 5 个 Epoch 保存一次 ---
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(MODEL_SAVE_DIR1, f"fusion_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            logger_instance.info(f">>> Checkpoint saved: {save_path}. Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")
    # logger_instance.info(f"TLS {EPOCHS}: {losses}")

    logger_instance.info("训练完成")
    return save_path, model


# ==============================================================================
# 9. 程序入口 (Main Execution)
# ==============================================================================
def train_and_save(train_path, train_vector, sys_vector):
    # ------------------- A. 加载原始 Excel 数据 -------------------
    # A. 准备数据源
    try:
        df = pd.read_excel(train_path)  # 使用 pandas 读取 Excel 文件，返回 DataFrame

        logger_instance.info(f"成功加载原始数据，共 {len(df)} 条。")  # 打印加载成功并显示工单条数
    except Exception as e:
        logger_instance.info(
            f"致命错误：无法加载或解析 Excel 文件 '{train_path}'。请检查路径和格式。")  # 如果 Excel 文件不存在或格式错误，打印错误并退出
        exit()
    df['问题序号'] = df['问题序号'].astype('str')


    index_path = train_vector  # 保存历史工单索引路径，用于 HC 专家检索
    FORCE_GEN = False
    
    # 1. 为 Expert 1 (UX) 生成数据 (无需 RAG Handler)
    if FORCE_GEN:
        # B. 初始化重型组件：LLM 引擎与 RAG 检索器
        logger_instance.info(f"初始化 VLLM 模型：{VLLM_MODEL_PATH}...")  # 提示正在初始化 vLLM 大模型
        llm_generator = VllmModel(model_path=VLLM_MODEL_PATH,
                                  tensor_parallel_size=TENSOR_PARALLEL_SIZE)  # 创建 VllmModel 实例，用于批量推理
        # model_path: 模型权重路径
        # tensor_parallel_size: GPU 并行数量，用于多卡推理
        # 初始化 RAG Handler
        rag_handler = RAGHandler(
            bug_index_path=train_vector,  # 历史工单 FAISS 索引路径
            doc_index_path=sys_vector,  # 系统组件文档 FAISS 索引路径
            embedding_model_name=EMBEDDING_MODEL  # Embedding 模型路径，用于文本向量化
        )
        ux_case_list = generate_expert_data(
            df=df,  # 原始工单 DataFrame
            expert_name='UX',  # 当前生成专家为 UX
            template_func=EXPERT_UX_TEMPLATE,  # UX 专家 Prompt 模板
            rag_handler=rag_handler,  # 传入 handler 但不使用检索方法
            index_path=index_path,  # 历史工单索引路径（未被 UX 使用）
            llm_generator=llm_generator
        )
        # 2. 为 Expert 2 (SS) 生成数据
        ss_case_list = generate_expert_data(
            df=df,  # 传入参数
            expert_name='SS',  # 传入参数
            template_func=lambda desc, doc_context: EXPERT_SS_TEMPLATE(desc, doc_context),  # 传入参数
            rag_handler=rag_handler,  # 传入参数
            index_path=index_path,  # 历史工单索引路径（未被 UX 使用）
            llm_generator=llm_generator  # 传入参数
        )

        # 3. 为 Expert 3 (HC) 生成数据
        hc_case_list = generate_expert_data(
            df=df,  # 原始工单 DataFrame
            expert_name='HC',  # 当前专家为历史一致性 HC
            template_func=lambda desc, example_str: EXPERT_HC_TEMPLATE(desc, example_str),
            # HC 专家 Prompt 模板（依赖历史工单 RAG 上下文）
            rag_handler=rag_handler,  # 用于实时检索历史工单
            index_path=index_path,  # 历史工单索引路径（未被 UX 使用）
            llm_generator=llm_generator  # 历史工单索引路径
        )
    else:
        
        ux_case_ = load_cases('UX')
        ux_case_list = []
        for ind, row in df.iterrows():
            tick = str(row['问题序号'])
            if not ux_case_.get(tick):
                logger_instance.error(f"UX-{tick}")
                continue
            ux_case_list.append(ux_case_.get(tick))
        ss_case_ = load_cases('SS')
        ss_case_list = []
        for ind, row in df.iterrows():
            tick = str(row['问题序号'])
            if not ss_case_.get(tick):
                logger_instance.error(f"SS-{tick}")
                continue
            ss_case_list.append(ss_case_.get(tick))

        # 3. 为 Expert 3 (HC) 生成数据
        hc_case_ = load_cases('HC')
        hc_case_list = []
        for ind, row in df.iterrows():
            tick = str(row['问题序号'])
            if not hc_case_.get(tick):
                logger_instance.error(f"HC-{tick}")
                continue
            hc_case_list.append(hc_case_.get(tick))

    # 对刚刚生成的数据进行向量化并且保存
    logger_instance.info(f"Train Data Size: {len(ux_case_list)} {len(ss_case_list)} {len(hc_case_list)}")
    logger_instance.info(f"加载向量模型 {EMBEDDING_MODEL}")
    embedder = build_embedder(EMBEDDING_MODEL, device='npu')  # 初始化 Embedding 模型
    # device='npu' 指定使用 NPU 加速
    fusion_data = build_fusion_dataset_from_case_lists(
        ux_cases=ux_case_list,  # 传入参数
        ss_cases=ss_case_list,  # 传入参数
        hc_cases=hc_case_list,  # 传入参数
        embed_model=embedder,  # 传入参数
        require_all_experts=True,  # 传入参数
        batch_size=BATCH_SIZE,  # 批处理大小，控制显存使用
    )

    savep, model = train_main(fusion_data)  # 训练
    print(savep)


train_and_save(args.dap, args.tvec, args.svec)
