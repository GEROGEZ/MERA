import datetime
import os  # 文件路径操作
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import json
from typing import List, Dict, Union
from tqdm import tqdm
from vllm import LLM, SamplingParams
# --- 检索增强生成 (RAG) 相关库 ---
from langchain_huggingface import HuggingFaceEmbeddings  # 文本向量化
from langchain_community.vectorstores import FAISS  # 向量数据库操作
from langchain_core.documents import Document  # 标准文档结构

# --- 自定义业务逻辑模块导入 ---
from parse_data import extract_field  # 从模型返回文本中解析 severity 字段
from prepare_fusion_data import build_fusion_dataset_from_case_lists, build_embedder  # 特征工程
from fusion_infer import infer_fusion_to_xlsx  # 融合层推理工具
from case import Case  # 封装单条 Bug 案例的类
import time

import argparse

from format_logger import logger_instance
from config import BATCH_SIZE, EMBEDDING_MODEL, VLLM_MODEL_PATH, TENSOR_PARALLEL_SIZE, FORCE_GEN

parser = argparse.ArgumentParser(description='Parameters of Magnitude',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dap", help="Data Path", default="./data/test_dataset_20260408_17.xlsx")
parser.add_argument("--tvec", help="Data Path", default="./vector/history_faiss_index")
parser.add_argument("--svec", help="Data Path", default="./vector/pdf_faiss_index")
parser.add_argument("--tpth", help="Data Path", default="./modelz/fusion_model_epoch_4.pth")
args = parser.parse_args()


# ==============================================================================
# 1. VllmModel：高效大模型推理引擎封装
# ==============================================================================
class VllmModel:
    """
    基于 vLLM 框架封装的模型调用类，实现高性能批量推理。
    """

    def __init__(self, model_path, tensor_parallel_size, temperature=0.3, topp=1.0):
        """
        初始化 LLM 引擎。
        :param model_path: 模型权重存放路径。
        :param tensor_parallel_size: GPU 并行数量（切分模型的显卡数）。
        :param temperature: 温度系数。0.3 表示在稳定性和创造性之间取得平衡。
        :param topp: 核采样参数，用于控制生成多样性。
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.temperature = temperature
        self.topp = topp

        # 自动从模型目录加载聊天模板 (chat_template)，确保 System/User 角色转换正确
        self.chat_template = json.load(
            open(f'{model_path}/tokenizer_config.json', 'r', encoding='utf-8')
        )['chat_template']

        # SamplingParams：控制生成长度和策略。max_tokens 设为 3072 以容纳长思维链。
        self.sampling_params = SamplingParams(temperature=temperature, top_p=topp, max_tokens=3072)

        # 初始化推理引擎：利用 80% 的 GPU 显存，最大上下文长度 4096。
        self.llm = LLM(model=self.model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.8,
                       max_model_len=4096)

    def call_chat(self, prompts: Union[List[str], str]):
        """
        批量调用对话功能。
        :param prompts: 待处理的 Prompt 列表。
        :return: 模型生成的纯文本回复列表。
        """
        # 将输入文本包装成模型要求的 Chat 结构（User 角色
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
        # 核心推理调用
        model_outputs = self.llm.chat(applied_prompts, self.sampling_params, chat_template=self.chat_template)
        real_outputs = []
        for model_output in model_outputs:
            prompt = model_output.prompt
            generated_text = model_output.outputs[0].text
            real_outputs.append(generated_text)
        # 提取推理结果中的生成文本内容
        return real_outputs


# ==============================================================================
# 2. 全局环境与路径配置
# ==============================================================================

EXAMPLE = """#####
样例{index}

问题: {description}
定级结果: {risk_level}

"""


####历史工单定级的数据
# ------------------------------------------------------------------

# 全局变量初始化


# ==============================================================================
# 3. RAGHandler 与 BugReportRAGSystem：双索引管理逻辑
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
                f"[Bug ID: {doc.metadata.get('问题单号', 'N/A')}, "
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


##设计一个读取本地历史工单的系统类
class BugReportRAGSystem:
    def __init__(self, excel_path: str = "/", embedding_model_name: str = EMBEDDING_MODEL):
        """
        初始化 RAG 系统
        """
        self.excel_path = excel_path
        self.embedding_model_name = embedding_model_name
        self.vector_store = None
        self.embeddings = None

        # 初始化 Embedding 模型
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
            bug_id = str(row['问题单号'])
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
# 4. 专家 Prompt 模板与生成引擎
# ==============================================================================


# JSON 约束 schema：确保模型输出能被程序稳定解析
# 定义 JSON 输出结构，用于 Instruct (LoRA 训练的数据格式)
OUTPUT_JSON_SCHEMA = """
{{
  "expert_name": "专家名称 (UX/SS/HC)",
  "reasoning_path": "详细的推理过程，必须引用上下文知识",
  "suggested_severity": "致命|严重|一般|提示",
  "confidence_score": 0.0 to 1.0
}}
"""
# 定义三种专家的 Prompt 生成函数（基于 Lambda 封装）
# UX 专家模板 (无需 RAG 上下文)
EXPERT_UX_TEMPLATE = lambda desc: f"""
你是一位高级用户体验 (UX) 分析师，专注于移动系统 Bug 对**用户使用流程**和**感知**的影响。你的目标是评估该 Bug 导致的**功能受损程度、用户流程中断的严重性**以及**用户感知的负面程度**。忽略任何底层代码或架构信息。请务必输出一个**合法的 JSON 对象**。
请严格根据以下四个等级进行判断：致命、严重、一般、提示。
[输入 Bug 工单]
问题描述: {desc}
[输出要求]
你必须输出一个包含以下键值的 JSON 对象，格式如下：
{OUTPUT_JSON_SCHEMA}
"""

# SS 专家模板 (需要 系统文档 RAG 上下文)
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

# HC 专家模板 (需要 历史工单 RAG 上下文)
EXPERT_HC_TEMPLATE = lambda desc, example_str: f"""你是一位测试流程合规官和历史数据审计专家，你的职责是对 Bug 工单进行定级（致命、严重、一般、提示），并严格参考历史已定级工单以保持定级标准的一致性。
 
请务必输出一个**合法的 JSON 对象**。
 
[输入 Bug 工单与历史数据]
问题描述: {desc}
---
历史工单
{example_str}
---
 
[定级流程 - 请严格按步骤执行]
 
**第一步：逐条相似性判断**
逐一检查每个历史样例，判断其与当前工单是否描述了**同一类功能缺陷或同一类用户场景**。
判断标准：
- "相似"：涉及相同的功能模块且问题表现类似（如都是旋转锁定失效、都是蓝牙连接问题等）
- "不相似"：虽然有个别关键词重叠，但描述的是不同功能或不同类型的问题
 
**第二步：筛选相似样例**
从所有历史样例中，仅保留与当前工单**相似**的样例。完全忽略不相似的样例，不相似的样例不应影响你的定级判断。
 
**第三步：基于筛选结果定级**
 
- **如果存在相似的历史样例**：
  统计相似样例中各定级的分布。你的定级**必须**与相似样例中**出现次数最多**的定级保持一致。这是强制要求，不允许偏离。
  在推理过程中必须明确说明：哪些样例相似、它们的定级是什么、你选择了哪个定级及原因。
 
- **如果没有任何相似的历史样例**：
  **完全独立定级**，不参考任何历史工单，仅依据问题描述本身做出判断。
 
[输出要求]
你必须输出一个包含以下键值的 JSON 对象，格式如下：
{OUTPUT_JSON_SCHEMA}
"""
from dataclasses import asdict


# -------------------------- 核心数据生成逻辑 --------------------------
def dump_cases(casel: list[Case], expert_name):
    file =f"./data/{expert_name}_cases_zym_rag_5_temperature_0.3_new_prompt_v2.json"
    if not os.path.exists(file):
        case_d = {}
    else:
        with open(file, 'r', encoding='utf-8') as fr:
            case_d = json.load(fr)
    for case in casel:
        tid = str(case.ticket_id)
        if tid not in case_d.keys():
            case_d[tid] = asdict(case)
        else:
            if case_d[tid]['predict_severity'] != case.predict_severity:
                logger_instance.warning(f"{tid} 定级结果变更 {case_d[tid]['predict_severity']} -> {case.predict_severity}")

    with open(file, 'w', encoding='utf-8') as fw:
        json.dump(case_d, fw, ensure_ascii=False, indent=4)


def batch_generate():
    tics = []
    with open('./logs/INFO.log', 'r', encoding='utf-8') as fr:
        for x in fr:
            if 'ERROR    | __main__:do_infe' in x:
                tic = x.split('-')[-1].strip('\n').strip()
                if tic not in tics:
                    tics.append(tic)
    return tics


# ticx = batch_generate()
# logger_instance.info(f"REST:{len(ticx)}")


def generate_expert_data(df: pd.DataFrame, expert_name: str, template_func: callable, rag_handler: RAGHandler,
                         index_path: str, llm_generator=None):
    """
    核心循环：为给定专家生成所有 Bug 的推理数据。
    """
    logger_instance.info(f"\n--- 正在为专家: {expert_name} 生成训练数据 ---")

    batch_prompts = []
    batch_description = []
    # df = df[df['问题单号'].isin(ticx)]
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
    logger_instance.info(f"开始 VLLM 批处理推理，总任务数: {len(batch_prompts)}")
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
            description = batch_description[reali]
            prompt = batch_prompts[reali]
            original_data = df.iloc[reali].to_dict()
            try:
                # 找到 JSON 块的开始和结束
                json_str = outputs[j].strip().split("```json")[1].split("```")[0].strip()
                case1: Case = Case(
                    ticket_id=original_data['问题单号'],
                    description=description,
                    original_severity=original_data['级别'],
                    prompt=prompt,
                    completion=json_str,
                    expert=expert_name,
                    predict_severity=extract_field(json_str)
                )

            except Exception as e:
                logger_instance.info(
                    f"\n[警告] Bug ID {original_data['问题单号']} JSON 解析失败: {e}. 原始输出: {outputs[:100]}...")
                case1: Case = Case(
                    ticket_id=original_data['问题单号'],
                    description=description,
                    original_severity=original_data['级别'],
                    prompt=prompt,
                    expert=expert_name,
                    predict_severity=extract_field(json_str))
            infer_data_cur_list.append(case1)
        dump_cases(infer_data_cur_list, expert_name)
        infer_data_list.extend(infer_data_cur_list)

    # 4. 保存为 JSONL 格式
    logger_instance.info(f"✅ 专家 {expert_name} 推理完成，共 {len(infer_data_list)} 条")
    return infer_data_list


def load_cases(expert_name):
    cases = {}
    with open(f"./data/{expert_name}_cases_zym_rag_5_temperature_0.2_new_prompt_v1.json", 'r', encoding='utf-8') as fr:
        cased = json.load(fr)
    for tick, info in cased.items():
        cases[tick] = Case(**info)
    return cases


def do_infer(test_file, tvec, svec, best_model_path):
    # 加载数据
    try:
        df = pd.read_excel(test_file)
        logger_instance.info(f"成功加载原始数据，共 {len(df)} 条。")
    except Exception as e:
        logger_instance.info(f"致命错误：无法加载或解析 Excel 文件 '{test_file}'。请检查路径和格式。")
        exit()
    df['问题单号'] = df['问题单号'].astype('str')
    # df = df[:1]
    # 初始化 VLLM 模型实例
    # 开始计算时间
    start_time = time.time()
    FORCE_GEN = True
    if FORCE_GEN:
        logger_instance.info(f"初始化 VLLM 模型：{VLLM_MODEL_PATH}...")
        llm_generator = VllmModel(model_path=VLLM_MODEL_PATH, tensor_parallel_size=TENSOR_PARALLEL_SIZE)

        # 初始化 RAG Handler
        rag_handler = RAGHandler(
            bug_index_path=tvec,
            doc_index_path=svec,
            embedding_model_name=EMBEDDING_MODEL
        )
        start_time = time.time()
        index_path = tvec
        hc_case_list = generate_expert_data(
                    df=df,
                    expert_name='HC',
                    template_func=lambda desc, example_str: EXPERT_HC_TEMPLATE(desc, example_str),
                    rag_handler=rag_handler,
                    index_path=index_path, llm_generator=llm_generator
                )
        ux_case_list = generate_expert_data(
            df=df,
            expert_name='UX',
            template_func=EXPERT_UX_TEMPLATE,
            rag_handler=rag_handler,  # 传入 handler 但不使用检索方法
            index_path=index_path, llm_generator=llm_generator
        )

        # 2. 为 Expert 2 (SS) 生成数据
        ss_case_list = generate_expert_data(
            df=df,
            expert_name='SS',
            template_func=lambda desc, doc_context: EXPERT_SS_TEMPLATE(desc, doc_context),
            rag_handler=rag_handler,
            index_path=index_path, llm_generator=llm_generator
        )
        
    else:
        # 3. 为 Expert 3 (HC) 生成数据
        hc_case_ = load_cases('HC')
        hc_case_list = []
        for ind, row in df.iterrows():
            tick = str(row['问题单号'])
            if not hc_case_.get(tick):
                logger_instance.error(f"HC-{tick}")
                continue
            hc_case_list.append(hc_case_.get(tick))
        # 1. 为 Expert 1 (UX) 生成数据 (无需 RAG Handler)
        ux_case_ = load_cases('UX')
        ux_case_list = []
        for ind, row in df.iterrows():
            tick = str(row['问题单号'])
            if not ux_case_.get(tick):
                logger_instance.error(f"UX-{tick}")
                continue
            ux_case_list.append(ux_case_.get(tick))

        ss_case_ = load_cases('SS')
        ss_case_list = []
        for ind, row in df.iterrows():
            tick = str(row['问题单号'])
            if not ss_case_.get(tick):
                logger_instance.error(f"SS-{tick}")
                continue
            ss_case_list.append(ss_case_.get(tick))

        

    # 测试集长度
    case_num = len(ux_case_list)

    # 对刚刚生成的数据进行向量化并且保存
    embedder = build_embedder(EMBEDDING_MODEL, device='npu')
    fusion_data = build_fusion_dataset_from_case_lists(
        ux_cases=ux_case_list,
        ss_cases=ss_case_list,
        hc_cases=hc_case_list,
        embed_model=embedder,
        require_all_experts=True,
        batch_size=BATCH_SIZE,
    )

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_xlsx = F"./output/ds_32B_final_{now}.xlsx"
    out_xlsx_detailed = f'./output/ds_32B_final_{now}_detailed.xlsx'
    infer_fusion_to_xlsx(
        fusion_data=fusion_data,
        model_ckpt_path=best_model_path,
        out_xlsx_path=out_xlsx,
        batch_size=32,
        hidden_dim=128,
        num_classes=4,
        device="npu",  # 自动选 npu/cuda/cpu；你也可以写 "npu"
    )

    # 1. 准备所有列表
    all_lists = [ux_case_list, ss_case_list, hc_case_list]

    # 2. 构建字典映射
    # 描述字典：因为 description 都一样，我们直接汇总所有列表，后进的会覆盖先进的（结果一致）
    desc_map = {case.ticket_id: case.description for cases in all_lists for case in cases}

    # 专家维度的独立字典
    ux_prompt_map = {case.ticket_id: case.prompt for case in ux_case_list}
    ux_comp_map = {case.ticket_id: case.completion for case in ux_case_list}

    ss_prompt_map = {case.ticket_id: case.prompt for case in ss_case_list}
    ss_comp_map = {case.ticket_id: case.completion for case in ss_case_list}

    hc_prompt_map = {case.ticket_id: case.prompt for case in hc_case_list}
    hc_comp_map = {case.ticket_id: case.completion for case in hc_case_list}

    # 2. 读取 Excel 文件
    df = pd.read_excel(out_xlsx)

    # 4. 填充数据
    # 只需要一列公共的描述
    df['description'] = df['ticket_id'].map(desc_map)

    # 填充不同维度的专家内容
    df['ux_prompt'] = df['ticket_id'].map(ux_prompt_map)
    df['ux_completion'] = df['ticket_id'].map(ux_comp_map)

    df['ss_prompt'] = df['ticket_id'].map(ss_prompt_map)
    df['ss_completion'] = df['ticket_id'].map(ss_comp_map)

    df['hc_prompt'] = df['ticket_id'].map(hc_prompt_map)
    df['hc_completion'] = df['ticket_id'].map(hc_comp_map)

    df.to_excel(out_xlsx_detailed, index=False)

    logger_instance.info(f"处理完成！文件已保存至: {out_xlsx_detailed}")

    # 结束时间
    end_time = time.time()
    all_time = end_time - start_time
    case_time = all_time / case_num
    logger_instance.info(f'总时长为{all_time},平均每条工单耗时为{case_time}')

for i in range(1):
    do_infer(args.dap, args.tvec, args.svec, args.tpth)
