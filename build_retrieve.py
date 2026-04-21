# ------------------------- 导入库 -------------------------
import argparse
import warnings
warnings.filterwarnings('ignore')

import os  # 文件路径操作
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pandas as pd  # Excel 数据读取与 DataFrame 操作

from typing import List, Dict  # 类型注解
from langchain_huggingface import HuggingFaceEmbeddings  # 文本向量化
from langchain_community.vectorstores import FAISS  # 向量检索
from langchain_core.documents import Document  # 文档封装对象

from format_logger import logger_instance

parser = argparse.ArgumentParser(description='Parameters of Magnitude',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dap", help="Train Data Path", default="./data/train_dataset_20260408_17.xlsx")
args = parser.parse_args()


# ------------------------- BugReport RAG 系统 -------------------------

class BugReportRAGSystem:
    """
    历史问题单 RAG 系统
    功能：
        1. 加载历史 Excel 问题单
        2. 文本清洗与 Document 封装
        3. 构建/更新 FAISS 向量索引
        4. 支持相似问题单查询
    """

    def __init__(self, excel_path: str,
                 embedding_model_name: str = "/home/ma-user/work/zuoyimin_after_mid/model/bge-small-zh-v1.5"):
        """
        初始化 BugReport RAG 系统
        :param excel_path: Excel 文件路径
        :param embedding_model_name: Embedding 模型路径
        """
        self.excel_path = excel_path  # 保存 Excel 路径
        self.embedding_model_name = embedding_model_name  # Embedding 模型路径
        self.vector_store = None  # FAISS 向量存储对象
        self.embeddings = None  # Embedding 模型实例

        # 初始化 Embedding 模型
        logger_instance.info(f"加载 Embedding 模型: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,  # 模型路径
            model_kwargs={'device': 'cpu'},  # 推理使用 CPU
            encode_kwargs={'normalize_embeddings': True}  # 输出向量归一化
        )
        logger_instance.info("模型加载完成。")

    # ---------------- 数据加载与预处理 ----------------
    def load_and_preprocess_data(self) -> List[Document]:
        """
        读取 Excel 并转换为 Document 对象
        """
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"找不到文件: {self.excel_path}")  # 文件不存在报错

        logger_instance.info(f"读取&清洗数据:{self.excel_path}")
        df = pd.read_excel(self.excel_path)  # 读取 Excel
        df.dropna(subset=['问题单描述'], inplace=True)  # 删除缺失描述行

        documents = []  # 存储 Document 对象
        logger_instance.info(f"开始构建Document 总训练集:{df.shape[0]}")
        for _, row in df.iterrows():
            bug_id = str(row['问题单号'])  # 问题单号
            description = str(row['问题单描述']).strip()  # 问题描述
            severity = str(row['级别']).strip()  # 问题定级

            metadata = {
                "bug_id": bug_id,  # 元数据
                "severity": severity
            }
            doc = Document(page_content=description, metadata=metadata)  # 封装 Document
            documents.append(doc)  # 添加到列表

        logger_instance.info(f"Document 构建完成，共加载 {len(documents)} 条有效工单。")
        return documents  # 返回 Document 列表

    # ---------------- 构建或更新向量索引 ----------------

    def build_or_update_vector_store(self, index_path="faiss_index"):
        """
        构建或增量更新 FAISS 向量数据库
        :param index_path: 索引保存路径
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
            # else:
            #     logger_instance.info("没有新的工单需要更新。")  # 无新增数据提示
        else:
            logger_instance.info(f"FAISS索引不存在，构建向量数据库:{index_path}")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)  # 添加新文档
            self.save_index(index_path)  # 保存索引
            logger_instance.info("向量数据库构建完成。")

    # ---------------- 保存向量索引 ----------------
    def save_index(self, folder_path="faiss_index"):
        """保存索引到本地"""
        if self.vector_store:
            self.vector_store.save_local(folder_path)  # 构建向量索引
            logger_instance.info(f"索引已保存至 {folder_path}")  # 保存索引

    # ---------------- 加载向量索引 ----------------
    def load_index(self, folder_path="faiss_index"):
        """从本地加载索引"""
        if os.path.exists(folder_path):
            self.vector_store = FAISS.load_local(
                folder_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # 允许反序列化
            )
            logger_instance.info("索引加载成功。")
        else:
            logger_instance.info("未找到本地索引，请先构建。")

    def retrieve_similar_bugs(self, query: str, k: int = 3) -> List[Dict]:
        """
        根据查询文本检索相似问题单
        :param query: 查询内容
        :param k: 返回 top-k 个相似问题单
        :return: 包含 bug_id、description、severity、similarity_score 的字典列表
        """
        if not self.vector_store:
            raise ValueError("向量数据库未初始化！")  # 未初始化报错

        results = self.vector_store.similarity_search_with_score(query, k=k)  # 检索向量相似度

        retrieved_info = []  # 检索向量相似度
        for doc, score in results:
            info = {
                "bug_id": doc.metadata.get("bug_id"),  # 历史工单号
                "description": doc.page_content,  # 工单描述
                "severity": doc.metadata.get("severity"),  # 工单定级
                "similarity_score": round(score, 4)  # 相似度分数
            }
            retrieved_info.append(info)  # 添加到结果列表

        return retrieved_info  # 返回结果列表


def build_train_data_vector(train_path):
    save_path = './vector/history_faiss_index'
    rag_system = BugReportRAGSystem(excel_path=train_path)
    rag_system.build_or_update_vector_store(index_path=save_path)
    logger_instance.info(f"向量数据库检测")
    # ---------------- 构建或更新向量索引 ----------------
    new_bug_query = "晚上人脸识别突然用不了了，一直提示失败"
    logger_instance.info(f">>> 检索: '{new_bug_query}' 的相似数据")
    # ---------------- 检索示例 ----------------
    results = rag_system.retrieve_similar_bugs(new_bug_query, k=1)
    for i, res in enumerate(results):
        logger_instance.info(f"[匹配结果 {i + 1}]")
        logger_instance.info(f"历史单号: {res['bug_id']}")  # 输出历史工单号
        logger_instance.info(f"历史定级: {res['severity']}")  # 输出历史定级
        logger_instance.info(f"问题描述: {res['description']}")  # 输出工单描述
    # print(save_path, flush=True)


build_train_data_vector(args.dap)
