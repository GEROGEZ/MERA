import warnings
warnings.filterwarnings('ignore')
import argparse
# ------------------------- 导入标准库 -------------------------
import os  # 文件路径与目录操作
from typing import Union, List, Dict  # 类型注解
import json  # JSON 数据处理
import torch  # PyTorch 框架
from PIL import Image  # 图像处理
from pdf2image import convert_from_path  # PDF 渲染为图像
from tqdm import tqdm  # 可视化进度条

# ------------------------- 导入第三方库 -------------------------
from langchain_huggingface import HuggingFaceEmbeddings  # 向量化接口
from langchain_community.vectorstores import FAISS  # 向量检索
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本切割
from langchain_core.documents import Document  # 文档对象
from langchain_community.document_loaders import PyPDFLoader  # PDF 文本加载器

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor  # 多模态大模型

from format_logger import logger_instance
from config import MM_LLM_PATH, EMBEDDING_MODEL, DEVICE

parser = argparse.ArgumentParser(description='Parameters of Magnitude',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dap", help="Train Data Path", default="./data/huawei_mate60.pdf")
args = parser.parse_args()

pdf_prompt = """
你现在是一名资深手机系统专家，熟悉手机操作系统的所有组件、页面结构、系统服务和交互逻辑。

你的任务如下：

润色与重写手机系统组件说明
我将提供一段关于手机系统某组件的说明文档。你需要：

1. 对文档进行专业化、结构化的润色和改写。

2. 修正语言问题，使表达更清晰准确。

3. 输出结构化的组件说明，包括但不限于：

  · 功能概述

  · 页面与界面布局

  · 关键入口位置

  · 主要操作流程

  · 系统交互逻辑

  · 依赖的系统服务或组件
手机说明文档如下：{doc}
"""


# ------------------- Qwen3-VL 封装 -------------------
class Qwen3VLModel:
    """
    该类负责 Qwen3-VL 模型的一键式初始化与图文混合推理。

    设计细节：
    - 设备映射 (device_map): 自动处理跨 NPU/GPU/CPU 的张量切片，支持大模型在有限显存下的运行。
    - 精度控制 (torch_dtype): 在高性能设备上自动采用 float16 以换取双倍的推理速度。
    - 处理流 (Processor): 统一管理图像缩放、Normalization 以及文本分词 (Tokenizer) 的同步。
    """

    def __init__(self, model_name_or_path: str, device: str = "cpu"):
        """
        初始化模型引擎。
        :param model_name_or_path: 预训练权重的本地路径或 HuggingFace ID。
        :param device: 推理后端设备，如 'npu', 'cuda', 'cpu'。
        """
        self.device = device  # 保存设备信息
        logger_instance.info(f"加载模型：{model_name_or_path}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,  # 模型路径
            device_map="auto",  # 自动设备分配
            dtype=torch.float16 if device != "cpu" else torch.float32  # 高性能设备使用 float16
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)  # 加载预处理器
        logger_instance.info("模型加载完成。")


    def _load_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """
        内部工具函数：支持路径输入或已有的 PIL 对象，确保图像最终为标准的 RGB 格式。
        RGB 格式转换是视觉模型的关键，防止透明通道 (RGBA) 导致的通道数匹配错误。
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")  # PIL 对象确保 RGB
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError("image must be a path or PIL.Image object")  # 类型错误
        return img  # 返回标准 RGB 图像

    def call_chat(self, prompt: str, images: Union[List[Union[str, Image.Image]], None] = None,
                  max_new_tokens: int = 5120, do_sample: bool = False) -> str:
        """
        执行多模态对话推理的核心入口。

        推理逻辑：
        1. 构造标准 Messages 列表，符合 OpenAI 风格的对话格式。
        2. 调用 apply_chat_template 将 Prompt 和图像交错嵌入。
        3. 利用 generate 函数生成预测 Token 流。
        4. 切除 Input 部分，仅保留模型生成的 Answer。
        """
        messages = [{"role": "user", "content": []}]  # 初始化对话消息
        if images:
            if not isinstance(images, list):
                images = [images]
            for img in images:
                pil_img = self._load_image(img)  # 统一 RGB 处理
                messages[0]["content"].append({"type": "image", "image": pil_img})  # 添加图像到消息

        messages[0]["content"].append({"type": "text", "text": prompt})  # 添加文本 Prompt
        # 构造输入张量
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # 移动到指定设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():  # 禁用梯度计算
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)

        # 去掉 prompt 部分
        output_ids = [out[len(in_ids):] for in_ids, out in zip(inputs["input_ids"], generated_ids)]
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output_text or prompt  # 返回生成文本或原 Prompt


# ------------------- 增强文档 RAG 系统 -------------------
class EnhancedDocumentRAGSystem:
    """
    增强版 RAG 系统总控类。

    该类实现了“数据注入 (Ingestion) -> 知识增强 (Augmentation) -> 持久化存储 (Storage) -> 检索 (Retrieval)”的全生命周期管理。

    特别优化：
    - 追加写入 JSONL：防止大文档处理到中途断电或崩坏导致的数据丢失。
    - 语义分段：不仅仅按字符切分，而是基于 Markdown 标题和换行符进行语义敏感型切分。
    """

    def __init__(self, mm_llm_model_path: str, embedding_model_name: str, output_file: str, mm_lll_device: str):
        self.mm_processor = Qwen3VLModel(mm_llm_model_path, mm_lll_device)  # 初始化多模态 LLM
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)  # 初始化向量化
        self.vector_store: FAISS = None  # FAISS 向量存储
        self.output_file = output_file  # 输出 JSONL 路径

    """
        初始化 RAG 系统各个子组件。
        :param mm_llm_model_path: Qwen3-VL 路径。
        :param embedding_model_name: BGE 模型路径，用于将增强后的文本编码。
        :param output_file: JSONL 存档文件路径。
        """

    def enhance_pdf_pages(self, pdf_path: str) -> List[Document]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"{pdf_path} not found.")  # 文件不存在报错
        logger_instance.info(f"开始处理系统知识PDF:{pdf_path}")
        # PDF -> PIL Images
        images = convert_from_path(pdf_path, fmt="jpeg", thread_count=4)  # PDF 渲染为图像
        loader = PyPDFLoader(pdf_path)  # PDF 文本加载器
        raw_pages = loader.load()  # 原始文本列表

        if len(images) != len(raw_pages):
            logger_instance.info(f"[WARN] PDF page count ({len(raw_pages)}) != images ({len(images)})")
        enhanced_docs = []  # 增强文本列表
        # 确保输出文件的目录存在
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        for page, img in tqdm(zip(raw_pages, images), total=len(raw_pages), desc="Enhancing pages"):
            logger_instance.info(f"开始提取&总结PDF知识: 第{page.metadata.get('page', None)}页. 共 {page.metadata.get('total_pages', None)}页")
            try:  # 调用多模态模型生成增强文本
                enhanced_text = self.mm_processor.call_chat(
                    prompt=pdf_prompt.format(doc=page.page_content),
                    images=[img],
                    max_new_tokens=5120
                )
            except Exception as e:  # 推理失败回退
                logger_instance.info(f"[ERROR] MM-LLM enhance failed: {e}, using raw text")
                enhanced_text = page.page_content  # 回退原始文本

            # ====== 实时追加写入 JSONL 文件 ======
            logger_instance.info(f"更新PDF知识: 第{page.metadata.get('page', None)}页. \n原文: {page.page_content}. \n模型总结内容: {enhanced_text}")
            page_info = {
                "page_number": page.metadata.get("page", None),  # 页面编
                "original_text": page.page_content,  # 原文
                "enhanced_text": enhanced_text  # 增强文本
            }

            # 修复：使用纯追加模式(a)，每行记录一个 JSON 对象。
            # 这不仅解决了覆盖漏洞，还极大提升了写入速度。
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(page_info, ensure_ascii=False) + "\n")

            # 构建 Document 对象
            doc = Document(page_content=enhanced_text, metadata={"source_file": pdf_path})
            enhanced_docs.append(doc)

        # 文本切割
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70, separators=["\n\n", "\n", "."])
        return splitter.split_documents(enhanced_docs)

    def build_or_load_vector_store(self, pdf_path: str, index_path: str = "enhanced_doc_faiss_index"):
        if os.path.exists(index_path):
            logger_instance.info("检测到已有本地索引，加载并更新...")
            self.vector_store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            logger_instance.info(f"本地FAISS索引不存在，构建向量数据库:{index_path}")
            documents = self.enhance_pdf_pages(pdf_path)
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.save_index(index_path)  # 保存索引
        """
        构建或加载 FAISS 向量存储
        :param pdf_path: PDF 文件路径
        :param index_path: FAISS 索引路径
        """

    def save_index(self, folder_path: str = "enhanced_doc_faiss_index"):
        if self.vector_store:
            os.makedirs(folder_path, exist_ok=True)
            self.vector_store.save_local(folder_path)
            logger_instance.info(f"索引已保存至 {folder_path}")
        """
        保存 FAISS 向量索引
        :param folder_path: 索引保存目录
        """

    def retrieve_similar_documents(self, query: str, k: int = 3) -> List[Dict]:
        """
        相似文档检索
        :param query: 查询文本
        :param k: 返回 Top-k 文档
        :return: 文档列表，每项包含 content、metadata、score
        """
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [{"content": doc.page_content, "metadata": doc.metadata, "score": score} for doc, score in results]


# ================= 使用示例 =================
def build_system_vector(sys_file):
    save_path = './vector/pdf_faiss_index'
    pdf_data = './vector/enhanced_ouput.jsonl'
    rag_system = EnhancedDocumentRAGSystem(mm_llm_model_path=MM_LLM_PATH, mm_lll_device=DEVICE,
                                           embedding_model_name=EMBEDDING_MODEL, output_file=pdf_data)
    rag_system.build_or_load_vector_store(sys_file, index_path=save_path)
    logger_instance.info(f"向量数据库检测")
    rag_system.save_index(folder_path=save_path)
    query = "联系人功能总结"
    results = rag_system.retrieve_similar_documents(query)
    for r in results:
        logger_instance.info(f'{r["content"], r["metadata"], r["score"]}')
    print(save_path, flush=True)


build_system_vector(args.dap)
