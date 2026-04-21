MM_LLM_PATH = "/home/ma-user/work/zuoyimin_after_mid/model/Qwen3-VL-8B-Instruct"
EMBEDDING_MODEL = "/home/ma-user/work/zuoyimin_after_mid/model/bge-small-zh-v1.5"
DEVICE = "npu"
VLLM_MODEL_PATH = "/home/ma-user/work/models/ds_32b"  # 替换为你的 DeepSeek 模型路径
TENSOR_PARALLEL_SIZE = 4  # 替换为你的原始 Excel 文件路径
BATCH_SIZE = 400  # 降低批次大小，为 RAG 留出内存空间
MODEL_SAVE_DIR = "./model"
MODEL_SAVE_DIR1 = "./modelw"
MODEL_SAVE_DIR2 = "./modelz"
FORCE_GEN=True

