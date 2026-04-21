source /usr/local/Ascend/ascend-toolkit/set_env.sh

source /usr/local/Ascend/nnal/atb/set_env.sh

# fuser -k /dev/davinci_manager  > log2 2>&1 &

# 数据分割
# construct_result=`CUDA_VISIBLE_DEVICES=4,5,6,7 python data_split.py --dap "./data/Beta结项验收数据集0401shit (2).xlsx"`

train_path="./data/train_dataset_20260408_14.xlsx"
test_path="./data/test_dataset_20260408_14.xlsx"

echo 训练集： $train_path
echo 测试集： $test_path
# 构造训练集向量
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7  python build_retrieve.py --dap $train_path
tvec=./vector/history_faiss_index
echo 训练集向量化结果 $tvec

# 构造系统知识向量，一般不需重复构造

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7  python build_pdf_retrieve.py --dap ./data/huawei_mate60.pdf

svec=./vector/pdf_faiss_index
echo 系统知识向量化结果 $svec

# 开始训练

# ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 python train.py --dap $train_path --tvec $tvec --svec $svec

train_pth=./model/fusion_model_epoch_20.pth
echo 训练输出模型 $train_pth

# 开始测试
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 python infer.py --dap $test_path --tvec $tvec --svec $svec --tpth $train_pth
