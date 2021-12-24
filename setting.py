"""
# File       :  setting.py
# Time       :  2021/11/29 6:26 下午
# Author     : Qi
# Description:
"""
import torch

UNK = 0
PAD = 1
BATCH_SIZE = 64
# 训练数据
TRAIN_FILE = 'data/train.txt'
# 验证数据
DEV_FILE = 'data/dev.txt'
# 测试数据
TEST_FILE = 'data/test.txt'
# 模型存放位置
SAVE_FILE = 'model/model.pt'
# encoder decoder 层数
LAYERS = 6
# embedding 维度
D_MODEL = 512
# 第一个全联接层纬度数
D_FF = 1024
# 多头注意力头数
H_NUM = 8
DROPOUT = 0.1
EPOCHS = 20
MAX_LENGTH = 60
# 英文单词数
SRC_VOCAB = 5493
# 中文单词数
TGT_VOCAB = 3196

BLEU_REFERENCES = '../data/bleu/references.txt'
BLEU_CANDIDATE = '../data/bleu/candidate.txt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')