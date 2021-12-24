"""
# File       :  one_trans.py
# Time       :  2021/12/1 12:05 下午
# Author     : Qi
# Description:
"""



# 初始化模型
import torch

from nltk import word_tokenize
from my_utils import get_word_dict, subsequent_mask
from torch.autograd import Variable
from setting import SAVE_FILE, DEVICE, LAYERS, D_MODEL, D_FF, DROPOUT, H_NUM, TGT_VOCAB, SRC_VOCAB
import numpy as np


def init_model():
    from setting import LAYERS, D_MODEL, D_FF, DROPOUT, H_NUM, TGT_VOCAB, SRC_VOCAB
    from model import make_model
    # 模型的初始化
    model = make_model(
        SRC_VOCAB,
        TGT_VOCAB,
        LAYERS,
        D_MODEL,
        D_FF,
        H_NUM,
        DROPOUT
    )
    return model

model = init_model()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cn_idx2word, cn_word2idx, en_idx2word, en_word2idx = get_word_dict()

model.load_state_dict(torch.load(SAVE_FILE, map_location=torch.device('cpu')))

def sentence2id(sentence):
    """
    句子转 ID list
    :param sentence: 'I am a boy'
    :return: [['2', '5', '90', '9', '192', '3']]
    """
    en = []
    en.append(['BOS'] + word_tokenize(sentence.lower()) + ['EOS'])

    sentence_id = [[int(en_word2idx.get(w, 0)) for w in e] for e in en]
    return sentence_id

def src_handle(X):
    """
    将句子 id 列表转换为tensor，并且生成输入的mask矩阵
    :param X: 句子 id 列表
    :return: 单词列表id的list和对输入的mask
    """
    src = torch.from_numpy(np.array(X)).long().to(DEVICE)
    src_mask = (src != 0).unsqueeze(-2)
    return src, src_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    :param model: 模型
    :param src: 输入
    :param src_mask: 输入的mask
    :param max_len: 最大长度
    :param start_symbol: 开始标志
    :return: 预测的单词ID
    """
    memory = model.encode(src, src_mask)

    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]

        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def output(out):
    translation = []

    for i in range(1, out.size(1)):
        sym = cn_idx2word[out[0, i].item()]
        if sym != 'EOS':
            translation.append(sym)
        else:
            break

    # 打印模型翻译输出的中文句子结果
    print("translation: %s" % " ".join(translation))
    return ''.join(translation)

def machine_translate(sentence):
    """
    单句翻译
    :param sentence: 句子
    :return: 中文句子
    """
    src, src_mask = src_handle(sentence2id(sentence))
    out = greedy_decode(model, src, src_mask, max_len=50, start_symbol=int(cn_word2idx.get('BOS')))
    cn_result = output(out)
    return cn_result