"""
# File       :  model.py
# Time       :  2021/11/29 6:26 下午
# Author     : Qi
# Description:
"""
import copy
import math
from setting import LAYERS, D_MODEL, D_FF, DROPOUT, H_NUM, DEVICE
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.functional import F

class Embeddings(nn.Module):
    """
    embedding 层
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    位置编码层
    """
    def __init__(self, d_model, dropout, max_len = 5000):
        """
        初始化
        :param d_model:embedding 维度
        :param dropout:dropout 参数
        :param max_len:最大句子长度
        """
        super(PositionalEncoding, self).__init__()

        # dropout 层
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个 (max_len, d_model) 维度的全零矩阵，存放位置编码
        pe = torch.zeros(max_len, d_model, device=DEVICE)

        # position -> (max_len, 1)
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)

        # sin/cos公式中的分母
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * (-math.log(10000.0) / d_model))

        # 填充 pe 矩阵
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一维
        pe = pe.unsqueeze(0)

        # Adds a persistent buffer to the module.向模块添加持久缓冲区。
        # 反向传播不会更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x

def attention(query, key, value, mask=None, dropout=None):
    """
    其中 h 是多头注意力的头数目，这里放在一起同时计算
    :param query:(batch_size, h, sequence_len, embedding_dim)
    :param key:(batch_size, h, sequence_len, embedding_dim)
    :param value:(batch_size, h, sequence_len, embedding_dim)
    :param mask:
    :param dropout:
    :return:
    """
    d_k = query.size(-1)
    # 多维矩阵相乘需保证最后两维匹配
    # (batch_size, h, sequence_len, embedding_dim) X (batch_size, h, embedding_dim, sequence_len)
    # scores --> (batch_size, h, sequence_len, sequence_len)
    # scores 中每一行代表着长度为sequence_len的句子中每个单词与其他单词的相似度
    scores = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))

    # 如果有 mask 先进行 mask 操作
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # softmax
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # (batch_size, h, sequence_len, sequence_len) X (batch_size, h, sequence_len, embed_dim)
    #  --> (batch_size, h, sequence_len, embed_dim)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    """
    深拷贝 N 份 module， 参数不共享
    :param module: 传入的module
    :param N: 份数
    :return: nn.ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeaderAttention(nn.Module):
    """
    初始化权重矩阵 W_Q, W_K, W_V
    Embedding x 与权重矩阵相乘，得到 K, Q, V
    分解 K, Q, V 得到每个 head 的权重
    输出 (batch_size, sequence_len, embed_dim)
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        初始化
        :param h:多头注意力头数目
        :param d_model: embed 维度数
        :param dropout: dropout 参数
        """
        super(MultiHeaderAttention, self).__init__()
        # 检测维度是否可被头数整除，因为每个注意力机制函数只负责最终输出序列中一个子空间
        assert d_model % h == 0

        # 每个 head 的维度
        self.d_k = d_model // h
        self.h = h

        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_num = query.size(0)

        # query --> (batch_size, h, sequence_len, embedding_dim/h)
        query, key, value = [linear(x).view(batch_num, -1, self.h, self.d_k).transpose(1, 2) \
                             for linear, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 将多个头的注意力矩阵concat起来
        # 输入：x --> (batch_size, h, sequence_len, embed_dim/h(d_k))
        # 输出：x --> (batch_size, sequence_len, embed_dim)
        x = x.transpose(1, 2).contiguous().view(batch_num, -1, self.h * self.d_k)

        # (batch_size, sequence_len, embed_dim)
        return self.linears[-1](x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (torch.sqrt(std**2 + self.eps)) + self.b_2

class SublayerConnection(nn.Module):
    """
    将 Multi-Head Attention 和 Feed Forward 层连在一起
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 返回Layer Norm 和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    """
    前向传播
    x 维数 (batch_size, sequence_len, embed_dim)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化
        :param d_model: embedding 维度
        :param d_ff: 隐藏单元个数
        :param dropout: dropout参数
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        将embedding输入进行multi head attention
        得到 attention之后的结果
        """
        x = self.sublayer[0](x, lambda x:self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终hidden表示结果
        m = memory

        # self-attention的q，k和v均为decoder hidden
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))

        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 复制N个decoder layer
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class LabelSmoothing(nn.Module):
    """
    标签平滑处理
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        """
        损失函数是KLDivLoss，那么输出的y值得是log_softmax
        具体请看pytorch官方文档，KLDivLoss的公式
        """
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    """
    损失函数
    简单的计算损失和进行参数反向传播更新训练的函数
    """

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()

class NoamOpt:
    """
    优化器
    Optim wrapper that implements rate.
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def make_model(src_vocab, tgt_vocab, N=LAYERS, d_model=D_MODEL, d_ff=D_FF, h=H_NUM, dropout=DROPOUT):
    c = copy.deepcopy
    attn = MultiHeaderAttention(h, d_model).to(DEVICE)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)
    ).to(DEVICE)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)
