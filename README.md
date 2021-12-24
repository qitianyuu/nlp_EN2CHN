## 机器翻译 英译汉 基于 pyTorch 的 Transformer
pyTorch 实现多头注意力机制的小项目，参考文章[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) 和[这个代码仓库](https://github.com/taoztw/Transformer)


`setting.py`:模型相关参数，文件目录的配置文件。  
`utils.py`:一些工具函数。  
`data_pre.py`:数据的预处理，得到输出模型的batch数据和相关的mask矩阵  
`model.py`:模型文件。  
`train.py`:进行模型的训练。和最好模型的保存。  
`test.py`:对测试集句子的测试输出。  
`bleu_score.py`:对机器翻译评分。  
`one_trans.py`:实现单个句子进行翻译。  
`app.py`:通过使用one_trans文件封装的单个句子翻译的方法，实现flask api  

### 模型训练数据
使用**14533**条翻译数据进行训练。  
数据文件格式：en`\t`cn
  

### 结果评估
使用BLEU算法进行翻译效果评估[BLEU](https://www.cnblogs.com/by-dream/p/7679284.html)
BLEU算法评价结果：  
    
    对399条翻译句子效果进行评估
    验证集:0.1075088492716548，n-gram权重：(1,0,0,0)
          0.03417978514554449,n-gram权重：(1,0.2,0,0)
### flask 接口
 `/translation` post 方法
```json
{
  "sentence": "I am very happy"
}
// return
{
  "result": "翻译结果",
  "msg": 'success',
  "code": 200
}
```

## 运行项目
1. `python train.py`：训练模型，保存模型
2. `python app.py`启动flask。
