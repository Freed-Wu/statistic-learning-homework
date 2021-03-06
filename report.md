---
documentclass: ctexart
title: 技术方案
author: Wu Zhenyu (SA21006096)
institute: USTC
bibliography: refs/main.bib
---

## 任务

本次大赛提供源自真实业务的脱敏数据做为训练集，参赛队伍基于训练集构建视频理解算
法模型。

赛题根据难度分为2个赛道：

1. 视频广告秒级语义解析
2. 多模态视频广告标签

两个任务均以视频，音频，文本三个模态做为输入，参赛选手利用模型对视频广告进行理
解，最终得分需要综合精度指标和效率指标得出。

### 视频广告秒级语义解析 Video Ads Content Structuring

对于给定测试视频样本，通过算法将视频在时序上进行“幕”的分段，并且预测出每一段在
呈现形式、场景、风格等三个维度上的标签，使用Mean Average Precision(MAP)进行评分。

### 多模态视频广告标签 Multimodal Video Ads Tagging

对于给定的测试视频样本，通过算法预测出视频在呈现形式、场景、风格等三个维度上的
标签，使用Global Average Precision(GAP)进行评分。

---

赛道二明显比赛道一简单。赛道二相当于省去了赛道一分段的任务，即将视频每一段的标
签叠加起来即得到了视频的标签。故选赛道二。

## Model

比赛提供的 baseline 如下，关于改进见 [Tricks](#Tricks)。

### 提取特征

- 视频：提供了原始视频和一个检测子的 npy 格式的文件，用
NextVLAD[@lin2018nextvlad] 提取特征。
- 音频：同视频。
- 文本：提供了通过 OCR 和 ASR 提取得到的 npy 格式的文本。用
Bert[@devlin2019bert] 提取特征。
- 图片：将视频最中间一帧用 ResNet[@he2015deep] 提取特征。

### 特征融合

将各个模态特征直接 cat 后输入到 SE[@hu2019squeezeandexcitation] 模块。输出结果
用 Logistics 分类。

## Tricks

### 数据集的重新分配

baseline 的方案是

- 训练集：4500
- 验证集：500
- 测试集：1000

大概在 35 或 40 个 epoch 达到验证集的最好结果。每5个 epoch 验证一次，结果比上次
好的模型保留，拿验证集最好结果对应的模型去测试，会导致结果测试比验证结果稍差一
点。

后调整为：

- 训练集：5000
- 验证集：0
- 测试集：1000

直接将第 35 或 40 的 epoch 的模型拿去测试。涨点。原因不言自名。但确实不参加比赛
想不到这个想法。

### 数据的预处理

baseline 提取特征的网络中，文本的数据来源是对视频 OCR 和对音频 ASR 。注意到 OCR
和 ASR 得到的结果为有少量乱码一样的英文搀在中文台词之间。通过正则表达式将中文以
外的的字符包括标点符号全部过滤掉再输入网络，涨点。原因应该是减少了数据噪声，降
低了学习难度。还尝试了保留阿拉伯数字或者标点符号，以及把少于2个字符的句子滤掉，
效果不明显。现在认为 Tokenize 时本来就会去除标点符号。这一尝试没有意义。

### 移除性能较差的模态

最初一共有 4 个模态，音频，视频，文本（OCR，ASR），图片。发现图片和音频模态表现
很差，删了这2个模态后会涨点。应该是这2个模态在 baseline 中本身实现性能就不好的
缘故。例如图片模态的输入应该是视频中的关键帧，baseline 只是简单地抽取了最中间的
一帧做关键帧，结果自然不好。但直到比赛结束也没有把这个关键帧的网络实现起来:(

### 改变模态融合的方式

baseline 使用了 early fusion，就是把多个模态提取出的特征 cat 成一个新张量后送到
backbone。换成 late fusion，即每个模态提取出的特征分别送到 backbone 后输出的结
果（概率）再取平均（也尝试了最大等，但不如平均）效果更好。个人认为不同模态输出
的特征有差异，比如文本属于人造的符号语言，信息量大，冗余度低，而视频恰恰相反。
直接将他们的特征 cat 并不妥当。现在基本上 early fusion 的论文都是
cross-modality attention 的方法，把一种模态的 V 和 K 和另一种模态的 Q 送去
attention ，想必是这种简单的 cat 已经被淘汰了。但当时不知道，只是试出来改成
late fusion 好。

### 自监督学习

改了提取特征的网络，对所有数据（无标签，包括测试集）进行类似
SimCLR[@chen2020simple] 的操作。涨点。原因是利用了测试集的信息。

### essemble

涨点。原因显然:)

## 总结

还有不少失败的 trick, 例如加了对抗扰动，即FGSM[@goodfellow2015explaining]，希望
能训练一个对扰动更加稳健的模型。结果越训练越差。现在看来，当初没有对 FGSM 没有
理解正确就下手写代码了，应该先弄懂再搞。

还考虑视频裁减之后会不会好一点。但没来的及试了。

还有 GAP，当初没明白GAP 算分的原理，误以为手动改预测的概率也能提高GAP。实际上
GAP 是由准确度计算得到的，概率只用来给预测的可能结果排序。

感觉有不少技巧是瞎试出来，试出来结果好就用了，当时也不清楚原因。希望在今后的学
习过程中能知其然更知其所以然。
