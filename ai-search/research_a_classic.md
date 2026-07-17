# 研究笔记 A：经典基石——第一幕·符号时代 + 第二幕·嵌入时代

> 幻灯片主线：一条查询的旅程 = **查询理解 → 召回 → 排序 → 结果呈现 → 评估闭环**。
> 本笔记逐篇论文标注：它改造了骨架里的哪一环、为什么改、改了之后数字上发生了什么变化。

---

## 0. 骨架速览（写在最前面，方便后续对照）

```
用户输入 query
   │
   ▼
【查询理解】 分词 / 意图识别 / 实体链接 / query embedding
   │
   ▼
【召回 Retrieval】 从亿级/十亿级候选中捞出千级候选
   ├─ 符号时代：倒排索引 Boolean 匹配（Unicorn）
   └─ 嵌入时代：ANN 近邻检索（EBR / Que2Search / DPR），与倒排索引混合
   │
   ▼
【排序 Ranking】 粗排（轻量模型，候选千→百）→ 精排（重模型，候选百→十）
   └─ 精排的效率难题由 ColBERT 的 late interaction 折中解决
   │
   ▼
【结果呈现】 展示 / 摘要 / 高亮
   │
   ▼
【评估闭环】 A/B 实验、人工评估、日志回流成训练数据
```

---

## 1. Unicorn（Facebook, VLDB 2013）

### 直觉
把整个 Facebook 社交图谱想象成一座巨型图书馆的"索引卡片柜"：每张卡片对应一个属性（比如"是 David 的朋友""住在纽约""喜欢唐顿庄园"），卡片后面挂着一串满足这个属性的人名（posting list）。搜索一句话，本质上就是把几张卡片"叠在一起做集合运算"——交集（AND）、并集（OR）、以及"顺着一张卡片再翻到下一张卡片"（多跳 apply）。Unicorn 的洞察是：**搜索 = 集合运算**，图遍历也可以用倒排索引的语言来表达。

### 核心数学
Unicorn 本质上没有复杂的概率模型，其"数学"是集合代数：

$$
\text{result} = \text{PostingList}(t_1) \;\square\; \text{PostingList}(t_2) \;\square\; \cdots
$$

其中 $\square \in \{\cap, \cup, \setminus\}$ 对应 `and` / `or` / `difference` 算子；$t_i$ 是前缀化的属性词条（term），例如 `friend:10003` 表示"fbid 为 10003 的人的好友"。

多跳查询用 **apply** 算子表达：

$$
\text{apply}(E, \text{edge\_type}) = \bigcup_{v \in E} \text{PostingList}(\text{edge\_type} : v)
$$

含义：先算出第一步的结果集合 $E$（比如"我在纽约的朋友"），再把每个结果节点当作新的种子，去查它们的另一类关系边（比如"works\_at"边），从而实现"我在纽约的朋友的雇主"这种二跳查询。

排序阶段还引入 **static rank**（与查询无关的重要性打分），用于让每个 posting list 按重要性排序，从而支持"取 top-K 而非全量遍历"。

### 架构图要素
- **组件**：Query Parser（自然语言 → 查询树）→ Term/Posting List 存储（内存倒排索引，按 fbid 或 static rank 排序）→ 算子执行引擎（and/or/difference/apply）→ Leaf/Aggregator 分片架构（数据分片在数千台商用机器上，Aggregator 汇总多分片结果）→ 增量索引更新管道（Hive 批量索引 + 实时流式更新，秒级新鲜度）。
- **数据流向**：用户自然语言 query → 解析成算子树（如 `(apply (and (term friend:10003) (term lives-in:111)) works-at)`）→ 分发到各分片并行执行 posting list 交并差 → 按 static rank 截断 top-K → 返回候选实体集合。

### 在骨架中的位置
Unicorn 是**召回环节的地基基础设施**：它把"关系型的图遍历"改造成"倒排索引上的集合运算"，取代了此前 Facebook 内部多套独立的关键词搜索后端（PPS、Typeahead 等）。它的下游是排序（用 static rank 做粗略排序），上游是自然语言到查询树的解析（可以视为"查询理解"的早期形态）。重要的是，**它不仅是符号时代的产物，也是嵌入时代的宿主**——EBR 和 Que2Search 后来都选择把向量检索"塞进"Unicorn 的倒排索引里做混合检索，而不是另起炉灶，这条线索贯穿整个"经典基石"部分。

### 关键数字
- 索引规模：**万亿级（trillions）边**，连接**百亿级（tens of billions）**用户和实体。
- 部署规模：数千台商用服务器（thousands of commodity servers）。
- 服务能力：每天回答**数十亿次查询**，延迟在**数百毫秒**级别。
- 数据新鲜度：Facebook 每天新增 25 亿条内容、27 亿次点赞，通过实时流式管道让索引在秒级内保持最新。

### 原文链接
- VLDB 2013 论文 PDF（已验证可访问）：https://www.vldb.org/pvldb/vol6/p1150-curtiss.pdf
- Meta 工程博客（图文并茂讲解 posting list / and-or-apply，已验证可访问）：https://engineering.fb.com/2013/03/06/core-infra/under-the-hood-building-out-the-infrastructure-for-graph-search/
- ACM DL 条目：https://dl.acm.org/doi/10.14778/2536222.2536239

---

## 2. Facebook Embedding-Based Retrieval / EBR（KDD 2020, arXiv:2006.11632）

### 直觉
符号时代的倒排索引只认"字面是否匹配"——你搜"讲鬼故事的地方"，索引里没有"鬼故事"三个字的文档永远进不了候选集，哪怕它叫"万圣节恐怖夜活动"。EBR 的思路是：**把 query 和 document 都嵌入同一个向量空间，让语义相近的东西在空间里靠得近**，检索问题从"字符串匹配"变成"在高维空间里找最近的邻居"——即使一个字都不重合，只要语义近，也能被召回。

### 核心数学
**统一嵌入（Unified Embedding）**：query 塔的输入不只是查询文本，还包括 searcher 的位置、社交图谱嵌入等上下文特征；document 塔同理。两塔各自编码成稠密向量 $u(q)$、$v(d)$。

**距离度量**：

$$
D(q,d) = 1 - \cos(u(q), v(d))
$$

**Triplet Loss**（训练目标，来自人脸识别 FaceNet 的思想）：

$$
L = \sum_{i=1}^{N} \max\Big(0,\; D(q^{(i)}, d^{(i)}_+) - D(q^{(i)}, d^{(i)}_-) + m\Big)
$$

符号含义：$q^{(i)}$ 是第 $i$ 条样本的 query；$d^{(i)}_+$ 是与之相关的正例文档（点击过的结果）；$d^{(i)}_-$ 是不相关的负例文档；$m$ 是人为设定的 margin（间隔超参数）。**这条公式想优化的是**：让正例的距离比负例的距离至少小 $m$，一旦差距超过 $m$，这个样本就"太简单"、梯度为零、不再参与学习——模型的注意力被自动集中在"难分的样本"上。论文发现 margin 的取值会带来 5%–10% 的 KNN recall 波动，是极其敏感的超参数。

**为什么随机负例不够 / 为什么曝光未点击的负例也不够**——这是全文最重要的工程洞察：
- 如果用"曝光但未点击"作为负例，模型学到的负例全部是"字面上有点像但用户没点"的**难例**，而线上检索面对的候选池里绝大多数文档跟 query 毫不相关（**易例**）。只用难例训练会让训练分布和线上分布产生系统性偏差（train-serving mismatch），论文实验显示这样训练出来的模型 recall 明显更差。
- 反过来，如果全部用随机负例，模型只需要学会"文本字面是否搭边"就能把损失降到很低，学不会区分社交/地理位置等"看起来沾边但其实不该被检索到"的相似候选——这就是需要 **Hard Negative Mining** 的原因。
- 解法：**在线难负例挖掘**（batch 内随机借用同 batch 里别的 query 的正例作负例，k=2 最优）+ **离线难负例挖掘**（用上一版模型对候选库做 ANN 检索，取排名 101–500 名作为难负例，而不是最难的第 1–2 名，因为那些位置常是漏标的假负例）；最终按约 **100:1（易:难）** 的比例混合两类负例效果最好。

### 架构图要素
- **组件**：Query 塔（文本 character n-gram embedding + 位置特征 + 社交图谱嵌入 → sum-pooling/拼接 → 稠密向量）、Document 塔（同构）、Triplet Loss 训练目标、离线 ANN 召回评估（Recall@K）、Faiss 粗量化（coarse quantization，做 IVF 式聚类分桶）+ 乘积量化（Product Quantization / OPQ，做压缩打分）、**Hybrid 检索**（把 embedding 检索结果作为新的 term 融合进 Unicorn 原有倒排索引，与布尔匹配并行跑在同一套系统里，而非另建一套独立索引）、Embedding 作为排序特征回流给下游 Ranking、人工评估反馈闭环（针对新召回结果做人工相关性标注，再回流成训练数据修正精度问题）。
- **数据流向**：query 文本+上下文 → Query 塔实时推理出向量 → 与预先离线算好并量化索引的 Document 向量做 ANN 检索（粗量化定位候选簇 → PQ 精确打分）→ 与传统倒排索引召回结果合并（hybrid）→ 候选集流入排序层（embedding cosine 相似度也作为排序特征之一）→ 结果展示 → 用户点击日志与人工标注回流，用于难负例挖掘和模型迭代。

### 在骨架中的位置
EBR 直接改造**召回环节**：从"纯 Boolean 倒排匹配"升级为"embedding 近似最近邻检索 + 倒排索引"的**混合召回**。上游承接查询理解产出的 query 表示，下游把相似度分数作为一个新特征喂给排序层。EBR 论文特别强调没有另起一套独立的向量检索系统，而是把 ANN 结果编码后"塞进"Unicorn 的倒排索引框架里做联合检索——这是工程上"全栈优化"的核心，避免了维护两套候选集、两套索引的高昂成本。

### 关键数字
- 事件（Events）搜索垂类：统一嵌入相对纯文本嵌入，**召回率提升约 18%**（线上 A/B）。
- 群组（Groups）搜索垂类：**召回率提升约 16%**。
- 引入离线难负例挖掘的二阶段模型相对 baseline：**召回率提升 3.4%**。
- 仅用"难正例"（hard positives）训练即可达到与点击数据相近的召回水平，而数据量只有点击数据的 **4%**。
- margin 超参数取值不同可造成 **5%–10%** 的 KNN recall 波动。

### 原文链接
- arXiv 摘要页（已验证可访问）：https://arxiv.org/abs/2006.11632
- PDF：https://arxiv.org/pdf/2006.11632
- DOI（KDD 2020）：https://doi.org/10.1145/3394486.3403305

---

## 3. Que2Search（Facebook, KDD 2021）

### 直觉
EBR 用的是"字符 n-gram + 浅层网络"理解 query 和文档，对短查询、多语言、带图片的电商场景（Facebook Marketplace）力不从心。Que2Search 相当于给 EBR 的双塔"换脑子"：把简单的词袋表示换成跨语言预训练模型 XLM/XLM-R，再把图片也编码进同一个向量空间，并且像"先学会认字母，再学会认单词，最后学会认整句话"一样，用两阶段课程学习循序渐进地训练难度递增的负例。

### 核心数学
**架构**：Query 塔（3-gram 哈希稀疏特征 + 国家 embedding + 2 层 XLM 编码器）与 Document 塔（title/description 原文经共享的 6 层 XLM-R + 二者的 3-gram 稀疏特征 + 图片经 GrokNet 预训练模型编码、多图用 Deep Sets 聚合）分别通过注意力做多路特征的**late fusion**：

$$
w = \text{softmax}\big(\text{proj}(\psi_1 \Vert \psi_2 \Vert \cdots \Vert \psi_N)\big), \qquad f = \sum_{i=1}^N w_i \psi_i
$$

$\psi_i$ 是第 $i$ 路特征通道（文本、n-gram、图片等）编码出的向量，$\Vert$ 表示拼接。**这一步想解决的问题**是不同模态/通道的重要性不是固定的（比如短 query 更依赖 3-gram，长 query 更依赖 XLM），用可学习的注意力权重自动分配。

**两阶段训练（Curriculum Learning）**：

*第一阶段——batch 内负例，带温度缩放的多分类交叉熵*：

$$
\text{loss}_i = -\log \frac{\exp(s \cdot \cos(q_i, d_i))}{\sum_{j=1}^{B} \exp(s \cdot \cos(q_i, d_j))}
$$

$B$ 是 batch size，$s$（论文取 15–20）是缩放常数（等价于温度的倒数 $1/\tau$），用来拉大正负样本相似度之间的差距、加速收敛。训练到该阶段收敛为止（早停）。

*第二阶段——课程学习，挖掘 batch 内最难负例*：

$$
n^{q_i} = \arg\max_{j \neq i} \cos(q_i, d_j), \qquad
\text{loss}_i = \max\big(0,\; -[\cos(q_i, d_i) - \cos(q_i, d_{n^{q_i}})] + \text{margin}\big)
$$

即对每个 query，把同一 batch 内除自己正例外相似度最高的那个文档当作"最难负例"，再用 margin rank loss 精细拉开距离（margin 取 0.1–0.2 效果最好）。论文强调：**必须先完成第一阶段的收敛，再开始第二阶段**，颠倒顺序（先难后易）效果会变差——这正是"课程学习"从易到难的核心思想在检索场景的体现。

**多任务学习**：额外加一个"用 document 向量预测该商品历史上被哪些 query 搜索过"的多标签分类任务（4.5 万个高频 query 类别，多标签交叉熵，每个正确类别标签概率均摊为 $1/k$），强迫模型学到商品和搜索意图之间更本质的语义关联。

### 架构图要素
- **组件**：Query 塔（3-gram + 国家 + 2 层 XLM，P99 延迟 1.5ms，需实时推理）、Document 塔（title/description 的 6 层 XLM-R + 3-gram + GrokNet 图片向量 + Deep Sets 聚合，离线/近线预计算）、注意力融合层、多任务分类头（doc→query 类别预测）、梯度混合式多模态融合（Gradient Blending：单模态损失 + 全模态损失加权求和，防止某个模态主导训练）、两阶段训练流水线（in-batch softmax → 课程学习难负例挖掘）、NLP Service（统一推理服务，调用 Predictor 平台算 query 向量）、Unicorn ANN 索引（复用 Unicorn 做召回）、两阶段排序（轻量 GBDT 粗排 → 深度模型精排，cosine 相似度作为强特征注入两个阶段）。
- **数据流向**：query 请求 → NLP Service 调用 Predictor 平台实时算出 query 向量 → 向 Unicorn 发起 ANN 检索请求，召回 top 商品 → 候选进入 GBDT 粗排（各分片先选头部结果）→ 深度模型精排（引入 Que2Search cosine 相似度特征）→ 结果展示 → 线上 A/B 与人工相关性标注回流优化训练数据和 ANN 超参数。

### 在骨架中的位置
Que2Search 同时触及**查询理解**（用 XLM/XLM-R 生成更准的 query 语义表示，是"query and document understanding system"的定位）和**召回**（作为 embedding 检索通道接入 Unicorn）与**排序**（cosine 相似度作为排序强特征）。它是 EBR 思路在电商搜索（Facebook Marketplace）场景下的深化版：把"用字符 n-gram 近似语义"升级为"用预训练语言模型 + 多模态理解语义"，同时用课程学习解决了 EBR 中"难负例挖掘调参易失败"的痛点。

### 关键数字
- 离线相关性指标（offline relevance）：绝对提升 **>5%**。
- 线上用户参与度（online engagement）：提升 **>4%**。
- Query 塔推理延迟：CPU 上 **P99 < 1.5ms**（2 层 XLM，128 维输出）。
- 线上 A/B：EBR 召回通道使参与度提升约 **2.8 个百分点**，排序侧引入相似度特征提升约 **1.24 个百分点**。
- 消融实验（外包标注数据集上，AUC 千分位提升）：课程学习 **+0.8‰**，多任务学习 **+0.5‰**，梯度混合多模态融合 **+0.7‰**。

### 原文链接
- Meta Research 官方页面（含摘要与 PDF 下载，已验证可访问）：https://research.facebook.com/publications/que2search-fast-and-accurate-query-and-document-understanding-for-search-at-facebook/
- 论文 PDF 直链：https://research.facebook.com/file/1360877564310294/Que2Search-Fast-and-Accurate-Query-and-Document-Understanding-for-Search-at-Facebook.pdf
- ACM DL：https://dl.acm.org/doi/10.1145/3447548.3467127
- 注：Que2Search 未发布 arXiv 版本，以上 Meta Research 官方链接和 ACM DOI 为权威来源。

---

## 4. DPR：Dense Passage Retrieval（Karpukhin et al., EMNLP 2020, arXiv:2004.04906）

### 直觉
开放域问答（"珠穆朗玛峰有多高？"）过去几乎全靠 BM25/TF-IDF 这类字面匹配去几百万篇维基百科段落里找答案所在段落。DPR 证明了一件很朴素但当时颇具冲击力的事：**只用几万条问答样本训练一个双塔模型，就能在语义层面超过经过几十年打磨的 BM25**——它把"关键词命中"换成了"问题向量与段落向量的点积"，而且训练技巧极其简单：**同一个 batch 里，别人的正确答案就是我的负例**，不需要额外挖掘负例的复杂流水线。

### 核心数学
**双编码器**：问题编码器 $E_Q$、段落编码器 $E_P$（两个独立的 BERT），相似度用点积：

$$
\text{sim}(q, p) = E_Q(q)^{\top} E_P(p)
$$

**In-batch Negatives 损失**：给定一个 batch 里的 $B$ 个 (question, positive passage) 对 $\{(q_i, p_i)\}_{i=1}^{B}$，对第 $i$ 条样本，把 batch 内其余 $B-1$ 个问题的正例段落全部当作 $q_i$ 的负例，用 softmax 归一化后取负对数似然：

$$
L(q_i, p_i^+, p_{i,1}^-, \dots, p_{i,n}^-) = -\log \frac{\exp(\text{sim}(q_i, p_i^+))}{\exp(\text{sim}(q_i, p_i^+)) + \sum_{j=1}^{n}\exp(\text{sim}(q_i, p_{i,j}^-))}
$$

在纯 in-batch negative 的实现中，负例集合 $\{p_{i,j}^-\}$ 就是 batch 内其他问题的正例段落，即分母退化为对 batch 内所有 $B$ 个段落求和：

$$
L = -\log \frac{\exp(\text{sim}(q_i,p_i))}{\sum_{j=1}^{B}\exp(\text{sim}(q_i,p_j))}
$$

**这条公式想优化的是**：不需要显式采样负例、不需要额外计算，训练时一次前向传播就"免费"获得了 $B-1$ 个负例（batch size 越大，负例越丰富，效果越好），把检索问题转化为 batch 内的一个多分类问题（谁是我的正确段落）。论文实际训练中 batch size 取 128，并额外混入 1 个 BM25 难负例段落以提升区分度。

### 架构图要素
- **组件**：Question Encoder（BERT）、Passage Encoder（BERT，独立参数）、In-batch Negative 采样器、FAISS 向量索引（离线对全部维基百科段落建索引）、下游 Reader/生成式 QA 模型（DPR 只负责检索，检索结果交给抽取式或生成式阅读理解模型作答）。
- **数据流向**：问题文本 → Question Encoder 实时编码成向量 → FAISS 检索预先离线编码好的段落向量库，取 top-K 段落 → 交给 Reader（如 BERT 抽取式阅读理解）在 top-K 段落里定位答案 span → 输出最终答案。这条链路后来被视为 RAG（检索增强生成）范式的直接前身。

### 在骨架中的位置
DPR 改造的是开放域问答场景里的**召回环节**：用稠密向量检索取代 BM25 稀疏检索，成为 reader/生成模型的上游供给。它与 EBR 是同一时期（2020 年）、互相独立却殊途同归的工作——EBR 面向工业级社交搜索、强调工程全栈与难负例挖掘；DPR 面向学术开放域问答、强调"in-batch negatives 有多简单多有效"，两者共同确立了双塔向量召回的范式，也是 ColBERT、Que2Search 等后续工作对照的基线。

### 关键数字
- Natural Questions 数据集，Top-20 段落检索准确率：DPR **78.4%** vs. BM25 **59.1%**（绝对提升约 **19.3 个百分点**，与摘要中"9%–19% 绝对提升"一致）。
- 端到端问答 Exact Match（NQ）：DPR-based 系统 **41.5%** vs. 此前 ORQA 的 **33.3%**，创下当时新的 SOTA。

### 原文链接
- arXiv 摘要页（已验证可访问）：https://arxiv.org/abs/2004.04906
- PDF：https://arxiv.org/pdf/2004.04906
- 会议版本：EMNLP 2020

---

## 5. ColBERT：Contextualized Late Interaction over BERT（Khattab & Zaharia, SIGIR 2020, arXiv:2004.12832）

### 直觉
把"双塔检索"和"交叉编码器精排"想象成两种极端的相亲方式：双塔是"先各自写一份自我介绍浓缩成一句话摘要，再比对两份摘要像不像"——快，但信息损失大；交叉编码器是"让两个人当面把所有细节都聊一遍再打分"——准，但一次只能撮合一对，慢到无法应对海量候选。ColBERT 的折中方案是：**先各自把自己拆解成很多个"关键词级别的自我介绍"（每个 token 一个向量），见面时不需要重新聊天，只需要拿这些关键词一一比对，各自找全场最像自己的那个词，再把最像的分数加起来**——既保留了细粒度的词级别交互，又能把文档那一半的向量提前算好、离线存起来。

### 核心数学
**Late Interaction / MaxSim**：query 经 BERT 编码为一组 token 级向量 $\{E_{q_1}, \dots, E_{q_N}\}$，document 同理编码为 $\{E_{d_1}, \dots, E_{d_M}\}$（均做归一化，使点积等价于 cosine 相似度）。相关性分数：

$$
S_{q,d} = \sum_{i=1}^{N} \max_{j=1}^{M} E_{q_i} \cdot E_{d_j}^{\top}
$$

符号含义：对 query 中每个 token 向量 $E_{q_i}$，在 document 的所有 token 向量里找**最相似**的那一个（$\max_j$，即 MaxSim），得到这个 query token 在这篇文档里"最好的落脚点"分数；再把 $N$ 个 query token 各自的最优匹配分数**求和**，作为整篇文档的最终得分。**这条公式想优化的问题**是：既不像双塔那样把整个文档压成一个向量（丢失词级别细节），也不像交叉编码器那样对每个 query-document 对都跑一次完整的 Transformer 前向（计算量爆炸）；因为 document 端的 token 向量可以**离线预计算并建索引**，在线阶段只需要做一堆向量点积和取 max/sum，代价极低。

### 架构图要素
- **组件**：Query Encoder（BERT + 线性降维层，query 用 `[MASK]` 填充做 query augmentation 增强表达）、Document Encoder（同结构 BERT，离线批量编码全库文档为 token 级向量矩阵）、向量相似度索引（token 级向量可用向量近邻索引做剪枝，支持端到端检索而不只是重排）、MaxSim 打分层（在线阶段的轻量计算）。
- **数据流向**：文档库离线经 Document Encoder 编码为 token 级向量矩阵并建索引 → 查询到来时 Query Encoder 实时编码出 query token 向量 → 与候选文档的 token 向量矩阵做 MaxSim 计算并求和 → 按分数排序输出。既可以放在"粗排候选之后做精排"，也可以用其剪枝友好的特性直接做端到端召回。

### 在骨架中的位置
ColBERT 主要发力**排序环节**（尤其是精排），是双塔（bi-encoder，只在召回阶段快）与交叉编码器（cross-encoder，只在精排阶段准但慢）之间的第三条路：它把交叉编码器"query-document 全交互"的精度部分保留在 token 级别，又把交叉编码器"每次都要重新跑完整前向"的开销通过预计算 document 向量摊掉，因此既能作为**高效精排器**替换传统 cross-encoder，也因为剪枝友好可以直接下沉到**召回环节**做端到端检索。这是后续 late-interaction 类模型（ColBERTv2、ColPali 等）的开山之作。

### 关键数字
- 相较基于 BERT 的交叉编码器精排：**推理速度快两个数量级（100×）**，**每次查询所需 FLOPs 减少四个数量级（10,000×）**。
- MS MARCO Passage Ranking（Dev 集）MRR@10：约 **0.360**，与 BERT-base 级别的交叉编码器效果相当，同时效率远超所有非 BERT 基线。

### 原文链接
- arXiv 摘要页（已验证可访问）：https://arxiv.org/abs/2004.12832
- PDF：https://arxiv.org/pdf/2004.12832
- 会议版本：SIGIR 2020

---

## 6. 补充概念（简洁版，供架构图与过渡页使用）

### 6.1 双塔（Bi-Encoder） vs. 交叉编码器（Cross-Encoder）的权衡

**直觉**：双塔是"提前把每个人的自我介绍写好存档"，交叉编码器是"每次都要让两人当面详谈"。

**核心矛盾**：
- 双塔：$score(q,d) = f(q) \cdot g(d)$，$q$ 和 $d$ 独立编码，互不看见对方——**可以离线预计算全部 document 向量并建 ANN 索引**，在线只需算一次 query 向量 + 一次近邻查找，延迟低、可扩展到十亿级候选，但因为编码时看不到对方，损失了 token 级的精细交互，精度上限较低。
- 交叉编码器：$score(q,d) = f(q \oplus d)$，$q$ 和 $d$ 拼接后一起过 Transformer，每一层都能互相 attend——精度最高，但**无法预计算**（因为每个 query-document 对的拼接输入都不同），对每个候选都要重新跑一次完整前向，只能用于"候选已经很少"的精排阶段。
- ColBERT 的 late interaction 正是在这两极之间插入的第三种范式（见第 5 节）。

**在骨架中的位置**：这组权衡决定了骨架为什么必须是"召回（双塔/倒排，快但粗）→ 排序（交叉编码器/复杂特征，慢但准）"的分层结构，而不是从头到尾用一个模型。

### 6.2 级联排序架构（Retrieval → 粗排 → 精排）

**直觉**：像面试招聘的"简历筛选 → 电话面试 → 现场面试"三级漏斗——每一轮候选人数量骤减，但对每个候选人投入的时间和精力骤增。

**结构**：
- **召回（Retrieval）**：候选池从数亿/数十亿降到数千，用倒排索引/ANN 等轻量方法，算力预算最低、单候选成本最低。
- **粗排（Pre-ranking / L1）**：候选从数千降到数百，用轻量模型（如小型 GBDT、简单双塔打分），可以引入比召回阶段更多的特征。
- **精排（Ranking / L2）**：候选从数百降到十几到几十，用最重的模型（深度网络、交叉编码器、多特征融合），因为候选集小，单次推理成本可以很高。

**核心权衡**：**候选数量与单候选算力成本成反比**，级联架构保证总算力预算可控——如果直接对亿级候选跑精排模型，算力开销是不可承受的；如果只用召回的粗糙打分做最终排序，效果又不够精细。Que2Search 论文中提到的"轻量 GBDT 粗排 + 深度模型精排"就是这一范式的工业实例。

### 6.3 ANN 原理直觉：HNSW 与 IVF-PQ 为什么可行

**直觉**：在十亿级向量里找最近邻，逐一计算距离是不可能的——ANN 的核心思想是"用可控的精度损失换取指数级的速度提升"，两条主流路径：

- **HNSW（Hierarchical Navigable Small World）**：把所有向量组织成一个**多层图**，越上层节点越稀疏、边越长（像坐飞机直达远方城市），越下层节点越密集、边越短（像市内公交站一站一站挪）。查询时从最上层的稀疏图开始贪心地往"更近的邻居"走，每层收敛后下沉一层继续细化，最终在最底层的稠密图里锁定近似最近邻。**为什么可行**：小世界网络的性质保证了从任意起点出发，经过 $O(\log N)$ 步贪心跳转就能接近目标，避免了线性扫描。
- **IVF-PQ（Inverted File + Product Quantization）**：分两步压缩搜索空间。**IVF（粗量化）**先用 k-means 把全部向量聚成几千个簇（类似给向量分"街区"），查询时只需要计算 query 到各簇中心的距离，选出最近的几个簇，把搜索范围从全库收窄到少数几个"街区"。**PQ（乘积量化，精排/打分）**把每个高维向量切成若干段子向量，每段独立用小型码本量化压缩，将原本需要存储的浮点向量压缩成几个字节的编码，距离计算也变成查表操作，大幅降低内存占用和计算量。EBR 论文中提到 PQ 的子空间切分数一般在 $x > d/4$ 后精度收益就很有限，是常见的调参经验。**为什么可行**：搜索问题被拆成"先用廉价的粗筛缩小范围，再用压缩后的向量做精确但低成本的打分"两步，避免了在全量高维空间里做精确距离计算。

**在骨架中的位置**：ANN 是**召回环节**的通用引擎，EBR、Que2Search、DPR 等所有依赖向量检索的方法最终都要落地到 HNSW 或 IVF-PQ 这类具体的索引结构上，才能在毫秒级延迟内完成十亿级候选的近似最近邻查询。

---

## 附：所有原文链接汇总（均已验证可访问）

| 论文 | 链接 |
|---|---|
| Unicorn（VLDB 2013） | https://www.vldb.org/pvldb/vol6/p1150-curtiss.pdf |
| Unicorn 图解博客（Meta Engineering） | https://engineering.fb.com/2013/03/06/core-infra/under-the-hood-building-out-the-infrastructure-for-graph-search/ |
| Facebook EBR（KDD 2020） | https://arxiv.org/abs/2006.11632 |
| Que2Search（KDD 2021） | https://research.facebook.com/publications/que2search-fast-and-accurate-query-and-document-understanding-for-search-at-facebook/ |
| DPR（EMNLP 2020） | https://arxiv.org/abs/2004.04906 |
| ColBERT（SIGIR 2020） | https://arxiv.org/abs/2004.12832 |

## 未能逐字核对的内容（供后续引用时留意）

- Que2Search 的消融实验千分位数字（课程学习 +0.8‰、多任务 +0.5‰、梯度混合 +0.7‰）与线上 A/B 百分点（EBR +2.8pp、排序 +1.24pp）来自第三方中文技术解读（6aiq.com，作者标注引用自原论文图表），未能直接从 ACM 论文原文的图表中逐字核对像素级数值，建议在幻灯片中标注为"约"或降低精度使用（如"提升个位数千分点"）。论文摘要中明确认领的硬数字（离线相关性 >5%、线上参与度 >4%、P99 延迟 <1.5ms）已通过 Meta Research 官方页面直接核对，可放心引用。
- ColBERT 的 MS MARCO MRR@10 = 0.360 数值来自第三方汇总表格及后续 ColBERTv2 论文中的对照表，未从 ColBERT 原始 SIGIR 论文正文逐字确认；摘要中"两个数量级更快、四个数量级更少 FLOPs"已在 arXiv 摘要原文中逐字确认，可高置信度引用。
- EBR 论文中"+18% events / +16% groups"的线上召回率提升数字来自搜索引擎对论文正文的摘录，未能直接打开论文 Table 4 逐字核对；建议幻灯片中标注数量级（"两位数百分比提升"）而非精确到个位数，或在正式使用前用 PDF 阅读工具二次核对原文表格。
