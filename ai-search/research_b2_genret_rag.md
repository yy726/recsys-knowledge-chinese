# 研究笔记 B2：生成式检索（DSI）与 RAG

> 幻灯片主线：「一条查询的旅程」= 查询理解 → 召回 → 排序 → 结果呈现 → 评估闭环。
> 生成式检索的定位：把「召回 + 索引」两幕直接替换成一个模型的参数记忆。
> RAG 的定位：把「结果呈现」从十条蓝链换成一段生成式答案，但检索器仍是前面几幕搭好的召回系统。

---

## 主题一：生成式检索（Generative Retrieval / DSI 家族）

### 1. 直觉

**DSI = 把图书馆的索引卡片全部背进脑子里。**

传统检索系统是「图书馆 + 检索员」的分离结构：文档被编码成向量存进一个外部索引（书架 + 卡片目录），查询来了之后检索员（dual encoder + MIPS/ANN）去查表、比对、排序。DSI 做的事情是把整座图书馆的目录背下来，直接问模型「哪本书里有答案」，模型脱口而出书号（docid）——不再有单独的索引数据结构，语料库的知识被压缩进 Transformer 的参数权重里。索引（indexing）和检索（retrieval）从两个独立系统，变成了同一个模型的两种训练任务：「记忆」和「回忆」。

这个类比也解释了它最大的软肋：背书的人换了新书要重新背（语料更新代价高，见 DSI++），而且书architecture太大时（百万级文档）脑子记不住那么多号码（见 Pradeep et al. 规模化实验）。

### 2. 核心数学

**2.1 检索问题的重新表述**

传统 IR 把查询 $q \in \mathcal{Q}$ 映射到文档集合的子集 $\{d_1,\ldots,d_n\} \subseteq \mathcal{D}$，经由 docid 中转。DSI 用一个 seq2seq 模型直接学习：

$$
q \;\longmapsto\; j \in \mathcal{Y}
$$

其中 $j$ 是文档 $d_j$ 的标识符（docid）。检索变成标准的条件生成：

$$
j^* = \arg\max_j \Pr_\theta(j \mid q)
$$

对比三种范式的核心差异（DSI 论文 Table 1）：
- BM25/TF-IDF：稀疏向量匹配 $\arg\max_j \mathbf{v}_q^\top \mathbf{v}_{d_j}$，索引 = 倒排表
- Dual Encoder：稠密向量匹配（近似 MIPS），索引 = 向量表
- **DSI**：索引 = 训练模型本身；检索 = 跑一次模型推理，索引这个动作被约化为「训练」这个机器学习中最基本的操作

**2.2 双任务训练目标（indexing + retrieval）**

DSI 用同一个 seq2seq 模型联合优化两类样本：

$$
\mathcal{L} = \underbrace{-\sum_{(d_j,\,j)} \log P_\theta(j \mid d_j)}_{\text{indexing 任务：记住文档→docid 的映射}} \;+\; \underbrace{-\sum_{(q,\,j)} \log P_\theta(j \mid q)}_{\text{retrieval 任务：学会查询→docid 的映射}}
$$

直白解释：indexing 任务教模型「这篇文档对应几号」（相当于把文档内容压进参数），retrieval 任务教模型「这个问题应该去几号文档找答案」。只用 retrieval 样本训练不足以泛化到未见过的检索（论文原话），必须靠 indexing 样本让模型先"记住"语料库。

**2.3 docid 表示方案（对结果影响巨大）**

论文核心消融维度，三种主要方案：
1. **Atomic docid（无结构原子标识符）**：每篇文档分配一个独立的新 token，解码器输出层直接扩展一个 softmax 分类头，一步到位但词表随语料线性膨胀。
2. **Naive string docid（朴素字符串）**：把整数 docid 直接当字符串符号序列生成（如"31441"逐位输出），复用已有词表，可扩展到大语料。
3. **Semantic string docid（语义字符串 / 层次聚类）**：先对文档做层次聚类（如对文档的稠密表示做 $k$-means 递归二分），docid 变成一串反映聚类路径的数字前缀（如"3-1-4"表示第3类下第1子类下第4篇），语义相近的文档共享 docid 前缀，解码时天然带有 beam search 友好的层次结构。

结论：**结构化的 docid（语义聚类式或朴素字符串式）比原子式更能扩展到大语料**，因为解码器词表不必随文档数线性增长；语义聚类式在多组实验中综合表现最佳。

**2.4 NCI 的前缀感知解码（PAWA）**

NCI 在 DSI 语义 docid 思路上，把解码器改造为「前缀感知、权重自适应」（Prefix-Aware Weight-Adaptive decoder, PAWA），根据已生成的 docid 前缀动态调整下一步的输出分布，并引入查询生成（query generation）增强训练信号、以及一致性正则化（consistency-based regularization）抑制过拟合语义聚类噪声。

**2.5 SEAL 的 FM-index 约束解码**

SEAL 不给文档分配人工 docid，而是**用文档中出现过的任意 n-gram 本身作为标识符**。用 FM-index（Ferragina–Manzini index，一种压缩后缀数组）对整个语料建立索引，解码时用约束生成（constrained decoding）确保模型只能生成语料中真实出现过的 n-gram 片段，生成的 n-gram 再通过 FM-index 高效映射回其所在的原文文档。这样文档表示不需要预先设计的层次结构，避免了 DSI/NCI 的"聚类 docid 设计"这一额外工程环节。

### 3. 架构图要素

**组件：**
- 语料预处理：文档 → docid 分配（atomic / naive string / semantic clustering / FM-index n-gram）
- 单一 Transformer（encoder-decoder，如 T5）：既是索引存储介质，也是检索执行引擎
- 两路训练数据流：`(document, docid)` 索引样本 + `(query, docid)` 检索样本，可选混入 query generation 生成的伪查询样本弥合分布差异
- 推理时：beam search 解码 docid 序列（或 SEAL 中的约束 n-gram 解码）得到 top-k 候选文档

**数据流（文字描述）：**
文档语料 → 离线阶段：docid 编码方案确定（聚类 / FM-index 建立）→ 与合成查询（query generation，如 DSI-QG、NCI）配对 → 灌入 seq2seq 模型做多任务训练（indexing loss + retrieval loss）→ 训练收敛后，模型参数即索引 → 线上查询进入模型 → beam search 直接吐出 docid 列表 → 无需额外的向量检索或倒排查表步骤。

### 4. 在骨架中的位置与上下游

生成式检索**替换的是「召回 + 索引」这两幕**：传统骨架里"查询理解"之后是"召回"（从倒排索引或向量索引里捞出候选），DSI 把这一步压缩成"模型对 query 做一次前向推理直接吐出 docid"，索引本身不再是独立的数据结构，而是模型训练的副产物。下游的"排序"环节在纯 DSI 范式里被弱化甚至消失（beam search 得分本身自带排序），"结果呈现"和"评估闭环"两幕基本不受影响，仍然复用原有链路。

它与 dense retrieval 的权衡：dense retrieval（dual encoder + ANN）索引更新只需增量插入向量，语料变化友好；DSI 索引更新等价于重新训练/微调模型，语料动态更新是天生短板（催生了 DSI++、IncDSI 等后续工作）。dense retrieval 在千万级以上语料上依然稳健，而生成式检索在百万级语料上的有效性仍是未解难题（见下方 Pradeep et al. 结论）。

### 5. 关键数字

**DSI 原始论文（Tay et al., NeurIPS 2022，NQ 数据集，语料规模 10k/100k/320k 文档）：**
- Base 规模 T5：在最小语料（10k）上 Hits@1 从 dual encoder 的 12.4% 提升到 DSI 的 33.9%（提升超过 20 个点）；在语料规模扩大 30 倍（320k）后，提升幅度收窄到约 7 个点
- 11B 参数 T5：小语料上比 dual encoder 提升超过 25 个点，大语料上提升超过 15 个点——**模型越大，DSI 相对优势越明显**
- 零样本（zero-shot）设置下，DSI 的 Hits@1 比 BM25 高 14 个点
- 结论：DSI 显著优于 dual encoder 基线，且随模型规模持续提升（这是该论文最大的卖点）

**NCI（Wang et al., NeurIPS 2022）：**
- 相较最佳基线，NQ320k 上 Recall@1 相对提升 +21.4%，TriviaQA 上 R-Precision 相对提升 +16.8%

**SEAL（Bevilacqua et al., NeurIPS 2022）：**
- 在 KILT 基准（含 NQ）上，passage-level R-precision 平均比更成熟的检索方案高出至少 10 个点，同时内存占用比同类系统更轻量

**规模化挑战：Pradeep et al., "How Does Generative Retrieval Scale to Millions of Passages?" (EMNLP 2023, arXiv 2305.11841)**
- 首次在 MS MARCO 全量语料（8.8M passages）上系统评测生成式检索，模型规模测到 11B 参数
- 核心结论三条：(1) **合成查询（synthetic queries）作为文档表示在索引阶段的重要性被反复验证为核心**，弱化了原始 DSI 的"直接用文档全文做索引样本"策略；(2) 此前提出的多种架构改动，在**扣除计算成本**后基本无效；(3) **单纯扩大模型参数量的收益有上限**，无法像 LLM 那样持续 scaling
- 总体判断：生成式检索在小语料上与 SOTA dual encoder 具备竞争力，但**扩展到百万级语料仍是重要且尚未解决的挑战**——这是目前对生成式检索最权威的"泼冷水"结论，适合作为幻灯片里的平衡视角

**DSI++（Mehta et al., 2022/2023, arXiv 2212.09744）：**
- 持续索引新文档会导致模型对已索引文档产生明显的灾难性遗忘（forgetting），训练过程中出现不稳定的"遗忘事件"
- 提出用生成式记忆（对已索引文档采样伪查询）在持续索引新文档时同步回放，缓解遗忘——这是应对"动态语料更新"难题的代表性方案

**TIGER（Rajput et al., NeurIPS 2023, arXiv 2305.05065）：**
- 把生成式检索思路搬进推荐系统：用预训练文本编码器 + RQ-VAE 残差量化，为每个 item 生成一串离散的语义 ID（tuple of semantic tokens），再用 seq2seq Transformer 在用户历史序列上自回归预测下一个 item 的语义 ID
- 显著优于当时 SOTA 的序列推荐模型；对**冷启动/长尾 item**（无历史交互记录）的泛化能力明显提升，是语义 ID 最常被引用的落地价值

**工业落地情况（2025-2026，可搜索到的新进展）：**
- Semantic ID 方案已在 **Meta、YouTube、淘宝（Taobao）、快手（Kuaishou）** 等公司的生产推荐系统中落地，用于解决冷启动泛化、embedding 表稳定性、支撑大规模动态语料等问题
- **OneRec（快手，2025年2月）**：统一 encoder-decoder 生成式推荐架构，引入基于强化学习的迭代偏好对齐（Iterative Preference Alignment, IPA）机制，用用户反馈和奖励模型精炼生成结果
- **OneRec-V2（快手，2025年6月）**：转向完全自回归解码器，改进多领域对齐，联合优化推荐、搜索、广告任务
- **DAS（Dual-Aligned Semantic IDs）**：已在快手在线广告系统中部署上线
- **EGA-V1/V2（美团，2025）**：把候选召回、排序、竞价建模统一进单一生成式接口，用于在线广告，实现 token 级出价与实时端到端推理
- 电商搜索方向也出现 semantic cluster ID + RL 专家引导的生成式检索工作（2026年新论文），显示这一路线正从推荐系统向搜索场景延伸

### 6. 原文链接（已核实）

- DSI: Tay et al., "Transformer Memory as a Differentiable Search Index", NeurIPS 2022 — https://arxiv.org/abs/2202.06991
- NCI: Wang et al., "A Neural Corpus Indexer for Document Retrieval", NeurIPS 2022 — https://arxiv.org/abs/2206.02743
- SEAL: Bevilacqua et al., "Autoregressive Search Engines: Generating Substrings as Document Identifiers", NeurIPS 2022 — https://arxiv.org/abs/2204.10628
- DSI-QG: Zhuang et al., "Bridging the Gap Between Indexing and Retrieval for Differentiable Search Index with Query Generation" — https://arxiv.org/abs/2206.10128
- 规模化挑战：Pradeep et al., "How Does Generative Retrieval Scale to Millions of Passages?", EMNLP 2023 — https://arxiv.org/abs/2305.11841
- DSI++: Mehta et al., "DSI++: Updating Transformer Memory with New Documents" — https://arxiv.org/abs/2212.09744
- TIGER: Rajput et al., "Recommender Systems with Generative Retrieval", NeurIPS 2023 — https://arxiv.org/abs/2305.05065

---

## 主题二：RAG 演进

### 1. 直觉

**RAG = 让模型考试时可以开卷，但答题（生成）的手还是自己的手。**

参数化语言模型把知识"死记硬背"进权重里，知识陈旧、无法追溯来源、也无法针对新知识局部更新。RAG 的思路是给模型配一个可以随时翻阅的外部资料库（非参数化记忆）：先检索出相关文档，再把文档"喂"给生成模型，让生成过程"看着资料作答"。这条演化线的关键分水岭在于"翻书"这件事的粒度和主动权：REALM 是"预训练阶段就学会怎么翻书"，RAG（Lewis et al.）是"给定资料后怎么把多篇资料揉进一次生成"的两种加权公式，FiD 是"翻书效率工程"（怎么塞下更多资料还不爆显存），Self-RAG 是"自己决定要不要翻书、翻完书自己批改答案"，2025-2026 的 Agentic RAG / Deep Research 则是"从翻一次书变成带着问题多轮跑图书馆、边查边想、写研究报告"。

### 2. 核心数学

**2.1 REALM：检索即隐变量，MIPS 反向传播**

REALM 把检索建模为一个隐变量 $z$（从语料库 $\mathcal{Z}$ 中挑出的文档），联合建模检索概率和生成概率：

$$
p(y \mid x) = \sum_{z \in \mathcal{Z}} p(y \mid z, x)\, p(z \mid x)
$$

检索分布用最大内积搜索（MIPS）的 softmax 形式给出：

$$
p(z \mid x) = \frac{\exp\big(f(x,z)\big)}{\sum_{z'} \exp\big(f(x,z')\big)}, \qquad f(x,z) = \mathrm{Embed}_{\text{input}}(x)^\top \mathrm{Embed}_{\text{doc}}(z)
$$

直白解释：$x$ 是掩码语言模型的输入，$z$ 是候选文档，$f(x,z)$ 是查询向量和文档向量的内积（打分函数）。这个检索步骤要对**百万级文档**做 argmax/softmax 近似，训练时通过反向传播联合更新检索器和语言模型的参数——这是"端到端联合训练检索器"的关键：检索器不是外挂的固定模块，而是随着语言建模目标一起被梯度更新的。工程上的难点是文档 embedding 会随参数更新过时，REALM 采用**异步刷新 MIPS 索引**（每隔几百步用最新参数重新计算并重建索引）来缓解这个"陈旧索引"问题。

**2.2 RAG（Lewis et al.）：两种边缘化公式**

RAG 把检索到的 top-k 文档 $z$ 当作隐变量，在生成序列 $y = (y_1,\ldots,y_N)$ 上边缘化，给出两种公式：

RAG-Sequence（整个生成序列共用同一篇/组检索文档）：

$$
p_{\text{RAG-Seq}}(y \mid x) \approx \sum_{z \in \text{top-}k(p(\cdot\mid x))} p_\eta(z \mid x)\, \prod_{i=1}^{N} p_\theta(y_i \mid x, z, y_{1:i-1})
$$

RAG-Token（每生成一个 token 都可以重新加权/切换检索文档）：

$$
p_{\text{RAG-Tok}}(y \mid x) \approx \prod_{i=1}^{N} \sum_{z \in \text{top-}k(p(\cdot\mid x))} p_\eta(z \mid x)\, p_\theta(y_i \mid x, z, y_{1:i-1})
$$

其中 $p_\eta(z\mid x)$ 是 DPR 式双塔检索器给出的文档相关性分布，$p_\theta$ 是 seq2seq 生成器（BART）。直白解释：RAG-Sequence 是"先选定几篇参考资料，通篇按这几篇写完"；RAG-Token 是"每写一个字都可以换一篇参考资料来看"，粒度更细但计算更贵。两者共享同一个训练目标——端到端最大化生成序列的对数似然，检索器和生成器联合微调（但检索器的文档 index 本身在微调阶段通常保持固定，不像 REALM 那样重建索引）。

**2.3 FiD：解码器侧融合，编码器独立编码**

FiD 把检索到的 $k$ 篇文档**独立**编码（可并行），再在解码器侧把所有文档的编码拼接起来做统一的交叉注意力：

$$
\mathbf{H}_i = \mathrm{Encoder}\big([q \,;\, p_i]\big), \quad i = 1,\ldots,k
\qquad\qquad
y = \mathrm{Decoder}\big(\mathrm{Concat}(\mathbf{H}_1,\ldots,\mathbf{H}_k)\big)
$$

直白解释：编码器只需要分别理解"问题+第 $i$ 篇文档"，这一步计算量随文档数线性增长但可以并行处理；真正融合多篇文档信息的动作放在解码器的交叉注意力里一次性完成（"Fusion-in-Decoder"）。这避免了把所有文档拼成一个长序列再整体编码（那样自注意力开销是文档数的平方级），使得 FiD 能够比 RAG 检索并利用多得多的文档（几十篇）而不至于显存爆炸。

**2.4 Self-RAG：反思 token 作为词表的一部分**

Self-RAG 把检索决策和自我评估也变成"生成"的一部分，扩展词表加入反思 token，标准的下一词预测同时预测这些控制符号：

$$
p(y_t, r_t \mid x, y_{<t})
$$

四类反思 token：
- **Retrieve**：判断当前步骤是否需要检索（按需检索，而非无脑固定检索）
- **ISREL**（is relevant）：判断检索到的段落是否与问题相关
- **ISSUP**（is supported）：判断生成内容是否被检索段落所支持（事实一致性）
- **ISUSE**（is useful）：判断最终生成内容整体上是否有用

直白解释：模型在生成过程中会"自言自语"式地吐出这些标记，用来决定"要不要去查资料""查到的资料靠不靠谱""我写的话有没有依据""这段回答整体有没有用"，从而让检索变成模型自主决策的动作，而不是外部流程写死的"每次都检索 top-k"。

### 3. 架构图要素

**REALM：** 输入 $x$（带掩码的句子）→ 检索器（双塔 embedding，MIPS 索引）→ 检索出候选文档 $z$ → 拼接 $x$ 和 $z$ 输入 knowledge-augmented encoder → 预测掩码 token → 梯度同时回传给检索器和编码器 → 每隔若干步异步重建 MIPS 索引。

**RAG（Lewis）：** 输入 $x$ → DPR 双塔检索器（query encoder + 预先编码好的 Wikipedia 稠密向量索引）→ top-k 段落 → BART 生成器逐个/统一编码段落 → RAG-Sequence 或 RAG-Token 两种方式做边缘化解码 → 输出 $y$。

**FiD：** 输入 $x$ → 检索器（BM25 或 DPR）→ $k$ 篇段落 → 编码器**并行独立**编码每个"问题+段落"对 → 解码器把所有段落编码拼接成一个大的 memory bank，统一做交叉注意力生成答案。

**Self-RAG：** 输入 $x$ → LM 先判断是否需要 Retrieve → 若需要，调用检索器取回段落 → 并行生成候选延续片段并标注 ISREL/ISSUP → 按反思 token 打分做 segment-level beam search，挑出最优延续 → 结尾标注 ISUSE 综合评估。

**Agentic RAG / Deep Research（2025-2026）：** 用户任务 → LLM 规划器先生成一份多步研究计划（可选：请用户确认/修改计划）→ 循环执行：搜索引擎调用 / 网页浏览 / 工具调用（代码执行、数据分析）→ 阅读、推理、判断信息是否足够 → 不够则继续下一轮检索（迭代式执行，interleaved tool use，类 ReAct 模式）→ 信息充分后融合多源证据撰写带引用的研究报告。

### 4. 在骨架中的位置与上下游

RAG 系列改变的是**「结果呈现」**这一幕：从返回十条蓝链（标题+摘要+URL 列表）变成直接生成一段综合性答案（可带引用）。但 RAG 的 retriever 本身**并不是新发明的组件**——无论是 REALM 的 MIPS 检索器、RAG 论文里的 DPR 双塔、FiD 依赖的 BM25/DPR，还是 Self-RAG 按需调用的检索模块，本质上都是前面"查询理解 → 召回 → 排序"这几幕已经搭建好的检索系统（dense retrieval / 稀疏检索 / 混合检索）。RAG 是在检索系统之上加了一层"生成式结果呈现"，检索质量的上限依然决定了 RAG 答案质量的上限（"检索不到就编不出真话"）。到了 Agentic RAG / Deep Research 阶段，这一层进一步演化成会自己规划、多轮调用检索与工具、自我纠错的智能体，某种程度上把"查询理解"（分解子问题）、"召回"（多轮检索）、"结果呈现"（综合报告）三幕重新揉进了一个循环智能体里，是对整条骨架的一次收拢与重构。

### 5. 关键数字

**REALM（Guu et al., ICML 2020, arXiv 2002.08909）：**
- 在三个主流 Open-QA 基准上相较此前 SOTA（含显式和隐式知识存储两类方法）**绝对准确率提升 4–16 个百分点**
- 同时带来可解释性与模块化的额外收益（检索到的具体文档可作为答案依据展示）
- 原始实验用 ScaNN 库做 MIPS，开源版本改为暴力矩阵乘法

**RAG（Lewis et al., NeurIPS 2020, arXiv 2005.11401）：**
- 在三个开放域问答任务上取得当时的 SOTA，超过纯参数化 seq2seq 模型和任务专用的 retrieve-and-extract 架构
- 在语言生成任务上，RAG 生成的文本比纯参数化 SOTA 基线**更具体、更多样、更符合事实**
- RAG-Sequence 与 RAG-Token 两种边缘化公式在不同任务上各有优势，是后续几乎所有 RAG 变体讨论"检索粒度"问题的起点

**FiD（Izacard & Grave, EACL 2021, arXiv 2007.01282）：**
- 通过增加检索段落数量持续提升效果，在 Natural Questions 与 TriviaQA 上取得当时开放域问答的 SOTA（参考后续文献引用的数值：大模型版本 NQ EM≈51.4，TriviaQA EM≈67.6，具体数值请以论文正文表格为准）
- 核心贡献是证明"解码器侧融合"可以让模型有效利用远多于此前方法（如 RAG）的检索段落数量，而不产生编码器侧的平方级注意力开销

**Self-RAG（Asai et al., ICLR 2024, arXiv 2310.11511）：**
- 7B 和 13B 参数规模的 Self-RAG，在开放域 QA、推理、事实校验等多任务上**显著优于 ChatGPT 以及带检索增强的 Llama2-chat**
- 在长文本生成任务上，相较这些基线在**事实性和引用准确率**上有显著提升
- 关键机制：反思 token 让检索变成"按需触发"而非"无脑固定 top-k"，避免了不必要或不相关检索段落拖累生成质量

**Agentic RAG / Deep Research（2025-2026 进展）：**
- **OpenAI Deep Research**（2025年2月25日发布，系统卡：https://cdn.openai.com/deep-research-system-card.pdf）：基于为网页浏览与数据分析优化的 o3 模型版本，可自主进行多步网络研究，号称能在数十分钟内完成人类需要数小时才能完成的信息综合任务；2025年4月起 Plus/Team/Enterprise/Edu 用户每月 25 次额度、Pro 用户 250 次，并推出基于 o4-mini 的轻量版；2025年7月17日起接入可视化浏览器（随 ChatGPT agent 能力扩展）
- **Gemini Deep Research**：2024年末推出，2025-2026 年基于 Gemini 2.5 Pro（百万级 token 上下文、稀疏 MoE 架构、"思考模型"）持续升级；核心特征是**先生成研究计划、经用户确认后再执行**"规划 → 搜索 → 阅读 → 推理 → 撰写"的完整闭环；2026年4月21日推出 Deep Research Max（基于 Gemini 3.1 Pro），提供"速度优先"与"全面性优先"两种智能体模式
- **学术定义（Agentic RAG Survey, arXiv 2501.09136，2025年1月）**：判定一个 RAG 方法是否"真正 agentic"的三条标准——(1) 自主策略：LLM 自主选择和组织检索/推理策略；(2) 迭代执行：支持基于中间结果动态调整的多轮执行；(3) 交织工具调用：遵循类 ReAct 的"行动-观察"交替模式

### 6. RAG 与传统搜索栈的关系

RAG 的 retriever 不是独立发明的新技术栈，而是**直接复用前几幕已经建好的检索系统**：查询理解阶段的改写/扩展、召回阶段的稀疏/稠密/混合检索、排序阶段的重排模型，都可以原封不动地作为 RAG pipeline 的输入。RAG 论文本身用的就是标准 DPR 双塔（对应"召回"这一幕的 dense retrieval 技术），FiD、Self-RAG 也都可以插拔式换用 BM25、DPR 或其他检索器。这意味着在幻灯片的骨架图上，RAG 不需要画一整套新的检索路径，只需要在"结果呈现"这一幕之前插入"生成"这个新组件，并把"召回+排序"的输出作为生成的条件输入——这正是把 RAG 定位为"结果呈现层革新"而非"检索层革新"的依据。Agentic RAG / Deep Research 则是例外：它把查询理解、召回、结果呈现重新打包进一个多轮迭代的智能体循环，某种程度上模糊了骨架中原本清晰的阶段边界。

### 7. 原文链接（已核实）

- REALM: Guu et al., "REALM: Retrieval-Augmented Language Model Pre-Training", ICML 2020 — https://arxiv.org/abs/2002.08909
- RAG: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020 — https://arxiv.org/abs/2005.11401
- FiD: Izacard & Grave, "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", EACL 2021 — https://arxiv.org/abs/2007.01282
- Self-RAG: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", ICLR 2024 — https://arxiv.org/abs/2310.11511
- Agentic RAG Survey: "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG" — https://arxiv.org/abs/2501.09136
- OpenAI Deep Research 系统卡（2025年2月）— https://cdn.openai.com/deep-research-system-card.pdf ；发布说明 — https://openai.com/index/introducing-deep-research/
- Gemini Deep Research Agent 官方文档 — https://ai.google.dev/gemini-api/docs/deep-research

---

## 附：与骨架的一句话映射（供制图参考）

| 环节 | 传统搜索栈 | 生成式检索（DSI）的替换 | RAG 的复用/新增 |
|---|---|---|---|
| 查询理解 | 改写、扩展、意图识别 | 不变（仍需先理解 query） | 不变；Agentic RAG 会把它升级为"任务规划" |
| 召回 | 倒排索引 / dense retrieval + ANN | **被模型参数记忆整体替换**：一次前向推理直接吐出 docid | 复用召回系统输出作为生成的条件输入 |
| 排序 | 学习排序 / 重排模型 | beam search 得分部分承担排序功能 | 复用排序结果决定送入生成器的段落顺序/数量（FiD 尤其依赖） |
| 结果呈现 | 十条蓝链 + 摘要 | 不适用（DSI 本身不生成答案，只返回 docid） | **核心改造点**：从链接列表变为生成式答案，可带引用 |
| 评估闭环 | CTR/NDCG/人工标注 | Recall@k / Hits@k（IR 传统指标） | 事实性、引用准确率、有用性（Self-RAG 式反思指标） |
