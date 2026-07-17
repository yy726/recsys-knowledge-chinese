# 经典搜索引擎技术在 2025-2026 工业搜索系统中的现状核查

核查日期：2026-07-15
方法：WebSearch + web_fetch 检索官方博客、论文（arXiv/ACL/KDD/RecSys）、法庭文件报道等公开资料，禁止命令行抓取。

---

## 1. BM25 / 倒排索引

**【2026 结论】** BM25 + 倒排索引依然活着，但角色从"唯一检索器"降级为"混合检索（hybrid retrieval）中的一路 lexical channel"，通常通过 RRF 与向量检索融合，是当前 Elasticsearch/OpenSearch/Vespa 的默认工业实践。

**【证据】**
- Elasticsearch 8.8（2023）为 Search API 加入 RRF 支持，8.9 正式推出"combine vector, keyword, and semantic retrieval with hybrid search"，2024-2025 年 Elasticsearch Labs 持续发文讲解加权 RRF：[Weighted reciprocal rank fusion (RRF) in Elasticsearch](https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf)；[Elastic Search 8.9 hybrid search 发布博客](https://www.elastic.co/blog/whats-new-elastic-enterprise-search-8-9-0)
- OpenSearch 官方博客系统讲解 BM25 + dense + RRF/normalization 的混合检索管道最佳实践：[Building effective hybrid search in OpenSearch: techniques and best practices](https://opensearch.org/blog/building-effective-hybrid-search-in-opensearch-techniques-and-best-practices/)
- Vespa 官方文档/博客：BM25 作为 first-phase 候选生成，向量相似度作 second-phase 精排，再用 RRF 在 global-phase 融合两路信号，强调"BM25+ANN 在 web 规模下 <100ms"：[Vespa Hybrid Search Tutorial](https://docs.vespa.ai/en/learn/tutorials/hybrid-search.html)；[Redefining Hybrid Search Possibilities with Vespa](https://blog.vespa.ai/redefining-hybrid-search-possibilities-with-vespa/)
- 2025 年学术研究显示 BM25 在 BEIR 零样本场景下仍是有竞争力的基线，尤其在接入 reranker 后可比肩最好的 dense 模型：[arXiv:2504.12879](https://arxiv.org/pdf/2504.12879)

**【与 LLM 的关系】** 被增强并存——BM25/倒排索引仍是 hybrid retrieval 中不可或缺的一路"精确词面匹配"信号，通过 RRF 与向量/LLM embedding 融合，而非被取代。

---

## 2. 中文分词 / 分词技术

**【2026 结论】** 中文分词在倒排索引侧依然必要（LLM 的子词 tokenizer 与面向检索的分词是两套不同体系、用途不同），LLM 时代改变的是"要不要用倒排索引"而非"要不要分词"。

**【证据】**
- 中文技术博客系统讨论"LLM 时代是否还需要倒排索引"，结论是倒排索引作用在 LLM 时代发生演变（更多服务稠密检索/向量索引），但对传统 lexical 检索仍是刚需：[在 LLM 时代我们是否还需要倒排索引？](https://yangwenbo.com/articles/llm-and-inverted-index.html)
- jieba（结巴中文分词）仍在持续维护和讨论，且专门保留 `cut_for_search` 搜索引擎模式用于构建倒排索引：[GitHub - fxsjy/jieba](https://github.com/fxsjy/jieba)；[jieba分词原理解析：一个老牌中文分词器的工程智慧（2025）](https://asterzephyr.github.io/posts/jieba_blog/)
- 中国科学院计算所团队 2024 年综述系统讨论大语言模型时代信息检索技术的演变，包括分词/索引环节的角色变化：[大语言模型时代的信息检索综述（ACL CCL 2024）](https://aclanthology.org/2024.ccl-2.6.pdf)

**【与 LLM 的关系】** 共存而非替代——LLM 的 BPE/WordPiece 子词分词器服务于模型输入侧，jieba 等面向倒排索引的中文分词服务于检索侧，两者职责不同、并未互相取代（此项证据链相对较弱，多为个人技术博客而非官方一手材料）。

---

## 3. Term Weight / 词权重

**【2026 结论】** 静态词权重（TF-IDF/BM25 式打分）正被学习型稀疏检索（Learned Sparse Retrieval，如 SPLADE、ELSER）和深度词袋模型增强，头部电商已把学习型词权重模型部署到线上系统，但完全推翻倒排索引式打分尚未成为普遍现状，多是"增强/并存"。

**【证据】**
- Elastic 官方将其学习型稀疏编码器 ELSER 在 8.11 版本（2023 年末）正式 GA 用于生产环境，持续在 8.19 文档中维护：[Elastic Search 8.11: ELSER model is now GA](https://www.elastic.co/blog/whats-new-elastic-search-8-11-0)；[ELSER 文档 8.19](https://www.elastic.co/guide/en/machine-learning/8.19/ml-nlp-elser.html)
- Pinecone 官方集成 Naver 的 SPLADE 模型做稀疏向量生成，支持 sparse-dense 混合向量公开预览；SPLADE-v3 于 2024 年 3 月发布：[SPLADE for Sparse Vector Search Explained | Pinecone](https://www.pinecone.io/learn/splade/)；[Introducing support for sparse-dense embeddings | Pinecone](https://www.pinecone.io/blog/sparse-dense/)；[GitHub - naver/splade](https://github.com/naver/splade)
- 淘宝团队 2024 年论文提出"Deep Bag-of-Words"模型，为中文电商 query 中不同词（品牌/类目 vs 修饰词）学习差异化权重，并已部署上线：[Deep Bag-of-Words Model: An Efficient and Interpretable Relevance Architecture for Chinese E-Commerce (arXiv:2407.09395)](https://arxiv.org/pdf/2407.09395)
- KDD'23 论文提出端到端学习词权重的方法，是学习型词权重研究的代表作：[End-to-End Query Term Weighting (KDD'23)](https://doi.org/10.1145/3580305.3599815)

**【与 LLM 的关系】** 部分被学习模型增强/替代——SPLADE/ELSER 等由 BERT 类 transformer 打分取代静态 IDF 式权重，是"神经网络化"而非直接的"LLM 化"，目前尚未看到大规模用生成式 LLM 直接做词权重打分的工业案例。

---

## 4. 意图识别 / 类目识别

**【2026 结论】** 头部工业搜索系统正快速把意图识别/query 理解从传统多级判别式 pipeline（BERT 分类器）切换为统一生成式 LLM 或 LLM 蒸馏出的小模型，2025-2026 年已有多篇一手工业论文佐证。

**【证据】**
- 小红书（Xiaohongshu）2026 年论文 QP-OneModel：用统一生成式 LLM 取代"BERT 判别式模型组成的孤立 pipeline"，一个模型同时做意图理解、query 改写等多任务，并生成意图描述作为下游排序的新语义信号：[QP-OneModel: A Unified Generative LLM for Multi-Task Query Understanding in Xiaohongshu Search (arXiv:2602.09901)](https://arxiv.org/html/2602.09901)
- Amazon 2025 年论文 REIC：用 RAG + LLM 做大规模意图分类，解决传统分类器"意图/类目体系频繁变动需要重训练"的痛点，已在 KDD'25 workshop 及 EMNLP'25 industry track 发表：[REIC: RAG-Enhanced Intent Classification at Scale (arXiv:2506.00210)](https://arxiv.org/abs/2506.00210)；[Amazon Science 页面](https://www.amazon.science/publications/reic-rag-enhanced-intent-classification-at-scale)
- Amazon 2025 年论文提出"rationale-guided distillation"，用 LLM 生成的推理依据蒸馏出 110M 参数的轻量 cross-encoder，效果匹敌 7B 参数 LLM 但推理快 50 倍，用于 query-product 相关性/意图判断：[Rationale-guided distillation for e-commerce relevance classification - Amazon Science](https://www.amazon.science/publications/rationale-guided-distillation-for-e-commerce-relevance-classification-bridging-large-language-models-and-lightweight-cross-encoders)
- 淘宝 TaoSR1（2025）用"思考模型"（CoT + RL）处理否定、平替、问答类复杂 query 的相关性/意图判断，是把 LLM 推理能力注入原本由 BERT 主导环节的典型案例：[TaoSR1: The Thinking Model for E-commerce Relevance Search (arXiv:2508.12365)](https://arxiv.org/abs/2508.12365)

**【与 LLM 的关系】** 被替代/被蒸馏——原判别式分类器逐步被生成式 LLM 直接推理，或被 LLM 蒸馏出的小模型取代，尤其在长尾、复杂语义、类目体系频繁变化的场景。

---

## 5. 相关性 BERT（交叉编码器精排）

**【2026 结论】** 交叉编码器（cross-encoder，BERT 系）仍是精排主力，但正与 LLM 推理型排序器（CoT/RL 驱动）及 RankGPT 蒸馏排序器并存，形成"BERT 做基础相关性判断 + LLM 只在难例/头部结果上做增强"的分层架构。

**【证据】**
- Google 反垄断案（US v. Google）法庭文件披露：Google 的核心排序模型 RankEmbed（内部称 RankEmbed BERT）是 dual-encoder 深度学习模型，用 70 天搜索日志 + 人工质量评分训练；2025 年 9 月 2 日法官 Amit Mehta 裁定 Google 须向"合格竞争对手"披露其训练数据（不含模型本身）：[Google's RankEmbed system forced to share secrets in antitrust remedy (ppc.land, 2025-09-04)](https://ppc.land/googles-rankembed-system-forced-to-share-secrets-in-antitrust-remedy/)
- 淘宝 TaoSR1（2025）明确指出"BERT-based models excel at semantic matching but lack complex reasoning capabilities"，因此在 BERT 之外引入 LLM 思考模型作为互补而非替代：[TaoSR1 (arXiv:2508.12365)](https://arxiv.org/abs/2508.12365)
- RankGPT 证明 GPT-3.5/GPT-4 是强 zero-shot listwise reranker，但推理成本过高不适合直接上生产；后续 Rank-DistiLLM 等工作把 RankGPT 的排序能力蒸馏进小型开源 cross-encoder，效果媲美 LLM 排序器但效率高出几个数量级：[Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs for Passage Re-Ranking (arXiv:2405.07920)](https://arxiv.org/html/2405.07920v4)
- Cohere Rerank、BAAI bge-reranker-v2-m3、jina-reranker-v2 等 cross-encoder 仍在 2025-2026 年被 Pinecone 等向量数据库作为 hosted reranker API 提供，是生产系统的标配组件（信息来自综合检索结果，非单一官方一手材料，供参考）。

**【与 LLM 的关系】** 共存/部分蒸馏替代——LLM 直接做全链路排序成本过高，工业界更多采用"BERT 精排 + LLM 蒸馏出的排序器 or 仅对难例调用 LLM 推理"的混合方案。

---

## 6. 人工相关性分档标注 + nDCG 评价

**【2026 结论】** 人工分档标注和 nDCG 依然是行业和学术界的黄金标准，但 LLM 标注（如 UMBRELA）已被主流评测（TREC）正式纳入流程用于规模化和成本削减，学术界同时警告 LLM 尚不能完全替代人工。

**【证据】**
- Google《Search Quality Rater Guidelines》最新版本于 2025 年 9 月 11 日更新（在 2026 年仍是现行版本），新增 AI Overview 评分示例并扩展 YMYL 定义（含选举/政务信息）：[Google updates search quality raters guidelines adding AI Overview examples & YMYL definitions (Search Engine Land)](https://searchengineland.com/google-updates-search-quality-raters-guidelines-adding-ai-overview-examples-ymyl-definitions-461908)；[官方 PDF，标注 "General Guidelines: September 11, 2025"](https://guidelines.raterhub.com/searchqualityevaluatorguidelines.pdf)；[Search Quality Raters Guidelines update 历次更新 - Google Search Central Blog](https://developers.google.com/search/blog/2023/11/search-quality-rater-guidelines-update)
- UMBRELA（复现 Bing 的 LLM 相关性评估方法）已被正式用于 TREC 2024 RAG Track 对 19 个团队共 77 个 run 的自动化评估：[UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor (arXiv:2406.06519)](https://arxiv.org/abs/2406.06519)；[TREC RAG 2024 Evaluation Overview](https://trec-rag.github.io/annoucements/evaluation/)
- TREC 2025 RAG Track 官方 overview 显示：仍以人工评估 sub-narrative 相关性（0-4 分制）为主，并同时报告 nDCG@30、nDCG@100、recall@100 等经典 IR 指标，同时将自动化（LLM）评估与人工评估做一致性对比分析：[Overview of the TREC 2025 Retrieval Augmented Generation (RAG) Track (arXiv:2603.09891)](https://arxiv.org/pdf/2603.09891)
- 学术界明确指出 LLM 标注的局限性：[LLM-based relevance assessment still can't replace human relevance assessment (arXiv:2412.17156, 2024)](https://arxiv.org/abs/2412.17156)

**【与 LLM 的关系】** 被增强而非替代——LLM 标注广泛用于规模化和预筛，但 Google 与 TREC 仍将人工评估保留为 gold standard。

---

## 7. 缓存召回 / 结果缓存

**【2026 结论】** 缓存依然是现代搜索/RAG 系统的重要基础设施，但形态从"精确 query 字符串缓存"演进为"语义缓存（semantic cache）"——在 embedding 空间匹配相似 query 并复用检索结果或生成内容。

**【证据】**
- 2025 年系统论文 CacheRAG 提出针对知识图谱问答场景的 RAG 语义缓存系统：[CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering (arXiv:2604.26176)](https://arxiv.org/abs/2604.26176)
- 2026 年论文讨论开放网络 RAG 场景下具备"新鲜度感知"和风险约束的语义缓存机制：[Risk-Constrained Freshness-Aware Semantic Caching for Open-Web Retrieval-Augmented LLMs (arXiv:2607.04281)](https://arxiv.org/pdf/2607.04281)
- 行业文章总结生产环境语义缓存的落地位置（通常放在文档检索之前）与效果（削减 30%-70% 的 LLM API 调用）：[Semantic Caching for RAG Systems - The Production Gap](https://boringbot.substack.com/p/semantic-caching-for-rag-systems)；[Semantic Caching Explained: A Complete Guide for AI, LLMs, and RAG Systems (2025-12)](https://www.blog.qualitypointtech.com/2025/12/semantic-caching-explained-complete.html)

**【与 LLM 的关系】** 被增强/新形态——从精确匹配缓存演进为向量相似度匹配缓存，并进一步扩展到 LLM 推理侧的 KV-cache/中间结果缓存（此项证据以学术论文和技术博客为主，缺少大厂官方一手案例，证据强度中等偏弱）。

---

## 8. LTR / GBDT / LambdaMART 融合排序

**【2026 结论】** LambdaMART/GBDT 仍是很多工业排序系统的技术基石（bedrock），2025 年仍在被顶级平台生产部署和迭代，但同年已出现严谨的线上 A/B 证据显示神经排序模型在部分电商场景开始反超 GBDT——"神经 vs 树模型"格局正在松动而非定论。

**【证据】**
- Airbnb 2025 年论文《Beyond Pairwise Learning-To-Rank At Airbnb》开篇即称"pairwise learning-to-rank (LTR) models" 是"当今搜索排序技术栈的基石（bedrock of search ranking tech stacks today）"，文中提出的 all-pairwise 改进方案已在 Airbnb 线上部署并通过线上实验验证：[Beyond Pairwise Learning-To-Rank At Airbnb (arXiv:2505.09795)](https://arxiv.org/abs/2505.09795)
- 2025 年 RecSys 论文基于 OTTO 电商生产数据集和 8 周线上 A/B 实验，系统对比 LambdaMART（GBDT）与多种深度神经网络架构，指出 LambdaMART 长期是成功 LTR 系统的支柱，但一个简单 DNN 架构在总点击量和营收上开始反超强 GBDT 基线（销量持平）：[Industry Insights from Comparing Deep Learning and GBDT Models for E-Commerce Learning-to-Rank (arXiv:2507.20753, RecSys 2025)](https://arxiv.org/abs/2507.20753)
- Airbnb 团队 KDD'24 论文《Learning to Rank for Maps at Airbnb》显示 Airbnb 仍在持续发表 LTR 相关生产研究，是该技术栈仍在活跃迭代的旁证（信息来自搜索结果摘要，未逐字核对原文，供参考）。

**【与 LLM 的关系】** 基本独立于 LLM 议题——LTR/GBDT 与神经排序的竞争主要是"树模型 vs 深度学习"层面的工程选择，LLM 目前更多作用于精排之上游的语义特征生成，尚未看到 LLM 直接取代底层 LTR 框架的工业案例。

---

## 9. 查询词推荐（搜索联想 / 相关搜索）

**【2026 结论】** 经典 query suggestion（自动补全/相关搜索）依然广泛存在且仍带来可观点击贡献，但 AI 搜索产品把"猜你想问"做成了对话式的"追问建议（follow-up prompt）"新形态，本质相同、呈现形式从下拉列表变为可点击对话按钮。

**【证据】**
- TechCrunch 报道 Google 于 2025 年 3 月推出 AI Mode，支持复杂多步提问，产品设计上强调持续对话式交互：[Google Search's new 'AI Mode' lets users ask complex, multi-part questions (TechCrunch, 2025-03-05)](https://techcrunch.com/2025/03/05/google-searchs-new-ai-mode-lets-users-ask-complex-multi-part-questions/)
- Nielsen Norman Group 分析文章指出 AI Mode 在每次回答末尾提供"可点击的 follow-up prompt 建议"，并将其与 Perplexity 的 related questions 机制类比为共享的产品范式：[Google AI Mode: Powerful Search, Poor Usability (NN/g)](https://www.nngroup.com/articles/google-ai-mode/)
- 综合行业统计文章显示传统 autocomplete 仍在贡献约 12%-15% 的点击率、约 23% 的用户使用建议词，是仍然活跃的经典功能（此数据来自 SEO 聚合类网站的二手统计，未找到 Google 官方一手数据源，证据强度较弱，仅供参考）：[Google Search Statistics 2025-2026 相关聚合文章](https://www.growthnavigate.com/google-search-statistics)

**【与 LLM 的关系】** 被增强/延伸形态——传统"预测式补全"的产品逻辑被搬进生成式对话界面，变成"AI 主动生成追问选项"，目标一致（降低用户表达门槛、延长会话）但载体从搜索框下拉变为对话内按钮。

---

## 九行总表

| # | 概念 | 2026 状态 | 一句话证据 |
|---|------|-----------|-----------|
| 1 | BM25 / 倒排索引 | 角色变化：降级为 hybrid 检索中的一路 lexical channel | Elasticsearch 8.9/OpenSearch/Vespa 均把 BM25+向量+RRF 作为默认 hybrid 架构 |
| 2 | 中文分词 | 仍在用，与 LLM tokenizer 并存、职责不同 | jieba 的 `cut_for_search` 模式仍是构建中文倒排索引的通用工具，未被 LLM BPE 分词取代 |
| 3 | Term Weight 词权重 | 角色变化：被学习型稀疏检索增强 | Elastic ELSER 8.11 GA、Pinecone 集成 SPLADE、淘宝 Deep BoW 模型均已上线生产 |
| 4 | 意图识别/类目识别 | 被替代/蒸馏：转向 LLM 或 LLM 蒸馏小模型 | 小红书 QP-OneModel、Amazon REIC、淘宝 TaoSR1 均是 2025-2026 一手工业案例 |
| 5 | 相关性 BERT 交叉编码器 | 仍是主力，与 LLM 蒸馏排序器/推理型排序共存 | Google RankEmbed BERT 经反垄断法庭披露仍是核心排序器；TaoSR1 用 LLM 补足 BERT 推理短板 |
| 6 | 人工分档标注 + nDCG | 仍是黄金标准，LLM 标注（UMBRELA）辅助规模化 | Google Rater Guidelines 2025-09-11 仍在更新；TREC 2024/2025 官方用 UMBRELA 做自动化评估但保留人工 |
| 7 | 缓存召回 | 角色延续，新形态为语义缓存 | CacheRAG/RAGCache 等 2025 系统论文把缓存做到 embedding 相似度匹配层 |
| 8 | LTR/GBDT/LambdaMART | 仍是基石，但神经排序开始局部反超 | Airbnb 2025 论文称 LambdaMART 为"排序技术栈基石"；RecSys 2025 生产 A/B 显示 DNN 开始反超 GBDT |
| 9 | 查询词推荐 | 角色延续+延伸：经典联想仍在，AI 追问是新形态 | Google AI Mode（2025-03 上线）与 Perplexity 均以对话式 follow-up 取代/补充下拉联想词 |

