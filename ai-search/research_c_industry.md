# 2025-2026 四大厂 AI 搜索动态调研
（供《从倒排索引到生成式检索——LLM 时代的工业级搜索》幻灯片使用）

调研基准日：2026-07-15。骨架五环节：**查询理解 → 召回 → 排序 → 结果呈现 → 评估闭环**。

---

## 一、Meta

### 1.1 产品时间线

| 时间 | 事件 |
|---|---|
| 2013 | Graph Search 上线，底层检索引擎 **Unicorn** 发布——支撑千亿级社交图谱边、每日数十亿次查询的内存/闪存倒排索引系统 |
| 2017 | Engineering at Meta 发布 *Under the Hood: Photo Search*，机器学习图像理解+Unicorn 检索结合 |
| 2020 (KDD) | 论文 *Embedding-based Retrieval in Facebook Search*（Huang et al.）：Facebook 搜索长期以布尔匹配（倒排索引）为主，该文首次系统性地把**统一向量嵌入（unified embedding）+ 双塔召回**接入 Facebook 搜索，与倒排索引做**混合召回** |
| 2021 (KDD) | 论文 *Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook*：基于 XLM 的多任务、多模态查询/文档理解模型，P99 延迟 <1.5ms（CPU），部署于 Facebook 主搜索 |
| 2023 | *Que2Engage*：将 Que2Search 扩展到 Facebook Marketplace，兼顾语义相关性与engagement 的多目标召回 |
| 2025-04-29 | 发布独立 **Meta AI App**，作为脱离社交平台容器的独立 AI 助手入口 |
| 2025-06-12 | Meta 以 **143 亿美元**投资/收购 Scale AI，创始人 Alexandr Wang 加入并领导新设立的 **Meta Superintelligence Labs (MSL)** |
| 2025-04 | Llama 4 系列（Scout / Maverick）发布，成为彼时 Meta AI 助手在 Facebook/Instagram/WhatsApp/Messenger 的底层模型；Llama 4 口碑不及预期 |
| 2025-12 | Meta AI 与 CNN、Fox News、Fox Sports、Le Monde、News Corp、《今日美国》等媒体达成内容合作，为 Meta AI 补充实时新闻信息源 |
| 2025-12（报道） | 有报道称 Meta 下一代模型（代号 Avocado）延期至 2026Q1，并从开源策略转向闭源 |
| 2026-04-08 | **Muse Spark** 发布——MSL 成立以来首个自研大模型，闭源 |
| **2026-06-15** | Facebook Search 内上线 **"AI Mode"**：由 Muse Spark 驱动的自然语言问答式搜索，答案来自 Facebook Groups、Reels、Instagram、Threads 上的公开内容，而非传统蓝链列表 |

### 1.2 技术架构要点（映射到骨架）

- **召回**：从 Unicorn 的纯倒排索引（布尔匹配）→ EBR 论文引入的双塔嵌入向量召回，与倒排索引**混合检索（hybrid retrieval）**；AI Mode 时代进一步扩展召回源为 Groups/Reels/Instagram/Threads 的公开内容，外加 Bing 索引 + Meta 自建爬虫补充网页内容。
- **查询理解**：Que2Search 用 XLM 多任务/多模态模型做查询-文档联合表征；AI Mode 用 Muse Spark（LLM）做自然语言意图理解，取代早期的关键词匹配。
- **排序**：piRank、Que2Engage 等论文体现"相关性+engagement"多目标排序，具有强社交图谱个性化特征（这是 Meta 区别于 Google/OpenAI 的核心差异点——排序信号里大量利用社交关系/图结构，而非纯网页权威度）。
- **结果呈现**：AI Mode 是生成式摘要+来源摘录，而非蓝链列表，与 Google AI Overviews/AI Mode 高度类似，但语料库以"人们公开说了什么"为主，弱化传统网页内容。
- **评估闭环**：官方未披露具体质量评估体系；Forbes 报道明确指出"我们不知道 AI Mode 如何加权信息源、如何应对错误信息"，这是当前最大的不确定性/风险点。

### 1.3 商业数字

- 摩根士丹利分析师 Brian Nowak 预测：若 AI Mode 留住 **10 亿用户**（约占 Facebook 30 亿月活的 1/3），并将每日查询的 **10% 货币化**，可为 Meta 带来**超过 100 亿美元/年**的营收（逻辑：留存用户规模 × 查询变现率 × 单次查询广告价值）。
- 消息公布当日（2026-06-15）Meta 股价上涨近 5%，报收接近 595 美元，但年内仍下跌约 8%。
- Meta AI 助手：官方口径为"到年底将成为全球使用最多的 AI 助手"，月活约 6 亿（具体统计口径与年份存在报道差异，未能精确核实到 2026 年年中的最新月活数字）。

### 1.4 与其他家的差异化定位

- 唯一一家**从社交图谱基础设施（Unicorn）原生演化到生成式搜索**的公司，历史脉络最长、最连续（Unicorn→EBR→Que2Search→Que2Engage→AI Mode），体现"同一骨架的持续升级"而非推倒重来。
- 语料库差异化：主打"人的公开发言"（UGC/社交内容）而非网页权威内容，这是与 Google/OpenAI/Perplexity 最大的不同——**检索对象是社交图谱节点而非网页文档**。
- 商业化路径与 Google 类似（广告变现），但直接竞争对象被分析师明确定义为 Google 搜索。

---

## 二、Google

### 2.1 产品时间线

| 时间 | 事件 |
|---|---|
| 2023-05-10 | Google I/O 发布 **SGE (Search Generative Experience)**，Search Labs 实验性功能 |
| 2023-05-25 | SGE 向 Labs 等候名单用户开放 |
| 2023-08 → 2023-11 | SGE 扩展至印度、日本，随后 120+ 国家/地区（仍限 Labs） |
| 2024-05-14 | Google I/O：SGE **更名为 AI Overviews**，向美国全量用户开放（脱离 Labs） |
| 2025-03-26 | AI Overviews 扩展至德国等更多欧洲市场 |
| **2025-05-20** | Google I/O：**AI Mode** 面向美国全量用户开放（此前需 Labs 注册），采用**定制版 Gemini 2.5 模型**支持 **query fan-out**（查询扇出）技术——将复杂问题拆解为多条子查询并行检索；Deep Search 功能可对复杂问题发出数百条子查询 |
| 2025-09 | DOJ 反垄断案救济令生效：法官 Amit Mehta 要求 Google 向"合格竞争对手"共享 **RankEmbed** 相关的底层用户数据（不含模型/算法/后训练 LLM 本身） |
| 2026 Q1 财报 | Google 宣称搜索广告收入 AI 驱动增长 19%，"无可测量的付费点击蚕食" |
| **2026 I/O** | 官方披露 AI Mode 月活**超过 10 亿**，查询量**每季度翻倍以上** |

### 2.2 技术架构要点（映射到骨架）

- **查询理解 + 召回**：AI Mode 用定制 Gemini 2.5 模型做 **query fan-out**——一条查询被拆解为约 9 条子查询（不同垂类差异大：软件类 11.7 条、旅行 10.8 条、职业 9.8 条、本地检索仅 3.79 条），每条子查询并行检索后再综合。Deep Search 场景子查询可达数百条。
- **召回/排序**：反垄断庭审文件披露 **RankEmbed**（RankEmbed BERT）—— 一个将 query 与 document 编入向量空间的双塔深度学习排序模型，用于语义相关性判断；训练数据包含 70 天搜索日志、用户交互行为、人工质量评估（human quality raters）。这是 Google 传统排序管线中 LLM/embedding 化的直接证据。另有报道称 AI Overviews 使用名为 "FastSearch" 的检索系统而非直接复用传统链接排序结果。
- **结果呈现**：分为两层产品——AI Overviews（搜索结果页顶部摘要卡片，与传统十条蓝链共存）与 AI Mode（独立对话式全屏回答标签页，更接近 Perplexity/ChatGPT 的体验）。
- **评估闭环**：延续 Google 传统的人工质量评估员（human quality raters）体系，结合大规模点击/停留等行为数据做在线 A/B 测试，这一体系通过反垄断诉讼首次被要求部分对外披露训练数据（但模型与算法仍是商业机密，未被强制公开）。

### 2.3 商业数字（规模与左右互搏）

- **覆盖率**：2026 年 AI Overviews 覆盖约 **48%~50%** 的搜索查询（不同方法论测算范围 21%~65%）；垂类差异显著：健康类 71~88%、教育类 83%，娱乐类 37%，交易/购物类仅 19%。
- **CTR 冲击**：Seer Interactive 测得 AIO 场景下自然点击率一度暴跌 **61%**（2025-09 低点），到 2026-02 反弹至 2.4%，但仍比无 AIO 查询的基线 3.8% 低约 37%。Pew Research 行为研究：有 AI Overview 时用户点击传统结果概率仅 8%，无 AI Overview 时为 15%（相对下降 47%）。
- **补偿机制**：被 AI Overviews 引用的品牌可获得 35% 更多自然点击、91% 更多付费点击。
- **左右互搏**：AI Overviews 中广告占比从 2025 年 1 月约 3% 的搜索结果页快速上升到 2025 年 11 月约 40%；官方称"无可测量的付费点击蚕食"，Q4 2025 搜索广告收入 630.7 亿美元（同比+17%）。但同期第三方数据显示 Google 搜索引荐给网站的流量同比下降约 33%（截至 2025-11），AI Overviews 引用来源里只有 17% 来自传统自然结果前十名（2024 年年中该比例为 76%）；广告主层面 84% 反馈 AI Max 广告效果中性或负面。这构成了 Google 传统搜索业务与生成式搜索之间清晰的"左右互搏"张力。

### 2.4 差异化定位

Google 是唯一在**同一搜索结果页内并存三层产品**（十条蓝链 + AI Overviews 摘要 + 独立 AI Mode 标签页）的厂商，试图在不破坏现有万亿美元广告业务的前提下嫁接生成式体验，因此表现出最强的"渐进式改造"特征，而非另起炉灶。

---

## 三、OpenAI

### 3.1 产品时间线

| 时间 | 事件 |
|---|---|
| 2024-07 | **SearchGPT** 原型发布（邀请制） |
| 2024-10-31 | 正式推出 **ChatGPT Search**，向所有用户开放 |
| 2024-10-31 起 | 依赖 **Bing 索引**提供网页检索能力，ChatGPT-User 爬虫负责实时抓取页面；SEMrush 对 8000 万条 ChatGPT 查询的研究显示 46% 触发了搜索；另一项研究显示 SearchGPT 引用来源与 Bing 前十结果重合度高达 **87%** |
| **2025-02-02** | 发布 **Deep Research**：基于 o3 优化版模型，可自主进行多步骤网络研究，数十分钟内完成人类需数小时的分析任务；初期仅 Pro 用户（200 美元/月）可用，限 100 次/月 |
| 2025-04 | Deep Research 用量限制放宽：Plus/Team/Enterprise/Edu 25 次/月，Pro 250 次/月，Free 5 次/月（免费版由轻量化 o4-mini 版本驱动） |
| **2025-07-17** | 发布 **ChatGPT Agent**：融合 Operator（可执行操作的远程浏览器）+ Deep Research（网页综合能力）+ ChatGPT 对话能力，同时上线购物研究（shopping research）功能 |
| **2025-10-21** | 发布 **ChatGPT Atlas** 浏览器（先 macOS，iOS/Windows/Android 后续跟进）；支持 agent mode 直接在网页内执行任务（如 Instacart 购物代理，测试阶段） |
| 2025 年内 | 报道称 OpenAI 网页爬虫抓取量自 2025-08 起**增长约 3 倍**，正在构建自有网页索引以降低对 Bing 的依赖，但截至 2026 年年中仍未完全摆脱 Bing 依赖 |
| **2026-01-16** | 官宣在 ChatGPT 中正式上线**广告**，先覆盖美国 Free / ChatGPT Go 层级用户 |
| 2026-02-09 | 广告功能正式滚动上线；形式为回答下方的单张 "sponsored" 卡片，按对话主题（而非关键词）匹配 |
| 2026-03 | 广告扩展至加拿大、澳大利亚、新西兰 |
| 2026-05-05 | 自助广告投放平台 **Ads Manager** 面向所有美国广告主开放，无最低消费门槛 |
| 2026-05 | 宣布将扩展至英国、墨西哥、巴西、日本、韩国 |

### 3.2 技术架构要点（映射到骨架）

- **召回**：核心仍**依赖 Bing 索引**（截至 2026 年年中未完全独立），同时通过 OAI-SearchBot 等自有爬虫快速扩张自建索引，意图逐步过渡到"检索主权"；Deep Research/Agent 场景则是**实时多步骤浏览**而非静态索引查询，属于"检索即行动"的新范式。
- **查询理解/编排**：由推理模型（o3/o4-mini/GPT-5.x 系列）决定何时调用搜索工具、如何拆解任务，ChatGPT Agent 把"深度研究+浏览器操作+对话"编排成统一 agent 循环。
- **排序**：高度复现 Bing 的排序结果（引用与 Bing 前十重合率 87%），OpenAI 自身在此基础上做二次相关性筛选与引用选择，而非自建完整排序模型（与 Google/Perplexity/Meta 均有自研排序模型形成鲜明对比）。
- **结果呈现**：对话式回答+来源引用（ChatGPT Search）、长篇结构化研究报告（Deep Research）、浏览器内嵌代理执行（Atlas）三种形态并存；2026 年起在回答下方插入独立的赞助卡片广告。
- **评估闭环**：主要依赖用户反馈（点赞/点踩）与 RLHF，尚未见对外披露类似 Google 人工质量评估员规模化体系的信息（信息不完整，未能核实）。

### 3.3 商业数字

- 广告定价：CPM 25~60 美元，CPC 3~5 美元，B2B 点击可达 8~15 美元；Plus/Pro/Enterprise 用户保持无广告。
- 市场份额：截至 2026 年 1 月，ChatGPT 占全球 AI 聊天助手流量约 **60.7%**（Similarweb），远超 Gemini（15.0%）与 Copilot（13.2%）；ChatGPT Search 每周处理 2.5~5 亿次查询。

### 3.4 差异化定位

OpenAI 是唯一**从对话模型切入、反向补齐检索能力**的公司（其余三家都是先有检索/索引基础设施再叠加生成能力）。其"借道 Bing、自建索引在追赶"的路径，与 Google/Meta/Perplexity 均自建多年检索基础设施形成结构性差异；浏览器 Atlas 与购物代理的组合，显示其正把"搜索"重新定义为"可执行任务的浏览器 agent"，而不只是问答。

---

## 四、Perplexity

### 4.1 产品时间线

| 时间 | 事件 |
|---|---|
| 2025-01 | 发布自研模型 **Sonar**（基于 Llama 3.3 70B 微调，面向"网页落地"问答，128K 上下文，采用 Cerebras 晶圆级推理，速度约 1200 token/s） |
| 2025-07 | **Comet 浏览器**发布，先向 200 美元/月的 Max 订阅用户开放 |
| 2025-08 | 推出 **Comet Plus**：5 美元/月订阅层，与 CNN、Condé Nast、《华盛顿邮报》等出版商按约 **80%** 比例分享收入（覆盖访问、引用、agent 操作等场景） |
| 2025 下半年（报道） | Perplexity 宣布**不做广告**变现，转向订阅与 agentic 产品 |
| 2025-08-12（报道日期） | Perplexity 向 Google 提出**340.5 亿美元收购 Chrome 浏览器**的非邀约要约（未被接受） |
| 2025-10-02 | Comet 浏览器**全球免费开放** |
| 2025-10 | 收购 AI 设计工具团队 **Visual Electric**（原 Sequoia 投资项目），组建新的 agents 体验团队 |
| 2025-11-19 | 与美国 **GSA（联邦总务署）**签署"直接对政府"协议，联邦机构以每机构 0.25 美元的价格获得 Perplexity 使用权 |
| 2025（报道，具体月份未明确） | 与 **Snapchat** 达成合作，将 Perplexity 对话式搜索嵌入 Snapchat（据报道协议金额约 4 亿美元） |
| 2026-01-29 | 与 **Microsoft** 签署约 **7.5 亿美元** Azure 云服务三年合约（同期与 Amazon 存在云服务纠纷） |
| 2026-05-06 | Snap 官宣与 Perplexity 的合作"友好终止" |
| **2026-06-05/08** | 完成新一轮融资，融资额约 **2 亿美元**，估值约 **200 亿美元**（累计融资约 17.2 亿美元）；官方口径是为 Comet 免费化和"agent 经济入口"布局 |
| 2026-03 | ARR（年化经常性收入）突破约 **4.5 亿美元**（一说 2 月至 3 月单月环比增长约 50%），目标 2026 年底达到约 6.56 亿美元 ARR（不同信源对同期 ARR 数字存在差异，如另一处提及"截至 2026 年年中 ARR 约 2 亿美元"，两者口径可能不同，未能完全核实并统一） |

### 4.2 技术架构要点（映射到骨架）

- **召回**：**自建索引**（覆盖超 2000 亿个 URL），通过自有爬虫 **PerplexityBot** 做实时抓取 + API 数据接入，用机器学习模型预测哪些 URL 需要被索引及索引优先级/更新频率（"按需索引"而非静态全量索引）；基础设施规模为数万 CPU、数百 TB 内存，可支持每秒数万次索引操作。
- **查询理解 + 排序**：候选文档召回后做**段落级排序与抽取（passage ranking & extraction）**，再交给生成模型合成答案；采用**模型路由（model routing）**——简单请求路由到小型内部模型，复杂多步推理路由到更强模型（遵循"用够用的最小模型"原则），Pro 层可路由至 Claude Sonnet 4.6 / GPT-4o / Gemini，Max 层可路由至 Claude Opus 4.7 / GPT-5.4，并提供 "Perplexity Computer"、"Model Council" 等更高级编排能力。
- **结果呈现**：RAG 生成式回答+行内引用（inline citations）为标配；Comet 浏览器进一步把"结果呈现"延伸为可在网页内直接执行的 agent 操作，而不仅是一段文字回答。
- **评估闭环**：与其余三家依赖点击率/停留时长/人工评估员的模式不同，Perplexity 用 **Comet Plus 的出版商收入分成机制（80/20 分成）** 构建了一种"以内容伙伴经济利益驱动内容质量与可持续供给"的另类评估/激励闭环。

### 4.3 商业数字

- 2026 年 6 月完成 2 亿美元融资，估值约 200 亿美元；ARR 数据在不同报道间存在差异（约 2 亿~4.5 亿美元区间，需谨慎使用）。
- 市场份额：截至 2026 年 5 月（Statcounter），Perplexity 占 AI 聊天助手引荐流量约 **7.67%**，位列第三，高于 Google 自家 Gemini 的 7.03%；但整体量级远小于 ChatGPT Search（约每周 5000 万次查询 vs ChatGPT 2.5~5 亿次）。
- 明确放弃广告变现，是四家中唯一公开宣称"不做广告"的公司。

### 4.4 差异化定位

Perplexity 是四家中**唯一没有存量搜索广告业务包袱、完全从零自建"检索+生成"全栈**的公司，其"自建索引 + 按需爬取 + 模型路由 + 出版商分成"的组合，在骨架五环节里最接近教科书式的现代 RAG 架构，也是最适合用来给学生讲解"生成式检索全栈"的范例；但其规模（索引覆盖、查询量）与 Google/Bing 相比仍是数量级差距。

---

## 五、横向对比表（骨架五环节）

| 环节 | Meta | Google | OpenAI | Perplexity |
|---|---|---|---|---|
| **查询理解** | Que2Search 系列（XLM 多任务/多模态编码）→ AI Mode 由 Muse Spark（LLM）做自然语言意图理解 | 定制 Gemini 2.5 做 **query fan-out**，一条查询拆成约 9 条并行子查询（Deep Search 可达数百条） | 推理模型（o3/o4-mini/GPT-5.x）驱动工具调用与任务拆解，ChatGPT Agent 统一编排 | 查询分解 + **模型路由**（按复杂度选择内部小模型或前沿大模型） |
| **召回** | Unicorn 倒排索引 + EBR 双塔向量召回（混合检索），语料涵盖 Groups/Reels/Instagram/Threads + Bing/自建爬虫网页 | 传统倒排索引 + **RankEmbed** 双塔嵌入召回（反垄断庭审披露）+ AI Overviews 疑似使用独立的 "FastSearch" | 主要**依赖 Bing 索引**，自建爬虫/索引增长中但未完全独立（截至 2026 年中） | **完全自建索引**（超 2000 亿 URL）+ PerplexityBot 实时按需爬取，ML 预测索引优先级 |
| **排序** | piRank/Que2Engage：相关性 + engagement 多目标、强社交图谱个性化排序 | RankEmbed 等神经排序信号 + 传统信号融合，多阶段漏斗；人工质量评估员训练数据 | 高度复现 Bing 排序（引用与 Bing 前十重合率 87%），自身做二次筛选，未见自研完整排序模型 | 候选文档**段落级排序与抽取**后再送入生成模型合成 |
| **结果呈现** | AI Mode：生成式摘要，来源为公开社交内容而非蓝链 | 三层共存：十条蓝链 + AI Overviews 摘要卡 + 独立 AI Mode 对话标签页 | 对话回答+引用 / Deep Research 长报告 / Atlas 浏览器内 agent 执行 / 2026 起插入 sponsored 卡片 | RAG 生成回答+行内引用；Comet 浏览器把呈现延伸为可执行的网页内 agent 操作 |
| **评估闭环** | 未公开披露，社交互动信号推测参与排序训练；错误信息治理机制不明（Forbes 明确指出"未知"） | 人工质量评估员 + 海量点击/停留在线 A/B（RankEmbed 训练用 70 天日志+人工评分，部分数据被反垄断令要求共享） | 主要靠用户反馈/RLHF，公开信息有限 | **Comet Plus 出版商收入分成（80/20）**替代传统点击评估，作为内容供给可持续性的经济闭环 |
| **商业化/差异化定位** | 广告变现，社交内容为差异化语料，分析师预期 $10B+/年新增收入 | 广告为主，AI Overviews/AI Mode 与传统搜索"左右互搏"（覆盖48-50%查询，CTR 一度降61%后部分反弹） | 2026 年初上线广告（sponsored 卡片）+ 订阅（Plus/Pro/Enterprise）+ Atlas 浏览器承载购物代理 | 订阅（Pro $20/月、Max $200/月）+ Comet Plus 出版商分成，**明确不做广告** |

---

## 六、跨厂横向数据：AI 搜索对传统 SEO/流量的冲击

- **零点击搜索**：2026 年前四个月，Google 搜索**零点击率达 68.01%**（SparkToro/Similarweb），较 2024 年的 60.45% 明显上升；触发 AI Overviews 的查询零点击率平均高达 **83%**，未触发的查询约 60%；移动端零点击率 77%，桌面端 46.5%。
- **搜索市场份额**：传统搜索引擎份额上，Google 仍占据 Cloudflare Radar 引荐流量的 87.63%（2026 年 5 月）、Statcounter 全球搜索引擎份额 90.39%；AI 聊天助手内部份额上 ChatGPT 60.7%、Gemini 15.0%、Copilot 13.2%、Perplexity 位列第三（7.67%，引荐流量口径，高于 Google 自家 Gemini 的 7.03%）。
- **总量级对比**：所有 AI 聊天机器人（ChatGPT、Gemini、Claude、Perplexity）合计引荐给网站的流量仅占 Google 引荐流量的极小比例（一处报道称合计约为 DuckDuckGo 单独引荐流量的 1/5），说明 2026 年年中 AI 搜索仍主要蚕食"信息类/研究类查询"，导航类查询仍高度集中在 Google。
- **出版商流量影响**：Google 搜索引荐给网站的流量同比下降约 33%（截至 2025 年 11 月），AI Overviews 引用来源里仅 17% 来自自然结果前十名（2024 年年中为 76%），显示 AI 摘要正在与自然结果排名"脱钩"。

---

## 七、引用链接列表

**Meta**
- Forbes（原始起点报道，已用 web_fetch 完整读取）：[Facebook Launches Search Engine AI Tool That Could Make Meta $10 Billion A Year, Analyst Says](https://www.forbes.com/sites/maryroeloffs/2026/06/15/facebook-launches-search-engine-ai-tool-that-could-make-meta-10-billion-a-year-analyst-says/)
- Meta 官方新闻稿（已用 web_fetch 完整读取）：[New AI Tools to Help You Make Things Happen on Facebook](https://about.fb.com/news/2026/06/new-ai-tools-to-help-you-make-things-happen-on-facebook/)
- [Benzinga: Meta's New AI Search Could Generate $10 Billion A Year](https://www.benzinga.com/markets/tech/26/06/53215944/metas-new-ai-search-could-generate-10-billion-a-year-and-challenge-googles-dominance)
- [Meta AI 助手 MAU 与 Llama 演进](https://ai.meta.com/blog/future-of-ai-built-with-llama/)
- [Llama 4 发布](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Meta 转向闭源 AI / Avocado 延期报道](https://www.digitimes.com/news/a20251211PD206/meta-llama-development-2026.html)
- [Embedding-based Retrieval in Facebook Search (KDD 2020 论文)](https://arxiv.org/pdf/2006.11632)
- [Que2Search 论文 (KDD 2021)](https://www.semanticscholar.org/paper/Que2Search:-Fast-and-Accurate-Query-and-Document-at-Liu-Rangadurai/8a4343e08656398f27ec3e8ef0b4fa3e1bfc724b)
- [Que2Engage 论文](https://arxiv.org/abs/2302.11052)
- [Unicorn 架构 2013 Engineering at Meta 博客](https://engineering.fb.com/2013/03/06/core-infra/under-the-hood-building-out-the-infrastructure-for-graph-search/)
- [Meta AI App 发布](https://about.fb.com/news/2025/04/introducing-meta-ai-app-new-way-access-ai-assistant/)
- [Meta AI 引入更多实时新闻内容合作 (2025-12)](https://about.fb.com/news/2025/12/bringing-more-real-time-news-and-content-to-meta-ai/)
- [Muse Spark 发布](https://about.fb.com/news/2026/04/introducing-muse-spark-meta-superintelligence-labs/)

**Google**
- [Google AI Overviews 统计 (2026)](https://thestacc.com/blog/google-ai-overview-statistics/)
- [Omnibound: Google AI Overviews Statistics 2026](https://www.omnibound.ai/blog/google-ai-overviews-statistics)
- [Google AI Mode I/O 2025 官方博客](https://blog.google/products-and-platforms/products/search/google-search-ai-mode-update/)
- [Query Fan-Out 原创研究](https://www.ekamoira.com/blog/query-fan-out-original-research-on-how-ai-search-multiplies-every-query-and-why-most-brands-are-invisible)
- [ppc.land: Google's RankEmbed system forced to share secrets in antitrust remedy](https://ppc.land/googles-rankembed-system-forced-to-share-secrets-in-antitrust-remedy/)
- [Search Engine Journal: Google Antitrust Case: AI Overviews Use FastSearch, Not Links](https://www.searchenginejournal.com/google-antitrust-case-ai-overviews-use-fastsearch-not-links/555220/)
- [DOJ 反垄断案文件（2025-05-29）](https://justice.gov/atr/media/1402131/dl?inline=)
- [Alphabet Q1 2026 财报报道](https://tech-insider.org/alphabet-q1-2026-earnings-109-billion-google-cloud-surge/)
- [Google Search 广告与 AI Overviews 蚕食争议报道](https://media-entertainment.news-articles.net/content/2026/06/26/google-s-ai-generated-summaries-the-new-publisher-risk.html)
- [Google I/O 2024：SGE 更名为 AI Overviews](https://blog.google/products/search/generative-ai-google-search-may-2024/)

**OpenAI**
- [OpenAI's SearchGPT relies on Bing's index](https://www.thekeyword.co/news/openai-s-chatgpt-search-relies-on-bing-s-index)
- [Seer Interactive: 87% of SearchGPT Citations Match Bing's Top Results](https://www.seerinteractive.com/insights/87-percent-of-searchgpt-citations-match-bings-top-results)
- [OpenAI: Introducing deep research](https://openai.com/index/introducing-deep-research/)
- [Campus Technology: New OpenAI 'Deep Research' Agent](https://campustechnology.com/articles/2025/02/12/new-openai-deep-research-agent-turns-chatgpt-into-a-research-analyst.aspx)
- [OpenAI: Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/)
- [Axios: OpenAI's ChatGPT agent combines deep research and online shopping](https://www.axios.com/2025/07/17/chatgpt-agent-open-ai-web-deep-research)
- [OpenAI: Introducing ChatGPT Atlas](https://openai.com/index/introducing-chatgpt-atlas/)
- [MIT Technology Review: OpenAI's new Atlas browser](https://www.technologyreview.com/2025/10/27/1126673/openai-new-atlas-browser/)
- [OpenAI Help Center: Ads in ChatGPT](https://help.openai.com/en/articles/20001047-ads-in-chatgpt)
- [ChatGPT Ads Launch 2026 分析](https://almcorp.com/blog/chatgpt-ads-aggressive-placement-pricing-analysis/)
- [SEO Sherpa: OpenAI Is Building Its Own Web Index](https://seosherpa.com/openai-is-building-its-own-web-index/)
- [Botify: OpenAI Has Tripled Their Crawl of the Web](https://www.botify.com/blog/openai-tripled-web-crawl)

**Perplexity**
- [ByteByteGo: How Perplexity Built an AI Google](https://blog.bytebytego.com/p/how-perplexity-built-an-ai-google)
- [Perplexity 官方: Architecting and Evaluating an AI-First Search API](https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api)
- [Perplexity 官方: Meet New Sonar](https://www.perplexity.ai/hub/blog/meet-new-sonar)
- [Perplexity 官方: Introducing the Perplexity Search API](https://www.perplexity.ai/hub/blog/introducing-the-perplexity-search-api)
- [TechTimes: Perplexity Raises $200 Million for Comet](https://www.techtimes.com/articles/318028/20260608/perplexity-raises-200-million-comet-ai-browser-agent-economy-front-door.htm)
- [Yahoo Finance: Perplexity finalizes $20 billion valuation round](https://finance.yahoo.com/news/perplexity-finalizes-20-billion-valuation-232758524.html)
- [Digiday: How Perplexity's new revenue model works](https://digiday.com/media/how-perplexity-new-revenue-model-works-according-to-its-head-of-publisher-partnerships/)
- [Perplexity AI Statistics 2026 (ARR/用户)](https://www.getpanto.ai/blog/perplexity-ai-statistics)
- [CNBC: Perplexity offers to buy Google's Chrome browser for $34.5 billion](https://www.cnbc.com/2025/08/12/perplexity-google-chrome-ai.html)
- [TechCrunch: Perplexity acquires the team behind Visual Electric](https://techcrunch.com/2025/10/02/perplexity-acquires-the-team-behind-sequioa-backed-ai-design-startup-visual-electric/)
- [GSA: GSA and Perplexity Sign First Direct to Government Deal](https://www.gsa.gov/about-gsa/newsroom/news-releases/gsa-perplexity-sign-first-direct-to-gov-deal-11192025)
- [Bloomberg: Perplexity Inks Microsoft AI Cloud Deal Amid Dispute With Amazon](https://www.bloomberg.com/news/articles/2026-01-29/perplexity-inks-microsoft-ai-cloud-deal-amid-dispute-with-amazon)
- [TechCrunch: Snap says its $400M deal with Perplexity 'amicably ended'](https://techcrunch.com/2026/05/06/snap-says-its-400m-deal-with-perplexity-amicably-ended/)

**跨厂横向数据（零点击/市场份额）**
- [SparkToro: In 2026, Less than One Third of Google Searches Still Send a Click](https://sparktoro.com/blog/in-2026-less-than-one-third-of-google-searches-still-send-a-click/)
- [Search Engine Land: Google zero-click searches reach 68% in early 2026](https://searchengineland.com/google-zero-click-searches-2026-study-479717)
- [TechnologyChecker: Search Engine Market Share 2026](https://technologychecker.io/blog/search-engine-market-share)
- [StackMatix: AI Search Market Share 2026](https://www.stackmatix.com/blog/ai-search-market-share-2026)

---

## 八、未能完全核实/存在信源冲突的信息（供撰写幻灯片时谨慎处理或标注"据报道"）

1. **Perplexity ARR 数字**：不同信源给出"2026年3月ARR约4.5亿美元/环比增50%"与"截至2026年年中ARR约2亿美元"两套不一致的数字，未能进一步核实统一，建议在幻灯片中只使用区间或注明信源。
2. **AI Overviews 覆盖率**：不同工具/研究方法给出 21%~65% 的宽区间，Google 官方未给出精确数字（仅称"约50%"），48%这个数字来自第三方 SEO 行业统计，非 Google 官方披露。
3. **Meta AI 助手月活体量**：约 6 亿月活的表述出自较早期报道（"到今年年底成为全球第一"），未找到 2026 年年中的官方最新月活确认数字。
4. **Snapchat-Perplexity 合作金额（4亿美元）**及**Apple 收购 Perplexity 传闻**：均为媒体报道口径，未见官方确认金额或交易细节，Apple 一事被明确描述为"早期内部讨论、可能不会有正式要约"。
5. **零点击搜索 68.01%**：主要来自 SparkToro/Similarweb 一家机构方法论，其他信源（Statista 等）口径不同，未做交叉验证。
6. **OpenAI 是否/何时完全摆脱 Bing 依赖**：截至 2026 年年中的公开信息显示"自建索引在增长"但"仍未完全独立"，没有找到官方明确的时间表或摆脱依赖的确切声明。
