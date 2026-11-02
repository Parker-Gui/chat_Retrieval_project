# MovieGraph：基于 RAG 与 Knowledge Graph 的电影问答实验

本项目是信息检索课程 Project，使用 MetaQA 电影数据集，对比以下三种问答方法：

1. **Pure LLM**：不检索外部知识，直接由 DeepSeek 回答。
2. **Vector RAG**：从 Neo4j Vector Index 检索相关三元组，再由 DeepSeek 根据证据回答。
3. **KG Retrieval**：由 DeepSeek 将问题转换为 Cypher，在 Neo4j 中执行图关系查询并返回答案。

项目基于开源 Android Chatbox 进行二次开发。中期阶段重点是后端核心检索功能和可复现
实验，Android 前端与完整聊天流程将在 final 阶段继续完成。

## 一、当前完成情况

中期项目已经完成：

- MetaQA 数据预处理与固定 300 题 benchmark；
- Neo4j Knowledge Graph 导入与 1-hop、2-hop、3-hop 查询验证；
- 基于 `all-MiniLM-L6-v2` 的 Vector Index；
- Pure LLM、Vector RAG、KG Retrieval 三套回答 Pipeline；
- DeepSeek JSON 结构化输出、答案解析与错误记录；
- 三组完整实验及统一的 Exact Match、Precision、Recall、F1 和 Latency 评测；
- PowerShell 一键实验脚本与断点续跑功能。

中期不包含 Hybrid RAG，也不要求 Android 与后端完成端到端集成。

## 二、实验内容与结果

实验使用同一份固定 MetaQA benchmark，共 300 道问题，其中 1-hop、2-hop、3-hop
各 100 道。三种方法使用同一个 DeepSeek 配置和同一套答案集合评分代码，主要变量是
获取知识的方式。

### 2.1 三种方法的基本原理

- **Pure LLM**：不提供检索证据，用于观察模型依靠内部知识能够达到的基线性能。
- **Vector RAG**：将知识图谱三元组编码成向量，按语义相似度返回 Top-5 三元组，再让
  DeepSeek 仅根据这些证据回答。它适合语义相似的事实检索，但不能保证多条三元组组成
  完整的多跳路径。
- **KG Retrieval**：使用 Text2Cypher 将自然语言问题转换成只读 Cypher，通过 Neo4j
  显式遍历图关系。它能够直接执行 1-hop、2-hop 和 3-hop 关系查询。

### 2.2 总体结果

| 方法         | 成功题数 |      Exact Match |        Precision |           Recall |               F1 | 平均 Latency |
| ------------ | -------: | ---------------: | ---------------: | ---------------: | ---------------: | -----------: |
| Pure LLM     |  300/300 |            8.67% |           0.3057 |           0.4586 |           0.3196 |      1270 ms |
| Vector RAG   |  300/300 |           15.00% |           0.2249 |           0.2024 |           0.2008 |      1110 ms |
| KG Retrieval |  300/300 | **96.00%** | **0.9745** | **0.9832** | **0.9773** |      1596 ms |

### 2.3 分 hop 的 Exact Match

| 方法         |            1-hop |            2-hop |            3-hop |
| ------------ | ---------------: | ---------------: | ---------------: |
| Pure LLM     |           25.00% |            0.00% |            1.00% |
| Vector RAG   |           43.00% |            0.00% |            2.00% |
| KG Retrieval | **95.00%** | **96.00%** | **97.00%** |

结果说明：Vector RAG 对简单 1-hop 查询有明显帮助，但 Top-5 相似三元组很难稳定覆盖
完整的多跳关系；KG Retrieval 能够显式执行图路径，因此在三种 hop 数量下都保持稳定。
KG Retrieval 的平均延迟略高，但准确率提升非常明显。

Vector Retrieval 的单独检索实验还得到：Hits@1 为 20.67%，Hits@3 为 30.00%，
Hits@5 为 32.67%，Recall@5 为 23.06%，MRR 为 0.254。这说明 Vector RAG 的主要
限制来自证据检索阶段。

正式汇总结果保存在：

```text
experiments/midterm/metrics/answer_pure_llm_metrics.json
experiments/midterm/metrics/answer_vector_rag_metrics.json
experiments/midterm/metrics/answer_kg_retrieval_metrics.json
experiments/midterm/metrics/vector_rag_metrics.json
experiments/midterm/metrics/kg_retrieval_metrics.json
```

逐题 predictions、Prompt 调试历史和小样本指标只保存在本地，已通过 `.gitignore`
排除，不上传 GitHub。

## 三、环境要求

目前的部署与实验脚本以 Windows PowerShell 为准。

必须安装：

- Git；
- Python 3.11；
- JDK 17；
- 可访问 DeepSeek API 的网络；
- 一个可用的 `DEEPSEEK_API_KEY`。

可选安装：

- Android Studio：midterm 实验不需要，final 做 Android 集成时再使用；
- Docker：当前默认使用本地 Neo4j ZIP，不要求 Docker。

请不要使用 Python 3.14。部分依赖可能找不到对应的 wheel，导致 `pydantic-core` 尝试
通过 Rust 编译并安装失败。本项目统一使用 Python 3.11。

## 四、从 GitHub 获取项目

打开 PowerShell，进入准备存放课程项目的目录：

```powershell
git clone https://github.com/Parker-Gui/chat_Retrieval_project.git
cd chat_Retrieval_project
```

如果已经 clone，只需进入项目目录并获取最新代码：

```powershell
git pull
```

## 五、创建 Python 环境并安装依赖

以下命令均在项目根目录执行。

### 5.1 创建 Python 3.11 Virtual Environment

```powershell
py -3.11 -m venv backend\venv311
```

激活环境：

```powershell
.\backend\venv311\Scripts\Activate.ps1
```

如果 PowerShell 禁止运行激活脚本，可仅对当前窗口临时允许：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\backend\venv311\Scripts\Activate.ps1
```

也可以不激活环境，后续直接使用：

```powershell
.\backend\venv311\Scripts\python.exe
```

### 5.2 安装 Python 依赖

```powershell
.\backend\venv311\Scripts\python.exe -m pip install --upgrade pip
.\backend\venv311\Scripts\python.exe -m pip install -r backend\requirements.txt
```

## 六、配置 DeepSeek 与 Neo4j

### 6.1 配置 DeepSeek

复制配置模板：

```powershell
Copy-Item backend\env.example backend\.env
```

打开 `backend/.env`，至少填写：

```env
DEEPSEEK_API_KEY=你的真实API_KEY
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

不要把真实 API Key 写入 `env.example`，也不要提交 `.env`。

### 6.2 配置 Neo4j

复制配置模板：

```powershell
Copy-Item neo4j\.env.example neo4j\.env
```

默认开发配置为：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=moviegraph-dev
```

### 6.3 确认 JDK 17

```powershell
java -version
```

如果系统默认 Java 不是 17，在当前 PowerShell 窗口设置：

```powershell
$env:JAVA_HOME = "你的JDK17目录"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
java -version
```

Android Studio 自带的 JBR 17 也可以使用。

## 七、准备 MetaQA 数据

固定的 300 题 benchmark 已包含在仓库中，但为了导入 Neo4j 和建立 Vector Index，
仍然需要 MetaQA 原始 Knowledge Base。

从 MetaQA 官方仓库下载数据：

```text
https://github.com/yuyuz/MetaQA
```

将以下目录复制到 `data/raw/metaqa/`：

```text
data/raw/metaqa/
├── kb/
│   └── kb.txt
├── 1-hop/
│   └── vanilla/qa_test.txt
├── 2-hop/
│   └── vanilla/qa_test.txt
└── 3-hop/
    └── vanilla/qa_test.txt
```

本项目默认只处理 test split，不需要音频数据和 NTM 改写数据。

处理原始数据：

```powershell
.\backend\venv311\Scripts\python.exe .\scripts\data\prepare_metaqa.py
```

重新生成固定 benchmark：

```powershell
.\backend\venv311\Scripts\python.exe .\scripts\data\create_midterm_benchmark.py
```

## 八、安装 Neo4j 并导入 Knowledge Graph

项目脚本会下载 Neo4j Community 5.26.32、设置本地密码、启动数据库、导入 MetaQA
三元组并运行验证：

```powershell
.\scripts\neo4j\setup_neo4j.ps1
```

首次运行需要下载 Neo4j，时间取决于网络速度。完成后可以访问：

```text
Neo4j Browser: http://localhost:7474
Bolt:          bolt://localhost:7687
```

之后只启动或停止 Neo4j：

```powershell
.\scripts\neo4j\start_neo4j.ps1
.\scripts\neo4j\stop_neo4j.ps1
```

单独验证图数据：

```powershell
.\backend\venv311\Scripts\python.exe .\scripts\neo4j\verify_graph.py
```

## 九、建立 Vector Index

第一次运行 Vector RAG 前，需要对 134,741 条 MetaQA 三元组生成 embedding，并在
Neo4j 中建立 cosine Vector Index：

```powershell
.\scripts\run_step5.ps1
```

脚本会自动：

1. 启动 Neo4j；
2. 下载或加载 `all-MiniLM-L6-v2`；
3. 生成 384 维 embedding；
4. 写入 `KnowledgeChunk` 节点；
5. 创建 Neo4j Vector Index；
6. 运行 Vector Retrieval 单独评测。

第一次 embedding 全部三元组需要一定时间。脚本支持断点续跑，不会重复计算已经完成的
节点。如需强制重建：

```powershell
.\scripts\run_step5.ps1 -Rebuild
```

## 十、运行中期实验

### 10.1 先运行 Smoke Tests

```powershell
.\backend\venv311\Scripts\python.exe .\scripts\evaluation\smoke_test_answer_pipelines.py
.\backend\venv311\Scripts\python.exe .\scripts\evaluation\smoke_test_kg_retrieval.py
```

第一个测试使用 Stub，不消耗 DeepSeek API；第二个测试连接本地 Neo4j，但不调用
DeepSeek。全部显示 `PASS` 后再运行正式实验。

### 10.2 小规模检查

建议每种方法先运行 3 道题，检查 API、Vector Index 和 Neo4j 是否正常：

```powershell
.\backend\venv311\Scripts\python.exe .\scripts\evaluation\evaluate_answer_systems.py --system pure_llm --limit 3
.\backend\venv311\Scripts\python.exe .\scripts\evaluation\evaluate_answer_systems.py --system vector_rag --limit 3
.\backend\venv311\Scripts\python.exe .\scripts\evaluation\evaluate_answer_systems.py --system kg_retrieval --limit 3
```

### 10.3 完整三组实验

```powershell
.\scripts\run_midterm.ps1
```

该命令会依次运行：

```text
Pure LLM → Vector RAG → KG Retrieval
```

每种方法使用同一份 300 题 benchmark。运行过程中每道题都会立刻写入本地 prediction
文件，因此意外中断后再次运行同一命令即可从未完成的位置继续。

只运行某一种方法：

```powershell
.\scripts\run_midterm.ps1 -System pure_llm
.\scripts\run_midterm.ps1 -System vector_rag
.\scripts\run_midterm.ps1 -System kg_retrieval
```

只重试 API 或运行失败的问题：

```powershell
.\scripts\run_midterm.ps1 -RetryErrors
```

只有在修改 Prompt 或评分逻辑后，才建议使用 `-RetryWrong`。它会重新调用所有回答错误的
问题，产生额外 API 调用：

```powershell
.\scripts\run_midterm.ps1 -RetryWrong
```

### 10.4 输出位置

正式汇总 metrics：

```text
experiments/midterm/metrics/
```

逐题 predictions：

```text
experiments/midterm/predictions/
```

Predictions 不上传 GitHub。如果需要给组员分析具体错误，应通过课程组内部方式单独共享。

## 十一、评测指标

- **Exact Match**：预测答案集合与标准答案集合完全一致才记为 1，是本实验的主要指标。
- **Precision**：预测出的实体中有多少属于标准答案，用于衡量额外错误答案。
- **Recall**：标准答案中有多少被系统找出，用于衡量遗漏答案。
- **F1**：Precision 与 Recall 的调和平均，用于综合衡量答案准确性和完整性。
- **Latency**：从输入问题到 Pipeline 返回结果的耗时。
- **Hits@K、MRR**：用于 Vector Retrieval 单独实验，衡量相关证据是否出现在前 K 条结果
  以及第一条相关证据的排名。
- **Cypher Execution Success Rate**：用于 KG Retrieval，衡量生成的 Cypher 是否能通过
  安全检查并成功执行。

所有答案在评分前都会进行大小写和空格标准化，并按照集合比较，答案顺序不影响结果。

## 十二、项目结构

```text
chat_Retrieval_project/
├── app/                         # Android 客户端，final 阶段继续集成
├── backend/
│   ├── app/
│   │   ├── rag/                 # 三种回答 Pipeline、生成器和共享 State
│   │   └── retrieval/
│   │       ├── vector/          # Vector Retriever 与本地 embedding 适配
│   │       └── kg/              # Text2Cypher KG Retriever
│   ├── main.py                  # FastAPI 服务入口
│   ├── env.example              # DeepSeek 配置模板
│   └── requirements.txt         # Python 依赖
├── data/
│   ├── raw/metaqa/              # MetaQA 原始数据，本地保存
│   └── processed/               # 处理后的图和问答数据，本地生成
├── experiments/
│   ├── shared/benchmark/        # 固定 300 题 benchmark
│   └── midterm/
│       ├── configs/             # 可复现实验配置
│       ├── metrics/             # 正式汇总指标
│       └── predictions/         # 逐题结果，仅本地保存
├── neo4j/
│   ├── cypher/                  # Schema 和查询示例
│   ├── .env.example             # Neo4j 配置模板
│   └── docker-compose.yml       # 可选 Docker 配置
├── scripts/
│   ├── data/                    # 数据预处理和 benchmark 生成
│   ├── neo4j/                   # Neo4j 安装、启动、导入与验证
│   ├── indexing/                # Vector Index 构建
│   ├── evaluation/              # 三组实验、评测和 Smoke Tests
│   ├── run_step5.ps1            # Vector Index 与检索实验
│   └── run_midterm.ps1          # 中期三组完整实验
└── README.md
```

核心执行流程：

```text
MetaQA 原始数据
    ↓
数据预处理
    ↓
Neo4j Knowledge Graph + Vector Index
    ↓
Pure LLM / Vector RAG / KG Retrieval
    ↓
统一答案集合评分
    ↓
Metrics 与本地 Predictions
```

## 十三、Midterm Report 建议范围

Midterm report 由组员根据当前代码和实验结果完成，建议包含：

1. **Introduction / Background**
   - 课程问题背景；
   - LLM factual QA 的限制；
   - RAG 与 Knowledge Graph 的作用。
2. **Dataset and Benchmark**
   - MetaQA 数据集；
   - Knowledge Graph 三元组；
   - 1-hop、2-hop、3-hop；
   - 固定 300 题采样方法。
3. **Technical Solutions**
   - Neo4j 图模型；
   - Pure LLM baseline；
   - Vector RAG Pipeline；
   - Text2Cypher KG Retrieval Pipeline；
   - DeepSeek JSON 输出与错误处理。
4. **Evaluation Method**
   - 三组方法的控制变量；
   - Exact Match、Precision、Recall、F1、Latency；
   - Vector Retrieval 和 Cypher 的辅助指标。
5. **Results and Analysis**
   - 总体结果；
   - 分 hop 结果；
   - Vector RAG 多跳能力不足的原因；
   - KG Retrieval 的优势、延迟和同名实体限制。
6. **Current Limitations and Next Plan**
   - 当前只有核心检索实验；
   - Hybrid RAG 与 Android 端到端功能留到 final。

Midterm report 不应把 Hybrid RAG 写成已经完成的中期实验，也不需要重点描述 Android UI。

## 十四、Final 阶段计划

Final 计划完成：

1. 实现 Hybrid RAG，同时使用 Vector Retrieval 与 KG Retrieval；
2. 设计证据合并、去重和冲突处理策略；
3. 将最佳 Pipeline 接入 FastAPI `/chat`；
4. 完成 Android Chat UI 与后端通信；
5. 在界面中展示回答、来源和错误提示；
6. 根据时间决定是否加入用户反馈和 Online Search fallback；
7. 在相同 benchmark 上增加 Hybrid RAG，形成四组对照；
8. 进行 Ablation Study，例如只使用 Vector、只使用 KG、两者结合；
9. 完成 final report 和项目演示。
