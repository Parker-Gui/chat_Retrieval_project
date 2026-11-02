# MovieGraph：基于 RAG 与 Knowledge Graph 的电影问答实验

本项目是信息检索课程 Project，使用 MetaQA 电影数据集比较三种问答方法：

1. **Pure LLM**：不检索外部知识，直接由 DeepSeek 回答。
2. **Vector RAG**：从 Neo4j Vector Index 检索相关三元组，再由 DeepSeek 根据证据回答。
3. **KG Retrieval**：使用 Text2Cypher 生成 Cypher，在 Neo4j 中执行图关系查询。

项目基于开源 [Android Chatbox](https://github.com/alvinluo-tech/chatbox) 二次开发。
中期阶段重点是核心检索功能与三组对照实验；Android 前端、Hybrid RAG 和完整聊天流程
留到 final 阶段完成。

## 一、快速开始

### Step 1：Clone 项目

```powershell
git clone https://github.com/Parker-Gui/chat_Retrieval_project.git
cd chat_Retrieval_project
```

### Step 2：创建 Python 3.11 环境并安装依赖

```powershell
py -3.11 -m venv backend\venv311
.\backend\venv311\Scripts\python.exe -m pip install --upgrade pip
.\backend\venv311\Scripts\python.exe -m pip install -r backend\requirements.txt
```

必须使用 Python 3.11。Python 3.14 可能导致 `pydantic-core` 编译失败。

### Step 3：配置 DeepSeek 和 Neo4j

```powershell
Copy-Item backend\env.example backend\.env
Copy-Item neo4j\.env.example neo4j\.env
```

打开 `backend/.env`，填写自己的 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=你的真实API_KEY
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

`neo4j/.env` 默认内容可以直接使用：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=moviegraph-dev
```

`.env` 包含本地密码和 API Key，不要上传 GitHub。

### Step 4：准备 MetaQA 数据

从 MetaQA 官方仓库下载数据：[https://github.com/yuyuz/MetaQA](https://github.com/yuyuz/MetaQA)。

将以下文件复制到项目对应位置：

```text
data/raw/metaqa/kb/kb.txt
data/raw/metaqa/1-hop/vanilla/qa_test.txt
data/raw/metaqa/2-hop/vanilla/qa_test.txt
data/raw/metaqa/3-hop/vanilla/qa_test.txt
```

然后运行数据处理脚本：

```powershell
.\backend\venv311\Scripts\python.exe .\scripts\data\prepare_metaqa.py
.\backend\venv311\Scripts\python.exe .\scripts\data\create_midterm_benchmark.py
```

第二条命令使用固定 random seed 42，生成 300 题 benchmark，其中每种 hop 各100题。

### Step 5：安装 Neo4j 并导入 Knowledge Graph

运行前确认电脑已经安装 JDK 17，然后执行：

```powershell
.\scripts\neo4j\setup_neo4j.ps1
```

该脚本会自动下载 Neo4j Community 5.26.32、启动数据库、导入 MetaQA 三元组并验证
1-hop、2-hop 和 3-hop 图路径。

### Step 6：建立 Vector Index

```powershell
.\scripts\run_step5.ps1
```

该脚本会下载或加载 `all-MiniLM-L6-v2`，为 134,741 条三元组生成 embedding，建立
Neo4j Vector Index，并运行 Vector Retrieval 单独评测。第一次运行时间较长，但支持
断点续跑。

### Step 7：运行中期三组实验

```powershell
.\scripts\run_midterm.ps1
```

脚本会自动运行检查，并依次执行：

```text
Pure LLM → Vector RAG → KG Retrieval
```

三种方法都使用同一份 300 题 benchmark。实验会保存每道题的状态，意外中断后重新运行
同一个命令即可继续，不需要从头开始。

如果出现 API 临时失败，修复网络或 API Key 后运行：

```powershell
.\scripts\run_midterm.ps1 -RetryErrors
```

除非修改了 Prompt 或评测代码，否则不要使用 `-RetryWrong`，它会产生大量额外 API 调用。

## 二、实验内容

### 2.1 Pure LLM

不向 DeepSeek 提供检索证据，用于测量模型仅依靠内部知识回答 MetaQA 问题的能力。
它是三组实验中的 no-retrieval baseline。

### 2.2 Vector RAG

使用 `all-MiniLM-L6-v2` 将知识图谱三元组编码成384维向量，并通过 cosine similarity
检索 Top-5 三元组。DeepSeek 只能根据检索结果回答，因此该实验能够观察语义检索对
1-hop 和多跳问题的影响。

### 2.3 KG Retrieval

DeepSeek 根据自然语言问题生成只读 Cypher，Neo4j 再执行明确的图关系路径。与 Vector
RAG 相比，它不只查找相似文本，还能显式执行 1-hop、2-hop 和 3-hop 查询。

## 三、评测方法

三组方法使用相同的 benchmark、DeepSeek 配置和答案评分代码。主要指标为：

- **Exact Match**：预测答案集合与标准答案集合完全一致才算正确；
- **Precision**：预测答案中正确实体的比例；
- **Recall**：标准答案中被成功找出的比例；
- **F1**：Precision 与 Recall 的综合指标；
- **Latency**：回答一道题所需的时间；
- **Hits@K 和 MRR**：用于单独评价 Vector Retrieval；
- **Cypher Execution Success Rate**：用于评价生成的 Cypher 能否成功执行。

答案会先进行大小写和空格标准化，再按照集合比较，答案顺序不影响分数。

## 四、实验结果

### 4.1 总体结果

| 方法         | 成功题数 |      Exact Match |        Precision |           Recall |               F1 | 平均 Latency |
| ------------ | -------: | ---------------: | ---------------: | ---------------: | ---------------: | -----------: |
| Pure LLM     |  300/300 |            8.67% |           0.3057 |           0.4586 |           0.3196 |      1270 ms |
| Vector RAG   |  300/300 |           15.00% |           0.2249 |           0.2024 |           0.2008 |      1110 ms |
| KG Retrieval |  300/300 | **96.00%** | **0.9745** | **0.9832** | **0.9773** |      1596 ms |

### 4.2 分 hop 的 Exact Match

| 方法         |            1-hop |            2-hop |            3-hop |
| ------------ | ---------------: | ---------------: | ---------------: |
| Pure LLM     |           25.00% |            0.00% |            1.00% |
| Vector RAG   |           43.00% |            0.00% |            2.00% |
| KG Retrieval | **95.00%** | **96.00%** | **97.00%** |

Vector RAG 对简单 1-hop 问题有帮助，但 Top-5 相似三元组不一定能够连接成完整的
多跳路径，因此在 2-hop 和 3-hop 上明显下降。KG Retrieval 直接执行图路径，在三种
hop 数量下都保持了稳定结果，但平均 Latency 略高。

Vector Retrieval 单独实验得到 Hits@1 20.67%、Hits@3 30.00%、Hits@5 32.67%、
Recall@5 23.06%、MRR 0.254，说明 Vector RAG 的主要限制来自检索阶段。

正式汇总 metrics 位于：

```text
experiments/midterm/metrics/
```

逐题 predictions 只保存在本地，不上传 GitHub。

## 五、项目结构

```text
chat_Retrieval_project/
├── app/                         # Android 客户端，final 阶段继续开发
├── backend/
│   ├── app/
│   │   ├── rag/                 # 三种回答 Pipeline、Generator 和共享 State
│   │   └── retrieval/
│   │       ├── vector/          # Vector Retriever 与 embedding
│   │       └── kg/              # Text2Cypher KG Retriever
│   ├── main.py                  # FastAPI 服务入口
│   ├── env.example              # DeepSeek 配置模板
│   └── requirements.txt         # Python 依赖
├── data/
│   ├── raw/metaqa/              # MetaQA 原始数据，本地保存
│   └── processed/               # 处理后的图和问答数据
├── experiments/
│   ├── shared/benchmark/        # 固定 300 题 benchmark
│   └── midterm/
│       ├── configs/             # 实验配置
│       ├── metrics/             # 正式汇总结果
│       └── predictions/         # 逐题结果，仅本地保存
├── neo4j/                       # Neo4j 配置与 Cypher Schema
├── scripts/
│   ├── data/                    # 数据处理
│   ├── neo4j/                   # Neo4j 安装、导入和启动
│   ├── indexing/                # Vector Index 构建
│   ├── evaluation/              # 实验与评测
│   ├── run_step5.ps1            # 建立 Vector Index
│   └── run_midterm.ps1          # 运行中期三组实验
└── README.md
```

整体流程为：

```text
MetaQA → 数据处理 → Neo4j Graph / Vector Index
       → 三种回答方法 → 统一评分 → Metrics
```

## 六、Midterm Report 应写的范围

Midterm report 建议包含：

1. Project background，以及 LLM factual QA 的问题；
2. MetaQA 数据集、Knowledge Graph 和固定 300 题 benchmark；
3. Neo4j 数据模型与三种方法的 Pipeline；
4. Pure LLM、Vector RAG、KG Retrieval 的实现方法；
5. Exact Match、Precision、Recall、F1、Latency 等评测方法；
6. 总体结果和分 hop 结果；
7. Vector RAG 多跳能力不足的原因；
8. KG Retrieval 的优势、延迟和同名实体限制；
9. 当前限制与 final 阶段计划。

Midterm report 不要把 Hybrid RAG 写成已经完成的实验，也不需要重点描述 Android UI。

## 七、Final 阶段计划

Final 计划完成：

1. 实现 Hybrid RAG，合并 Vector Retrieval 与 KG Retrieval；
2. 设计证据合并、去重和冲突处理策略；
3. 将最佳 Pipeline 接入 FastAPI `/chat`；
4. 完成 Android Chat UI 与后端通信；
5. 根据时间决定是否加入用户反馈和 Online Search fallback；
6. 在相同 benchmark 上增加 Hybrid RAG，形成四组对照；
7. 进行 Vector-only、KG-only 和 Hybrid 的 Ablation Study；
8. 完成 final report 和项目演示。
