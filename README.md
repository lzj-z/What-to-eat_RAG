# AI 私厨 · 智能食谱 RAG 问答系统

基于检索增强生成（RAG）技术构建的本地食谱问答系统，支持菜品推荐、制作步骤查询、按食材/分类/难度筛选，并可选接入 Neo4j 知识图谱实现关联推理。

---

## 功能特性

- **智能对话问答**：自然语言提问，自动识别意图并给出菜品推荐或详细步骤
- **三路混合检索**：向量语义检索 + BM25 关键词检索 + 菜名精确匹配，通过 RRF 融合排序
- **查询路由与重写**：LLM 自动判断问题类型（推荐列表 / 详细步骤 / 通用问答）并优化搜索词
- **元数据过滤**：支持按分类（荤菜/素菜/汤品/主食/早餐/饮品）和难度（简单/中等/困难/非常困难）过滤
- **知识图谱增强**（可选）：接入 Neo4j 实现食材→菜品关联查询和相似菜品推理
- **流式输出**：SSE 实时流式返回回答，RAG 过程可视化展示
- **菜品检索面板**：按分类浏览食谱，支持关键词搜索

---

## 技术栈

| 层级 | 技术 |
|---|---|
| LLM | DeepSeek Chat (`deepseek-chat`) |
| Embedding | 阿里 DashScope `text-embedding-v3` |
| 向量数据库 | FAISS（本地持久化） |
| 关键词检索 | BM25 + jieba 中文分词 |
| 知识图谱 | Neo4j（可选，离线自动降级） |
| RAG 框架 | LangChain |
| 后端服务 | Flask + Flask-CORS |
| 前端 | 原生 HTML / CSS / JS |

---

## 项目结构

```
Eat what_RAG/
├── server.py                  # Flask HTTP 服务入口
├── main.py                    # RAG 系统主控（初始化、检索、生成）
├── config.py                  # 全局配置（模型、路径、检索参数）
├── neo4j.txt                  # Neo4j 连接配置（uri/user/password）
├── LLM/
│   ├── init_llm.py            # LLM 与 Embedding 客户端初始化
│   └── .env                   # API Key 环境变量（不提交 Git）
├── RAG_moudle/
│   ├── data_preparation.py    # 数据加载、Markdown 分块、元数据提取
│   ├── index_construction.py  # FAISS 索引构建与增量更新
│   ├── retrieval_optimization.py  # 混合检索、RRF 融合、多样化检索
│   ├── generation_integration.py  # 查询路由、重写、回答生成
│   ├── graph_extraction.py    # Neo4j 数据导入
│   └── graph_retrieval.py     # 知识图谱检索（1跳/2跳推理）
├── Dish/                      # 食谱 Markdown 文件
│   ├── breakfast/
│   ├── dinner and lunch/
│   │   ├── meat/
│   │   ├── vegetable/
│   │   └── soup/
│   ├── staple/
│   └── drink/
├── page/                      # 前端静态文件
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── images/
└── vector_index/              # FAISS 索引（运行后自动生成）
```

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- （可选）Neo4j 4.x / 5.x

### 2. 安装依赖

```bash
pip install flask flask-cors python-dotenv \
    langchain langchain-community langchain-core langchain-text-splitters \
    faiss-cpu jieba rank_bm25 neo4j openai
```

### 3. 配置 API Key

在 `LLM/` 目录下创建 `.env` 文件：

```env
DEEPSEEK_APIKEY=your_deepseek_api_key
DEEPSEEK_BASEURL=https://api.deepseek.com

DASHSCOPE_APIKEY=your_dashscope_api_key
DASHSCOPE_BASEURL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 4. 配置 Neo4j（可选）

在项目根目录创建 `neo4j.txt`（不存在则自动跳过图谱，系统正常运行）：

```
uri=bolt://localhost:7687
user=neo4j
password=your_password
```

### 5. 启动服务

```bash
python server.py
```

服务启动后访问：[http://localhost:5000](http://localhost:5000)

首次启动会自动读取 `Dish/` 目录、构建向量索引（约需数十秒），之后启动直接加载缓存。

---

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/status` | 查询系统初始化状态 |
| POST | `/api/ask` | 普通问答（返回完整 JSON） |
| POST | `/api/ask/stream` | 流式问答（SSE，推荐） |
| POST | `/api/search` | 按分类检索菜品 |

### 流式问答示例

```bash
curl -X POST http://localhost:5000/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "推荐几个简单的素菜"}'
```

SSE 事件类型：`step`（过程日志）、`answer_start`、`answer_chunk`、`answer_end`、`error`

---

## 添加食谱

在 `Dish/` 对应子目录下新增 Markdown 文件，格式参考已有文件（使用 `#`/`##`/`###` 标题划分菜名、食材、步骤、技巧章节）。

系统通过 MD5 指纹检测文件变化，重启后自动增量更新索引，无需手动操作。

---

## 配置说明

在 `config.py` 中可调整以下参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `data_path` | `./Dish` | 食谱目录路径，可通过环境变量 `DATA_PATH` 覆盖 |
| `top_k` | `5` | 检索返回的文档块数（list 路由自动扩大为 3 倍） |
| `temperature` | `0.1` | 生成温度，越低越稳定 |
| `max_tokens` | `2048` | 单次生成最大 token 数 |
| `use_graph_rag` | `True` | 是否启用知识图谱增强（连接失败自动降级） |
