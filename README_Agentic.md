# Agentic RAG 数据问答系统

## 项目概述

本项目实现了基于Agentic RAG架构的智能数据问答系统。系统使用ReAct (Reasoning + Action) agent，通过MCP (Model Context Protocol) 协议与数据库和RAG工具服务交互，回答用户关于数据的问题。

## 架构特点

### Agentic RAG 架构
传统的RAG系统直接将检索功能集成到agent中，而Agentic RAG将RAG功能作为独立的MCP工具服务，让agent可以智能地决定是否使用检索功能。

```
传统RAG: 用户查询 → Agent直接调用RAG → 获取上下文 → 回答问题
Agentic RAG: 用户查询 → Agent通过MCP调用RAG工具 → 获取上下文 → 回答问题
                                      ↑
                              独立运行的RAG MCP服务
```

## 核心组件

1. **ReAct Agent** (`agent.py`): 核心智能体，支持动态工具选择
2. **API服务器** (`api.py`): FastAPI实现的RESTful API
3. **RAG工具服务** (`rag_mcp_server.py`): 独立的MCP工具服务器
4. **RAG处理器** (`rag_utils.py`): 向量检索和文档处理

## 快速开始

### 1. 环境准备

```bash
# 使用uv安装依赖
uv pip install -e .

# 或者使用pip
pip install -e .
```

### 2. 配置环境变量

编辑 `.env` 文件：
```bash
OPENAI_API_KEY=your_api_key
MODEL_BASE_URL=https://your_model_api
MODEL_NAME=gpt-4o-mini
```

### 3. 启动服务

确保数据库MCP服务已启动（端口8006），然后启动本项目：

```bash
# 启动RAG MCP服务
uv run python rag_mcp_server.py

# 启动主API服务（在新的终端）
uv run python api.py
```

## API使用

### 基本问答

```bash
curl -X POST "http://localhost:8010/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "卫片表中有哪些字段？"}'
```

### 响应格式

```json
{
  "answer": "卫片表包含以下字段：...",
  "retry_count": 0,
  "tool_calls": [
    {
      "name": "rag_search",
      "args": {"query": "卫片表字段信息"}
    },
    {
      "name": "database_query",
      "args": {"query": "SELECT * FROM 卫片表 LIMIT 5"}
    }
  ],
  "tool_responses": [...],
  "sql_queries": ["SELECT * FROM 卫片表 WHERE..."]
}
```

## Agentic RAG 工具

### 可用的RAG工具

1. **rag_search**: 基本RAG搜索，返回格式化文本
2. **rag_search_with_context**: 返回结构化结果
3. **rag_search_tables**: 专门搜索表结构信息

### 工具使用示例

Agent会自动判断何时使用RAG工具。当用户询问数据库结构相关问题时：

1. Agent调用`rag_search("卫片表字段信息")`获取表结构
2. 基于检索到的信息构造SQL查询
3. 执行数据库查询
4. 综合结果返回回答

## 支持的文档格式

- CSV文件
- PDF文件
- JSON文件（包括专门的表字段说明.json）
- Excel文件（.xlsx, .xls）
- YAML文件

## 开发

### 测试MCP工具

```bash
# 测试RAG功能
uv run python rag_mcp_server.py --test
```

### 查看系统架构

```bash
curl http://localhost:8010/graph
```

## 架构优势

1. **解耦**: RAG服务可以独立扩展和部署
2. **标准化**: 使用标准MCP协议通信
3. **动态调用**: Agent智能选择是否使用RAG工具
4. **多服务**: 支持连接多个RAG服务
5. **可观测**: 完整的工具调用链追踪

## 应用场景

- 企业内部数据智能问答
- 文档知识库的问答系统
- 数据库结构查询和解释
- 结合结构化和非结构化数据的知识问答

## 技术栈

- **LangChain & LangGraph**: 构建agent工作流
- **FastMCP**: MCP协议实现
- **FastAPI**: Web API框架
- **ChromaDB**: 向量数据库存储
- **Python 3.11+**: 运行环境

## 未来扩展

- 支持更多的文档格式
- 集成更多的MCP工具服务
- 支持多语言问答
- 添加可视化界面
- 支持分布式部署