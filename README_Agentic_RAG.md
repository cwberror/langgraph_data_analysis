# Agentic RAG 架构说明

## 架构概述

本项目采用Agentic RAG架构，将RAG功能作为独立的MCP工具提供，与数据库查询工具一样被agent调用。

### 组件说明

1. **主API服务器** (`api.py`) - FastAPI应用，提供REST API接口
2. **Agent核心** (`agent.py`) - 实现ReAct agent逻辑
3. **RAG MCP服务器** (`rag_mcp_server.py`) - 独立的MCP工具服务器，提供RAG搜索功能
4. **数据库工具服务器** - 提供数据库查询功能的MCP服务器（外部）

### 数据流

```
用户请求 -> API服务器 -> Agent -> MCP客户端 -> RAG工具服务器/数据库工具服务器
```

## 启动服务

1. 启动数据库工具服务器（外部服务）
2. 启动主API服务器：
   ```bash
   python api.py
   ```

API服务器启动时会自动启动RAG MCP服务器。

## MCP工具说明

### RAG搜索工具

- **工具名称**: `rag_search`
- **功能**: 搜索数据库上下文以获取相关信息
- **参数**:
  - `query`: 搜索查询，应该包含相关的表名、字段名或业务问题
- **返回**: 相关文档内容的字符串

### RAG搜索工具（结构化结果）

- **工具名称**: `rag_search_with_context`
- **功能**: 搜索数据库上下文并返回结构化结果
- **参数**:
  - `query`: 搜索查询，应该包含相关的表名、字段名或业务问题
- **返回**: 包含搜索结果的字典

## 开发说明

### 添加新的MCP工具

1. 在`rag_mcp_server.py`中添加新的工具函数
2. 使用`@mcp.tool`装饰器注册工具
3. 重启API服务器以使更改生效

### 修改现有工具

1. 修改`rag_mcp_server.py`中的工具函数实现
2. 重启API服务器以使更改生效

## 测试

可以通过以下方式测试RAG工具：

1. 启动API服务器
2. 使用curl或Postman发送请求到`/ask`端点
3. 在问题中包含需要数据库上下文的信息

示例：
```bash
curl -X POST "http://localhost:8010/ask" -H "Content-Type: application/json" -d '{"question": "卫片表中有哪些字段？"}'
```