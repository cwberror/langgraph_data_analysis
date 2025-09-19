# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Python data analysis project that implements an intelligent ReAct (Reasoning + Action) agent with Agentic RAG architecture. The agent uses LangChain, LangGraph, and MCP (Model Context Protocol) to interact with tools and databases for answering user questions about data. The RAG functionality is now implemented as a separate MCP tool service, following the Agentic RAG pattern where agents can dynamically choose to use RAG tools based on query requirements.

## Key Components
- `agent.py`: ReAct agent core logic with enhanced Agentic RAG support
- `api.py`: FastAPI implementation for serving the agent as a web API
- `rag_utils.py`: RAG processor implementation for searching database documentation
- `rag_mcp_server.py`: Standalone MCP server providing RAG tools
- `pyproject.toml`: Project dependencies including FastAPI, LangChain, LangGraph, and FastMCP
- `.env`: Environment configuration with API keys and model settings

## Architecture
The agent follows an Agentic RAG architecture with a graph-based workflow:
1. User questions are received via API endpoint
2. The agent initializes with both database query tools and RAG search tools via MCP
3. Uses LangGraph to orchestrate the reasoning process over available MCP tools
4. The agent can dynamically choose to use RAG tools for database context retrieval
5. Calls LLM with tools via MCP protocol to generate informed responses
6. Displays tool calls, database responses, RAG context, and final answers

### Agentic RAG Implementation
- RAG functionality is implemented as a separate MCP server (`rag_mcp_server.py`)
- Agent connects to RAG MCP server via MCP protocol
- Agent can dynamically decide whether to use RAG tools based on the query context
- RAG tools provide database structure information, field descriptions, and metadata
- This decoupling allows for independent scaling and deployment of RAG services

## Development Commands
- Start RAG MCP Server: `uv run python rag_mcp_server.py`
- Start main API server: `uv run python api.py`
- Test RAG functionality: `uv run python rag_mcp_server.py --test`
- Install dependencies: `uv pip install -e .`
- The project uses Python 3.11+

## Key Dependencies
- LangChain for LLM interactions and tool calling
- LangGraph for workflow orchestration
- LangGraph-MCP-Adapters for MCP client functionality
- FastMCP for Model Context Protocol server implementation
- FastAPI for RESTful API
- ChromaDB for vector storage
- Pandas for data manipulation
- MultiServerMCPClient for connecting to multiple MCP servers

## Environment Variables
The project requires:
- OPENAI_API_KEY: API key for the model provider
- MODEL_BASE_URL: Base URL for the model API
- MODEL_NAME: Specific model to use
- MODEL_PROVIDER: The model provider name

## Important Implementation Details
- Agentic RAG: RAG functionality is now implemented as an independent MCP tool service
- The agent connects to both database MCP server and RAG MCP server simultaneously
- The system prompt guides the agent to use RAG tools for database context when needed
- RAG tools include: `rag_search`, `rag_search_with_context`, and `rag_search_tables`
- API response includes all tool calls (both database and RAG) and extracted SQL queries
- The architecture supports horizontal scaling - RAG MCP server can be deployed independently

## MCP Servers
The project uses two MCP servers:
1. **Database Tool Server** (`http://localhost:8006/mcp`): Provides database query tools
2. **RAG Tool Server** (`http://localhost:8007/mcp`): Provides RAG search tools

## Example RAG Tool Usage
When agent needs database context information:
1. Agent decides to call `rag_search` tool via MCP protocol
2. RAG MCP server searches vector database for relevant documentation
3. Returns formatted results with table structures and field information
4. Agent uses this information to construct accurate database queries
- 该项目的github代码仓库地址是：https://github.com/cwberror/langgraph_data_analysis.git