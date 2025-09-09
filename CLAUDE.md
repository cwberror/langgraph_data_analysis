# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Python data analysis project that implements an intelligent ReAct (Reasoning + Action) agent. The agent uses LangChain, LangGraph, and MCP (Model Context Protocol) to interact with tools and databases for answering user questions about data.

## Key Components
- `main.py`: Core implementation of the ReAct agent with automatic answer evaluation and retry mechanisms
- `pyproject.toml`: Project dependencies including FastAPI, LangChain, LangGraph, and Streamlit
- `.env`: Environment configuration with API keys and model settings

## Architecture
The agent follows a graph-based workflow:
1. Receives user questions through an interactive CLI
2. Uses LangGraph to orchestrate the reasoning process
3. Calls LLM (Qwen3-Coder) with tools via MCP protocol
4. Automatically evaluates answer quality
5. Retries up to 5 times with feedback if answers are insufficient
6. Displays tool calls, database responses, and final answers

## Development Commands
- Run the agent: `python main.py`
- Install dependencies: `pip install -e .` or `uv pip install -e .`
- The project uses Python 3.11+

## Key Dependencies
- LangChain for LLM interactions and tool calling
- LangGraph for workflow orchestration
- FastMCP for Model Context Protocol integration
- Pandas for data manipulation
- Streamlit for potential UI components

## Environment Variables
The project requires:
- OPENAI_API_KEY: API key for the model provider
- MODEL_BASE_URL: Base URL for the model API
- MODEL_NAME: Specific model to use
- MODEL_PROVIDER: The model provider name

## Important Implementation Details
- The agent automatically retries up to 5 times if answers are deemed insufficient
- Tool calls are displayed in the console for transparency
- Database interactions are handled through MCP protocol
- Answer quality is evaluated by a separate LLM call