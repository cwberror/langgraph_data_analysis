#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReAct Agent实现模块
负责实现ReAct agent的核心逻辑，包括与MCP工具的交互、RAG集成等
"""

import asyncio
import os,sys
import json
from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


# 加载环境变量
load_dotenv()

# 定义agent状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_question: str
    retry_count: int
    # 多智能体状态管理字段
    current_agent: str  # 当前执行的智能体 ("rag_agent" 或 "db_agent")
    rag_context: str    # RAG智能体提供的上下文信息
    table_info: str     # 表结构信息
    field_info: str     # 字段信息
    db_query: str       # 数据库查询语句
    db_result: str      # 数据库查询结果
    loop_count: int     # 循环计数，用于限制最大循环次数

async def should_continue_async(state: AgentState):
    """
    检查是否需要继续执行工具调用或切换智能体

    Args:
        state (AgentState): 当前agent状态，包含消息历史和智能体状态

    Returns:
        str: "continue"表示需要继续执行工具调用，
             "switch_to_db"表示切换到数据库智能体，
             "switch_to_rag"表示切换到RAG智能体，
             "end"表示结束流程
    """
    # 获取当前状态中的消息列表
    messages = state["messages"]
    current_agent = state.get("current_agent", "rag_agent")
    rag_context = state.get("rag_context", "")

    # 获取循环计数
    loop_count = state.get("loop_count", 0)
    max_loops = 5  # 限制最大循环次数为5

    # 如果循环次数超过限制，强制结束流程
    if loop_count >= max_loops:
        print(f"已达到最大循环次数 {max_loops}，强制结束流程")
        return "end"

    # 如果没有消息，直接结束流程
    if not messages:
        return "end"

    # 获取最后一条消息
    last_message = messages[-1]

    # 检查最后一条消息是否为AI消息且包含工具调用
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"检测到工具调用: {[tc['name'] for tc in last_message.tool_calls]}")
        return "continue"

    # 根据当前智能体和状态决定下一步操作
    if current_agent == "rag_agent" and rag_context:
        # RAG智能体已完成工作，切换到数据库智能体
        return "switch_to_db"
    elif current_agent == "db_agent" and state.get("db_result", ""):
        # 数据库智能体已完成工作，结束流程
        return "end"
    else:
        # 继续当前智能体的工作
        return "continue"


async def rag_agent_async(state: AgentState, tools):
    """
    RAG智能体节点函数，负责检索相关表和字段信息

    Args:
        state (AgentState): 当前agent状态
        tools (list): 可用的工具列表

    Returns:
        dict: 更新后的状态字典
    """
    messages = state["messages"]
    original_question = state.get("original_question", "")
    retry_count = state.get("retry_count", 0)
    rag_context = state.get("rag_context", "")

    if not messages:
        messages = []

    # 初始化模型并绑定工具
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0.3 if retry_count > 0 else 0,  # 重试时增加一些随机性
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("MODEL_BASE_URL")
    )
    model_with_tools = model.bind_tools(tools)

    # 构造RAG智能体的系统提示
    retry_instruction = ""
    if retry_count > 0:
        retry_instruction = f"\n\n注意: 这是第{retry_count + 1}次尝试回答。请仔细分析之前的回答不足之处，提供更准确、完整的答案。"

    system_prompt_content = f'''
        你是一个RAG智能体，专门负责检索与用户问题相关的数据库表结构和字段信息。
        请遵循以下步骤：
        1：分析用户问题，理解用户需要查询的数据内容
        2：调用"rag_search"工具获取与用户问题相关的表名、字段名信息
        3：提取并整理表结构信息和字段信息
        4：将这些信息保存到状态中，供数据库查询智能体使用

        用户问题: {original_question}
        {retry_instruction}
        '''

    system_prompt = SystemMessage(content=system_prompt_content)
    response_messages = [system_prompt] + messages

    response = await model_with_tools.ainvoke(response_messages)
    print(f"[RAG智能体响应]: {response.content if hasattr(response, 'content') else '无内容'}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[RAG智能体工具调用]: {[tc['name'] for tc in response.tool_calls]}")

    # 增加循环计数
    loop_count = state.get("loop_count", 0) + 1

    return {
        "messages": [response],
        "current_agent": "rag_agent",
        "loop_count": loop_count
    }


async def db_agent_async(state: AgentState, tools):
    """
    数据库查询智能体节点函数，负责执行数据库查询

    Args:
        state (AgentState): 当前agent状态
        tools (list): 可用的工具列表

    Returns:
        dict: 更新后的状态字典
    """
    messages = state["messages"]
    original_question = state.get("original_question", "")
    retry_count = state.get("retry_count", 0)
    rag_context = state.get("rag_context", "")
    table_info = state.get("table_info", "")
    field_info = state.get("field_info", "")

    if not messages:
        messages = []

    # 初始化模型并绑定工具
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0.3 if retry_count > 0 else 0,  # 重试时增加一些随机性
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("MODEL_BASE_URL")
    )
    model_with_tools = model.bind_tools(tools)

    # 构造数据库查询智能体的系统提示
    retry_instruction = ""
    if retry_count > 0:
        retry_instruction = f"\n\n注意: 这是第{retry_count + 1}次尝试回答。请仔细分析之前的回答不足之处，提供更准确、完整的答案。"

    system_prompt_content = f'''
        你是一个数据库查询智能体，专门负责根据RAG智能体提供的表结构和字段信息构造并执行SQL查询。
        请遵循以下步骤：
        1：分析RAG智能体提供的表结构信息和字段信息
           表信息：{table_info}
           字段信息：{field_info}
        2：根据用户问题和提供的表字段信息，构造准确的SQL查询语句
        3：调用数据库查询工具执行SQL查询
        4：分析查询结果并生成最终回答

        用户问题: {original_question}
        {retry_instruction}
        '''

    system_prompt = SystemMessage(content=system_prompt_content)
    response_messages = [system_prompt] + messages

    response = await model_with_tools.ainvoke(response_messages)
    print(f"[数据库查询智能体响应]: {response.content if hasattr(response, 'content') else '无内容'}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[数据库查询智能体工具调用]: {[tc['name'] for tc in response.tool_calls]}")

    # 增加循环计数
    loop_count = state.get("loop_count", 0) + 1

    return {
        "messages": [response],
        "current_agent": "db_agent",
        "loop_count": loop_count
    }

async def create_agent_graph(tools):
    """
    创建多智能体执行图

    Args:
        tools (list): 可用的工具列表

    Returns:
        CompiledGraph: 编译后的执行图
    """
    # 分离RAG工具和数据库工具
    rag_tools = [tool for tool in tools if tool.name.startswith("rag")]
    db_tools = [tool for tool in tools if not tool.name.startswith("rag")]

    # 如果没有明确的工具分离，就使用所有工具
    if not rag_tools:
        rag_tools = tools
    if not db_tools:
        db_tools = tools

    # 创建异步版本的节点函数
    async def async_rag_agent(state: AgentState):
        return await rag_agent_async(state, rag_tools)

    async def async_db_agent(state: AgentState):
        return await db_agent_async(state, db_tools)

    # 构建多智能体图
    graph = StateGraph(AgentState)
    graph.add_node("rag_agent", async_rag_agent)
    graph.add_node("db_agent", async_db_agent)

    # 为每个智能体创建专用的工具节点
    rag_tool_node = ToolNode(rag_tools)
    db_tool_node = ToolNode(db_tools)
    graph.add_node("rag_tools", rag_tool_node)
    graph.add_node("db_tools", db_tool_node)

    # 定义边和条件边
    graph.add_edge(START, "rag_agent")
    graph.add_conditional_edges(
        "rag_agent",
        should_continue_async,
        {
            'continue': 'rag_tools',
            'switch_to_db': 'db_agent',
            'end': END
        }
    )
    graph.add_conditional_edges(
        "db_agent",
        should_continue_async,
        {
            'continue': 'db_tools',
            'end': END
        }
    )

    # 工具调用后返回到相应的智能体
    graph.add_edge("rag_tools", "rag_agent")
    graph.add_edge("db_tools", "db_agent")

    # 编译图
    compiled_graph = graph.compile()

    return compiled_graph


async def initialize_agent():
    """
    初始化agent应用

    Returns:
        CompiledGraph: 初始化后的agent应用，如果初始化失败则返回None
    """
    try:
        print("开始初始化MCP客户端...")
        # 初始化MCP客户端，首先只连接数据库工具服务器
        try:
            print("连接MCP服务器...")
            # 获取当前脚本目录
            project_root = os.path.dirname(os.path.abspath(__file__))
            rag_mcp_path = os.path.join(project_root, "rag_mcp_server.py")

            client = MultiServerMCPClient(
                {
                    "mcp_zxxc": {
                        "transport": "sse",
                        "url": "http://localhost:8006/mcp"
                    },
                    "rag_mcp": {
                        "transport": "stdio",
                        "command": sys.executable,  # Python解释器路径
                        "args": [rag_mcp_path],     # RAG MCP服务器脚本路径
                        "env": {
                            **os.environ,  # 传递当前环境变量
                            "PYTHONUNBUFFERED": "1",  # 确保Python输出不被缓冲
                            "PYTHONIOENCODING": "utf-8"  # 确保Python IO使用UTF-8编码
                        },
                        "cwd": project_root  # 设置工作目录
                    }
                }
            )
            print("数据库(SSE)和RAG(STDIO) MCP服务器连接成功")

            # 显示连接成功的消息
            print("正在获取工具...")
            tools = await client.get_tools()
            print(f"获取到 {len(tools)} 个工具")

            # 显示工具信息
            for i, tool in enumerate(tools):
                print(f"  工具 {i+1}: {tool.name}")

        except Exception as e:
            print(f"初始化MCP客户端失败: {e}")
            import traceback
            traceback.print_exc()
            print("创建一个最小化的agent，不使用MCP工具...")
            # 创建一个没有外部工具的agent
            from langchain_core.tools import tool

            @tool
            def fallback_search(query: str) -> str:
                """备用搜索工具"""
                return f"备用工具收到查询: {query}"

            tools = [fallback_search]
            print("使用备用工具继续初始化")
        
        # 工具已经在上面获取或创建了，直接使用

        # 创建agent图
        if tools:
            print("开始创建agent图...")
            agent_app = await create_agent_graph(tools)
            print("Agent应用初始化完成")
            return agent_app
        else:
            print("未获取到任何工具，无法创建agent")
            return None

    except Exception as e:
        print(f"初始化agent失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

async def ask_question(agent_app, question: str):
    """
    向agent提问并获取答案

    Args:
        agent_app (CompiledGraph): 已初始化的agent应用
        question (str): 用户的问题

    Returns:
        dict: 包含答案和相关信息的字典，如果出错则返回None
    """
    try:
        # 创建初始状态
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "original_question": question,
            "retry_count": 0,
            "current_agent": "rag_agent",  # 初始为RAG智能体
            "rag_context": "",
            "table_info": "",
            "field_info": "",
            "db_query": "",
            "db_result": "",
            "loop_count": 0  # 初始化循环计数
        }

        # 记录工具调用、响应和SQL查询
        tool_calls = []
        tool_responses = []
        sql_queries = []

        # 执行agent并收集过程信息
        final_state = None
        async for step in agent_app.astream(initial_state, stream_mode="values"):
            if "messages" in step and step["messages"]:
                message = step["messages"][-1]
                if isinstance(message, AIMessage):
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_calls.append({
                                "name": tool_call['name'],
                                "args": tool_call['args']
                            })
                            print(f"工具调用: {tool_call['name']}")
                            # 从工具调用参数中提取SQL查询
                            if 'args' in tool_call:
                                # 检查常见的SQL查询参数名
                                for key in ['query', 'sql', 'statement']:
                                    if key in tool_call['args'] and isinstance(tool_call['args'][key], str):
                                        # 添加SQL查询，不进行严格验证以支持中文SQL
                                        sql_queries.append(tool_call['args'][key])
                elif isinstance(message, ToolMessage):
                    try:
                        content = json.loads(message.content) if isinstance(message.content, str) else message.content
                        tool_responses.append(content)
                        print(f"工具响应: {content}")
                        # 从工具响应中提取SQL查询
                        if isinstance(content, dict):
                            # 检查常见的SQL查询参数名
                            for key in ['query', 'sql', 'statement']:
                                if key in content and isinstance(content[key], str):
                                    # 添加SQL查询，不进行严格验证以支持中文SQL
                                    sql_queries.append(content[key])
                    except:
                        tool_responses.append(message.content)
                        print(f"工具响应: {message.content}")
                        # 尝试从字符串响应中提取SQL语句
                        if isinstance(message.content, str):
                            # 简单的SQL提取（查找可能的SQL语句）
                            import re
                            sql_patterns = [
                                r'(SELECT\s+.*?;)',
                                r'(INSERT\s+INTO\s+.*?;)',
                                r'(UPDATE\s+.*?;)',
                                r'(DELETE\s+FROM\s+.*?;)'
                            ]
                            for pattern in sql_patterns:
                                matches = re.findall(pattern, message.content, re.IGNORECASE | re.DOTALL)
                                for match in matches:
                                    sql_queries.append(match)
            # 保存最终状态
            final_state = step

        # 从工具调用中额外提取SQL查询
        for tool_call in tool_calls:
            if 'args' in tool_call and isinstance(tool_call['args'], dict):
                for key in ['query', 'sql', 'statement']:
                    if key in tool_call['args'] and isinstance(tool_call['args'][key], str):
                        # 确保SQL查询不在sql_queries中才添加
                        if tool_call['args'][key] not in sql_queries:
                            sql_queries.append(tool_call['args'][key])

        # 提取最终答案
        answer = ""
        retry_count = 0
        if final_state and "messages" in final_state and final_state["messages"]:
            final_message = final_state["messages"][-1]
            if isinstance(final_message, AIMessage) and final_message.content:
                answer = final_message.content
                retry_count = final_state.get("retry_count", 0)

        return {
            "answer": answer,
            "retry_count": retry_count,
            "tool_calls": tool_calls,
            "tool_responses": tool_responses,
            "sql_queries": sql_queries
        }
    except Exception as e:
        print(f"处理问题时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None