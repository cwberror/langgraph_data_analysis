#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示Agentic RAG功能的脚本
展示如何通过MCP工具查询表字段信息
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List

async def test_agentic_rag_api():
    """通过API测试Agentic RAG功能"""
    print("演示Agentic RAG功能")
    print("=" * 50)

    api_url = "http://localhost:8010/ask"

    # 测试问题集
    test_questions = [
        "卫片表中有哪些字段？",
        "实地面积和实测面积有什么区别？",
        "违法类型字段都有哪些？",
        "如何查询年度内的卫片数据？"
    ]

    for question in test_questions:
        print(f"\n测试问题: {question}")
        print("-" * 30)

        try:
            async with aiohttp.ClientSession() as session:
                payload = {"question": question}

                async with session.post(api_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()

                        print(f"答案: {result['answer']}")
                        print(f"重试次数: {result['retry_count']}")
                        print(f"工具调用数量: {len(result['tool_calls'])}")

                        if result['tool_calls']:
                            print("工具调用详情:")
                            for tool_call in result['tool_calls']:
                                print(f"  - {tool_call['name']}: {tool_call['args']}")

                        if result['sql_queries']:
                            print("SQL查询:")
                            for sql_query in result['sql_queries']:
                                print(f"  - {sql_query}")
                    else:
                        print(f"API返回错误: {response.status}")
                        error_text = await response.text()
                        print(f"错误信息: {error_text}")

        except Exception as e:
            print(f"测试失败: {e}")

        print("\n")

async def test_rag_search_directly():
    """直接测试RAG搜索功能"""
    print("直接测试RAG搜索功能")
    print("=" * 50)

    # 先测试RAG MCP服务器
    servers = [
        ("http://localhost:8007/mcp/", "RAG工具"),
        ("http://localhost:8006/mcp/", "数据库工具")
    ]

    for server_url, name in servers:
        print(f"\n测试{name}: {server_url}")
        try:
            async with aiohttp.ClientSession() as session:
                # 获取工具列表
                list_tools_url = f"{server_url}tools"

                async with session.get(list_tools_url) as response:
                    if response.status == 200:
                        tools_result = await response.json()
                        print(f"找到的工具数量: {len(tools_result)}")

                        for i, tool in enumerate(tools_result):
                            print(f"  {i+1}. {tool.get('name', 'unknown')}: {tool.get('description', 'No description')}")
                    else:
                        print(f"获取工具列表失败: {response.status}")

        except Exception as e:
            print(f"{name}测试失败: {e}")

def summarize_architecture():
    """总结架构变化"""
    print("Agentic RAG架构总结")
    print("=" * 50)

    print("""
传统的RAG vs Agentic RAG:

[传统RAG]
用户查询 → Agent直接调用RAG函数 → 获取上下文 → 回答问题

[Agentic RAG]  ← 当前架构
用户查询 → Agent通过MCP协议调用RAG工具 → 获取上下文 → 回答问题
              ↗
独立运行的RAG MCP服务器

主要优势:
1. 解耦 - RAG服务可以独立扩展和部署
2. 标准化 - 使用标准MCP协议通信
3. 动态调用 - Agent可以智能选择是否使用RAG工具
4. 多服务 - 可以同时连接多个RAG服务

工作流程:
1. API服务器启动时自动启动RAG MCP服务器
2. Agent初始化时连接到多个MCP服务器（数据库+检索）
3. Agent根据查询内容决定是否需要数据库上下文信息
4. 如果需要，调用rag_search工具获取相关表结构信息
5. 基于RAG结果构造准确的数据库查询
6. 返回完整的回答，包含工具调用记录
    """)

async def main():
    """主函数"""
    # 测试API
    await test_agentic_rag_api()

    # 直接测试底层服务
    await test_rag_search_directly()

    # 总结架构
    summarize_architecture()

    print("\n演示完成！")

if __name__ == "__main__":
    asyncio.run(main())