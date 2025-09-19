#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示Agentic RAG功能的脚本
展示如何通过MCP工具查询表字段信息
"""

import asyncio
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从rag_utils导入工具函数用于直接测试
from rag_utils import search_database_context
from rag_mcp_server import rag_search_tool, rag_search_with_context_tool

async def demo_agentic_rag():
    """演示Agentic RAG功能"""
    print("演示Agentic RAG功能")
    print("=" * 50)

    # 测试查询
    queries = [
        "卫片表包含哪些字段？",
        "地级市字段有哪些可选值？",
        "实测面积字段的单位是什么？",
        "违法类型字段的字典值有哪些？",
        "违法用地属于哪个字段的值？"
    ]

    print("1. 直接调用RAG工具函数:")
    for query in queries:
        print(f"\n查询: {query}")
        results = search_database_context(query)

        if results:
            result = results[0]
            print(f"结果: {result['content'][:300]}...")
        else:
            print("未找到相关文档")

    print("\n" + "=" * 50)
    print("2. 通过MCP工具接口调用:")

    for query in queries:
        print(f"\n查询: {query}")
        # 模拟MCP工具调用
        result = await rag_search_tool(query)
        print(f"结果: {result[:300]}...")

    print("\n" + "=" * 50)
    print("3. 通过MCP工具接口调用（结构化结果）:")

    for query in queries:
        print(f"\n查询: {query}")
        # 模拟MCP工具调用
        result = await rag_search_with_context_tool(query)
        if "results" in result and result["results"]:
            content = result["results"][0]["content"]
            print(f"结果: {content[:300]}...")
        else:
            print("未找到相关文档")

    print("\n演示完成!")

if __name__ == "__main__":
    asyncio.run(demo_agentic_rag())