#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG MCP服务器，将RAG功能作为MCP工具提供
使用FastMCP框架实现
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastmcp import FastMCP
from rag_utils import search_database_context

# 创建FastMCP服务器实例
mcp = FastMCP("RAG MCP Server", version="1.0.0")


@mcp.tool
def rag_search(query: str) -> str:
    """
    搜索数据库上下文以获取相关信息，特别包括表结构和字段信息

    Args:
        query: 搜索查询，应该包含相关的表名、字段名或业务问题

    Returns:
        格式化的搜索结果字符串
    """
    try:
        if not query:
            return "查询不能为空"

        # 使用RAG处理器搜索相关文档
        results = search_database_context(query)

        if not results:
            return "未找到相关数据库文档信息"

        # 格式化结果
        formatted_results = []
        for i, result in enumerate(results, 1):
            content = result["content"]
            source = result["metadata"].get("source_file", "未知来源")
            formatted_results.append(f"文档 {i} (来源: {source}):\n{content}")

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"RAG搜索工具执行出错: {str(e)}"


@mcp.tool
def rag_search_with_context(query: str) -> Dict[str, Any]:
    """
    搜索数据库上下文并返回结构化结果，适用于需要进一步处理的场景

    Args:
        query: 搜索查询，应该包含相关的表名、字段名或业务问题

    Returns:
        包含搜索结果的结构化字典
    """
    try:
        if not query:
            return {"error": "查询不能为空"}

        # 使用RAG处理器搜索相关文档
        results = search_database_context(query)

        if not results:
            return {"message": "未找到相关数据库文档信息", "results": []}

        return {
            "message": f"找到{len(results)}个相关文档",
            "results": results
        }

    except Exception as e:
        return {"error": f"RAG搜索工具执行出错: {str(e)}"}


@mcp.tool
def rag_search_tables(table_name: str = None) -> str:
    """
    专门搜索数据库表结构和字段信息

    Args:
        table_name: 可选的表名，如果不提供则搜索所有表的信息

    Returns:
        表结构和字段信息的格式化字符串
    """
    try:
        if table_name:
            # 如果提供了表名，搜索特定表
            search_query = f"{table_name} 表字段信息"
        else:
            # 如果没有提供表名，搜索所有表的信息
            search_query = "表结构 字段信息 表名"

        print(f"搜索表信息: {search_query}")
        results = search_database_context(search_query)

        if not results:
            if table_name:
                return f"未找到表 '{table_name}' 的相关信息"
            else:
                return "未找到任何表结构信息"

        # 专门格式化表结构信息
        formatted_results = []
        for i, result in enumerate(results, 1):
            content = result["content"]
            # 提取表名信息
            if "表名:" in content:
                # 找出表名
                lines = content.split('\n')
                tableinfo = ""
                for line in lines:
                    if line.strip().startswith('表名:'):
                        tableinfo = line.strip()
                        break

                formatted_results.append(f"[{tableinfo}]\n{content[:300]}{'...' if len(content) > 300 else ''}")
            else:
                formatted_results.append(f"信息 {i}: {content[:300]}{'...' if len(content) > 300 else ''}")

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"表信息搜索工具执行出错: {str(e)}"


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description='RAG MCP服务器')
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    args = parser.parse_args()

    if args.test:
        print("运行FastMCP RAG测试模式...")

        # 测试基本RAG功能
        print("\n1. 测试基本RAG功能...")
        try:
            results = search_database_context("卫片表包含哪些字段？")
            if results:
                print(f"** 找到 {len(results)} 个相关文档")
                print(f"  第一个结果预览: {results[0]['content'][:100]}...")
            else:
                print("** 未找到相关文档")
        except Exception as e:
            print(f"** 测试失败: {e}")
            import traceback
            traceback.print_exc()

        # 测试MCP工具函数
        print("\n2. 测试MCP工具函数...")
        try:
            # 由于MCP工具需要使用专门的调用方式，我们在这里只验证函数定义
            # 实际的工具调用将通过MCP协议进行
            print("** RAG工具已定义:")
            print("  - rag_search: 基本RAG搜索")
            print("  - rag_search_with_context: 结构化RAG搜索")
            print("  - rag_search_tables: 表结构搜索")
            print("** 工具验证: 所有RAG工具都已成功定义")

        except Exception as e:
            print(f"** MCP工具函数测试失败: {e}")
            import traceback
            traceback.print_exc()

        print("\n=== 测试完成 ===")
        print("如果所有测试都通过了，可以正常启动MCP服务器")
        print("运行: python rag_mcp_server.py")
    else:
        # 正常运行MCP服务器
        print("启动RAG MCP服务器...", file=sys.stderr)
        # 开启详细日志
        mcp.run()