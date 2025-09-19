#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示增强RAG功能的脚本
展示如何查询表字段信息
"""

import asyncio
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_utils import RAGProcessor

async def demo_enhanced_rag():
    """演示增强的RAG功能"""
    print("演示增强的RAG功能")
    print("=" * 50)
    
    # 创建RAG处理器实例
    rag_processor = RAGProcessor()
    
    # 测试查询
    queries = [
        "卫片表包含哪些字段？",
        "地级市字段有哪些可选值？",
        "实测面积字段的单位是什么？",
        "违法类型字段的字典值有哪些？",
        "违法用地属于哪个字段的值？"
    ]
    
    print("测试查询:")
    for query in queries:
        print(f"\n查询: {query}")
        results = rag_processor.search(query, k=1)
        
        if results:
            doc = results[0]
            print(f"结果: {doc.page_content[:300]}...")
        else:
            print("未找到相关文档")
    
    print("\n演示完成!")

if __name__ == "__main__":
    asyncio.run(demo_enhanced_rag())