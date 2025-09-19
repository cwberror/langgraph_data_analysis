#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重新初始化RAG向量存储的脚本"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_utils import initialize_rag

async def main():
    """主函数"""
    print("开始重新初始化RAG向量存储...")
    try:
        await initialize_rag("./database_docs", clear_first=True)
        print("RAG向量存储重新初始化完成!")
    except Exception as e:
        print(f"重新初始化RAG向量存储时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())