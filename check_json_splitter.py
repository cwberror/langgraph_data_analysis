#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查RecursiveJsonSplitter的正确用法"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from langchain_text_splitters import RecursiveJsonSplitter
    import inspect

    # 获取RecursiveJsonSplitter的构造函数签名
    sig = inspect.signature(RecursiveJsonSplitter.__init__)
    print("RecursiveJsonSplitter.__init__ signature:")
    print(sig)

    # 尝试创建实例
    try:
        splitter = RecursiveJsonSplitter()
        print("成功创建RecursiveJsonSplitter实例")
        print("RecursiveJsonSplitter methods:", [method for method in dir(splitter) if not method.startswith('_')])
    except Exception as e:
        print(f"创建RecursiveJsonSplitter实例时出错: {e}")

except Exception as e:
    print(f"导入RecursiveJsonSplitter时出错: {e}")