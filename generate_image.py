# -*- coding: utf-8 -*-
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import base64

# 加载环境变量
load_dotenv()

# 定义agent状态
class AgentState:
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_question: str
    retry_count: int
    max_retries: int

async def create_agent_graph_for_visualization():
    """创建用于可视化的agent图结构"""
    try:
        # 初始化MCP客户端
        client = MultiServerMCPClient(
            {
                "mcp_zxxc": {
                    "transport": "sse",
                    "url": "http://localhost:8006/mcp"
                },
            }
        )
        
        # 获取工具
        tools = await client.get_tools()
        print(f"可用工具: {[tool.name for tool in tools]}")
        
        # 创建图
        graph = StateGraph(AgentState)
        
        # 添加节点
        graph.add_node("agent", lambda state: state)  # 简化的agent节点
        graph.add_node("retry", lambda state: state)  # 重试节点
        
        # 创建工具节点
        if tools:
            tool_node = ToolNode(tools)
            graph.add_node("tools", tool_node)
        
        # 添加评估节点
        graph.add_node("evaluate", lambda state: state)
        
        # 添加边
        graph.add_edge(START, "agent")
        
        # 添加条件边
        if tools:
            graph.add_conditional_edges(
                "agent",
                lambda state: "continue" if tools else "end",
                {
                    'continue': 'tools',
                    'end': 'evaluate'
                }
            )
            graph.add_edge("tools", "agent")
        else:
            graph.add_edge("agent", "evaluate")
        
        graph.add_conditional_edges(
            "evaluate",
            lambda state: "end",
            {
                'end': END,
                'retry': 'retry'
            }
        )
        graph.add_edge("retry", "agent")
        
        return graph.compile()
    except Exception as e:
        print(f"创建图结构失败: {e}")
        return None

def generate_mermaid_graph(graph, output_path="langgraph_flow.md"):
    """
    生成Mermaid格式的流程图
    
    Args:
        graph: 编译后的LangGraph图对象
        output_path (str): 输出文件路径
    
    Returns:
        bool: 是否成功生成图表
    """
    try:
        # 获取图的Mermaid表示
        mermaid_code = graph.get_graph().draw_mermaid()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# LangGraph 流程图\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_code)
            f.write("\n```\n")
            f.write("\n\n## 说明\n")
            f.write("- START: 流程开始\n")
            f.write("- agent: AI代理节点\n")
            f.write("- tools: 工具调用节点\n")
            f.write("- evaluate: 答案评估节点\n")
            f.write("- retry: 重试节点\n")
            f.write("- END: 流程结束\n")
        
        print(f"Mermaid流程图已保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"生成Mermaid流程图失败: {e}")
        return False

def save_graph_image(graph, output_path="langgraph_flow.png"):
    """
    保存LangGraph流程图为PNG图片
    
    Args:
        graph: 编译后的LangGraph图对象
        output_path (str): 输出图片路径
    
    Returns:
        bool: 是否成功生成图片
    """
    try:
        # 生成PNG图片数据
        png_data = graph.get_graph().draw_mermaid_png()
        
        # 保存为文件
        with open(output_path, 'wb') as f:
            f.write(png_data)
        
        print(f"流程图已保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"生成流程图图片失败: {e}")
        return False

async def main():
    """主函数"""
    print("开始生成LangGraph流程图...")
    
    # 创建图结构
    graph = await create_agent_graph_for_visualization()
    
    if graph:
        # 生成Mermaid格式的流程图
        success_mermaid = generate_mermaid_graph(graph, "langgraph_flow.md")
        
        # 生成PNG图片格式的流程图
        success_img = save_graph_image(graph, "langgraph_flow.png")
        
        if success_mermaid or success_img:
            print("流程图生成完成!")
        else:
            print("流程图生成失败!")
    else:
        print("无法创建图结构，流程图生成失败!")

if __name__ == "__main__":
    asyncio.run(main())