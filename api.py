from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import asyncio
import uvicorn
from typing import Optional
import os
import sys
import time
import logging
from fastapi.middleware.cors import CORSMiddleware

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入agent功能
import agent
from rag_utils import search_database_context_with_scores, search_database_context_with_reranking


app = FastAPI(
    title="Data Analysis API",
    description="API for the intelligent ReAct agent that answers questions about data",
    version="1.0.0"
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该更具体地指定允许的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加中间件来记录请求处理时间
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

class QuestionRequest(BaseModel):
    question: str


class RagAccuracyRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str
    retry_count: int
    tool_calls: list
    tool_responses: list
    sql_queries: list


class RagAccuracyResponse(BaseModel):
    query: str
    results: list

class HealthResponse(BaseModel):
    status: str
    message: str

# 全局变量存储agent实例
agent_app = None

# 全局变量存储RAG MCP服务器进程
rag_mcp_process = None


@app.on_event("startup")
async def startup_event():
    """初始化agent应用"""
    global agent_app, rag_mcp_process
    logger.info("正在初始化agent应用...")
    try:

        # 启动RAG MCP服务器
        import subprocess
        import sys
        import os

        # 获取项目根目录
        project_root = os.path.dirname(os.path.abspath(__file__))
        rag_mcp_server_path = os.path.join(project_root, "rag_mcp_server.py")

        if os.path.exists(rag_mcp_server_path):
            logger.info("正在启动RAG MCP服务器...")
            # 启动RAG MCP服务器作为子进程
            rag_mcp_process = subprocess.Popen([
                sys.executable, rag_mcp_server_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"RAG MCP服务器已启动，进程ID: {rag_mcp_process.pid}")

            # 等待服务器启动
            import time
            time.sleep(2)
        else:
            logger.warning("RAG MCP服务器文件不存在")

        # 初始化agent
        agent_app = await agent.initialize_agent()
        if agent_app:
            logger.info("Agent应用初始化完成")
        else:
            logger.warning("Agent应用初始化失败")
    except Exception as e:
        logger.error(f"初始化agent失败: {e}")
        import traceback
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown_event():
    """关闭agent应用"""
    global rag_mcp_process
    logger.info("正在关闭应用...")
    try:
        # 关闭RAG MCP服务器
        if rag_mcp_process and rag_mcp_process.poll() is None:
            logger.info("正在关闭RAG MCP服务器...")
            rag_mcp_process.terminate()
            try:
                rag_mcp_process.wait(timeout=5)
                logger.info("RAG MCP服务器已关闭")
            except subprocess.TimeoutExpired:
                rag_mcp_process.kill()
                logger.info("RAG MCP服务器已被强制关闭")
    except Exception as e:
        logger.error(f"关闭应用时发生错误: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    status = "ok" if agent_app else "error"
    message = "Agent is ready" if agent_app else "Agent initialization failed"
    logger.info(f"健康检查: {status} - {message}")
    return HealthResponse(status=status, message=message)

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """接收用户问题并返回答案的端点"""
    logger.info(f"收到问题: {request.question}")

    if not agent_app:
        logger.error("Agent未初始化")
        raise HTTPException(status_code=500, detail="Agent not initialized")


    try:
        # 调用agent获取答案
        result = await agent.ask_question(agent_app, request.question)

        if result is None:
            raise HTTPException(status_code=500, detail="处理问题时发生错误")

        logger.info(f"问题处理完成，重试次数: {result['retry_count']}")


        return AnswerResponse(
            answer=result["answer"],
            retry_count=result["retry_count"],
            tool_calls=result["tool_calls"],
            tool_responses=result["tool_responses"],
            sql_queries=result["sql_queries"]
        )
    except Exception as e:
        logger.error(f"处理问题时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理问题时发生错误: {str(e)}")


@app.post("/rag_accuracy", response_model=RagAccuracyResponse)
async def test_rag_accuracy(request: RagAccuracyRequest):
    """测试RAG准确度，返回查询在向量库中检索到的前5个最相似的分块和重排序分数"""
    logger.info(f"收到RAG准确度测试请求: {request.query}")


    try:
        # 使用带重排序的搜索函数
        results = search_database_context_with_reranking(request.query, k=5)

        # 格式化结果
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "content": result["content"],
                "metadata": result["metadata"],
                "cosine_similarity": round(result.get("reranking_score", 0.0), 4)
            })

        logger.info(f"RAG准确度测试完成，找到 {len(formatted_results)} 个结果")


        return RagAccuracyResponse(
            query=request.query,
            results=formatted_results
        )
    except Exception as e:
        logger.error(f"RAG准确度测试时发生错误: {str(e)}")
        # 更新Langfuse跟踪错误状态（如果可用）
        if LANGFUSE_AVAILABLE and langfuse and trace_id:
            try:
                langfuse.trace(id=trace_id).update(
                    output={"error": str(e)}
                )
                langfuse.flush()
            except Exception as e:
                logger.warning(f"更新Langfuse跟踪错误状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"RAG准确度测试时发生错误: {str(e)}")

@app.get("/")
async def root():
    """根路径，提供API信息"""
    return {
        "message": "Data Analysis API",
        "description": "API for the intelligent ReAct agent that answers questions about data",
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "rag_accuracy": "/rag_accuracy"
        }
    }

@app.get("/graph")
async def get_agent_graph():
    """获取agent的LangGraph流程图"""
    global agent_app
    if not agent_app:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # 生成流程图的Mermaid代码
        mermaid_code = agent_app.get_graph().draw_mermaid()
        
        # 生成流程图的PNG文件
        png_data = agent_app.get_graph().draw_mermaid_png()
        
        # 保存PNG文件
        with open("langgraph_flow.png", "wb") as f:
            f.write(png_data)
        
        return {
            "mermaid_code": mermaid_code,
            "png_path": "langgraph_flow.png"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成流程图时发生错误: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)