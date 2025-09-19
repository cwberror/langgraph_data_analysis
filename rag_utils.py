# -*- coding: utf-8 -*-
"""RAG工具模块，用于处理数据库说明文件并提供检索增强功能"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    JSONLoader,
    UnstructuredExcelLoader,
)
from langchain.embeddings.base import Embeddings
from pydantic.v1 import BaseModel
import json
import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加载环境变量
load_dotenv()

class CustomEmbeddings(Embeddings, BaseModel):
    """自定义嵌入模型，使用BAAI/bge-m3模型直接生成嵌入向量"""

    class Config:
        # 允许额外的字段，这样我们可以添加model属性
        extra = "allow"
        # 允许任意类型的字段
        arbitrary_types_allowed = True

    def __init__(self):
        super().__init__()
        # 初始化BGE-M3模型
        try:
            # 检查是否有可用的GPU
            import torch
            if torch.cuda.is_available():
                print("检测到GPU，将在GPU上加载模型")
                self.model = SentenceTransformer('BAAI/bge-m3', device='cuda')
            else:
                print("未检测到GPU，将在CPU上加载模型")
                self.model = SentenceTransformer('BAAI/bge-m3')
            print("成功加载BAAI/bge-m3嵌入模型")
        except Exception as e:
            print(f"加载BAAI/bge-m3模型失败: {e}")
            # 如果模型加载失败，使用备用方案
            self.model = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入向量"""
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, normalize_embeddings=True)
                # 确保返回的是列表格式
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.tolist()
                elif isinstance(embeddings, list) and isinstance(embeddings[0], torch.Tensor):
                    embeddings = [emb.tolist() for emb in embeddings]
                return embeddings
            except Exception as e:
                print(f"使用BGE-M3模型生成文档嵌入时出错: {e}")
                # 回退到原始API方法
                return [self._fallback_embed_query(text) for text in texts]
        else:
            # 如果模型未加载，回退到原始API方法
            return [self._fallback_embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """为单个查询生成嵌入向量"""
        if self.model is not None:
            try:
                embedding = self.model.encode(text, normalize_embeddings=True)
                # 确保返回的是列表格式
                if isinstance(embedding, torch.Tensor):
                    return embedding.tolist()
                return embedding
            except Exception as e:
                print(f"使用BGE-M3模型生成查询嵌入时出错: {e}")
                # 回退到原始API方法
                return self._fallback_embed_query(text)
        else:
            # 如果模型未加载，回退到原始API方法
            return self._fallback_embed_query(text)

    def _fallback_embed_query(self, text: str) -> List[float]:
        """回退方法：调用本地embedding API"""
        api_url = os.getenv("EMBEDDING_API_URL", "http://127.0.0.1:8003")
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-bge-m3")

        try:
            print(f"调用本地embedding API: {api_url}, 模型: {model_name}")

            # 获取本地embedding服务的响应
            response = requests.post(
                f"{api_url}/v1/embeddings",
                json={
                    "input": text,
                    "model": model_name
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()

                # 尝试多种可能的响应格式
                embedding = None

                # BGE-M3/OpenAI格式的响应
                if "data" in result and result["data"] and isinstance(result["data"], list):
                    if isinstance(result["data"][0], dict) and "embedding" in result["data"][0]:
                        embedding = result["data"][0]["embedding"]
                    elif isinstance(result["data"][0], list):
                        embedding = result["data"][0]

                # 其他格式
                if not embedding:
                    for key in ["embedding", "embeddings", "vector"]:
                        if key in result and isinstance(result[key], list):
                            embedding = result[key]
                            break

                if embedding and isinstance(embedding, list) and len(embedding) > 0:
                    print(f"成功获取embedding，维度: {len(embedding)}")
                    return embedding
                else:
                    print(f"未能从响应中提取embedding，响应内容: {result}")

            print("尝试备用端点 /embeddings")
            try:
                # 尝试备用端点 /embeddings
                alt_response = requests.post(
                    f"{api_url}/embeddings",
                    json={
                        "input": text,
                        "model": model_name
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )

                if alt_response.status_code == 200:
                    alt_result = alt_response.json()

                    # 尝试多种备用格式
                    embedding = None

                    # 备用端点格式
                    if "data" in alt_result and alt_result["data"] and isinstance(alt_result["data"], list):
                        if isinstance(alt_result["data"][0], dict) and "embedding" in alt_result["data"][0]:
                            embedding = alt_result["data"][0]["embedding"]
                        elif isinstance(alt_result["data"][0], list):
                            embedding = alt_result["data"][0]

                    # 其他格式
                    if not embedding:
                        for key in ["embedding", "embeddings", "vector"]:
                            if key in alt_result and isinstance(alt_result[key], list):
                                embedding = alt_result[key]
                                break

                    if embedding and isinstance(embedding, list) and len(embedding) > 0:
                        print(f"备用端点成功获取embedding，维度: {len(embedding)}")
                        return embedding
                    else:
                        print(f"备用端点未能获取有效embedding")

            except Exception as alt_e:
                print(f"备用端点调用也失败: {alt_e}")

            # 所有API调用都失败，使用fallback
            print("使用fallback模式生成embedding")
            return self._fallback_embedding(text)

        except Exception as e:
            print(f"本地embeddings API调用异常: {e}, 先尝试备用端点")
            # 在异常处理中也尝试备用端点
            try:
                alt_response = requests.post(
                    f"{api_url}/embeddings",
                    json={
                        "input": text,
                        "model": model_name
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )

                if alt_response.status_code == 200:
                    alt_result = alt_response.json()

                    # 尝试备用端点格式
                    for key in ["embedding", "embeddings", "vector", "data"]:
                        if key in alt_result and isinstance(alt_result[key], list):
                            if key == "data" and len(alt_result[key]) > 0:
                                if isinstance(alt_result["data"][0], dict) and "embedding" in alt_result["data"][0]:
                                    embedding = alt_result["data"][0]["embedding"]
                                    print(f"备用端点异常处理后成功获取embedding，维度: {len(embedding)}")
                                    return embedding
                            else:
                                embedding = alt_result[key]
                                if isinstance(embedding, list) and len(embedding) > 0:
                                    print(f"备用端点异常处理后成功获取embedding，维度: {len(embedding)}")
                                    return embedding

            except Exception as alt_e:
                print(f"备用端点异常处理也失败: {alt_e}")

            print("使用fallback模式生成embedding")
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> List[float]:
        """回退嵌入方法，生成768维固定向量"""
        import hashlib
        import numpy as np

        # 使用多个哈希函数和文本特征生成固定维度向量
        vector = []

        # 基础文本特征
        text_length = len(text)
        char_sum = sum(ord(c) for c in text)
        char_mean = char_sum / text_length if text_length > 0 else 0

        # 使用不同的哈希算法
        for idx, algorithm in enumerate(['md5', 'sha1', 'sha256']):
            hash_obj = hashlib.new(algorithm)
            hash_obj.update(f"{text}_{algorithm}".encode())
            hash_digest = hash_obj.hexdigest()

            # 从哈希中提取数值特征
            for i in range(0, len(hash_digest), 4):
                if len(vector) >= 768:
                    break
                hex_chunk = hash_digest[i:i + 4]
                if len(hex_chunk) >= 2:
                    num = int(hex_chunk, 16)
                    # 归一化到0-1范围
                    normalized = (num % 10000) / 10000.0
                    vector.append(normalized)

        # 补充到768维
        while len(vector) < 768:
            vector.append(0.0)

        # 应用简单的线性变换
        vector = np.array(vector)

        # 使用文本特征进行微调
        feature_factor = (text_length % 100) / 100.0
        vector = vector * 0.9 + feature_factor * 0.1

        return vector.tolist()


class RAGProcessor:
    """RAG处理器，用于处理数据库说明文件并提供检索功能"""

    class Config:
        # 允许额外的字段，这样我们可以添加模型属性
        extra = "allow"
        # 允许任意类型的字段
        arbitrary_types_allowed = True

    def __init__(self, persist_directory: str = "./chroma_db_v2"):
        """初始化RAG处理器"""
        # 当前只使用自定义embeddings，不再保留OpenAI embeddings
        self.embeddings = CustomEmbeddings()

        self.persist_directory = persist_directory
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        # 初始化JSON分割器
        self.json_splitter = RecursiveJsonSplitter(max_chunk_size=1000)
        # 初始化重排序模型
        self._init_reranker()
        # 初始化向量存储
        self.vectorstore = None
        self._init_vectorstore()

    def _init_reranker(self):
        """初始化重排序模型"""
        try:
            # 检查是否有可用的GPU
            import torch
            if torch.cuda.is_available():
                print("检测到GPU，将在GPU上加载重排序模型")
                self.reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
                self.reranker_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3').to('cuda')
            else:
                print("未检测到GPU，将在CPU上加载重排序模型")
                self.reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
                self.reranker_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
            self.reranker_model.eval()
            print("成功加载BAAI/bge-reranker-v2-m3重排序模型")
        except Exception as e:
            print(f"加载BAAI/bge-reranker-v2-m3模型失败: {e}")
            self.reranker_tokenizer = None
            self.reranker_model = None
    
    def _init_vectorstore(self):
        """初始化向量存储"""
        try:
            # 尝试加载现有的向量存储
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"已初始化向量存储，路径: {self.persist_directory}")
        except Exception as e:
            print(f"初始化向量存储失败: {e}")
            # 创建新的向量存储
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            try:
                # 显示向量存储相关信息
                self.vectorstore.persist()
                print(f"已创建新的向量存储: {self.persist_directory}")
            except:
                print("创建向量存储时遇到错误")
    
    def _get_loader(self, file_path: str):
        """根据文件扩展名获取相应的文档加载器"""
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.csv':
            return CSVLoader(file_path=file_path)
        elif ext == '.pdf':
            return PyPDFLoader(file_path=file_path)
        elif ext == '.json':
            # 对于表字段说明.json文件，使用自定义处理方法
            if "表字段说明" in file_path:
                return self._create_json_field_documents(file_path)
            return JSONLoader(
                file_path=file_path,
                jq_schema=".",
                text_content=False
            )
        elif ext in ['.xlsx', '.xls']:
            return UnstructuredExcelLoader(file_path=file_path)
        elif ext in ['.yaml', '.yml']:
            # YAML文件使用JSONLoader处理
            return JSONLoader(
                file_path=file_path,
                jq_schema=".",
                text_content=False
            )
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def _create_json_field_documents(self, file_path: str):
        """为表字段说明.json文件创建文档，使用RecursiveJsonSplitter进行分块"""
        import json
        from langchain_core.documents import Document

        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 使用RecursiveJsonSplitter分割JSON数据
            json_docs = self.json_splitter.split_json(data, convert_lists=True)

            documents = []
            # 为每个分割后的文档添加元数据
            for i, doc in enumerate(json_docs):
                # 创建Document对象
                document = Document(
                    page_content=doc,
                    metadata={
                        "source": file_path,
                        "source_file": file_path,
                        "file_type": ".json",
                        "chunk_id": i
                    }
                )
                documents.append(document)

            print(f"成功从 {file_path} 创建了 {len(documents)} 个文档块")
            return documents

        except Exception as e:
            print(f"处理JSON文件 {file_path} 时出错: {e}")
            return []
    
    async def process_file(self, file_path: str) -> List[Document]:
        """处理单个文件并返回文档列表"""
        try:
            # 获取对应的文档加载器
            loader = self._get_loader(file_path)

            # 如果loader是一个文档列表（来自_create_json_field_documents），直接使用
            if isinstance(loader, list):
                documents = loader
            else:
                # 否则使用loader加载文档
                documents = loader.load()

            # 为所有文档添加元数据
            for doc in documents:
                # 确保不覆盖已有的元数据
                if "source_file" not in doc.metadata:
                    doc.metadata["source_file"] = file_path
                if "file_type" not in doc.metadata:
                    doc.metadata["file_type"] = os.path.splitext(file_path)[1]
                # 添加source字段以保持一致性
                if "source" not in doc.metadata:
                    doc.metadata["source"] = file_path

            print(f"成功加载文件: {file_path}, 文档数量: {len(documents)}")
            return documents
        except PermissionError:
            print(f"权限被拒绝，跳过文件: {file_path}")
            return []
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return []
    
    async def process_directory(self, directory_path: str) -> List[Document]:
        """处理目录下的所有支持文件"""
        all_documents = []
        supported_extensions = {'.csv', '.pdf', '.json', '.xlsx', '.xls', '.yaml', '.yml'}
        
        # 遍历目录下的所有文件
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # 过滤临时文件和其他不需要的文件
                if file.startswith('~$') or file.startswith('.') or file.startswith('__'):
                    continue
                    
                _, ext = os.path.splitext(file.lower())
                
                # 只处理支持的文件格式
                if ext in supported_extensions:
                    try:
                        documents = await self.process_file(file_path)
                        all_documents.extend(documents)
                    except PermissionError:
                        print(f"权限被拒绝，跳过文件: {file_path}")
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        return self.text_splitter.split_documents(documents)
    
    async def add_documents_to_vectorstore(self, documents: List[Document]):
        """将文档添加到向量存储"""
        if not documents:
            print("没有文档需要添加到向量存储")
            return

        try:
            # 分割文档
            split_docs = self.split_documents(documents)
            print(f"文档分割完成，总共有 {len(split_docs)} 个文档块")

            # 添加到向量存储
            self.vectorstore.add_documents(split_docs)
            self.vectorstore.persist()

            print(f"成功添加 {len(split_docs)} 个文档块到向量存储")
        except Exception as e:
            print(f"添加文档到向量存储时出错: {e}")

    def clear_vectorstore(self):
        """清除向量存储中的所有文档"""
        try:
            # 获取所有文档的IDs
            collection = self.vectorstore._collection
            all_ids = collection.get()["ids"]

            if all_ids:
                # 删除所有文档
                collection.delete(ids=all_ids)
                self.vectorstore.persist()
                print(f"成功清除向量存储中的 {len(all_ids)} 个文档")
            else:
                print("向量存储中没有文档需要清除")
        except Exception as e:
            print(f"清除向量存储时出错: {e}")
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """搜索相关文档"""
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"搜索时出错: {e}")
            return []

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """使用BGE重排序模型对文档进行重排序"""
        if not self.reranker_model or not self.reranker_tokenizer:
            print("重排序模型未加载，跳过重排序")
            return documents

        try:
            # 准备重排序的输入
            sentence_pairs = [[query, doc.page_content] for doc in documents]

            # 检查是否有GPU可用
            device = 'cuda' if torch.cuda.is_available() and next(self.reranker_model.parameters()).is_cuda else 'cpu'

            # 使用tokenizer处理输入，并将输入移动到相应设备
            with torch.no_grad():
                inputs = self.reranker_tokenizer(sentence_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                # 将输入移动到模型所在的设备
                inputs = {k: v.to(device) for k, v in inputs.items()}
                scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = torch.sigmoid(scores)

            # 将分数和文档配对并排序
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # 返回排序后的文档
            return [doc for doc, score in scored_docs]
        except Exception as e:
            print(f"重排序时出错: {e}")
            return documents

    def search_with_reranking(self, query: str, k: int = 4) -> List[Document]:
        """搜索相关文档并进行重排序"""
        try:
            # 首先进行向量搜索，获取更多候选文档
            candidate_docs = self.vectorstore.similarity_search(query, k=k*2)

            # 如果有重排序模型，则进行重排序
            if self.reranker_model and self.reranker_tokenizer:
                reranked_docs = self._rerank_documents(query, candidate_docs)
                # 返回前k个文档
                return reranked_docs[:k]
            else:
                # 如果没有重排序模型，直接返回前k个文档
                return candidate_docs[:k]
        except Exception as e:
            print(f"带重排序的搜索时出错: {e}")
            return []
    
    async def process_and_store_directory(self, directory_path: str):
        """处理目录并存储到向量数据库"""
        print(f"开始处理目录: {directory_path}")
        
        # 处理目录中的所有文件
        documents = await self.process_directory(directory_path)
        
        if not documents:
            print("未找到任何支持的文件")
            return
        
        print(f"总共处理了 {len(documents)} 个文档")
        
        # 添加到向量存储
        await self.add_documents_to_vectorstore(documents)
        
        print("文件处理和存储完成")

# 创建全局RAG处理器实例
rag_processor = RAGProcessor()

async def initialize_rag(directory_path: str = "./database_docs", clear_first: bool = True):
    """初始化RAG系统"""
    if clear_first:
        print("清除现有的向量存储...")
        rag_processor.clear_vectorstore()

    await rag_processor.process_and_store_directory(directory_path)

def search_database_context(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """搜索数据库上下文"""
    documents = rag_processor.search(query, k=k)

    results = []
    for doc in documents:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    return results


def search_database_context_with_scores(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """搜索数据库上下文并返回相似度分数"""
    try:
        # 使用带重排序的搜索方法获取文档
        docs = rag_processor.search_with_reranking(query, k=k)

        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": 0.0  # 重排序后不再提供具体的相似度分数
            })

        return results
    except Exception as e:
        print(f"带分数的搜索时出错: {e}")
        return []

def search_database_context_with_reranking(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """搜索数据库上下文并使用重排序，返回重排序分数"""
    try:
        # 首先进行向量搜索，获取更多候选文档
        candidate_docs = rag_processor.vectorstore.similarity_search(query, k=k*2)

        # 如果有重排序模型，则进行重排序
        if rag_processor.reranker_model and rag_processor.reranker_tokenizer:
            reranked_docs = rag_processor._rerank_documents(query, candidate_docs)
            # 返回前k个文档和它们的重排序分数
            results = []
            for doc in reranked_docs[:k]:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "reranking_score": 0.0  # 简化处理，实际应用中可以返回具体的分数
                })
            return results
        else:
            # 如果没有重排序模型，直接返回前k个文档
            results = []
            for doc in candidate_docs[:k]:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "reranking_score": 0.0
                })
            return results
    except Exception as e:
        print(f"带重排序的搜索时出错: {e}")
        return []

# RAG工具函数，可作为agent工具使用
def rag_search_tool(query: str) -> str:
    """RAG搜索工具函数"""
    try:
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
        return f"RAG搜索工具执行出错: {e}"

def test_embeddings():
    """测试自定义embedding API"""
    try:
        print("正在测试自定义embedding模型...")
        
        # 创建自定义embeddings
        embeddings = CustomEmbeddings()
        
        # 测试单个嵌入
        test_text = "这是一个测试文本"
        single_embedding = embeddings.embed_query(test_text)
        print(f"单个文本嵌入成功，向量维度: {len(single_embedding)}")
        
        # 测试批量嵌入
        test_texts = ["测试文本1", "测试文本2", "测试文本3"]
        batch_embeddings = embeddings.embed_documents(test_texts)
        print(f"批量嵌入成功，文档数量: {len(batch_embeddings)}")
        print(f"每个文档向量维度: {len(batch_embeddings[0]) if batch_embeddings else 0}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    # 测试自定义embeddings
    test_result = test_embeddings()
    
    if test_result:
        print("\n开始RAG系统测试...")
        asyncio.run(initialize_rag("./database_docs"))
        
        # 测试搜索功能
        query = "用户表结构说明"
        results = search_database_context(query)
        print(f"查询 '{query}' 的结果:")
        for result in results:
            print(f"内容: {result['content'][:100]}...")
            print(f"元数据: {result['metadata']}")
            print("-" * 50)
    else:
        print("嵌入模型测试失败，请检查配置")