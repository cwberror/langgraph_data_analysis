# 数据库说明文件处理和RAG集成使用说明

## 项目结构
- `rag_utils.py`: RAG工具模块，负责处理数据库说明文件并提供检索增强功能
- `main.py`: 主程序，已集成RAG功能到现有agent工作流中

## 支持的文件格式
- CSV文件 (.csv)
- PDF文件 (.pdf)
- JSON文件 (.json)
- Excel文件 (.xlsx, .xls)
- YAML文件 (.yaml, .yml)

## 特殊JSON处理 - 表字段说明文件

项目现在支持专门处理"表字段说明.json"文件，该文件包含数据库表结构信息。

### 处理流程
1. 读取"表字段说明.json"文件
2. 按表名分组字段信息
3. 为每个表创建结构化文档
4. 将文档嵌入到Chroma向量数据库中
5. 支持基于自然语言查询的字段信息检索

### 优势
- 提供更精确的表结构信息
- 支持字段描述和字典值查询
- 优化了相关性排序

## 使用方法

### 1. 准备数据库说明文件
将所有数据库说明文件放置在 `database_docs` 目录下（可配置）。

### 2. 初始化RAG系统
运行以下命令初始化RAG系统：
```bash
python rag_utils.py
```

### 3. 运行主程序
```bash
python main.py
```

## RAG功能说明

### 自动文件处理
RAG系统会自动识别并处理以下格式的文件：
- CSV: 使用CSVLoader处理
- PDF: 使用PyPDFLoader处理
- JSON: 使用专门的处理器处理"表字段说明.json"文件，其他JSON文件使用JSONLoader处理
- Excel: 使用UnstructuredExcelLoader处理
- YAML: 使用JSONLoader处理

### 文档向量化
- 使用RecursiveCharacterTextSplitter将文档分割成适当大小的块
- 使用自定义Embeddings类连接到本地embedding API
- 存储在Chroma向量数据库中

### 智能检索
- 在agent重试时自动检索相关数据库文档
- 将检索到的上下文信息添加到系统提示中
- 帮助agent生成更准确的答案
- 特别优化了对表字段信息的检索

## 查询示例

以下是一些可以使用的查询示例：

1. "卫片表有哪些字段"
2. "地级市字段的字典值有哪些"
3. "实测面积字段的描述是什么"
4. "违法类型字段的可能值"

## 自定义配置

### 修改数据库说明文件目录
在 `rag_utils.py` 中修改 `initialize_rag` 函数的 `directory_path` 参数。

### 调整文档分割参数
在 `RAGProcessor` 类中修改 `RecursiveCharacterTextSplitter` 的参数：
- `chunk_size`: 文档块大小
- `chunk_overlap`: 文档块重叠大小

### 调整检索参数
在 `search_database_context` 函数中修改 `k` 参数来控制返回的文档数量。