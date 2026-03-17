# 高中数学 RAG 数据清洗与结构化流水线开发需求 (Prompt for AI Coder)

## 1. 项目背景
我正在开发一个针对高中数学的 RAG 系统。目前第一阶段（本地 OCR）已经完成：试卷 PDF 已经被 MinerU 等本地工具转化为了包含 LaTeX 公式的标准 Markdown 文件。
现在的目标是：读取这些长篇 Markdown 文件，通过**带重叠的滑动窗口**按页/按块截取文本，利用**异步大模型 (智谱 GLM)** 进行语义提取，最终输出带有严格验证的结构化 **JSONL** 文件。

为保证代码的健壮性和可维护性，请将代码拆分为多个子文件，并添加详细、规整的中文中文注释。

---

## 2. 架构设计与文件划分
请生成以下 5 个 Python 文件：

1. `schemas.py` (数据模型与硬校验层)
2. `prompts.py` (提示词与 Few-Shot 管理)
3. `llm_parser.py` (大模型异步调用与解析引擎)
4. `pipeline.py` (状态机、文件读写与滑动窗口逻辑)
5. `main.py` (入口与统筹执行)

---

## 3. 各模块详细功能要求

### 📄 文件 1: `schemas.py`
**职责：使用 Pydantic 定义目标 JSON 结构，并实现脏数据拦截。**
- 定义 `MathQuestion` 模型，包含以下字段：
  - `question_id`: 字符串（基于时间戳或 UUID 生成）。
  - `preamble`: 字符串（公共题干背景，解决孤儿上下文问题，可为空）。
  - `question_sys`: 字符串（具体题目，必须包含 LaTeX 原文）。
  - `answer_sys`: 字符串（详细解析）。
  - `knowledge_points`: 字符串列表（知识点标签）。
- **硬校验逻辑 (Validators)：**
  - 检查 `question_sys` 长度，如果少于 10 个字符则抛出 ValueError（拦截噪音数据）。
  - 检查字段中是否包含“绝密★”、“扫描全能王”等常见水印词，若有则抛出异常或清洗掉。
- 定义 `ExtractionResult` 模型，包含一个 `questions: List[MathQuestion]` 列表，用于大模型 JSON Mode 的根节点接收。

### 📄 文件 2: `prompts.py`
**职责：存放系统提示词，防篡改与去噪。**
- 编写 `MATH_EXTRACTION_PROMPT`。必须包含以下严厉的指令：
  1. **原样复制原则：** 必须逐字逐句从原文复制，绝对禁止篡改、意译 LaTeX 公式（如把 `$x^2$` 改成 `x的平方`）。
  2. **跨页与去重原则：** 给定文本可能包含上一页的截断尾部和本页的新内容。请自动拼接截断的题目，并根据题号去除重复提取的题目。
  3. **去噪原则：** 忽略所有页眉、页脚、页码、试卷标题及“密封线内不要答题”等无意义内容。
  4. **公共题干原则：** 如果是一道大题包含 (1)(2) 两个小问，请将它们拆分为两个独立的 JSON 对象，但必须将大题的公共题干提取到 `preamble` 字段中。
- 提供一段极简的 Few-Shot 示例（一小段 Markdown 对应一个 JSON 输出），包含在 Prompt 中。

### 📄 文件 3: `llm_parser.py`
**职责：封装智谱 GLM 的异步调用。**
- 引入 `zhipuai` 的异步客户端 `AsyncZhipuAI`。
- 实现类 `GLMParser`，初始化时读取环境变量 `ZHIPUAI_API_KEY`。
- 实现异步方法 `async def extract_questions(self, text_chunk: str) -> List[MathQuestion]:`
  - 使用智谱支持 JSON Mode 的模型（如 `glm-4-flash` 或 `glm-4-plus`），设置 `response_format={"type": "json_object"}`。
  - 将 `MATH_EXTRACTION_PROMPT` 作为 system 传入，`text_chunk` 作为 user 传入。
  - 解析返回的 JSON 字符串，并使用 `schemas.py` 中的 `ExtractionResult` 进行 Pydantic 校验。
  - 必须包含 `try-except` 逻辑，遇到 API 限流或 Pydantic 解析失败时，支持使用 `tenacity` 库进行最多 3 次的指数退避重试 (Retry)。

### 📄 文件 4: `pipeline.py`
**职责：文本分块、跨页补偿、状态机与 JSONL 保存。**
- 实现 `MarkdownProcessor` 类。
- **状态机与 JSONL 续传逻辑：**
  - 方法 `init_state(output_jsonl_path)`: 读取目标 jsonl 文件，统计目前已经成功处理了多少个 chunk（行数或自定义的状态记录），以便脚本崩溃后重新运行时跳过已处理的块。
  - 方法 `append_to_jsonl(questions: List[MathQuestion])`: 实时追加写入，确保安全。
- **滑动窗口切分逻辑 (MFCC 思想)：**
  - 方法 `chunk_markdown_with_overlap(markdown_text, chunk_size=2000, overlap=400)`: 
  - 按字符数（或按 Markdown 标题层级）对全篇长文本进行切分。
  - 必须保证后一个 chunk 的前段，包含前一个 chunk 的尾段（约 400 字符重叠），以解决题目跨页/跨块被腰斩的问题。
- **异步批处理：**
  - 方法 `async def run_pipeline(self, md_filepath, output_jsonl_path, parser: GLMParser)`
  - 使用 `asyncio.gather` 或 `asyncio.Semaphore` 控制并发量（例如最大并发设为 5），向大模型发送请求。

### 📄 文件 5: `main.py`
**职责：统筹调度，提供优雅的控制台输出。**
- 加载 `.env` 文件获取 API Key。
- 配置 `logging`，输出 INFO 级别的日志。
- 初始化 `GLMParser` 和 `MarkdownProcessor`。
- 指定输入的 `data/sample.md` 路径和输出的 `output/extracted.jsonl` 路径。
- 运行 `asyncio.run(processor.run_pipeline(...))`。

---

## 4. 额外编码规范要求
1. 全部代码使用 Python 3.10+ 标准，包含完善的 Type Hint (类型注解)。
2. 对于容易报错的地方（如 JSON 解析、网络请求），务必写好具体的 Exception 捕获并输出日志，**不要直接 pass**。
3. 请直接输出这 5 个文件的完整代码内容。