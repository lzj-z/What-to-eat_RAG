"""
生成集成模块
"""

import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from LLM.init_llm import deepseekllm

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, model_name: str = "deepseek-chat", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        
        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
        """初始化大语言模型"""
        logger.info("正在初始化LLM: deepseekllm")

        self.llm = deepseekllm

        logger.info("LLM初始化完成")
    
    def generate_basic_answer(self, query: str, context_docs: List[Document],
                               graph_context: str = "") -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表
            graph_context: 图谱检索结果文本（可选）

        Returns:
            生成的回答
        """
        context = self._build_context(context_docs, graph_context=graph_context)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")

        # 使用LCEL构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def generate_step_by_step_answer(self, query: str, context_docs: List[Document],
                                      graph_context: str = "") -> str:
        """
        生成分步骤回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表
            graph_context: 图谱检索结果文本（可选）

        Returns:
            分步骤的详细回答
        """
        context = self._build_context(context_docs, graph_context=graph_context)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

- 根据实际内容灵活调整结构
- 不要强行填充无关内容
- 重点突出实用性和可操作性
- 不要回答食谱中没有的信息，如果食谱中没有检索到相关信息也不要擅自补充，就回答"抱歉，我的食谱没有这道菜哟，后期我再慢慢研究"

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询，不做任何修改）：
   - 包含具体菜品名称：如"宫保鸡丁怎么做"、"红烧肉的制作方法"
   - 包含具体食材名称：如"鸡翅"、"排骨"、"鲈鱼"、"虾"、"土豆"
   - 明确的制作询问：如"蛋炒饭需要什么食材"、"糖醋排骨的步骤"
   - 具体的烹饪技巧：如"如何炒菜不粘锅"、"怎样调制糖醋汁"
   - 包含食材+需求的组合：如"鸡翅怎么做"、"推荐鸡翅菜谱"、"鸡翅的做法"

2. **需要重写的查询**：
   - 过于宽泛且不含任何具体食材/菜品名：如"做菜"、"有什么好吃的"、"推荐个菜"
   - 纯分类词：如"川菜"、"素菜"、"简单的"
   - 口语化且无具体指向：如"想吃点什么"、"有饮品推荐吗"
   - **采购/持有食材意图**（需重写为"XX的做法"）：如"我买了XX"、"家里有XX"、"剩了些XX"
   - **多分类数量要求**（需重写保留所有约束）：如"两荤一素"、"一荤两素"需保留数量+荤素+忌口+偏好等全部约束

**核心原则：查询中出现的所有具体食材名和菜品名必须原封不动地保留在输出中，绝对不能丢弃、替换或泛化。**

重写原则：
- 必须保留原查询中所有具体食材/菜品关键词
- 采购意图（"买了/有/剩了"）→ 转换为"XX的做法推荐"
- 多分类数量要求 → 保留完整约束（荤素数量+食材偏好+忌口）
- 保持原意不变，可以适当补充烹饪术语
- 保持简洁性

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "推荐个菜" → "简单家常菜推荐"
- "川菜" → "经典川菜菜谱"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做"（保持原查询）
- "红烧肉需要什么食材" → "红烧肉需要什么食材"（保持原查询）
- "鸡翅" → "鸡翅的做法"（保留食材名，只补充意图）
- "排骨" → "排骨的做法"（保留食材名，只补充意图）
- "想吃鸡翅" → "鸡翅菜谱推荐"（保留食材名）
- "我买了黄鳝" → "黄鳝的做法推荐"（采购意图→做法推荐，保留食材名）
- "家里有半根黄瓜" → "黄瓜的做法推荐"（保留食材名）
- "今晚想吃两荤一素，喜欢包菜，不爱吃鱼" → "两荤一素菜谱推荐（含包菜，不含鱼类）"（保留完整约束）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response



    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
   例如：推荐几个素菜、有什么川菜、给我3个简单的菜

2. 'detail' - 用户想要具体的制作方法或详细信息
   例如：宫保鸡丁怎么做、制作步骤、需要什么食材

3. 'general' - 其他一般性问题
   例如：什么是川菜、制作技巧、营养价值

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        # 确保返回有效的路由类型
        if result in ['list', 'detail', 'general']:
            return result
        else:
            return 'general'  # 默认类型

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 由 LLM 根据相关性从候选文档中筛选并列举

        Args:
            query: 用户查询
            context_docs: 上下文文档列表（可能包含相关性较低的文档）

        Returns:
            列表式回答
        """
        if not context_docs:
            return "抱歉，没有找到相关的菜品信息。"

        # 构建候选菜品清单（只传菜名+简短描述，节省 token）
        candidates = []
        for doc in context_docs:
            dish_name  = doc.metadata.get('dish_name', '未知菜品')
            category   = doc.metadata.get('category', '')
            difficulty = doc.metadata.get('difficulty', '')
            # 取正文前80字作为简述
            brief = doc.page_content[:80].strip().replace('\n', ' ')
            meta = f"{category} | {difficulty}" if difficulty else category
            candidates.append(f"- {dish_name}（{meta}）：{brief}")
        candidate_text = "\n".join(candidates)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪推荐助手。以下是从食谱库中检索到的候选菜品，请根据用户的具体需求推荐菜品。

**候选菜品格式说明：**
每条候选菜品格式为：- 菜名（分类 | 难度）：简述
其中"难度"字段（如：简单、中等、困难、非常困难）是菜品的实际难度标签。

**推荐规则（严格遵守）：**
1. 仔细理解用户需求，包括数量要求（如"两荤一素"="2道荤菜+1道素菜"）、食材偏好、忌口、难度要求等。
2. 从候选菜品中筛选最符合的菜品，严格满足数量和分类要求。
3. 若用户有明确数量要求（如"两荤一素"），必须给出对应数量的推荐，不能多也不能少。
4. 若用户有忌口（如"不爱吃鱼"），必须排除含该食材的菜品。
5. 若用户有偏好食材（如"喜欢包菜"），优先推荐含该食材的菜品。
6. 若用户有难度要求（如"非常困难"），必须依据候选菜品的难度标签进行筛选，只推荐符合难度的菜品。
7. 若候选菜品完全无法满足需求，回答"抱歉，我的食谱库暂时没有完全符合要求的菜品，您可以换个方向试试～"。
8. 不要编造食谱库中没有的菜品。

**回答格式：**
- 一句话说明推荐理由或搭配逻辑
- 逐条列出推荐菜品（格式：菜名（分类 | 难度））
- 不需要详细做法

用户需求：{question}

候选菜品（来自食谱库）：
{candidates}

请给出推荐：""")

        chain = (
            {"question": RunnablePassthrough(), "candidates": lambda _: candidate_text}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(query)

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document],
                                      graph_context: str = ""):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表
            graph_context: 图谱检索结果文本（可选）

        Yields:
            生成的回答片段
        """
        context = self._build_context(context_docs, graph_context=graph_context)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document],
                                             graph_context: str = ""):
        """
        生成详细步骤回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表
            graph_context: 图谱检索结果文本（可选）

        Yields:
            详细步骤回答片段
        """
        context = self._build_context(context_docs, graph_context=graph_context)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容
- 重点突出实用性和可操作性
- 不要回答食谱中没有的信息，如果食谱中没有检索到相关信息也不要擅自补充，就回答"抱歉，我的食谱没有这道菜哟，后期我再慢慢研究"

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 8000,
                       graph_context: str = "") -> str:
        """
        构建上下文字符串

        Args:
            docs:          文档列表
            max_length:    最大字符数（单个文档超长时截断末尾，而非整个文档丢弃）
            graph_context: 图谱检索结果文本（非空时追加到上下文末尾）

        Returns:
            格式化的上下文字符串
        """
        if not docs:
            base = "暂无相关食谱信息。"
        else:
            context_parts = []
            current_length = 0

            for i, doc in enumerate(docs, 1):
                # 添加元数据信息
                metadata_info = f"【食谱 {i}】"
                if 'dish_name' in doc.metadata:
                    metadata_info += f" {doc.metadata['dish_name']}"
                if 'category' in doc.metadata:
                    metadata_info += f" | 分类: {doc.metadata['category']}"
                if 'difficulty' in doc.metadata:
                    metadata_info += f" | 难度: {doc.metadata['difficulty']}"

                doc_text = f"{metadata_info}\n{doc.page_content}\n"

                # 已达到总长度上限，停止追加
                if current_length >= max_length:
                    break

                # 单篇文档超长时截断末尾，保证至少能进入上下文
                remaining = max_length - current_length
                if len(doc_text) > remaining:
                    doc_text = doc_text[:remaining] + "\n...(内容已截断)"

                context_parts.append(doc_text)
                current_length += len(doc_text)

            sep = "\n" + "=" * 50 + "\n"
            base = sep + sep.join(context_parts)

        # 追加图谱上下文（非空时）
        if graph_context and graph_context.strip():
            graph_section = (
                "\n" + "=" * 50 + "\n"
                "【知识图谱补充信息】\n"
                + graph_context.strip()
                + "\n" + "=" * 50
            )
            return base + graph_section

        return base
