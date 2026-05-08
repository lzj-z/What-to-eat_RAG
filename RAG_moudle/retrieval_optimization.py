"""
检索优化模块
"""

import re
import logging
from typing import List, Dict, Any, Callable

import jieba
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _jieba_preprocess(text: str) -> List[str]:
    """
    使用 jieba 对中文文本进行搜索模式分词，替代默认的空格分词。

    默认的 BM25 预处理函数是 text.split()，对中文完全无效：
      "请给我推荐几个鸡翅的做法".split() → ['请给我推荐几个鸡翅的做法']  （整句变成1个token）

    使用 jieba.cut_for_search 后：
      → ['请', '给', '我', '推荐', '几个', '鸡翅', '的', '做法']  （"鸡翅"被正确切出）

    这样 BM25 才能按关键词精确匹配文档。
    """
    tokens = jieba.cut_for_search(text)
    # 过滤空白、纯标点、单字符停用词(的/了/和/是 等)
    return [t for t in tokens if t.strip() and len(t.strip()) >= 1]


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块

        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # BM25检索器 —— 关键修复：使用 jieba 中文分词代替默认空格分词
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5,
            preprocess_func=_jieba_preprocess,
        )

        logger.info("检索器设置完成（BM25 已启用 jieba 中文分词）")

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        三路混合检索：向量语义 + BM25关键词 + 菜名直接匹配，使用加权 RRF 融合

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # —— 第1路：向量语义检索 ——
        vector_docs = self.vector_retriever.invoke(query)
        # —— 第2路：BM25 关键词检索（已使用jieba分词） ——
        bm25_docs = self.bm25_retriever.invoke(query)
        # —— 第3路：菜品名称直接匹配（元数据精确检索） ——
        name_docs = self._dish_name_search(query, top_k=5)

        # 三路加权 RRF 融合
        reranked_docs = self._rrf_rerank_multi(
            doc_lists=[vector_docs, bm25_docs, name_docs],
            weights=[1.0, 1.5, 2.0],  # 菜名精确匹配 > BM25关键词 > 向量语义
            source_names=["向量语义", "BM25关键词", "菜名匹配"],
        )

        return reranked_docs[:top_k]

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索

        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量

        Returns:
            过滤后的文档列表
        """
        # 先进行混合检索，获取更多候选
        docs = self.hybrid_search(query, top_k * 3)

        # 应用元数据过滤
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break

            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break

        return filtered_docs

    # ---------- 菜品名称直接检索（第三路） ----------

    def _dish_name_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        菜品名称 + 食材内容双重匹配 —— 解决向量/BM25都漏检的情况。

        匹配策略（优先级递减）：
        1. 关键词⊂菜名，或菜名⊂查询（原有菜名精确匹配）
        2. 关键词出现在 chunk 内容中（食材原料搜索，如"黄鳝"→"响油鳝丝"）

        按 parent_id 去重，每道菜只取一个代表块。

        Args:
            query: 用户查询
            top_k: 最多返回的文档数

        Returns:
            通过菜名/食材匹配到的文档块列表
        """
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        # 两轮收集：第一轮菜名精确命中，第二轮内容命中（权重降低）
        name_matched: List[Document] = []
        content_matched: List[Document] = []
        seen_parent_ids: set = set()

        for chunk in self.chunks:
            dish_name = chunk.metadata.get('dish_name', '')
            parent_id = chunk.metadata.get('parent_id', '')
            if not dish_name or not parent_id:
                continue
            if parent_id in seen_parent_ids:
                continue

            # 第一轮：菜名匹配
            if any(kw in dish_name for kw in keywords) or dish_name in query:
                seen_parent_ids.add(parent_id)
                name_matched.append(chunk)
                continue

            # 第二轮：食材/内容匹配（仅检查含"原料"相关 chunk，避免噪音）
            content = chunk.page_content
            if any(kw in content for kw in keywords):
                content_matched.append(chunk)

        # 补充内容命中（去重后合并）
        for chunk in content_matched:
            parent_id = chunk.metadata.get('parent_id', '')
            if parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                name_matched.append(chunk)

        result = name_matched[:top_k]
        if result:
            dish_names = [c.metadata.get('dish_name', '') for c in result]
            logger.info(f"菜名/食材匹配检索: 关键词={keywords}, 命中菜品={dish_names}, 共{len(result)}道")
        return result

    # ---------- 多路 RRF 融合 ----------

    def _rrf_rerank_multi(self, doc_lists: List[List[Document]],
                          weights: List[float],
                          source_names: List[str] = None,
                          k: int = 60) -> List[Document]:
        """
        多路 RRF (Reciprocal Rank Fusion) 加权融合

        Args:
            doc_lists: 各检索路返回的文档列表
            weights: 每路的权重系数
            source_names: 各路名称（用于日志）
            k: RRF 平滑参数

        Returns:
            融合排序后的文档列表
        """
        if source_names is None:
            source_names = [f"源{i}" for i in range(len(doc_lists))]

        doc_scores: Dict[int, float] = {}
        doc_objects: Dict[int, Document] = {}

        for source_idx, (docs, weight) in enumerate(zip(doc_lists, weights)):
            name = source_names[source_idx]
            for rank, doc in enumerate(docs):
                doc_id = hash(doc.page_content)
                doc_objects[doc_id] = doc
                rrf_score = weight / (k + rank + 1)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
                logger.debug(f"{name} rank{rank+1}: +{rrf_score:.4f}")

        # 按最终分数降序排列
        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        result = []
        for doc_id, final_score in sorted_items:
            doc = doc_objects[doc_id]
            doc.metadata['rrf_score'] = round(final_score, 4)
            result.append(doc)

        counts = [len(d) for d in doc_lists]
        logger.info(f"RRF三路融合完成: {dict(zip(source_names, counts))}, "
                    f"权重={dict(zip(source_names, weights))}, 合并去重后{len(result)}个文档")

        return result

    # ---------- 智能多食材检索 ----------

    def smart_hybrid_search(self, rewritten_query: str, top_k: int = 5) -> List[Document]:
        """
        智能检索入口：自动检测多食材查询并分别检索，保证每种食材都有结果。

        - 单食材查询（如"红烧鸡翅怎么做"）→ 普通 hybrid_search
        - 多食材查询（如"鸡翅的做法，包菜的做法，排骨的做法"）→ multi_query_hybrid_search

        Args:
            rewritten_query: 经 query_rewrite 处理后的查询
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        sub_queries = self._split_multi_query(rewritten_query)
        if len(sub_queries) > 1:
            logger.info(f"检测到多食材查询，拆分为 {len(sub_queries)} 个子查询: {sub_queries}")
            return self.multi_query_hybrid_search(sub_queries, top_k=top_k)
        return self.hybrid_search(rewritten_query, top_k=top_k)

    @staticmethod
    def _split_multi_query(query: str) -> List[str]:
        """
        将逗号分隔的多食材查询拆分为子查询列表。

        例如:
          "鸡翅的做法，包菜的做法，排骨的做法"
          → ["鸡翅的做法", "包菜的做法", "排骨的做法"]

          "红烧鸡翅怎么做"  →  ["红烧鸡翅怎么做"]  （不拆分）
        """
        parts = re.split(r'[，,；;]', query)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 2]
        return parts if len(parts) >= 2 else [query]

    def multi_query_hybrid_search(self, sub_queries: List[str], top_k: int = 5) -> List[Document]:
        """
        多子查询并行检索 —— 对每个食材子查询独立检索，然后轮转合并结果。

        策略：
          1. 每个子查询独立调用 hybrid_search，取 top_k_per 个结果
          2. 轮转（Round-Robin）优先合并，保证每个子查询至少贡献 1 个文档
          3. 再按 RRF 分数补充剩余名额

        Args:
            sub_queries: 子查询列表（每个食材一条）
            top_k: 总共返回的文档块数

        Returns:
            多样化的文档块列表
        """
        n = len(sub_queries)
        top_k_per = max(2, (top_k + n - 1) // n + 1)   # 每个子查询多取一点确保有余量

        per_query_docs: List[List[Document]] = []
        for q in sub_queries:
            docs = self.hybrid_search(q, top_k=top_k_per)
            per_query_docs.append(docs)
            logger.info(f"子查询 '{q[:20]}': 检索到 {len(docs)} 个文档块")

        result: List[Document] = []
        seen: set = set()

        # --- 第一轮：每个子查询各出最优结果（保证多样性） ---
        for docs in per_query_docs:
            for doc in docs:
                doc_id = hash(doc.page_content)
                if doc_id not in seen:
                    result.append(doc)
                    seen.add(doc_id)
                    break   # 每个子查询第一轮只取 1 个

        # --- 第二轮：继续轮转取第 2 名（进一步补充多样性） ---
        for docs in per_query_docs:
            added_this_query = 0
            for doc in docs:
                doc_id = hash(doc.page_content)
                if doc_id not in seen:
                    if added_this_query == 0 and len(result) < top_k:
                        result.append(doc)
                        seen.add(doc_id)
                        added_this_query += 1
                    break

        # --- 第三轮：按 RRF 分数填满剩余名额 ---
        all_candidates: Dict[int, Document] = {}
        for docs in per_query_docs:
            for doc in docs:
                doc_id = hash(doc.page_content)
                if doc_id not in all_candidates:
                    all_candidates[doc_id] = doc

        by_score = sorted(
            all_candidates.values(),
            key=lambda d: d.metadata.get('rrf_score', 0),
            reverse=True,
        )
        for doc in by_score:
            if len(result) >= top_k:
                break
            doc_id = hash(doc.page_content)
            if doc_id not in seen:
                result.append(doc)
                seen.add(doc_id)

        logger.info(f"多查询合并完成: {n} 个子查询, 最终返回 {len(result)} 个文档块")
        return result[:top_k]

    # ---------- 关键词提取工具 ----------

    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        """
        从查询中提取有意义的食材/菜品关键词。
        使用 jieba 分词，然后过滤停用词，保留可能是食材或菜名的词。

        Args:
            query: 用户查询

        Returns:
            关键词列表
        """
        # 菜谱场景的停用词（不作为食材/菜名匹配的词）
        stop_words = {
            '怎么做', '怎么', '如何', '做法', '制作', '方法', '步骤', '教程',
            '推荐', '有哪些', '有什么', '什么', '哪些', '几个', '几道',
            '请问', '请给', '告诉', '帮我', '我想', '想吃', '想要',
            '可以', '能不能', '可不可以', '是否', '一下', '一些',
            '菜谱', '食谱', '菜品', '简单', '详细', '具体',
            '好吃', '美味', '今天', '今晚', '适合',
            '做菜', '烹饪', '烧菜', '给我', '能给',
        }

        # 使用 jieba 精确模式分词
        tokens = list(jieba.cut(query))

        keywords = []
        for token in tokens:
            token = token.strip()
            # 保留 ≥2 字的中文词，且不是停用词
            if len(token) >= 2 and re.match(r'^[\u4e00-\u9fff]+$', token) and token not in stop_words:
                keywords.append(token)

        return keywords
