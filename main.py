"""
RAG系统主程序
"""

import os
import re
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any

import jieba

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from RAG_moudle.data_preparation import DataPreparationModule
from RAG_moudle.index_construction import IndexConstructionModule
from RAG_moudle.retrieval_optimization import RetrievalOptimizationModule
from RAG_moudle.generation_integration import GenerationIntegrationModule
from RAG_moudle.graph_retrieval import GraphRetrievalModule

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self.graph_module = None       # GraphRetrievalModule（可选，连接失败时为 None）

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # # 检查API密钥
        # if not os.getenv("MOONSHOT_API_KEY"):
        #     raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        # 4. 初始化图谱检索模块（连接失败时静默降级）
        if self.config.use_graph_rag:
            print("🕸️  初始化图谱检索模块...")
            self.graph_module = GraphRetrievalModule(
                neo4j_txt_path=self.config.neo4j_txt_path
            )
            if self.graph_module.available:
                print("✅ Neo4j 图谱检索已就绪")
            else:
                print("⚠️  Neo4j 不可用，将使用纯向量 RAG")

        print("✅ 系统初始化完成！")

    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 计算数据目录指纹
        current_fingerprint = self._compute_data_fingerprint()
        saved_fingerprint = self._load_saved_fingerprint()

        # 2. 尝试加载已保存的索引
        vectorstore = self._load_index_if_valid(current_fingerprint, saved_fingerprint)

        if vectorstore is not None:
            # 指纹匹配，直接使用已保存索引
            print("✅ 成功加载已保存的向量索引（数据无变化）！")
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            # 需要重建索引（全量或增量）
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            existing_vectorstore = self.index_module.load_index()

            if existing_vectorstore is not None and saved_fingerprint is not None:
                # 索引存在但指纹不匹配 → 检查是否只有新增文件
                new_file_chunks = self._get_new_file_chunks(chunks, saved_fingerprint)
                if new_file_chunks and not self._has_modified_files(current_fingerprint, saved_fingerprint):
                    print(f"检测到 {len(set(c.metadata.get('dish_name','') for c in new_file_chunks))} 个新增菜品，增量更新索引...")
                    self.index_module.vectorstore = existing_vectorstore
                    self.index_module.add_documents(new_file_chunks)
                    vectorstore = existing_vectorstore
                else:
                    print("检测到数据文件已修改，重建完整索引...")
                    vectorstore = self.index_module.build_vector_index(chunks)
            else:
                print("未找到已保存的索引，构建新索引...")
                vectorstore = self.index_module.build_vector_index(chunks)

            print("保存向量索引...")
            self.index_module.save_index()
            self._save_fingerprint(current_fingerprint)

        # 3. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 4. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")

    def _compute_data_fingerprint(self) -> dict:
        """计算数据目录中所有 .md 文件的指纹（路径+mtime+size 的 MD5）
        “指纹”是用来**判断知识库数据有没有变化**，从而决定**不能直接复用旧索引**，还是**需要增量更新/全量重建索引**"""
        fingerprint = {}
        data_path = Path(self.config.data_path)
        for md_file in sorted(data_path.rglob("*.md")):
            stat = md_file.stat()
            # 使用正斜杠统一路径格式，避免跨平台差异
            key = md_file.relative_to(data_path).as_posix()
            content = f"{key}:{stat.st_mtime}:{stat.st_size}"
            fingerprint[key] = hashlib.md5(content.encode()).hexdigest()
        return fingerprint

    def _fingerprint_path(self) -> Path:
        return Path(self.config.index_save_path) / "data_fingerprint.json"

    def _load_saved_fingerprint(self) -> dict:
        fp_path = self._fingerprint_path()
        if fp_path.exists():
            try:
                return json.loads(fp_path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _save_fingerprint(self, fingerprint: dict):
        fp_path = self._fingerprint_path()
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        fp_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_index_if_valid(self, current_fp: dict, saved_fp: dict):
        """仅当指纹完全匹配时加载已保存索引，否则返回 None"""
        if saved_fp is None or current_fp != saved_fp:
            return None
        return self.index_module.load_index()

    def _get_new_file_chunks(self, all_chunks: List, saved_fp: dict) -> List:
        """返回只属于新增文件（不在 saved_fp 中）的文档块"""
        saved_keys = set(saved_fp.keys())
        new_chunks = []
        for chunk in all_chunks:
            source = chunk.metadata.get('source', '')
            # 将绝对路径转为 posix 相对路径做比较
            try:
                rel = Path(source).relative_to(self.config.data_path).as_posix()
            except ValueError:
                rel = Path(source).as_posix()
            if rel not in saved_keys:
                new_chunks.append(chunk)
        return new_chunks

    def _has_modified_files(self, current_fp: dict, saved_fp: dict) -> bool:
        """检查是否有已存在文件被修改（排除纯新增情况）"""
        for key, val in current_fp.items():
            if key in saved_fp and saved_fp[key] != val:
                return True
        return False

    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n❓ 用户问题: {question}")

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # 2. 智能查询重写（所有路由类型都走重写，提取关键词提升检索精准度）
        print("🤖 智能分析查询...")
        rewritten_query = self.generation_module.query_rewrite(question)
        print(f"📝 重写结果: {rewritten_query}")

        # 3. 检索相关子块（list 路由扩大 top_k，保证候选多样性）
        print("🔍 检索相关文档...")
        list_top_k = self.config.top_k * 3 if route_type == 'list' else self.config.top_k
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters,
                                                                             top_k=list_top_k)
        else:
            relevant_chunks = self.retrieval_module.smart_hybrid_search(rewritten_query, top_k=list_top_k)

        # 3.5 图谱检索（与向量检索并行增强，失败时静默跳过）
        graph_context = ""
        if self.graph_module and self.graph_module.available:
            print("🕸️  图谱检索...")
            graph_context = self.graph_module.graph_context_for_query(question, route_type=route_type)
            if graph_context:
                print(f"图谱补充了 {len(graph_context)} 字的结构化信息")
            else:
                print("图谱未找到相关结构化信息")

        # 显示检索到的子块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get('dish_name', '未知菜品')
                # 尝试从内容中提取章节标题
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith('#'):
                    # 如果是标题开头，提取标题（仅取第一行）
                    title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                    section_title = content_preview[:title_end].replace('#', '').strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        # 5. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：直接返回菜品名称列表
            print("📋 生成菜品列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")

            return self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            # 详细查询：获取完整文档并生成详细回答
            print("获取完整文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")
            else:
                print(f"对应 {len(relevant_docs)} 个完整文档")

            print("✍️ 生成详细回答...")

            # 根据路由类型自动选择回答模式
            if route_type == "detail":
                # 详细查询使用分步指导模式
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(
                        question, relevant_docs, graph_context=graph_context)
                else:
                    return self.generation_module.generate_step_by_step_answer(
                        question, relevant_docs, graph_context=graph_context)
            else:
                # 一般查询使用基础回答模式
                if stream:
                    return self.generation_module.generate_basic_answer_stream(
                        question, relevant_docs, graph_context=graph_context)
                else:
                    return self.generation_module.generate_basic_answer(
                        question, relevant_docs, graph_context=graph_context)

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件。

        仅在用户意图明确且单一地指向某个分类时才应用分类过滤。
        复杂查询（多分类、含具体食材/菜品名）不应用分类过滤，避免误杀相关结果。
        """
        filters = {}

        # ---- 分类过滤（保守策略） ----
        category_keywords = DataPreparationModule.get_supported_categories()
        matched_categories = [cat for cat in category_keywords if cat in query]

        if len(matched_categories) == 1:
            # 检查是否存在跨分类表达（如"一荤一素"、"荤素搭配"）
            cross_cat_patterns = ['一荤一素', '荤素', '素荤', '搭配', '一荤', '一素']
            has_cross_cat = any(p in query for p in cross_cat_patterns)

            # 检查查询是否同时提到了具体食材/菜品名称
            # 如果有，说明是复杂查询（如"想吃螃蟹，素菜帮我选"），不应强制单分类过滤
            has_food_name = self._query_has_food_keywords(query, set(matched_categories))

            if not has_cross_cat and not has_food_name:
                filters['category'] = matched_categories[0]

        # 多分类（>=2）或无分类匹配时，不应用分类过滤

        # ---- 难度过滤 ----
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters

    def _query_has_food_keywords(self, query: str, category_words: set) -> bool:
        """
        检查查询中是否包含具体的食材/菜品名称（排除分类名和通用修饰词）。
        用于判断是否应跳过分类过滤。

        例如:
          "推荐几个素菜" → False（只有分类词，无具体食材）
          "想吃螃蟹，素菜帮我选" → True（包含具体食材"螃蟹"）
          "不爱吃鱼，推荐荤菜" → True（包含具体食材"鱼"相关词"不爱吃鱼"中的"吃鱼"）
        """
        # 通用词和分类词，不算具体食材
        generic_words = category_words | {
            '推荐', '做法', '菜谱', '食谱', '简单', '今晚', '今天', '具体',
            '什么', '哪些', '几个', '几道', '帮我', '想做', '想吃', '请问',
            '不爱', '不要', '不吃', '详细', '分别', '附上', '请给', '怎么',
            '适合', '容易', '方便', '好吃', '美味', '制作', '烹饪', '步骤',
            '教程', '方法', '一道', '搭配', '选择', '喜欢', '请分',
            '能给', '告诉', '可以', '一些', '一个', '晚上', '中午',
            '早上', '明天', '有什么', '有哪些', '比较',
        }

        tokens = list(jieba.cut(query))
        for token in tokens:
            token = token.strip()
            if (len(token) >= 2
                    and re.match(r'^[\u4e00-\u9fff]+$', token)
                    and token not in generic_words):
                return True
        return False

    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品

        Args:
            category: 菜品分类
            query: 可选的额外查询条件

        Returns:
            菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")

        # 使用元数据过滤搜索
        search_query = query if query else category
        filters = {"category": category}

        docs = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=10)

        # 提取菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        return dish_names

    def get_ingredients_list(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息

        Args:
            dish_name: 菜品名称

        Returns:
            食材信息
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # 搜索相关文档
        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

        # 生成食材信息
        answer = self.generation_module.generate_basic_answer(f"{dish_name}需要什么食材？", docs)

        return answer

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  Handsome,我是你的ai私厨 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")

        # 初始化系统
        self.initialize_system()

        # 构建知识库
        self.build_knowledge_base()

        print("\n交互式问答 (输入'退出'结束):")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break

                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

        print("\n主人，下次想吃什么随时可以跟我说噢！")

        # 关闭图谱连接
        if self.graph_module:
            self.graph_module.close()


def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()

        # 运行交互式问答
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")


if __name__ == "__main__":
    main()
