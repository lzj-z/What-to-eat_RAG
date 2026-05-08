"""
本地 Web 服务器 - 将 RAG 系统暴露为 HTTP API
运行方式: python server.py
"""

import os
import sys
import json
import logging
import threading
from pathlib import Path
from typing import Generator

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from main import RecipeRAGSystem

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='page', static_url_path='')
CORS(app)

# 全局 RAG 系统实例
rag_system: RecipeRAGSystem = None
system_ready = False
init_error = None
init_lock = threading.Lock()

# ---- RAG 过程日志收集 ----
class ProcessLogger:
    """收集单次请求的 RAG 过程日志"""

    def __init__(self):
        self.steps = []

    def add(self, stage: str, message: str, data=None):
        step = {"stage": stage, "message": message}
        if data is not None:
            step["data"] = data
        self.steps.append(step)

    def to_list(self):
        return self.steps


def init_rag_system():
    global rag_system, system_ready, init_error
    try:
        logger.info("ai私厨开始初始化 ...")
        rag_system = RecipeRAGSystem()
        rag_system.initialize_system()
        rag_system.build_knowledge_base()
        system_ready = True
        logger.info("ai私厨系统初始化完成")
    except Exception as e:
        init_error = str(e)
        logger.error(f"ai私厨系统初始化失败: {e}")


# 启动时异步初始化
init_thread = threading.Thread(target=init_rag_system, daemon=True)
init_thread.start()


# ---- 路由 ----

@app.route('/')
def index():
    return send_from_directory('page', 'index.html')


@app.route('/api/status', methods=['GET'])
def status():
    graph_available = False
    graph_error = ""
    if system_ready and rag_system is not None and rag_system.graph_module is not None:
        graph_available = rag_system.graph_module.available
        graph_error = rag_system.graph_module.connect_error
    return jsonify({
        "ready": system_ready,
        "error": init_error,
        "graph_available": graph_available,
        "graph_error": graph_error,
    })


@app.route('/api/ask', methods=['POST'])
def ask():
    """普通问答接口，返回完整过程日志 + 最终回答"""
    global rag_system, system_ready, init_error

    if not system_ready:
        msg = init_error if init_error else "系统正在初始化，请稍候..."
        return jsonify({"error": msg}), 503

    data = request.get_json(force=True)
    question = (data.get('question') or '').strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    proc = ProcessLogger()

    try:
        # Step 1: 查询路由
        proc.add("route", f"正在分析查询类型...")
        route_type = rag_system.generation_module.query_router(question)
        proc.add("route", f"查询类型识别完成", {"route_type": route_type})

        # Step 2: 查询重写（所有类型都走重写）
        proc.add("rewrite", "正在智能重写查询...")
        rewritten_query = rag_system.generation_module.query_rewrite(question)
        proc.add("rewrite", "查询重写完成", {"original": question, "rewritten": rewritten_query})

        # Step 3: 检索（list 路由扩大 top_k）
        proc.add("retrieval", "正在检索相关文档...")
        list_top_k = rag_system.config.top_k * 3 if route_type == 'list' else rag_system.config.top_k
        filters = rag_system._extract_filters_from_query(question)
        if filters:
            proc.add("retrieval", f"应用元数据过滤: {filters}", {"filters": filters})
            relevant_chunks = rag_system.retrieval_module.metadata_filtered_search(
                rewritten_query, filters, top_k=list_top_k
            )
        else:
            relevant_chunks = rag_system.retrieval_module.smart_hybrid_search(
                rewritten_query, top_k=list_top_k
            )

        chunk_info = []
        for chunk in relevant_chunks:
            dish = chunk.metadata.get('dish_name', '未知')
            preview = chunk.page_content[:80].strip().replace('\n', ' ')
            score = round(chunk.metadata.get('rrf_score', 0), 4)
            chunk_info.append({"dish": dish, "preview": preview, "rrf_score": score})
        proc.add("retrieval", f"检索完成，找到 {len(relevant_chunks)} 个相关块", {"chunks": chunk_info})

        if not relevant_chunks:
            return jsonify({
                "answer": "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。",
                "process": proc.to_list()
            })

        # Step 3.5: 图谱检索
        graph_context = ""
        if rag_system.graph_module and rag_system.graph_module.available:
            proc.add("graph", "正在图谱检索关联实体...")
            graph_context = rag_system.graph_module.graph_context_for_query(question, route_type=route_type)
            if graph_context:
                proc.add("graph", "图谱检索完成", {"graph_context": graph_context[:300]})
            else:
                proc.add("graph", "图谱未命中相关节点")
        else:
            proc.add("graph", "知识图谱离线，已跳过（请先导入 Neo4j 数据）")

        # Step 4: 获取父文档
        proc.add("parent_doc", "正在获取完整食谱文档...")
        relevant_docs = rag_system.data_module.get_parent_documents(relevant_chunks)
        doc_info = [doc.metadata.get('dish_name', '未知') for doc in relevant_docs]
        proc.add("parent_doc", f"获取到 {len(relevant_docs)} 个完整文档", {"dishes": doc_info})

        # Step 5: 生成回答
        proc.add("generation", "正在生成回答...")
        if route_type == 'list':
            answer = rag_system.generation_module.generate_list_answer(question, relevant_docs)
        elif route_type == 'detail':
            answer = rag_system.generation_module.generate_step_by_step_answer(
                question, relevant_docs, graph_context=graph_context)
        else:
            answer = rag_system.generation_module.generate_basic_answer(
                question, relevant_docs, graph_context=graph_context)
        proc.add("generation", "回答生成完成")

        return jsonify({"answer": answer, "process": proc.to_list()})

    except Exception as e:
        logger.error(f"处理问题时出错: {e}", exc_info=True)
        return jsonify({"error": f"处理失败: {str(e)}"}), 500


@app.route('/api/ask/stream', methods=['POST'])
def ask_stream():
    """流式问答接口，以 SSE 格式逐步返回过程 + 回答"""
    global rag_system, system_ready, init_error

    if not system_ready:
        msg = init_error if init_error else "系统正在初始化，请稍候..."
        def err_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': msg}, ensure_ascii=False)}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    data = request.get_json(force=True)
    question = (data.get('question') or '').strip()

    def generate() -> Generator[str, None, None]:
        def send(event_type: str, **kwargs):
            payload = {"type": event_type, **kwargs}
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        if not question:
            yield send("error", message="问题不能为空")
            return

        try:
            # Step 1: 路由
            yield send("step", stage="route", message="正在分析查询类型...")
            route_type = rag_system.generation_module.query_router(question)
            yield send("step", stage="route", message=f"查询类型: {route_type}", data={"route_type": route_type})

            # Step 2: 重写（所有类型都走重写）
            yield send("step", stage="rewrite", message="正在智能重写查询...")
            rewritten_query = rag_system.generation_module.query_rewrite(question)
            yield send("step", stage="rewrite", message="查询重写完成",
                       data={"original": question, "rewritten": rewritten_query})

            # Step 3: 检索（list 路由扩大 top_k）
            yield send("step", stage="retrieval", message="正在混合检索（向量 + BM25）...")
            list_top_k = rag_system.config.top_k * 3 if route_type == 'list' else rag_system.config.top_k
            filters = rag_system._extract_filters_from_query(question)
            if filters:
                yield send("step", stage="retrieval", message=f"应用元数据过滤: {filters}", data={"filters": filters})
                relevant_chunks = rag_system.retrieval_module.metadata_filtered_search(
                    rewritten_query, filters, top_k=list_top_k
                )
            else:
                relevant_chunks = rag_system.retrieval_module.smart_hybrid_search(
                    rewritten_query, 
                    top_k=list_top_k
                )

            chunk_info = []
            for chunk in relevant_chunks:
                dish = chunk.metadata.get('dish_name', '未知')
                preview = chunk.page_content[:80].strip().replace('\n', ' ')
                score = round(chunk.metadata.get('rrf_score', 0), 4)
                chunk_info.append({"dish": dish, "preview": preview, "rrf_score": score})
            yield send("step", stage="retrieval", message=f"检索完成，找到 {len(relevant_chunks)} 个相关块",
                       data={"chunks": chunk_info})

            if not relevant_chunks:
                yield send("answer_end", text="抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。")
                return

            # Step 3.5: 图谱检索
            graph_context = ""
            if rag_system.graph_module and rag_system.graph_module.available:
                yield send("step", stage="graph", message="正在图谱检索关联实体...")
                graph_context = rag_system.graph_module.graph_context_for_query(
                    question, route_type=route_type
                )
                if graph_context:
                    yield send("step", stage="graph", message="图谱检索完成",
                               data={"graph_context": graph_context[:400],
                                     "char_count": len(graph_context)})
                else:
                    yield send("step", stage="graph", message="图谱未命中相关节点")
            else:
                graph_err = getattr(rag_system.graph_module, "connect_error", "") if rag_system.graph_module else ""
                yield send("step", stage="graph",
                           message="知识图谱离线，已跳过",
                           data={"offline": True, "reason": graph_err})

            # Step 4: 父文档
            yield send("step", stage="parent_doc", message="正在获取完整食谱文档...")
            relevant_docs = rag_system.data_module.get_parent_documents(relevant_chunks)
            doc_info = [doc.metadata.get('dish_name', '未知') for doc in relevant_docs]
            yield send("step", stage="parent_doc", message=f"获取到 {len(relevant_docs)} 个完整文档",
                       data={"dishes": doc_info})

            # Step 5: 流式生成
            yield send("step", stage="generation", message="正在生成回答...")
            yield send("answer_start")

            if route_type == 'list':
                answer = rag_system.generation_module.generate_list_answer(question, relevant_docs)
                yield send("answer_chunk", text=answer)
            elif route_type == 'detail':
                for chunk in rag_system.generation_module.generate_step_by_step_answer_stream(
                        question, relevant_docs, graph_context=graph_context):
                    yield send("answer_chunk", text=chunk)
            else:
                for chunk in rag_system.generation_module.generate_basic_answer_stream(
                        question, relevant_docs, graph_context=graph_context):
                    yield send("answer_chunk", text=chunk)

            yield send("answer_end")

        except Exception as e:
            logger.error(f"流式处理出错: {e}", exc_info=True)
            yield send("error", message=f"处理失败: {str(e)}")

    return Response(generate(), mimetype='text/event-stream',
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route('/api/search', methods=['POST'])
def search_by_category():
    """按分类搜索菜品"""
    if not system_ready:
        return jsonify({"error": "系统未就绪"}), 503

    data = request.get_json(force=True)
    category = (data.get('category') or '').strip()
    query = (data.get('query') or '').strip()

    if not category:
        return jsonify({"error": "分类不能为空"}), 400

    try:
        dishes = rag_system.search_by_category(category, query)
        return jsonify({"dishes": dishes, "category": category})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  AI 私厨 RAG 系统本地服务器")
    print("=" * 60)
    print("  访问地址: http://localhost:5000")
    print("  API 文档:")
    print("    GET  /api/status        - 系统状态")
    print("    POST /api/ask           - 普通问答")
    print("    POST /api/ask/stream    - 流式问答（推荐）")
    print("    POST /api/search        - 按分类搜索")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
