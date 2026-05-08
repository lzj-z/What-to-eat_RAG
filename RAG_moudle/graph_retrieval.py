"""
Neo4j 图谱检索模块

根据用户查询，执行不同策略的 Cypher 查询，将图结构化数据转换为
LLM 可读的文本上下文（graph_context），与向量检索结果融合后送给 LLM。

连接配置从 neo4j.txt 中读取（与 neo4j_test.py 保持一致）。
Neo4j 连接失败时静默降级（返回空字符串），不影响向量 RAG 正常运行。
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import jieba
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

logger = logging.getLogger(__name__)

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 常见食材词（用于 jieba 识别复合食材串）
_INGREDIENT_HINTS = [
    "鸡腿", "鸡翅", "鸡胸", "鸡蛋", "排骨", "猪肉", "牛肉", "羊肉", "虾",
    "鱼", "海鲜", "豆腐", "土豆", "黄瓜", "茄子", "包菜", "白菜", "菠菜",
    "蘑菇", "香菇", "西红柿", "番茄", "辣椒", "洋葱", "大葱", "生姜", "蒜",
    "花椒", "八角", "桂皮", "料酒", "生抽", "老抽", "蚝油", "豆瓣酱",
    "黄鳝", "鳝鱼", "鳝丝", "螃蟹", "生蚝", "鲈鱼", "鳕鱼", "海参",
    "鸡腿排骨", "五花肉", "里脊", "鸡架", "鱿鱼", "带鱼", "龙虾", "罗氏虾",
]


def _split_ingredient_string(raw: str) -> List[str]:
    """
    将复合食材字符串拆分为独立食材列表。

    例如:
      "鸡腿排骨" → ["鸡腿", "排骨"]
      "鸡腿和排骨" → ["鸡腿", "排骨"]
      "西红柿" → ["西红柿"]

    优先用 jieba 分词；fallback 用正则切分连接词。
    """
    # 先用常见连接词切分
    parts = re.split(r'[和与跟及、，,]', raw)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 2]
    if len(parts) >= 2:
        return parts

    # jieba 分词尝试拆分无连接词的复合串（如"鸡腿排骨"）
    if _JIEBA_AVAILABLE and len(raw) >= 4:
        tokens = list(jieba.cut(raw, cut_all=False))
        result = [t for t in tokens if len(t) >= 2 and re.match(r'^[\u4e00-\u9fff]+$', t)]
        if len(result) >= 2:
            return result

    # fallback: 返回原字符串本身
    return [raw] if len(raw) >= 2 else []


# ── Neo4j 配置加载（复用 neo4j_test.py 的解析逻辑）────────────────────────────

def _load_neo4j_config(txt_path: Path) -> Dict[str, str]:
    content = txt_path.read_text(encoding="utf-8")
    uri_m = re.search(r'uri\s*=\s*"([^"]+)"', content)
    user_m = re.search(r'user\s*=\s*"([^"]+)"', content)
    pw_m = re.search(r'password\s*=\s*"([^"]+)"', content)
    if not (uri_m and user_m and pw_m):
        raise ValueError(f"neo4j.txt 中未找到完整的 uri/user/password 配置: {txt_path}")
    return {
        "uri": uri_m.group(1),
        "user": user_m.group(1),
        "password": pw_m.group(1),
    }


# ── 主模块 ─────────────────────────────────────────────────────────────────────

class GraphRetrievalModule:
    """
    Neo4j 图谱检索模块。

    连接失败时 available=False，所有检索方法均返回空字符串，
    系统自动降级为纯向量 RAG，不抛出异常。
    """

    def __init__(self, neo4j_txt_path: Optional[str] = None):
        """
        Args:
            neo4j_txt_path: neo4j.txt 路径（默认项目根目录下的 neo4j.txt）
        """
        self.driver = None
        self.available = False
        self.connect_error: str = ""   # 连接失败原因，供页面展示
        self._connect(neo4j_txt_path)

    # ── 连接 ──────────────────────────────────────────────────────────────────

    def _connect(self, txt_path: Optional[str]):
        txt_path = Path(txt_path) if txt_path else _PROJECT_ROOT / "neo4j.txt"
        if not txt_path.exists():
            self.connect_error = f"neo4j.txt 不存在: {txt_path}"
            logger.warning(f"{self.connect_error}，图谱检索已禁用")
            return
        try:
            from neo4j import GraphDatabase

            config = _load_neo4j_config(txt_path)
            self.driver = GraphDatabase.driver(
                config["uri"], auth=(config["user"], config["password"])
            )
            self.driver.verify_connectivity()
            self.available = True
            logger.info("Neo4j 连接成功，图谱检索已启用")
        except ModuleNotFoundError:
            self.connect_error = "neo4j 包未安装，请执行: pip install neo4j"
            logger.warning(f"Neo4j 连接失败: {self.connect_error}")
            self.driver = None
        except Exception as e:
            self.connect_error = str(e)
            logger.warning(f"Neo4j 连接失败（已降级为纯向量 RAG）: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            try:
                self.driver.close()
            except Exception:
                pass

    def _run_query(self, cypher: str, params: Dict = None) -> List[Dict]:
        """执行 Cypher 查询，失败时返回空列表"""
        if not self.available or not self.driver:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(cypher, params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.warning(f"Cypher 查询失败: {e}\n查询: {cypher[:120]}")
            return []

    # ── Cypher 查询策略 ────────────────────────────────────────────────────────

    def _search_recipe_by_name(self, name: str) -> List[Dict]:
        """按菜名模糊查询菜谱及其食材、难度、分类"""
        cypher = """
        MATCH (r:Recipe)
        WHERE r.name CONTAINS $name
        OPTIONAL MATCH (r)-[req:REQUIRES]->(i:Ingredient)
        OPTIONAL MATCH (r)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(cat:RecipeCategory)
        RETURN r.name AS recipe,
               r.cuisineType AS cuisineType,
               r.cookTime AS cookTime,
               r.servings AS servings,
               collect(DISTINCT {name: i.name, amount: req.amount, unit: req.unit}) AS ingredients,
               d.name AS difficulty,
               cat.name AS category
        LIMIT 3
        """
        return self._run_query(cypher, {"name": name})

    def _search_recipes_by_ingredient(self, ingredient: str) -> List[Dict]:
        """按食材查找所有使用该食材的菜谱（1跳）"""
        cypher = """
        MATCH (r:Recipe)-[req:REQUIRES]->(i:Ingredient)
        WHERE i.name CONTAINS $ingredient
        OPTIONAL MATCH (r)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(cat:RecipeCategory)
        RETURN r.name AS recipe,
               i.name AS matchedIngredient,
               req.amount AS amount,
               req.unit AS unit,
               d.name AS difficulty,
               cat.name AS category
        ORDER BY r.name
        LIMIT 8
        """
        return self._run_query(cypher, {"ingredient": ingredient})

    def _search_similar_recipes(self, recipe_name: str, top_n: int = 5) -> List[Dict]:
        """通过共享食材发现相似菜谱（2跳推理）"""
        cypher = """
        MATCH (r1:Recipe {name: $name})-[:REQUIRES]->(i:Ingredient)<-[:REQUIRES]-(r2:Recipe)
        WHERE r2.name <> $name
        RETURN r2.name AS similarRecipe,
               count(i) AS sharedIngredients,
               collect(i.name)[0..5] AS commonIngredients
        ORDER BY sharedIngredients DESC
        LIMIT $top_n
        """
        return self._run_query(cypher, {"name": recipe_name, "top_n": top_n})

    def _search_by_category(self, category: str) -> List[Dict]:
        """按菜品分类查询菜谱列表"""
        cypher = """
        MATCH (r:Recipe)-[:BELONGS_TO_CATEGORY]->(cat:RecipeCategory)
        WHERE cat.name CONTAINS $category
        OPTIONAL MATCH (r)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        RETURN r.name AS recipe,
               cat.name AS category,
               d.name AS difficulty,
               r.cuisineType AS cuisineType
        ORDER BY r.name
        LIMIT 10
        """
        return self._run_query(cypher, {"category": category})

    def _search_by_difficulty(self, difficulty: str) -> List[Dict]:
        """按难度等级查询菜谱列表"""
        cypher = """
        MATCH (r:Recipe)-[:HAS_DIFFICULTY_LEVEL]->(d:DifficultyLevel)
        WHERE d.name CONTAINS $difficulty
        OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(cat:RecipeCategory)
        RETURN r.name AS recipe,
               d.name AS difficulty,
               cat.name AS category,
               r.cuisineType AS cuisineType
        ORDER BY r.name
        LIMIT 10
        """
        return self._run_query(cypher, {"difficulty": difficulty})

    def _get_recipe_steps(self, recipe_name: str) -> List[Dict]:
        """获取菜谱的详细烹饪步骤（用于 detail 路由）"""
        cypher = """
        MATCH (r:Recipe)-[cs:CONTAINS_STEP]->(s:CookingStep)
        WHERE r.name CONTAINS $name
        RETURN r.name AS recipe,
               s.stepNumber AS stepNumber,
               s.description AS description,
               s.methods AS methods,
               s.tools AS tools,
               s.timeEstimate AS timeEstimate
        ORDER BY toFloat(cs.step_order)
        LIMIT 20
        """
        return self._run_query(cypher, {"name": recipe_name})

    # ── 查询意图分析 ────────────────────────────────────────────────────────────

    def _classify_query(self, query: str) -> Dict[str, Any]:
        """
        分析查询意图，返回图谱检索策略。

        Returns:
            {
              "strategy": "recipe_name" | "ingredient" | "category" | "difficulty" | "none",
              "entities": List[str]  # 提取到的实体名称列表
            }
        """
        # ── 难度关键词
        difficulty_map = {
            "非常简单": ["非常简单", "最简单", "超简单", "很简单"],
            "简单": ["简单", "容易", "基础"],
            "中等": ["中等", "普通", "一般"],
            "困难": ["困难", "复杂", "难"],
            "非常困难": ["非常困难", "最难", "超难"],
        }
        for diff_label, keywords in difficulty_map.items():
            if any(kw in query for kw in keywords):
                return {"strategy": "difficulty", "entities": [diff_label]}

        # ── 分类关键词
        category_map = {
            "荤菜": ["荤菜", "肉类", "肉菜"],
            "素菜": ["素菜", "素食", "蔬菜"],
            "汤品": ["汤", "汤品", "汤类"],
            "早餐": ["早餐", "早饭"],
            "主食": ["主食", "米饭", "面条", "饭食"],
            "饮品": ["饮品", "饮料", "喝的"],
            "甜品": ["甜品", "甜点", "甜食"],
        }
        for cat_label, keywords in category_map.items():
            if any(kw in query for kw in keywords):
                # 同时检查"什么食材"意图 → 降级为 none（向量擅长）
                if "食材" not in query and "原料" not in query:
                    return {"strategy": "category", "entities": [cat_label]}

        # ── 相似菜谱意图（替代/类似）
        similar_patterns = ["类似", "相似", "像.*一样", "替代", "换.*做法", "推荐.*类似"]
        if any(re.search(p, query) for p in similar_patterns):
            # 尝试提取菜名
            recipe_candidates = self._extract_recipe_name_from_query(query)
            if recipe_candidates:
                return {"strategy": "similar", "entities": recipe_candidates}

        # ── 采购/持有意图（"买了X"、"家里有X"、"剩了X"）→ ingredient 策略
        purchase_m = re.search(
            r"(?:买[了到过]|家里[有剩]|剩[了余]|有(?:一些|些|一点)|手里有|冰箱[里有])\s*([\u4e00-\u9fff]{2,10})",
            query
        )
        if purchase_m:
            raw = purchase_m.group(1).strip()
            entities = _split_ingredient_string(raw)
            if entities:
                logger.info(f"图谱: 识别为采购/持有意图, 食材={entities}")
                return {"strategy": "ingredient", "entities": entities}

        # ── 食材意图（"用X做" / "有X的菜" / "含X"）
        ingredient_patterns = [
            r"用\s*([\u4e00-\u9fff]{2,8}?)\s*(?:做|烹饪|制作)",
            r"含有?\s*([\u4e00-\u9fff]{2,8}?)\s*(?:的菜|菜谱)",
            r"有\s*([\u4e00-\u9fff]{2,8}?)\s*的\s*(?:菜|菜谱|做法|食谱)",
            r"([\u4e00-\u9fff]{2,8}?)\s*(?:可以做|能做|做什么菜|怎么做菜)",
        ]
        for pattern in ingredient_patterns:
            m = re.search(pattern, query)
            if m:
                raw = m.group(1).strip()
                entities = _split_ingredient_string(raw)
                if entities:
                    return {"strategy": "ingredient", "entities": entities}

        # ── 菜名意图（直接问某道菜）
        recipe_candidates = self._extract_recipe_name_from_query(query)
        if recipe_candidates:
            return {"strategy": "recipe_name", "entities": recipe_candidates}

        return {"strategy": "none", "entities": []}

    @staticmethod
    def _extract_recipe_name_from_query(query: str) -> List[str]:
        """
        从查询中提取可能是菜名的词组（简单启发式：去掉问句词后剩余的中文词组）。
        返回候选菜名列表（可能为空）。
        """
        # 去掉问句词 + 采购/持有相关词
        cleaned = re.sub(
            r"(怎么做|怎么烹饪|的做法|如何做|制作方法|步骤|食材|需要什么|有什么食材"
            r"|怎么|如何|推荐|有哪些|有什么|请问|帮我|想吃|想做"
            r"|买[了到过]?|家里[有剩]?|剩[了余]?|手里有|冰箱[里有]?"
            r"|今天|今晚|昨天|刚才|一些|一点|一堆|几个|好多)",
            "", query
        ).strip()
        # 保留 2~12 字的中文词组
        candidates = re.findall(r"[\u4e00-\u9fff]{2,12}", cleaned)
        return candidates[:2]  # 最多取前2个候选

    # ── 格式化图谱上下文 ─────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_recipe_detail(records: List[Dict]) -> str:
        """格式化菜谱详情查询结果"""
        if not records:
            return ""
        parts = []
        for r in records:
            name = r.get("recipe", "")
            lines = [f"【图谱】菜谱: {name}"]
            if r.get("category"):
                lines.append(f"  分类: {r['category']}")
            if r.get("difficulty"):
                lines.append(f"  难度: {r['difficulty']}")
            if r.get("cuisineType"):
                lines.append(f"  菜系: {r['cuisineType']}")
            if r.get("cookTime"):
                lines.append(f"  烹饪时间: {r['cookTime']}")
            ings = [
                i for i in (r.get("ingredients") or [])
                if i and i.get("name")
            ]
            if ings:
                ing_strs = []
                for i in ings:
                    amt = f"{i.get('amount', '')}{i.get('unit', '')}".strip()
                    ing_strs.append(f"{i['name']}{'(' + amt + ')' if amt else ''}")
                lines.append(f"  食材: {', '.join(ing_strs)}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    @staticmethod
    def _fmt_ingredient_results(records: List[Dict], ingredient: str) -> str:
        """格式化按食材查询结果"""
        if not records:
            return ""
        recipe_names = [r.get("recipe", "") for r in records if r.get("recipe")]
        if not recipe_names:
            return ""
        return (
            f"【图谱】含有「{ingredient}」的菜谱（共{len(recipe_names)}道）:\n"
            + "\n".join(f"  - {n}（{r.get('category', '')} | {r.get('difficulty', '')}）"
                        for n, r in zip(recipe_names, records))
        )

    @staticmethod
    def _fmt_similar_results(records: List[Dict], recipe_name: str) -> str:
        """格式化相似菜谱查询结果"""
        if not records:
            return ""
        lines = [f"【图谱】与「{recipe_name}」相似的菜谱（共享食材）:"]
        for r in records:
            common = "、".join(r.get("commonIngredients") or [])
            lines.append(
                f"  - {r.get('similarRecipe', '')}（共享{r.get('sharedIngredients', 0)}种食材: {common}）"
            )
        return "\n".join(lines)

    @staticmethod
    def _fmt_list_results(records: List[Dict], label: str) -> str:
        """格式化分类/难度列表查询结果"""
        if not records:
            return ""
        lines = [f"【图谱】{label}（共{len(records)}道）:"]
        for r in records:
            name = r.get("recipe", "")
            diff = r.get("difficulty", "")
            cat = r.get("category", "")
            tag = " | ".join(filter(None, [cat, diff]))
            lines.append(f"  - {name}{'（' + tag + '）' if tag else ''}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_steps(records: List[Dict]) -> str:
        """格式化烹饪步骤查询结果"""
        if not records:
            return ""
        recipe = records[0].get("recipe", "")
        lines = [f"【图谱】{recipe} 烹饪步骤:"]
        for r in records:
            step_num = r.get("stepNumber", "?")
            desc = r.get("description", "")
            time_est = r.get("timeEstimate", "")
            time_str = f"（约{time_est}）" if time_est else ""
            lines.append(f"  步骤{step_num}{time_str}: {desc}")
        return "\n".join(lines)

    # ── 对外统一接口 ────────────────────────────────────────────────────────────

    def graph_context_for_query(self, query: str, route_type: str = "detail") -> str:
        """
        根据用户查询生成图谱上下文文本。

        Args:
            query:      用户原始查询
            route_type: 来自 query_router 的路由类型 ('list'/'detail'/'general')

        Returns:
            图谱上下文字符串（无相关结果时返回空字符串）
        """
        if not self.available:
            return ""

        intent = self._classify_query(query)
        strategy = intent["strategy"]
        entities = intent["entities"]

        if strategy == "none" or not entities:
            return ""

        context_parts = []

        if strategy == "recipe_name":
            for entity in entities:
                # 菜谱详情
                detail = self._search_recipe_by_name(entity)
                ctx = self._fmt_recipe_detail(detail)
                if ctx:
                    context_parts.append(ctx)
                # detail 路由额外返回步骤
                if route_type == "detail" and detail:
                    matched_name = detail[0].get("recipe", entity)
                    steps = self._get_recipe_steps(matched_name)
                    step_ctx = self._fmt_steps(steps)
                    if step_ctx:
                        context_parts.append(step_ctx)
                # 相似菜谱（general/list 路由时附加）
                if route_type != "detail" and detail:
                    matched_name = detail[0].get("recipe", entity)
                    similar = self._search_similar_recipes(matched_name, top_n=3)
                    sim_ctx = self._fmt_similar_results(similar, matched_name)
                    if sim_ctx:
                        context_parts.append(sim_ctx)

            # ── 降级回退：recipe_name 策略无结果时，尝试食材搜索
            if not context_parts:
                logger.info(f"图谱: recipe_name 无结果，回退为 ingredient 搜索, entities={entities}")
                for entity in entities:
                    records = self._search_recipes_by_ingredient(entity)
                    ctx = self._fmt_ingredient_results(records, entity)
                    if ctx:
                        context_parts.append(ctx)

        elif strategy == "ingredient":
            for entity in entities:
                records = self._search_recipes_by_ingredient(entity)
                ctx = self._fmt_ingredient_results(records, entity)
                if ctx:
                    context_parts.append(ctx)

        elif strategy == "similar":
            for entity in entities:
                records = self._search_similar_recipes(entity)
                ctx = self._fmt_similar_results(records, entity)
                if ctx:
                    context_parts.append(ctx)

        elif strategy == "category":
            for entity in entities:
                records = self._search_by_category(entity)
                ctx = self._fmt_list_results(records, f"{entity}菜谱列表")
                if ctx:
                    context_parts.append(ctx)

        elif strategy == "difficulty":
            for entity in entities:
                records = self._search_by_difficulty(entity)
                ctx = self._fmt_list_results(records, f"{entity}难度菜谱列表")
                if ctx:
                    context_parts.append(ctx)

        return "\n\n".join(context_parts)
