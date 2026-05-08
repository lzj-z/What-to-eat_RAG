"""
离线图数据提取模块

从 Dish/ 目录下的 Markdown 菜谱文件中提取结构化实体和关系，
生成 data/nodes.csv、data/relationships.csv、data/neo4j_import.cypher，
供手动导入 Neo4j 图数据库使用。

用法（独立运行）:
    python -m RAG_moudle.graph_extraction
    python RAG_moudle/graph_extraction.py
"""

import csv
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from LLM.init_llm import deepseekllm

logger = logging.getLogger(__name__)

# ── 常量 ─────────────────────────────────────────────────────────────────────

DATA_DIR = _PROJECT_ROOT / "data"

# 文件夹路径关键词 → 菜品分类标签
_FOLDER_CATEGORY_MAP: Dict[str, str] = {
    "meat":      "荤菜",
    "vegetable": "素菜",
    "soup":      "汤品",
    "breakfast": "早餐",
    "staple":    "主食",
    "drink":     "饮品",
    "dessert":   "甜品",
}

# ★ 数量 → 难度标签
_STAR_DIFFICULTY_MAP: Dict[int, str] = {
    1: "非常简单",
    2: "简单",
    3: "中等",
    4: "困难",
    5: "非常困难",
}

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _node_id(prefix: str, name: str) -> str:
    """生成确定性节点 ID（MD5）"""
    return hashlib.md5(f"{prefix}:{name}".encode("utf-8")).hexdigest()[:16]


def _infer_category_from_path(md_path: Path, dish_root: Path) -> str:
    """从文件夹路径推断菜品分类"""
    try:
        parts = md_path.relative_to(dish_root).parts
    except ValueError:
        parts = md_path.parts
    for part in parts:
        for keyword, label in _FOLDER_CATEGORY_MAP.items():
            if keyword in part.lower():
                return label
    return "其他"


def _parse_difficulty(content: str) -> str:
    """从 Markdown 正文提取难度（★ 数量）"""
    match = re.search(r"预估烹饪难度[：:]\s*(★+)", content)
    if match:
        stars = len(match.group(1).strip())
        return _STAR_DIFFICULTY_MAP.get(min(stars, 5), "中等")
    return "中等"


def _extract_recipe_name(content: str, filename: str) -> str:
    """从 H1 标题或文件名提取菜谱名称"""
    match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
    if match:
        name = match.group(1).strip()
        # 去掉常见后缀
        name = re.sub(r"的做法.*$", "", name).strip()
        return name
    # 降级：用文件名
    return Path(filename).stem


# ── LLM 提取 prompt ───────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """你是专业的菜谱数据提取助手。请从以下 Markdown 菜谱中提取结构化信息。

菜谱文件名: {dish_name}
菜品分类: {category}
菜谱内容:
{content}

请严格按照以下 JSON 格式返回（不要添加任何其他文字、注释或代码块标记）:
{{
  "cuisine_type": "菜系（如家常菜/川菜/粤菜/日式/西式，根据内容判断，无法判断时填'家常菜'）",
  "prep_time": "准备时间（如'10分钟'，无法提取时填空字符串）",
  "cook_time": "烹饪时间（如'30分钟'，无法提取时填空字符串）",
  "servings": "份量（如'2-3人份'，无法提取时填空字符串）",
  "ingredients": [
    {{
      "name": "食材标准名称（去掉量词后缀，如'鸡翅中'而非'鸡翅中若干只'）",
      "amount": "数量（纯数字或范围，如'10-12'，无则填空字符串）",
      "unit": "单位（如'只'/'克'/'毫升'，无则填空字符串）"
    }}
  ],
  "steps": [
    {{
      "order": 步骤序号（整数，从1开始）,
      "description": "步骤描述（保留原文，控制在120字以内）",
      "methods": "该步骤的烹饪方法（如'煎'/'炒'/'蒸'/'焯水'，多个用逗号分隔，无则填空字符串）",
      "tools": "该步骤使用的工具（如'炒锅'/'菜刀'，多个用逗号分隔，无则填空字符串）",
      "time_estimate": "该步骤预计时间（如'2-3分钟'，无则填空字符串）"
    }}
  ],
  "tips": ["烹饪技巧1", "烹饪技巧2"]
}}"""
)


def _call_llm_extract(md_path: Path, content: str, dish_name: str, category: str) -> Optional[Dict]:
    """调用 LLM 从 Markdown 提取结构化数据，返回 dict 或 None"""
    chain = (
        {"dish_name": RunnablePassthrough(),
         "category": lambda _: category,
         "content": lambda _: content[:6000]}  # 截断防止超长
        | _EXTRACTION_PROMPT
        | deepseekllm
        | StrOutputParser()
    )
    try:
        raw = chain.invoke(dish_name)
        # 清理可能的 markdown 代码块标记
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw.strip())
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"LLM 返回非法 JSON ({md_path.name}): {e} | 原始: {raw[:200]}")
        return None
    except Exception as e:
        logger.error(f"LLM 调用失败 ({md_path.name}): {e}")
        return None


# ── 节点 / 关系构建 ────────────────────────────────────────────────────────────

def _build_recipe_node(
    recipe_name: str,
    category: str,
    difficulty: str,
    llm_data: Dict,
    file_path: str,
) -> Dict:
    return {
        "nodeId": _node_id("recipe", recipe_name),
        "label": "Recipe",
        "name": recipe_name,
        "category": category,
        "difficulty": difficulty,
        "cuisineType": llm_data.get("cuisine_type", ""),
        "prepTime": llm_data.get("prep_time", ""),
        "cookTime": llm_data.get("cook_time", ""),
        "servings": llm_data.get("servings", ""),
        "tips": " | ".join(llm_data.get("tips", [])),
        "filePath": file_path,
    }


def _build_ingredient_nodes_and_rels(
    recipe_id: str, recipe_name: str, ingredients: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    nodes, rels = [], []
    for ing in ingredients:
        name = ing.get("name", "").strip()
        if not name:
            continue
        ing_id = _node_id("ingredient", name)
        nodes.append({
            "nodeId": ing_id,
            "label": "Ingredient",
            "name": name,
        })
        rels.append({
            "relationshipId": _node_id("rel_requires", f"{recipe_name}:{name}"),
            "startNodeId": recipe_id,
            "endNodeId": ing_id,
            "type": "REQUIRES",
            "amount": ing.get("amount", ""),
            "unit": ing.get("unit", ""),
            "stepOrder": "",
        })
    return nodes, rels


def _build_step_nodes_and_rels(
    recipe_id: str, recipe_name: str, steps: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    nodes, rels = [], []
    for step in steps:
        order = step.get("order", 0)
        step_id = _node_id("step", f"{recipe_name}:{order}")
        nodes.append({
            "nodeId": step_id,
            "label": "CookingStep",
            "name": f"{recipe_name}-步骤{order}",
            "description": step.get("description", ""),
            "stepNumber": str(order),
            "methods": step.get("methods", ""),
            "tools": step.get("tools", ""),
            "timeEstimate": step.get("time_estimate", ""),
        })
        rels.append({
            "relationshipId": _node_id("rel_step", f"{recipe_name}:{order}"),
            "startNodeId": recipe_id,
            "endNodeId": step_id,
            "type": "CONTAINS_STEP",
            "amount": "",
            "unit": "",
            "stepOrder": str(float(order)),
        })
    return nodes, rels


def _build_category_node_and_rel(recipe_id: str, category: str) -> Tuple[Dict, Dict]:
    cat_id = _node_id("category", category)
    node = {"nodeId": cat_id, "label": "RecipeCategory", "name": category}
    rel = {
        "relationshipId": _node_id("rel_cat", f"{recipe_id}:{category}"),
        "startNodeId": recipe_id,
        "endNodeId": cat_id,
        "type": "BELONGS_TO_CATEGORY",
        "amount": "", "unit": "", "stepOrder": "",
    }
    return node, rel


def _build_difficulty_node_and_rel(recipe_id: str, difficulty: str) -> Tuple[Dict, Dict]:
    diff_id = _node_id("difficulty", difficulty)
    node = {"nodeId": diff_id, "label": "DifficultyLevel", "name": difficulty}
    rel = {
        "relationshipId": _node_id("rel_diff", f"{recipe_id}:{difficulty}"),
        "startNodeId": recipe_id,
        "endNodeId": diff_id,
        "type": "HAS_DIFFICULTY_LEVEL",
        "amount": "", "unit": "", "stepOrder": "",
    }
    return node, rel


# ── CSV 输出 ──────────────────────────────────────────────────────────────────

_NODE_FIELDNAMES = [
    "nodeId", "label", "name", "category", "difficulty",
    "cuisineType", "prepTime", "cookTime", "servings", "tips", "filePath",
    "description", "stepNumber", "methods", "tools", "timeEstimate",
]

_REL_FIELDNAMES = [
    "relationshipId", "startNodeId", "endNodeId", "type",
    "amount", "unit", "stepOrder",
]


def _write_csv(
    nodes: List[Dict],
    relationships: List[Dict],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = output_dir / "nodes.csv"
    rels_path = output_dir / "relationships.csv"

    # 写节点（去重）
    seen_node_ids = set()
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_NODE_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for node in nodes:
            if node["nodeId"] not in seen_node_ids:
                seen_node_ids.add(node["nodeId"])
                # 补全缺失字段为空字符串
                row = {k: node.get(k, "") for k in _NODE_FIELDNAMES}
                writer.writerow(row)

    # 写关系（去重）
    seen_rel_ids = set()
    with open(rels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_REL_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for rel in relationships:
            if rel["relationshipId"] not in seen_rel_ids:
                seen_rel_ids.add(rel["relationshipId"])
                writer.writerow(rel)

    logger.info(f"已写入 {len(seen_node_ids)} 个节点到 {nodes_path}")
    logger.info(f"已写入 {len(seen_rel_ids)} 条关系到 {rels_path}")
    return nodes_path, rels_path


# ── Cypher 脚本生成 ────────────────────────────────────────────────────────────

def _generate_cypher(nodes: List[Dict], relationships: List[Dict], output_dir: Path):
    """生成 Neo4j 导入用的 Cypher 脚本"""
    cypher_path = output_dir / "neo4j_import.cypher"
    lines = [
        "// Neo4j 图数据导入脚本 - 由 graph_extraction.py 自动生成",
        "// 使用 MERGE 确保幂等性（可重复执行）",
        "",
        "// ── 创建约束和索引 ──────────────────────────────────────────",
        "CREATE CONSTRAINT recipe_id IF NOT EXISTS FOR (n:Recipe) REQUIRE n.nodeId IS UNIQUE;",
        "CREATE CONSTRAINT ingredient_id IF NOT EXISTS FOR (n:Ingredient) REQUIRE n.nodeId IS UNIQUE;",
        "CREATE CONSTRAINT step_id IF NOT EXISTS FOR (n:CookingStep) REQUIRE n.nodeId IS UNIQUE;",
        "CREATE CONSTRAINT category_id IF NOT EXISTS FOR (n:RecipeCategory) REQUIRE n.nodeId IS UNIQUE;",
        "CREATE CONSTRAINT difficulty_id IF NOT EXISTS FOR (n:DifficultyLevel) REQUIRE n.nodeId IS UNIQUE;",
        "",
        "// ── 导入节点 ────────────────────────────────────────────────",
    ]

    # 按 label 分组写节点
    label_order = ["RecipeCategory", "DifficultyLevel", "Ingredient",
                   "CookingStep", "Recipe"]
    nodes_by_label: Dict[str, List[Dict]] = {lbl: [] for lbl in label_order}
    seen_node_ids: set = set()
    for node in nodes:
        lbl = node.get("label", "")
        nid = node.get("nodeId", "")
        if nid and nid not in seen_node_ids:
            seen_node_ids.add(nid)
            if lbl in nodes_by_label:
                nodes_by_label[lbl].append(node)

    for label in label_order:
        group = nodes_by_label[label]
        if not group:
            continue
        lines.append(f"// {label} 节点")
        for node in group:
            props = _cypher_props(node, exclude={"nodeId", "label"})
            lines.append(
                f'MERGE (n:{label} {{nodeId: "{node["nodeId"]}"}}) '
                f"SET n += {{{props}}};",
            )
        lines.append("")

    lines.append("// ── 导入关系 ────────────────────────────────────────────────")
    seen_rel_ids: set = set()
    for rel in relationships:
        rid = rel.get("relationshipId", "")
        if rid in seen_rel_ids:
            continue
        seen_rel_ids.add(rid)
        rel_type = rel["type"]
        start = rel["startNodeId"]
        end = rel["endNodeId"]
        rel_props = {}
        if rel.get("amount"):
            rel_props["amount"] = rel["amount"]
        if rel.get("unit"):
            rel_props["unit"] = rel["unit"]
        if rel.get("stepOrder"):
            try:
                rel_props["step_order"] = float(rel["stepOrder"])
            except ValueError:
                pass
        rel_props["relationshipId"] = rid

        props_str = _cypher_props(rel_props, exclude=set())
        lines.append(
            f'MATCH (a {{nodeId: "{start}"}}), (b {{nodeId: "{end}"}}) '
            f'MERGE (a)-[r:{rel_type} {{relationshipId: "{rid}"}}]->(b) '
            f"SET r += {{{props_str}}};",
        )

    lines.append("")
    lines.append("// ── 完成 ────────────────────────────────────────────────────")

    cypher_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"已写入 Cypher 脚本到 {cypher_path}")
    return cypher_path


def _cypher_props(d: Dict, exclude: set) -> str:
    """将 dict 转为 Cypher 属性字符串，跳过空值和排除键"""
    parts = []
    for k, v in d.items():
        if k in exclude or v is None or v == "":
            continue
        if isinstance(v, float):
            parts.append(f"{k}: {v}")
        elif isinstance(v, int):
            parts.append(f"{k}: {v}")
        else:
            escaped = str(v).replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'{k}: "{escaped}"')
    return ", ".join(parts)


# ── 主提取流程 ────────────────────────────────────────────────────────────────

def run_graph_extraction(
    dish_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    skip_llm: bool = False,
) -> Dict[str, Path]:
    """
    主提取函数：遍历所有 .md 菜谱文件，提取图数据，输出 CSV + Cypher。

    Args:
        dish_root:  菜谱 Markdown 文件根目录（默认 Dish/）
        output_dir: 输出目录（默认 data/）
        skip_llm:   跳过 LLM 调用，仅用正则提取基础信息（用于测试）

    Returns:
        {"nodes_csv": Path, "relationships_csv": Path, "cypher": Path}
    """
    if dish_root is None:
        dish_root = _PROJECT_ROOT / "Dish"
    if output_dir is None:
        output_dir = DATA_DIR

    md_files = sorted(dish_root.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"在 {dish_root} 下未找到任何 .md 文件")

    logger.info(f"发现 {len(md_files)} 个菜谱文件，开始提取...")

    all_nodes: List[Dict] = []
    all_rels: List[Dict] = []
    success_count = 0

    for i, md_path in enumerate(md_files, 1):
        logger.info(f"[{i}/{len(md_files)}] 处理: {md_path.name}")
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"读取文件失败: {md_path} - {e}")
            continue

        # ── 基础信息（正则提取，快速可靠）
        recipe_name = _extract_recipe_name(content, md_path.name)
        category = _infer_category_from_path(md_path, dish_root)
        difficulty = _parse_difficulty(content)
        recipe_id = _node_id("recipe", recipe_name)
        file_path = str(md_path.relative_to(_PROJECT_ROOT).as_posix())

        # ── LLM 提取（食材、步骤、菜系等）
        llm_data: Dict = {}
        if not skip_llm:
            llm_data = _call_llm_extract(md_path, content, recipe_name, category) or {}
        if not llm_data:
            # 降级：最小化提取，只保留基础字段
            llm_data = {"ingredients": [], "steps": [], "tips": []}
            logger.warning(f"LLM 提取失败，使用最小化模式: {recipe_name}")

        # ── 构建节点和关系
        recipe_node = _build_recipe_node(
            recipe_name, category, difficulty, llm_data, file_path
        )
        all_nodes.append(recipe_node)

        ing_nodes, ing_rels = _build_ingredient_nodes_and_rels(
            recipe_id, recipe_name, llm_data.get("ingredients", [])
        )
        all_nodes.extend(ing_nodes)
        all_rels.extend(ing_rels)

        step_nodes, step_rels = _build_step_nodes_and_rels(
            recipe_id, recipe_name, llm_data.get("steps", [])
        )
        all_nodes.extend(step_nodes)
        all_rels.extend(step_rels)

        cat_node, cat_rel = _build_category_node_and_rel(recipe_id, category)
        all_nodes.append(cat_node)
        all_rels.append(cat_rel)

        diff_node, diff_rel = _build_difficulty_node_and_rel(recipe_id, difficulty)
        all_nodes.append(diff_node)
        all_rels.append(diff_rel)

        success_count += 1

    logger.info(
        f"提取完成: {success_count}/{len(md_files)} 个文件成功，"
        f"共 {len(all_nodes)} 个节点，{len(all_rels)} 条关系"
    )

    # ── 写出 CSV
    nodes_path, rels_path = _write_csv(all_nodes, all_rels, output_dir)

    # ── 生成 Cypher 脚本
    cypher_path = _generate_cypher(all_nodes, all_rels, output_dir)

    print("\n✅ 图数据提取完成！")
    print(f"   节点 CSV:    {nodes_path}")
    print(f"   关系 CSV:    {rels_path}")
    print(f"   Cypher 脚本: {cypher_path}")
    print("\n📌 下一步：在 Neo4j Browser 中执行 Cypher 脚本导入数据")
    print("   COPY 内容到 Neo4j Browser 或使用 cypher-shell:")
    print(f"   cypher-shell -u neo4j -p <密码> --file \"{cypher_path}\"")

    return {
        "nodes_csv": nodes_path,
        "relationships_csv": rels_path,
        "cypher": cypher_path,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser(description="菜谱图数据提取工具")
    parser.add_argument(
        "--dish-root", type=Path, default=None,
        help="菜谱 Markdown 根目录（默认: Dish/）",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="输出目录（默认: data/）",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="跳过 LLM 调用，只做基础提取（测试用）",
    )
    args = parser.parse_args()

    run_graph_extraction(
        dish_root=args.dish_root,
        output_dir=args.output_dir,
        skip_llm=args.skip_llm,
    )
