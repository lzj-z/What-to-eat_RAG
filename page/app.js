/* ==============================================
   AI 私厨 - 前端交互逻辑
   ============================================== */

// ---- 菜品数据（来自 page/images 目录） ----
const DISHES = {
    meat: [
        { name: "葱烧海参", img: "images/meat/葱烧海参.png" },
        { name: "咖喱炒蟹", img: "images/meat/咖喱炒蟹.png" },
        { name: "咕噜肉",   img: "images/meat/咕噜肉.png" },
        { name: "红烧鸡翅", img: "images/meat/红烧鸡翅.png" },
        { name: "黄焖鸡",   img: "images/meat/黄焖鸡.png" },
        { name: "鸡蛋火腿炒黄瓜", img: "images/meat/鸡蛋火腿炒黄瓜.png" },
        { name: "清蒸鲈鱼", img: "images/meat/清蒸鲈鱼.png" },
        { name: "可乐鸡翅", img: "images/meat/可乐鸡翅.png" },
        { name: "清蒸生蚝", img: "images/meat/清蒸生蚝.png" },
        { name: "糖醋里脊", img: "images/meat/糖醋里脊.png" },
        { name: "微波葱姜黑鳕鱼", img: "images/meat/微波葱姜黑鳕鱼.png" },
        { name: "西红柿炒鸡蛋", img: "images/meat/西红柿炒鸡蛋.png" },
        { name: "响油鳝丝", img: "images/meat/响油鳝丝.png" },
        { name: "油焖大虾", img: "images/meat/油焖大虾.png" },
        { name: "辣椒炒肉", img: "images/meat/辣椒炒肉.png" },
        { name: "芥末黄油罗氏虾", img: "images/meat/芥末黄油罗氏虾.png" },
        { name: "小龙虾",   img: "images/meat/小龙虾.png" },
    ],
    soup: [
        { name: "菌菇炖乳鸽", img: "images/soup/菌菇炖乳鸽.png" },
        { name: "排骨苦瓜汤", img: "images/soup/排骨苦瓜汤.png" },
        { name: "羊肉汤",     img: "images/soup/羊肉汤.png" },
        { name: "玉米排骨汤", img: "images/soup/玉米排骨汤.png" },
        { name: "紫菜蛋花汤", img: "images/soup/紫菜蛋花汤.png" },
    ],
    vegetable: [
        { name: "白灼菜心", img: "images/vegetable/白灼菜心.png" },
        { name: "炒青菜",   img: "images/vegetable/炒青菜.png" },
        { name: "地三鲜",   img: "images/vegetable/地三鲜.png" },
        { name: "酸辣土豆丝", img: "images/vegetable/酸辣土豆丝.png" },
        { name: "蒲烧茄子", img: "images/vegetable/蒲烧茄子.png" },
        { name: "手撕包菜", img: "images/vegetable/手撕包菜.png" },
    ],
    drink: [
        { name: "冰粉",     img: "images/drink/冰粉.png" },
        { name: "冬瓜茶",   img: "images/drink/冬瓜茶.png" },
        { name: "柠檬水",   img: "images/drink/柠檬水.png" },
        { name: "酸梅汤",   img: "images/drink/酸梅汤.png" },
        { name: "杨枝甘露", img: "images/drink/杨枝甘露.png" },
        { name: "长岛冰茶", img: "images/drink/长岛冰茶.png" },
    ],
    staple: [
        { name: "炒河粉",       img: "images/staple/炒河粉.png" },
        { name: "葱油拌面",     img: "images/staple/葱油拌面.png" },
        { name: "蛋炒饭",       img: "images/staple/蛋炒饭.png" },
        { name: "日式咖喱饭",   img: "images/staple/日式咖喱饭.png" },
        { name: "手工水饺",     img: "images/staple/手工水饺.png" },
        { name: "汤面",         img: "images/staple/汤面.png" },
        { name: "意大利肉酱面", img: "images/staple/意大利肉酱面.png" },
        { name: "老友猪肉粉",   img: "images/staple/老友猪肉粉.png" },
    ],
};

const CATEGORY_LABELS = {
    meat: "🥩 荤菜",
    soup: "🍲 汤品",
    vegetable: "🥬 素菜",
    drink: "🧃 饮品",
    staple: "🍚 主食",
};

// ---- 状态 ----
let isSending = false;
let currentDish = null;
let graphAvailable = false;   // 图谱是否在线

// ---- DOM 缓存 ----
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ================================================
//  初始化
// ================================================
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initCategoryFilter();
    renderDishes("all");
    pollSystemStatus();
    autoResizeInput();
});

// ---- Tab 切换 ----
function initTabs() {
    $$(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            $$(".tab-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            $$(".panel").forEach((p) => p.classList.remove("active"));
            $(`#panel-${btn.dataset.tab}`).classList.add("active");
        });
    });
}

// ---- 分类筛选 ----
function initCategoryFilter() {
    $$(".cat-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            $$(".cat-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            renderDishes(btn.dataset.category);
        });
    });
}

// ---- 渲染菜品卡片 ----
function renderDishes(category) {
    const grid = $("#dish-grid");
    let dishes = [];

    if (category === "all") {
        Object.keys(DISHES).forEach((cat) => {
            DISHES[cat].forEach((d) => dishes.push({ ...d, category: cat }));
        });
    } else {
        dishes = DISHES[category].map((d) => ({ ...d, category }));
    }

    grid.innerHTML = dishes
        .map(
            (d) => `
        <div class="dish-card" onclick="openDishModal('${d.name}', '${d.img}', '${d.category}')">
            <img class="dish-card-img" src="${d.img}" alt="${d.name}" loading="lazy">
            <div class="dish-card-info">
                <div class="dish-card-name">${d.name}</div>
                <div class="dish-card-cat">${CATEGORY_LABELS[d.category]}</div>
            </div>
        </div>`
        )
        .join("");
}

// ================================================
//  系统状态轮询
// ================================================
function pollSystemStatus() {
    const badge = $("#system-status");
    const check = () => {
        fetch("/api/status")
            .then((r) => r.json())
            .then((d) => {
                if (d.ready) {
                    graphAvailable = !!d.graph_available;
                    const graphTag = graphAvailable
                        ? ' <span class="status-graph-on" title="知识图谱在线">🕸️</span>'
                        : "";
                    badge.className = "status-badge ready";
                    badge.querySelector(".status-text").innerHTML =
                        "系统就绪" + graphTag;
                } else if (d.error) {
                    badge.className = "status-badge error";
                    badge.querySelector(".status-text").textContent = "初始化失败";
                } else {
                    setTimeout(check, 3000);
                }
            })
            .catch(() => {
                badge.className = "status-badge error";
                badge.querySelector(".status-text").textContent = "连接失败";
            });
    };
    check();
}

// ================================================
//  对话功能
// ================================================

// 快捷问题
function askQuickQuestion(q) {
    $("#chat-input").value = q;
    sendMessage();
}

// 输入框自动高度
function autoResizeInput() {
    const input = $("#chat-input");
    input.addEventListener("input", () => {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 150) + "px";
    });
}

// 键盘事件
function handleInputKeydown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// 发送消息（流式）
async function sendMessage() {
    const input = $("#chat-input");
    const question = input.value.trim();
    if (!question || isSending) return;

    isSending = true;
    $("#btn-send").disabled = true;

    // 清除欢迎卡片
    const welcome = $(".welcome-card");
    if (welcome) welcome.remove();

    // 用户消息
    appendMessage("user", question);
    input.value = "";
    input.style.height = "auto";

    // AI 消息占位
    const aiMsgId = "ai-msg-" + Date.now();
    appendMessage("ai", "", aiMsgId);

    // 清空 RAG 面板
    clearRagPanel();

    try {
        const resp = await fetch("/api/ask/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let answerText = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop(); // 保留不完整的行

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                const raw = line.slice(6).trim();
                if (!raw) continue;

                let event;
                try {
                    event = JSON.parse(raw);
                } catch {
                    continue;
                }

                switch (event.type) {
                    case "step":
                        addRagStep(event.stage, event.message, event.data);
                        break;

                    case "answer_start":
                        removeTypingIndicator(aiMsgId);
                        break;

                    case "answer_chunk":
                        answerText += event.text;
                        updateMessageContent(aiMsgId, renderMarkdown(answerText));
                        scrollChatToBottom();
                        break;

                    case "answer_end":
                        if (!answerText && event.text) {
                            answerText = event.text;
                            updateMessageContent(aiMsgId, renderMarkdown(answerText));
                        }
                        break;

                    case "error":
                        updateMessageContent(
                            aiMsgId,
                            `<span style="color:#c62828">⚠️ ${event.message}</span>`
                        );
                        break;
                }
            }
        }

        // 如果没有收到任何 answer 内容（非流式 fallback）
        if (!answerText) {
            removeTypingIndicator(aiMsgId);
            updateMessageContent(aiMsgId, '<span style="color:#999">未收到回答</span>');
        }
    } catch (err) {
        updateMessageContent(
            $("#" + aiMsgId)
                ? aiMsgId
                : null,
            `<span style="color:#c62828">⚠️ 网络错误: ${err.message}</span>`
        );
    } finally {
        isSending = false;
        $("#btn-send").disabled = false;
        scrollChatToBottom();
    }
}

// 追加消息到聊天区
function appendMessage(role, content, id) {
    const container = $("#chat-messages");
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    if (id) div.id = id;

    const avatarText = role === "user" ? "🧑" : "👨‍🍳";
    const bubbleContent =
        content ||
        `<div class="typing-indicator"><span></span><span></span><span></span></div>`;

    div.innerHTML = `
        <div class="msg-avatar">${avatarText}</div>
        <div class="msg-bubble">${bubbleContent}</div>
    `;
    container.appendChild(div);
    scrollChatToBottom();
}

// 更新消息内容
function updateMessageContent(id, html) {
    const msg = document.getElementById(id);
    if (!msg) return;
    const bubble = msg.querySelector(".msg-bubble");
    bubble.innerHTML = html;
}

// 移除打字动画
function removeTypingIndicator(id) {
    const msg = document.getElementById(id);
    if (!msg) return;
    const typing = msg.querySelector(".typing-indicator");
    if (typing) typing.remove();
}

// 滚动到底部
function scrollChatToBottom() {
    const container = $("#chat-messages");
    container.scrollTop = container.scrollHeight;
}

// ---- 简易 Markdown 渲染 ----
function renderMarkdown(text) {
    if (!text) return "";
    let html = text
        // 标题
        .replace(/^### (.+)$/gm, "<h3>$1</h3>")
        .replace(/^## (.+)$/gm, "<h2>$1</h2>")
        .replace(/^# (.+)$/gm, "<h1>$1</h1>")
        // 粗体 & 斜体
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        // 行内代码
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        // 无序列表
        .replace(/^[\-\*] (.+)$/gm, "<li>$1</li>")
        // 有序列表
        .replace(/^\d+\.\s(.+)$/gm, "<li>$1</li>")
        // 换行
        .replace(/\n/g, "<br>");

    // 将连续 li 包裹到 ul 中
    html = html.replace(/((?:<li>.*?<\/li><br>?)+)/g, (match) => {
        const inner = match.replace(/<br>/g, "");
        return `<ul>${inner}</ul>`;
    });

    return html;
}

// ================================================
//  RAG 过程面板
// ================================================
const STAGE_ICONS = {
    route:      "🧭",
    rewrite:    "✏️",
    retrieval:  "🔍",
    graph:      "🕸️",
    parent_doc: "📄",
    generation: "✨",
};

const STAGE_NAMES = {
    route:      "查询路由",
    rewrite:    "查询重写",
    retrieval:  "向量检索",
    graph:      "图谱检索",
    parent_doc: "父文档获取",
    generation: "回答生成",
};

function clearRagPanel() {
    $("#rag-steps").innerHTML = "";
}

function addRagStep(stage, message, data) {
    const container = $("#rag-steps");
    const step = document.createElement("div");
    step.className = "rag-step";
    step.dataset.stage = stage;

    let dataHtml = "";
    if (data) {
        if (data.chunks) {
            dataHtml = data.chunks
                .map(
                    (c) =>
                        `<div>📌 <strong>${c.dish}</strong> (RRF: ${c.rrf_score})<br>&nbsp;&nbsp;${c.preview}…</div>`
                )
                .join("");
        } else if (data.dishes) {
            dataHtml = `📋 匹配菜品: ${data.dishes.join("、")}`;
        } else if (data.rewritten) {
            dataHtml = `原始: ${data.original}<br>重写: <strong>${data.rewritten}</strong>`;
        } else if (data.route_type) {
            dataHtml = `路由类型: <strong>${data.route_type}</strong>`;
        } else if (data.filters) {
            dataHtml = `过滤条件: ${JSON.stringify(data.filters)}`;
        } else if (data.graph_context) {
            // 图谱检索结果：格式化展示
            const lines = data.graph_context.split("\n").filter(Boolean);
            const charCount = data.char_count || data.graph_context.length;
            dataHtml = `<div class="graph-context-preview">
                ${lines.map(line => {
                    if (line.startsWith("【图谱】")) {
                        return `<div class="graph-line graph-title">${line}</div>`;
                    } else if (line.startsWith("  -")) {
                        return `<div class="graph-line graph-item">${line.replace(/^  /, "")}</div>`;
                    } else if (line.startsWith("  ")) {
                        return `<div class="graph-line graph-prop">${line.replace(/^  /, "")}</div>`;
                    }
                    return `<div class="graph-line">${line}</div>`;
                }).join("")}
                <div class="graph-char-count">共 ${charCount} 字节图谱上下文已注入</div>
            </div>`;
        } else if (data.offline) {
            // 图谱离线状态
            const reason = data.reason
                ? `<br><span style="color:#b71c1c">原因: ${data.reason}</span>`
                : "";
            dataHtml = `<div class="graph-offline-hint">
                💡 运行 <code>python -m RAG_moudle.graph_extraction</code> 提取数据，
                再执行 <code>data/neo4j_import.cypher</code> 导入 Neo4j，即可开启图谱增强${reason}
            </div>`;
        }
    }

    step.innerHTML = `
        <div class="rag-step-header">
            <div class="rag-step-icon">${STAGE_ICONS[stage] || "📌"}</div>
            <div class="rag-step-title">${STAGE_NAMES[stage] || stage}</div>
        </div>
        <div class="rag-step-msg">${message}</div>
        ${dataHtml ? `<div class="rag-step-data">${dataHtml}</div>` : ""}
    `;
    container.appendChild(step);
    container.scrollTop = container.scrollHeight;
}

function toggleRagPanel() {
    const panel = $("#rag-panel");
    panel.classList.toggle("collapsed");
    const btn = panel.querySelector(".rag-toggle");
    btn.textContent = panel.classList.contains("collapsed") ? "▶" : "◀";
}

// ================================================
//  菜品详情弹窗
// ================================================
function openDishModal(name, img, category) {
    currentDish = { name, img, category };
    $("#modal-img").src = img;
    $("#modal-img").alt = name;
    $("#modal-name").textContent = name;
    $("#modal-category").textContent = CATEGORY_LABELS[category];
    $("#dish-modal").classList.add("show");
}

function closeDishModal() {
    $("#dish-modal").classList.remove("show");
    currentDish = null;
}

function closeModal(e) {
    if (e.target === e.currentTarget) closeDishModal();
}

function askAboutDish() {
    if (!currentDish) return;
    closeDishModal();

    // 切换到对话 Tab
    $$(".tab-btn").forEach((b) => b.classList.remove("active"));
    $('[data-tab="chat"]').classList.add("active");
    $$(".panel").forEach((p) => p.classList.remove("active"));
    $("#panel-chat").classList.add("active");

    // 填入问题并发送
    const q = `${currentDish.name}怎么做？请给出详细步骤。`;
    $("#chat-input").value = q;
    sendMessage();
}

// ESC 关闭弹窗
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDishModal();
});
