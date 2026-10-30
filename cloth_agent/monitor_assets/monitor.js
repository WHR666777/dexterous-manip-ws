/* 只消费服务端记录。文本全部使用 textContent，模型输出不作为 HTML 或脚本执行。 */
"use strict";
const $ = (id) => document.getElementById(id);
const statusNames = {
  running: "运行中", in_progress: "执行中", completed: "执行完成", success: "成功报告",
  failure: "失败", failed: "失败", error: "错误", interrupted: "已中断", paused: "已暂停",
  rejected: "已拒绝", aborted: "已取消", stuck: "无法继续", manual_hold: "人工保持锁定",
  planning_failed: "规划失败", cancelled: "已取消"
};
const badStatuses = new Set(["failure", "failed", "error", "planning_failed", "rejected", "stuck"]);
const pretty = (value) => typeof value === "string" ? value : JSON.stringify(value, null, 2);
const statusLabel = (value) => statusNames[value] || value || "等待记录";
const tone = (status) => badStatuses.has(status) ? "bad" :
  ["paused", "interrupted", "aborted", "manual_hold", "cancelled"].includes(status) ? "warn" : "info";
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function badge(text, color) {
  const node = element("span", "badge", text);
  node.dataset.tone = color;
  return node;
}
function setBadge(id, text, color) {
  $(id).textContent = text;
  $(id).dataset.tone = color;
}
function blankModel(directory, monotonic) {
  return {directory, monotonic, receivedAt: performance.now(), items: new Map(), cells: new Map(),
    metadata: null, result: null, selected: null, start: null, end: null, task: "",
    notices: [], stderr: "", imageKey: null, turnComplete: false, codexError: ""};
}
let model = blankModel("", 0);
let scheduled = false;
let imageVersion = 0;
let lastImage = "";

function cell(id) {
  if (!model.cells.has(id)) model.cells.set(id, {id, code: "", result: null, observations: {}});
  return model.cells.get(id);
}
function toolResult(item) {
  // MCP 文本中包含真正的 cell JSON；图片 base64 已在后端剥离。
  for (const block of item?.result?.content || []) {
    if (block.type !== "text") continue;
    try {
      const value = JSON.parse(block.text);
      if (value && typeof value === "object" && !Array.isArray(value)) return value;
    } catch (_) { /* 普通文字仍可在原始工具结果里查看。 */ }
  }
  return item?.result?.structuredContent || null;
}
function cellForItem(item) {
  const result = toolResult(item);
  if (Number.isInteger(result?.cell_id)) return model.cells.get(result.cell_id);
  // 运行中尚无 cell_id 时，用源码关联；已被其他完成回执明确占用的 cell
  // 不再作为候选。重复源码仍有多个未关联候选时保留未知，不凭 item 数字猜编号。
  const code = item?.arguments?.code;
  const claimed = new Set([...model.items.values()].filter((value) => value.id !== item?.id)
    .map((value) => toolResult(value)?.cell_id).filter(Number.isInteger));
  const matches = [...model.cells.values()].filter((value) =>
    value.id !== 0 && !claimed.has(value.id) && code && value.code === code);
  return matches.length === 1 ? matches[0] : null;
}
function executionView(item, local = cellForItem(item)) {
  // Codex 的工具结束事件可能尚未落盘，尤其在 cell 被中断后。已有 cell 终态
  // 优先于旧的 in_progress；完成回执中的附图错误仍保留，避免遗漏 MCP 包装反馈。
  const parsed = toolResult(item);
  const result = local?.result ? {...parsed, ...local.result,
    image_errors: parsed?.image_errors || local.result.image_errors} : parsed;
  const status = local?.result?.status || (local?.end !== undefined ? local.status : null)
    || parsed?.status || local?.status || item?.status || "in_progress";
  return {item, local, result, status,
    code: local?.code || item?.arguments?.code || pretty(item?.arguments || item?.command || "")};
}
function cellTitle(id) {
  return id === 0 ? "启动准备 · Cell #0" : `Cell #${id}`;
}
function consume(message) {
  const {source, data} = message;
  if (source === "reset") {
    model = blankModel(data.directory, data.monotonic);
    lastImage = "";
    imageVersion += 1;
  } else if (source === "metadata") model.metadata = data;
  else if (source === "result") model.result = data;
  else if (source === "task") model.task = data;
  else if (source === "stderr") model.stderr += data + "\n";
  else if (source === "notice") {
    if (model.notices.at(-1) !== data) model.notices.push(data);
  } else if (source === "session") {
    if (data.type === "session_started") model.start = data.timestamp_monotonic;
    if (data.type === "session_ended") {
      model.end = data.timestamp_monotonic;
      model.result = data.result;
    }
    if (data.type === "cell_started") {
      const value = cell(data.cell_id);
      value.start = data.timestamp_monotonic;
      value.status = "running";
      if ($("follow").checked) model.selected = {kind: "cell", id: data.cell_id};
    }
    if (data.type === "cell_returned") {
      const value = cell(data.cell_id);
      value.end = data.timestamp_monotonic;
      value.status = data.status;
    }
  } else if (source === "control") {
    // 保存实际人工保持/故障记录，不把它展示为持续采样的实时遥测。
    if (data.report || data.ok === false || data.reason) {
      model.notices.push(pretty(data));
    }
  } else if (source === "cell") {
    const value = cell(data.cell_id);
    if (data.kind === "py") value.code = data.value;
    else {
      value.result = data.value;
      value.status = data.value.status;
      value.observations = data.observations || {};
    }
    if (!model.selected || ($("follow").checked && data.cell_id === Math.max(...model.cells.keys()))) {
      model.selected = {kind: "cell", id: data.cell_id};
    }
  } else if (source === "codex") {
    if (data.item?.id) {
      const previous = model.items.get(data.item.id);
      const item = {...previous, ...data.item};
      item.eventType = data.type;
      model.items.set(item.id, item);
      if ($("follow").checked && item.type === "mcp_tool_call" && item.tool === "python"
          && (!previous || data.type === "item.started")) {
        model.selected = {kind: "tool", id: item.id};
      }
    }
    if (data.type === "turn.completed") model.turnComplete = true;
    if (data.type === "turn.failed" || data.type === "error") {
      model.codexError = pretty(data.error || data.message || data);
      model.notices.push(model.codexError);
    }
  }
  scheduleRender();
}

function selectedView() {
  if (model.selected?.kind === "tool") {
    const item = model.items.get(model.selected.id);
    return executionView(item);
  }
  const local = model.cells.get(model.selected?.id);
  const item = [...model.items.values()].find((value) => cellForItem(value)?.id === local?.id && local);
  return executionView(item, local);
}

function renderSummary() {
  const taskStatus = model.result?.status;
  const running = [...model.cells.values()].filter((value) => !value.end && !value.result && value.status === "running");
  $("task-state").textContent = taskStatus === "completed" ? "已结束 · agent 报告完成" :
    taskStatus ? statusLabel(taskStatus) : model.start !== null ? "运行中 · 按任务记录" : "等待记录";
  $("task-state").style.color = taskStatus ? (badStatuses.has(taskStatus) ? "var(--bad)" : "var(--ink)") : "var(--ink)";
  $("reason").textContent = model.result?.reason || model.codexError || "";
  const pending = [...model.items.values()].find((item) =>
    ["mcp_tool_call", "command_execution"].includes(item.type)
    && ["in_progress", "running"].includes(executionView(item).status));
  $("phase").textContent = taskStatus ? (taskStatus === "manual_hold" ? "任务已撤权 · 保持状态见独立入口" : "任务已结束") :
    running.length ? `执行 ${cellTitle(running.at(-1).id)}` : pending ? `等待 ${pending.tool || "工具"} 返回` :
    model.turnComplete ? "等待任务收尾" : model.start !== null ? "等待 agent 输出 / 下一步" : "等待任务启动";
  $("cell-count").textContent = `${[...model.cells.keys()].filter((id) => id !== 0).length} 个 agent cell 已记录${model.cells.has(0) ? " · 含启动准备" : ""}`;
  const connected = model.metadata?.hardware_connected;
  $("hardware").textContent = connected === true ? "已连接（启动时）" : connected === false ? "未连接（启动时）" : "尚无记录";
  $("task-summary").textContent = model.task ? `本次任务 · ${model.task.slice(0, 90)}${model.task.length > 90 ? "…" : ""}` : "本次任务 · 暂无描述";
  $("task-text").textContent = model.task;
  $("directory").textContent = model.directory;
  updateElapsed();
}
function updateElapsed() {
  if (model.start === null) { $("elapsed").textContent = "—"; return; }
  const now = model.monotonic + (performance.now() - model.receivedAt) / 1000;
  const duration = (model.end ?? now) - model.start;
  $("elapsed").textContent = duration < 0 ? "历史时钟不匹配" :
    `${Math.floor(duration / 60)}m ${Math.floor(duration % 60).toString().padStart(2, "0")}s`;
}

function renderTimeline() {
  const container = $("timeline");
  const oldScroll = container.scrollTop;
  const nodes = [];
  const entries = [...model.items.values()].map((item) => ({item, local: cellForItem(item)}));
  // 启动准备没有 Codex item，必须放在最前；其余缺少 item 的 cell 按编号插入
  // 已关联 cell 之前，而不是统一补到列表底部。Codex 文字/调用仍保留原始顺序。
  const linked = new Set(entries.filter((entry) => entry.local).map((entry) => entry.local.id));
  for (const local of [...model.cells.values()].sort((a, b) => a.id - b.id)) {
    if (linked.has(local.id)) continue;
    const index = local.id === 0 ? 0 : entries.findIndex((entry) => entry.local && entry.local.id > local.id);
    entries.splice(index < 0 ? entries.length : index, 0, {local});
  }
  for (const {item, local} of entries) {
    if (!item) {
      const node = element("article", "timeline-card tool");
      const row = element("div", "card-top");
      row.append(element("span", "", local.id === 0 ? "系统启动准备" : "Python cell"),
        badge(statusLabel(local.result?.status || local.status), tone(local.result?.status || local.status)));
      node.append(row, element("p", "card-text", cellTitle(local.id)),
        element("pre", "card-preview", local.code || "等待源码记录"));
      if (model.selected?.kind === "cell" && model.selected.id === local.id) node.classList.add("selected");
      node.addEventListener("click", () => {
        $("follow").checked = false;
        model.selected = {kind: "cell", id: local.id};
        model.imageKey = null;
        scheduleRender();
      });
      nodes.push(node);
      continue;
    }
    const isTool = item.type === "mcp_tool_call" || item.type === "command_execution";
    const node = element("article", `timeline-card ${isTool ? "tool" : "message"}`);
    const row = element("div", "card-top");
    row.append(element("span", "", isTool ? `${item.server || "Codex"} / ${item.tool || "command"}` : "Agent 公开输出"));
    if (isTool) {
      const {result, status} = executionView(item, local);
      row.append(badge(statusLabel(status), tone(status)));
      const selected = model.selected?.kind === "tool" ? model.selected.id === item.id :
        local && model.selected?.id === local.id;
      if (selected) node.classList.add("selected");
      node.addEventListener("click", () => {
        $("follow").checked = false;
        model.selected = {kind: "tool", id: item.id};
        model.imageKey = null;  // 回看步骤时同步选择该 cell 的附图。
        scheduleRender();
      });
      const id = local?.id ?? result?.cell_id;
      node.append(row, element("p", "card-text", Number.isInteger(id) ? cellTitle(id) :
        item.tool === "python" ? "Python cell · 等待编号关联" : `${item.tool || "命令"}调用`),
        element("pre", "card-preview", item.arguments?.code || pretty(item.arguments || item.command || "")));
      if (item.error) node.append(element("p", "card-text errors", pretty(item.error)));
    } else {
      node.append(row, element("p", "card-text", item.text || pretty(item.summary || item)));
    }
    nodes.push(node);
  }
  container.replaceChildren(...(nodes.length ? nodes : [element("p", "empty", "等待 agent 输出或 cell 记录")]));
  container.scrollTop = $("follow").checked ? container.scrollHeight : oldScroll;
}

function robotRow(label, text, color, target = $("robot-results")) {
  const row = element("div", "robot-row");
  row.append(element("span", "", label), badge(text, color));
  target.append(row);
}

function renderRequests(calls) {
  // 高频轮询属于程序内部通信，默认收起并按操作合并；完整顺序仍保留在原始 JSON。
  // ok 只表示这次请求有正常回复，不把它翻译为运动成功，也不将查询次数算成动作次数。
  const names = {state: "读取机器人状态", start_motion: "提交运动计划",
    motion_status: "查询运动进度", set_gripper: "发送手部指令",
    observe: "获取现场观测", pause: "请求暂停保持"};
  const groups = new Map();
  for (const call of calls) {
    const group = groups.get(call.operation) || {count: 0, failed: 0, seconds: 0};
    group.count += 1;
    group.failed += call.ok ? 0 : 1;
    group.seconds += Number(call.elapsed_s || 0);
    groups.set(call.operation, group);
    // 失败不能因为默认折叠而被隐藏，主反馈区仍明确列出失败的请求与原因。
    if (!call.ok) {
      robotRow(names[call.operation] || call.operation, "请求失败", "bad");
      if (call.reason) $("robot-results").append(element("p", "hint errors", call.reason));
    }
  }
  $("request-details").hidden = calls.length === 0;
  const queries = groups.get("motion_status")?.count || 0;
  $("request-heading").textContent = `内部通信记录 · ${calls.length} 次请求${queries ? `，其中 ${queries} 次运动进度查询` : ""}`;
  $("request-groups").replaceChildren();
  for (const [operation, group] of groups) {
    robotRow(`${names[operation] || operation} · ${group.count} 次`,
      `累计往返 ${group.seconds.toFixed(3)} s${group.failed ? ` · ${group.failed} 次失败` : ""}`,
      group.failed ? "bad" : "muted", $("request-groups"));
  }
}

function renderSelected() {
  const view = selectedView();
  const result = view.result;
  const number = view.local?.id ?? result?.cell_id;
  const hasCell = Number.isInteger(number);
  $("selected-title").textContent = hasCell ? cellTitle(number) : view.item ? `工具 · ${view.item.tool || view.item.type}` : "Python cell";
  const codeStatus = {completed: "代码执行完毕", error: "代码异常中止", running: "代码执行中",
    interrupted: "代码已中断", paused: "代码已暂停"}[view.status] || statusLabel(view.status);
  setBadge("cell-badge", model.selected ? codeStatus : "待执行", tone(view.status));
  $("selected-note").textContent = number === 0 ? "系统预先编写的启动准备代码，不属于 agent 生成代码" :
    view.item ? `来自 ${view.item.server || "Codex"} · ${view.item.tool || view.item.type}` :
    hasCell ? `${cellTitle(number)} · 源码在执行前保存` : "提交工具调用后即可查看代码";
  const code = view.code || "# 等待 Python cell";
  if ($("code").textContent !== code) $("code").textContent = code;
  $("duration").textContent = typeof result?.elapsed_s === "number" ? `${result.elapsed_s.toFixed(3)} s` : "";
  $("outcome").hidden = !result;
  $("outcome").dataset.tone = tone(view.status);
  // 异常末行通常含 agent 抛出的明确原因，优先呈现；完整调用栈放在错误详情中。
  const exceptionSummary = String(result?.exception || "").trim().split("\n").at(-1);
  const explanation = result?.reason || exceptionSummary;
  $("outcome").textContent = result ? `${hasCell ? "代码执行" : "工具调用"}：${codeStatus}${explanation ? `\n${explanation}` : ""}` : "";
  $("robot-results").replaceChildren();
  const motion = result?.control?.motion;
  // planning_failed 可由 Python 的 MotionResult 返回而不抛异常；现有记录只在
  // stdout 中出现时，明确标为“文本提示”，不能伪装为新增的结构化硬件反馈。
  const failure = [["stdout", result?.stdout], ["返回值", result?.value]]
    .map(([source, text]) => [source, String(text || "").split("\n").find((line) => line.includes("planning_failed"))])
    .find(([, line]) => line);
  if (failure) {
    robotRow(`根据${failure[0]}提示`, "有运动规划失败", "bad");
    $("robot-results").append(element("p", "hint errors", failure[1]));
  }
  if (motion) {
    robotRow("最近一次已提交运动 · 收尾快照", statusLabel(motion.status), tone(motion.status));
    if (failure) {
      // 规划失败的动作没有 start_motion，因此这份快照可能属于前一段成功动作。
      // 不通过快照的 completed 覆盖本 cell 输出中的 planning_failed。
      $("robot-results").append(element("p", "hint", "该快照仅描述已提交的运动；规划失败的动作未提交，不会更新这份运动状态。"));
    }
    if (motion.reason) $("robot-results").append(element("p", "hint", motion.reason));
  }
  renderRequests(result?.robot_calls || []);
  $("buffer-note").textContent = result ? "cell 正常返回不代表运动到位或抓取成功。" :
    "cell 执行期间 stdout 被缓存；等待返回后显示，不提供 cell 内实时进度。";
  $("stdout").textContent = result?.stdout || "暂无输出";
  $("value").textContent = result?.value ?? "—";
  $("value-section").hidden = result?.value === null || result?.value === undefined;
  const errors = [result?.exception, result?.stderr, view.item?.error && pretty(view.item.error),
    ...(result?.image_errors || [])].filter(Boolean).join("\n");
  $("error-section").hidden = !errors;
  $("errors").textContent = errors;
  $("raw").textContent = result ? pretty(result) : view.item?.result ? pretty(view.item.result) : "暂无结果";
  $("call-details").hidden = !view.item;
  $("call-info").textContent = view.item ? pretty({item_id: view.item.id,
    server: view.item.server, tool: view.item.tool, codex_event_status: view.item.status}) : "";
}

function allImages() {
  const images = [];
  for (const local of [...model.cells.values()].sort((a, b) => a.id - b.id)) {
    for (const [index, shown] of (local.result?.images || []).entries()) {
      if (!/^[0-9a-f]{32}$/.test(shown.observation_id || "")) continue;
      images.push({...shown, key: `${local.id}:${index}`, cellId: local.id,
        metadata: local.observations[shown.observation_id], errors: local.result.image_errors || []});
    }
  }
  return images;
}
function renderImages() {
  const images = allImages();
  const select = $("image-select");
  const currentView = selectedView();
  const currentCell = currentView.local?.id ?? currentView.result?.cell_id;
  if (!model.imageKey || ($("follow").checked && images.length)) {
    model.imageKey = images.filter((image) => image.cellId === currentCell).at(-1)?.key || images.at(-1)?.key;
  }
  select.replaceChildren(...(images.length ? images.map((image) => {
    const option = element("option", "", `Cell #${image.cellId} · ${image.crop ? "裁剪" : "全图"} · ${image.observation_id.slice(0, 8)}`);
    option.value = image.key;
    return option;
  }) : [element("option", "", "暂无附图")]));
  const shown = images.find((image) => image.key === model.imageKey);
  if (!shown) {
    $("observation").hidden = true;
    $("image-stage").querySelector("p").hidden = false;
    $("image-meta").replaceChildren();
    $("image-error").textContent = "";
    $("image-context").textContent = "图片来自 cell 的 show() 记录。";
    return;
  }
  select.value = shown.key;
  $("image-context").textContent = `来自 Cell #${shown.cellId}${shown.cellId !== currentCell ? " · 当前选中 cell 无对应新图，显示已有快照" : ""}`;
  const rows = [["观测 ID", shown.observation_id], ["原图裁剪", shown.crop ? pretty(shown.crop) : "无 · 全图"],
    ["相机帧号", shown.metadata?.timestamps?.color_frame_number ?? "未记录"],
    ["相机时间", shown.metadata?.timestamps?.color_timestamp_ms !== undefined ?
      `${shown.metadata.timestamps.color_timestamp_ms} ms` : "未记录"]];
  $("image-meta").replaceChildren(...rows.flatMap(([key, value]) => [element("dt", "", key), element("dd", "", value)]));
  const identity = `${shown.key}:${shown.observation_id}:${pretty(shown.crop)}`;
  if (identity === lastImage) return;
  lastImage = identity;
  const generation = ++imageVersion;
  const image = new Image();
  // 切换观测时先清除旧画面，避免新 ID/裁剪标注短暂配上上一张图片。
  $("observation").hidden = true;
  $("image-stage").querySelector("p").hidden = false;
  $("image-stage").querySelector("p").textContent = "正在加载观测快照";
  $("image-error").textContent = shown.errors.join("\n");
  image.onload = () => {
    if (generation !== imageVersion) return;
    const box = shown.crop || [0, 0, image.naturalWidth, image.naturalHeight];
    if (box.length !== 4 || !box.every(Number.isInteger) || box[0] < 0 || box[1] < 0 ||
        box[2] <= box[0] || box[3] <= box[1] || box[2] > image.naturalWidth || box[3] > image.naturalHeight) {
      $("image-error").textContent = "裁剪记录无效 Invalid crop";
      return;
    }
    const canvas = $("observation");
    canvas.width = box[2] - box[0]; canvas.height = box[3] - box[1];
    canvas.getContext("2d").drawImage(image, box[0], box[1], canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);
    canvas.hidden = false;
    $("image-stage").querySelector("p").hidden = true;
  };
  image.onerror = () => {
    if (generation !== imageVersion) return;
    $("observation").hidden = true;
    $("image-stage").querySelector("p").hidden = false;
    $("image-stage").querySelector("p").textContent = "观测图片不可用";
    $("image-error").textContent = "观测 PNG 无法加载 Observation image unavailable";
  };
  image.src = `/observations/${shown.observation_id}.png`;
}

function scheduleRender() {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    renderSummary(); renderTimeline(); renderSelected(); renderImages();
    $("notices").textContent = model.notices.join("\n\n") || "暂无异常";
    $("codex-stderr").textContent = model.stderr || "暂无输出";
    $("diagnostic-count").textContent = model.notices.length ? `· ${model.notices.length} 条记录` : "";
  });
}
$("follow").addEventListener("change", () => {
  if ($("follow").checked) {
    const latest = [...model.cells.keys()].sort((a, b) => a - b).at(-1);
    if (latest !== undefined) model.selected = {kind: "cell", id: latest};
    model.imageKey = null;
  }
  scheduleRender();
});
$("image-select").addEventListener("change", () => {
  $("follow").checked = false;
  model.imageKey = $("image-select").value;
  scheduleRender();
});
const events = new EventSource("/events");
events.onopen = () => setBadge("connection", "监控连接正常", "good");
events.onerror = () => setBadge("connection", "连接中断 · 正在重连", "warn");
events.onmessage = (event) => {
  try { consume(JSON.parse(event.data)); }
  catch (error) {
    model.notices.push(`页面读取事件失败 Event error: ${error.message}`);
    scheduleRender();
  }
};
setInterval(updateElapsed, 1000);
