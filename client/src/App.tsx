// 多图工作流画布 — multi-image workflow canvas.
//
// IMAGE nodes hold reference photos (with a role), the LLM node analyzes
// connected images + the user's input into a generation prompt, the 生成
// node renders images with Muse's pipeline (connected references are
// vision-analyzed and woven into the prompt), OUTPUT collects results.
// All state lives server-side: projects + canvas JSON + blob images +
// generation history.

import {
useCallback,
useEffect,
useMemo,
useRef,
useState,
type ChangeEvent,
type PointerEvent as ReactPointerEvent,
type ReactNode,
} from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { bytesToBase64 } from "@hatch/space-sdk/client";
import { api, type ApiResponse } from "./api";

// ------------------------------------------------------------------ types

type NodeType = "image" | "llm" | "generate" | "output";
type Orientation = "square" | "landscape" | "portrait";

interface ImageData {
  blobKey: string | null;
  url: string | null;
  role: string;
  name: string;
}
interface LlmData {
  input: string;
  system: string;
  mode: "prompt" | "chat";
  output: string;
  status: string;
  error?: string;
}
interface GenerateData {
  prompts: string;
  orientation: Orientation;
  count: number;
  status: string;
  error?: string;
}
interface OutputData {
  images: { blobKey: string; url: string }[];
}
type NodeData = ImageData | LlmData | GenerateData | OutputData;

interface WFNode {
  id: string;
  type: NodeType;
  x: number;
  y: number;
  data: NodeData;
}
interface WFEdge {
  id: string;
  from: string;
  to: string;
}
interface Viewport {
  x: number;
  y: number;
  k: number;
}
interface CanvasDoc {
  nodes: WFNode[];
  edges: WFEdge[];
  viewport: Viewport;
}

type Project = ApiResponse<typeof api, "listProjects">["projects"][number];
type HistoryItem = ApiResponse<typeof api, "listHistory">["items"][number];

const NODE_W: Record<NodeType, number> = {
  image: 236,
  llm: 348,
  generate: 328,
  output: 308,
};
const TYPE_LABEL: Record<NodeType, string> = {
  image: "IMAGE 参考图",
  llm: "LLM 提示词",
  generate: "生成",
  output: "OUTPUT 结果",
};
const TYPE_ACCENT: Record<NodeType, string> = {
  image: "#38bdf8",
  llm: "#f5b83d",
  generate: "#34d399",
  output: "#fb7185",
};
const ROLES = ["人物", "服装", "场景", "动作构图", "商品", "风格", "其他"];
const ORIENTATIONS: { value: Orientation; label: string }[] = [
  { value: "landscape", label: "横版" },
  { value: "portrait", label: "竖版" },
  { value: "square", label: "方形" },
];

let idCounter = 0;
function newId(prefix: string): string {
  idCounter += 1;
  return `${prefix}-${Date.now().toString(36)}-${idCounter}`;
}

function defaultData(type: NodeType): NodeData {
  switch (type) {
    case "image":
      return { blobKey: null, url: null, role: "人物", name: "" };
    case "llm":
      return { input: "", system: "", mode: "prompt", output: "", status: "idle" };
    case "generate":
      return { prompts: "", orientation: "landscape", count: 1, status: "idle" };
    case "output":
      return { images: [] };
  }
}

function parseCanvas(raw: string): CanvasDoc {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) throw new Error("bad");
    const doc = parsed as Partial<CanvasDoc>;
    const nodes = Array.isArray(doc.nodes) ? doc.nodes : [];
    // A node left "running" by a closed page is idle again on load.
    for (const n of nodes) {
      const d = n.data as { status?: string };
      if (d && d.status === "running") d.status = "idle";
    }
    return {
      nodes,
      edges: Array.isArray(doc.edges) ? doc.edges : [],
      viewport:
        doc.viewport && typeof doc.viewport.k === "number"
          ? doc.viewport
          : { x: 0, y: 0, k: 1 },
    };
  } catch {
    return { nodes: [], edges: [], viewport: { x: 0, y: 0, k: 1 } };
  }
}

function serialize(doc: CanvasDoc, viewport: Viewport): string {
  return JSON.stringify({ nodes: doc.nodes, edges: doc.edges, viewport });
}

/** Viewport that frames every node inside a w×h canvas area. */
function computeFit(nodes: WFNode[], w: number, h: number): Viewport {
  if (nodes.length === 0 || w <= 0 || h <= 0) return { x: 0, y: 0, k: 1 };
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const n of nodes) {
    minX = Math.min(minX, n.x);
    minY = Math.min(minY, n.y);
    maxX = Math.max(maxX, n.x + NODE_W[n.type]);
    maxY = Math.max(maxY, n.y + 320);
  }
  const bw = Math.max(1, maxX - minX);
  const bh = Math.max(1, maxY - minY);
  const k = Math.max(0.1, Math.min(1.4, (w - 48) / bw, (h - 48) / bh));
  return {
    k,
    x: (w - bw * k) / 2 - minX * k,
    y: (h - bh * k) / 2 - minY * k,
  };
}

// ------------------------------------------------------------- image utils

/** Downscale a picked file and return base64 (no data: prefix) + mime. */
async function fileToScaledBase64(
  file: File,
  maxDim: number,
  asPng: boolean,
): Promise<{ dataBase64: string; mimeType: string }> {
  const bmp = await createImageBitmap(file);
  const scale = Math.min(1, maxDim / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * scale));
  const h = Math.max(1, Math.round(bmp.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const cx = canvas.getContext("2d");
  if (!cx) throw new Error("浏览器不支持 canvas");
  cx.drawImage(bmp, 0, 0, w, h);
  const mimeType = asPng ? "image/png" : "image/jpeg";
  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, mimeType, 0.9),
  );
  if (!blob) throw new Error("图片处理失败");
  const buf = await blob.arrayBuffer();
  return { dataBase64: bytesToBase64(new Uint8Array(buf)), mimeType };
}

/** Fetch one of our own blob URLs and return it as PNG base64 for vision. */
async function urlToVisionImage(
  url: string,
): Promise<{ dataBase64: string; mimeType: string }> {
  const res = await fetch(url);
  const blob = await res.blob();
  const file = new File([blob], "ref", { type: blob.type || "image/png" });
  return fileToScaledBase64(file, 1024, true);
}

function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

async function blobUrlToDataUrl(url: string): Promise<string> {
  const res = await fetch(url);
  const blob = await res.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("读取图片失败"));
    reader.readAsDataURL(blob);
  });
}

// ------------------------------------------------------------- node shell

interface NodeActions {
  update: (id: string, patch: Partial<NodeData>) => void;
  remove: (id: string) => void;
  startConnect: (id: string) => void;
  endConnect: (id: string) => void;
  upload: (id: string, file: File) => void;
  runLLM: (id: string) => void;
  runGenerate: (id: string) => void;
  runAll: (id: string) => void;
  openLightbox: (url: string) => void;
}

function Port({
  side,
  active,
  onClick,
  label,
}: {
  side: "in" | "out";
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      onPointerDown={(e) => e.stopPropagation()}
      className="absolute z-20 flex items-center justify-center rounded-full"
      style={{
        top: 13,
        width: 22,
        height: 22,
        minHeight: 22,
        ...(side === "in" ? { left: -11 } : { right: -11 }),
      }}
    >
      <span
        className="block rounded-full border-2 transition-transform"
        style={{
          width: active ? 16 : 12,
          height: active ? 16 : 12,
          borderColor: active ? "#f5b83d" : "#4b5568",
          background: active ? "#f5b83d" : "#171a20",
        }}
      />
    </button>
  );
}

function NodeShell({
  node,
  pendingFrom,
  actions,
  children,
  onHeaderPointerDown,
}: {
  node: WFNode;
  pendingFrom: string | null;
  actions: NodeActions;
  children: ReactNode;
  onHeaderPointerDown: (node: WFNode, e: ReactPointerEvent) => void;
}) {
  const accent = TYPE_ACCENT[node.type];
  const canIn = node.type !== "image";
  const canOut = node.type !== "output";
  return (
    <div
      data-node="1"
      className="absolute select-none rounded-xl border bg-[#171a20] shadow-[0_10px_30px_rgba(0,0,0,0.45)]"
      style={{
        left: node.x,
        top: node.y,
        width: NODE_W[node.type],
        borderColor: "#2b303c",
      }}
    >
      <div
        className="flex cursor-grab items-center gap-2 rounded-t-xl px-3 active:cursor-grabbing"
        style={{ height: 38, background: "#1d212a", touchAction: "none" }}
        onPointerDown={(e) => onHeaderPointerDown(node, e)}
      >
        <span
          className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm"
          style={{ background: accent }}
        />
        <span className="font-mono text-xs font-semibold tracking-wide text-[#e8eaf0]">
          {TYPE_LABEL[node.type]}
        </span>
        <button
          type="button"
          aria-label={`删除节点 ${TYPE_LABEL[node.type]}`}
          className="ml-auto flex h-8 w-8 items-center justify-center rounded text-[#9aa3b2] hover:bg-[#262b36] hover:text-white"
          onPointerDown={(e) => e.stopPropagation()}
          onClick={() => actions.remove(node.id)}
        >
          ×
        </button>
      </div>
      {canIn && (
        <Port
          side="in"
          label={`${TYPE_LABEL[node.type]} 输入端口`}
          active={pendingFrom !== null && pendingFrom !== node.id}
          onClick={() => actions.endConnect(node.id)}
        />
      )}
      {canOut && (
        <Port
          side="out"
          label={`${TYPE_LABEL[node.type]} 输出端口`}
          active={pendingFrom === node.id}
          onClick={() => actions.startConnect(node.id)}
        />
      )}
      <div className="node-body px-3 py-3">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------- node bodies

function StatusLine({ status, error }: { status: string; error?: string }) {
  if (status === "running")
    return <p className="mt-2 font-mono text-xs text-[#f5b83d]">运行中…（由 Muse 执行）</p>;
  if (status === "error")
    return <p className="mt-2 text-xs text-[#fb7185]">{error ?? "运行失败"}</p>;
  if (status === "done")
    return <p className="mt-2 font-mono text-xs text-[#34d399]">✓ 完成</p>;
  return null;
}

function ImageNodeBody({
  node,
  data,
  actions,
}: {
  node: WFNode;
  data: ImageData;
  actions: NodeActions;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) actions.upload(node.id, f);
    e.target.value = "";
  };
  return (
    <div>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        aria-label="选择参考图文件"
        onChange={onFile}
      />
      {data.url ? (
        <div>
          <img
            src={data.url}
            alt={data.name || "参考图"}
            className="h-36 w-full cursor-zoom-in rounded-lg object-cover"
            onClick={() => actions.openLightbox(data.url as string)}
          />
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              className="flex-1 rounded-lg bg-[#262b36] px-2 text-xs text-[#e8eaf0]"
              onClick={() => fileRef.current?.click()}
            >
              替换图片
            </button>
            <button
              type="button"
              className="flex-1 rounded-lg bg-[#262b36] px-2 text-xs text-[#fb7185]"
              onClick={() =>
                actions.update(node.id, { blobKey: null, url: null, name: "" })
              }
            >
              移除图片
            </button>
          </div>
        </div>
      ) : (
        <button
          type="button"
          className="flex h-36 w-full flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-[#3a4152] text-[#9aa3b2]"
          onClick={() => fileRef.current?.click()}
        >
          <span className="text-2xl leading-none">＋</span>
          <span className="text-xs">点击上传参考图</span>
          <span className="font-mono text-[11px] text-[#5f6879]">PNG / JPG / WebP</span>
        </button>
      )}
      <label className="mt-3 block">
        <span className="font-mono text-[11px] tracking-wide text-[#9aa3b2]">
          图片角色（随图号传给模型）
        </span>
        <select
          aria-label="图片角色"
          className="mt-1 w-full rounded-lg border border-[#2b303c] bg-[#101216] px-2 text-sm text-[#e8eaf0]"
          value={data.role}
          onChange={(e) => actions.update(node.id, { role: e.target.value })}
        >
          {ROLES.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

function LlmNodeBody({
  node,
  data,
  actions,
}: {
  node: WFNode;
  data: LlmData;
  actions: NodeActions;
}) {
  const ta =
    "w-full rounded-lg border border-[#2b303c] bg-[#101216] px-2 py-2 text-sm text-[#e8eaf0] placeholder:text-[#5f6879]";
  return (
    <div>
      <div
        className="mb-2 flex rounded-lg bg-[#101216] p-1"
        role="group"
        aria-label="LLM 节点模式"
      >
        {(
          [
            { v: "prompt", label: "节点模式 · 写提示词" },
            { v: "chat", label: "聊天模式 · 看图问答" },
          ] as const
        ).map((m) => (
          <button
            key={m.v}
            type="button"
            aria-pressed={data.mode === m.v}
            className="flex-1 rounded-md px-1 text-xs"
            style={
              data.mode === m.v
                ? { background: "#f5b83d", color: "#171207", fontWeight: 600 }
                : { color: "#9aa3b2" }
            }
            onClick={() => actions.update(node.id, { mode: m.v })}
          >
            {m.label}
          </button>
        ))}
      </div>
      <label className="block">
        <span className="font-mono text-[11px] tracking-wide text-[#9aa3b2]">INPUT · 你的要求（用图1、图2…指代连入的参考图）</span>
        <textarea
          aria-label="LLM 输入要求"
          className={`${ta} mt-1 max-h-44 min-h-20`}
          value={data.input}
          placeholder="例如：图1是人物身份参考，图2只参考站姿，图3参考场景与光线…"
          onChange={(e) => actions.update(node.id, { input: e.target.value })}
        />
      </label>
      <label className="mt-2 block">
        <span className="font-mono text-[11px] tracking-wide text-[#9aa3b2]">SYSTEM · 系统提示词（可留空用默认）</span>
        <textarea
          aria-label="LLM 系统提示词"
          className={`${ta} mt-1 max-h-28 min-h-12`}
          value={data.system}
          placeholder="留空则使用默认的提示词专家设定"
          onChange={(e) => actions.update(node.id, { system: e.target.value })}
        />
      </label>
      <button
        type="button"
        className="mt-3 w-full rounded-lg bg-[#f5b83d] font-semibold text-[#171207] disabled:opacity-40"
        disabled={data.status === "running" || !data.input.trim()}
        onClick={() => actions.runLLM(node.id)}
      >
        {data.status === "running" ? "Muse 分析中…" : "Run LLM"}
      </button>
      <label className="mt-3 block">
        <span className="font-mono text-[11px] tracking-wide text-[#9aa3b2]">OUTPUT · 模型输出（可编辑，会自动填入相连的生成节点）</span>
        <textarea
          aria-label="LLM 输出"
          className={`${ta} mt-1 max-h-44 min-h-20`}
          value={data.output}
          placeholder="运行后，Muse 整理好的提示词会出现在这里"
          onChange={(e) => actions.update(node.id, { output: e.target.value })}
        />
      </label>
      <StatusLine status={data.status} error={data.error} />
    </div>
  );
}

function GenerateNodeBody({
  node,
  data,
  actions,
  refThumbs,
  hasLlm,
}: {
  node: WFNode;
  data: GenerateData;
  actions: NodeActions;
  refThumbs: { url: string; role: string }[];
  hasLlm: boolean;
}) {
  const ta =
    "w-full rounded-lg border border-[#2b303c] bg-[#101216] px-2 py-2 text-sm text-[#e8eaf0] placeholder:text-[#5f6879]";
  const running = data.status === "running";
  return (
    <div>
      <label className="block">
        <span className="font-mono text-[11px] tracking-wide text-[#9aa3b2]">PROMPTS · 实际生图提示词</span>
        <textarea
          aria-label="生图提示词"
          className={`${ta} mt-1 max-h-44 min-h-20`}
          value={data.prompts}
          placeholder="直接写提示词，或先跑相连的 LLM 节点自动填入"
          onChange={(e) => actions.update(node.id, { prompts: e.target.value })}
        />
      </label>
      <div className="mt-2">
        <span className="font-mono text-[11px] tracking-wide text-[#9aa3b2]">
          IMAGES · 连入的参考图（生成时由 Muse 先看图，再融入提示词）
        </span>
        {refThumbs.length > 0 ? (
          <div className="mt-1 flex flex-wrap gap-1.5">
            {refThumbs.map((t, i) => (
              <span key={i} className="relative inline-block">
                <img
                  src={t.url}
                  alt={`图${i + 1} ${t.role}`}
                  className="h-11 w-11 rounded-md object-cover"
                />
                <span className="absolute bottom-0 left-0 rounded-tr-md bg-black/70 px-1 font-mono text-[10px] text-white">
                  图{i + 1}·{t.role}
                </span>
              </span>
            ))}
          </div>
        ) : (
          <p className="mt-1 text-xs text-[#5f6879]">未连接参考图，将按纯文字生图</p>
        )}
      </div>
      <div className="mt-3 flex items-end gap-2">
        <label className="flex-1">
          <span className="font-mono text-[11px] tracking-wide text-[#9aa3b2]">画幅</span>
          <div className="mt-1 flex rounded-lg bg-[#101216] p-1" role="group" aria-label="画幅">
            {ORIENTATIONS.map((o) => (
              <button
                key={o.value}
                type="button"
                aria-pressed={data.orientation === o.value}
                className="flex-1 rounded-md text-xs"
                style={
                  data.orientation === o.value
                    ? { background: "#34d399", color: "#0b1712", fontWeight: 600 }
                    : { color: "#9aa3b2" }
                }
                onClick={() => actions.update(node.id, { orientation: o.value })}
              >
                {o.label}
              </button>
            ))}
          </div>
        </label>
        <label>
          <span className="font-mono text-[11px] tracking-wide text-[#9aa3b2]">数量</span>
          <select
            aria-label="生成数量"
            className="mt-1 rounded-lg border border-[#2b303c] bg-[#101216] px-2 text-sm text-[#e8eaf0]"
            value={data.count}
            onChange={(e) =>
              actions.update(node.id, { count: Number(e.target.value) })
            }
          >
            {[1, 2, 3, 4].map((n) => (
              <option key={n} value={n}>
                {n} 张
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="mt-3 flex gap-2">
        <button
          type="button"
          className="flex-1 rounded-lg bg-[#34d399] font-semibold text-[#0b1712] disabled:opacity-40"
          disabled={running || !data.prompts.trim()}
          onClick={() => actions.runGenerate(node.id)}
        >
          {running ? "生成中…" : "生成"}
        </button>
        {hasLlm && (
          <button
            type="button"
            className="flex-1 rounded-lg bg-[#f5b83d] font-semibold text-[#171207] disabled:opacity-40"
            disabled={running}
            onClick={() => actions.runAll(node.id)}
          >
            一键运行（LLM＋生成）
          </button>
        )}
      </div>
      <StatusLine status={data.status} error={data.error} />
    </div>
  );
}

function OutputNodeBody({
  node,
  data,
  actions,
}: {
  node: WFNode;
  data: OutputData;
  actions: NodeActions;
}) {
  if (data.images.length === 0) {
    return (
      <p className="py-6 text-center text-xs text-[#5f6879]">
        生成结果会出现在这里
      </p>
    );
  }
  return (
    <div className="grid grid-cols-2 gap-2">
      {data.images.map((im, i) => (
        <figure key={`${im.blobKey}-${i}`} className="relative">
          <img
            src={im.url}
            alt={`生成结果 ${i + 1}`}
            className="aspect-square w-full cursor-zoom-in rounded-lg object-cover"
            onClick={() => actions.openLightbox(im.url)}
          />
          <a
            href={im.url}
            download
            target="_blank"
            rel="noreferrer"
            aria-label={`下载生成结果 ${i + 1}`}
            className="absolute right-1 top-1 flex h-8 w-8 items-center justify-center rounded-md bg-black/70 text-sm text-white"
            onClick={(e) => e.stopPropagation()}
          >
            ⬇
          </a>
        </figure>
      ))}
      <button
        type="button"
        className="col-span-2 rounded-lg bg-[#262b36] px-2 text-xs text-[#9aa3b2]"
        onClick={() => actions.update(node.id, { images: [] })}
      >
        清空结果
      </button>
    </div>
  );
}

// ------------------------------------------------------------------- app

export function App() {
  const queryClient = useQueryClient();
  const [projectId, setProjectId] = useState<number | null>(() => {
    const saved = Number(localStorage.getItem("wf-project-id"));
    return Number.isFinite(saved) && saved > 0 ? saved : null;
  });
  const [doc, setDoc] = useState<CanvasDoc | null>(null);
  const [viewport, setViewport] = useState<Viewport>({ x: 0, y: 0, k: 1 });
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [pendingFrom, setPendingFrom] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [mouse, setMouse] = useState<{ x: number; y: number } | null>(null);
  const [lightbox, setLightbox] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [showAddMenu, setShowAddMenu] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [renameText, setRenameText] = useState("");
  const [notice, setNotice] = useState<string | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const loadedForRef = useRef<number | null>(null);
  const savedJsonRef = useRef("");
  const seededRef = useRef(false);
  const docRef = useRef<CanvasDoc | null>(null);
  docRef.current = doc;
  const viewportRef = useRef(viewport);
  viewportRef.current = viewport;

  // ------------------------------------------------------------ projects
  const projectsQuery = useQuery({
    queryKey: ["projects"],
    queryFn: () => api.listProjects({}),
  });
  const projects = useMemo(
    () => projectsQuery.data?.projects ?? [],
    [projectsQuery.data],
  );

  const seedMutation = useMutation({
    mutationFn: () => api.seedStarter({}),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["projects"] }),
  });

  useEffect(() => {
    if (projectsQuery.data && projects.length === 0 && !seededRef.current) {
      seededRef.current = true;
      seedMutation.mutate();
    }
  }, [projectsQuery.data, projects.length, seedMutation]);

  useEffect(() => {
    if (projects.length === 0) return;
    if (projectId === null || !projects.some((p) => p.id === projectId)) {
      const first = projects[0];
      if (first) setProjectId(first.id);
    }
  }, [projects, projectId]);

  useEffect(() => {
    if (projectId !== null) {
      localStorage.setItem("wf-project-id", String(projectId));
    }
  }, [projectId]);

  const currentProject: Project | undefined = projects.find(
    (p) => p.id === projectId,
  );

  // -------------------------------------------------------------- canvas
  const canvasQuery = useQuery({
    queryKey: ["canvas", projectId],
    queryFn: () => api.getCanvas({ projectId: projectId as number }),
    enabled: projectId !== null,
  });

  useEffect(() => {
    if (!canvasQuery.data || projectId === null) return;
    if (loadedForRef.current === projectId) return;
    const parsed = parseCanvas(canvasQuery.data.canvas);
    setDoc(parsed);
    // Open framed on the whole workflow so every node is reachable at once,
    // especially on a phone-width canvas.
    const el = containerRef.current;
    const fit = el
      ? computeFit(parsed.nodes, el.clientWidth, el.clientHeight)
      : parsed.viewport;
    setViewport(fit);
    setPendingFrom(null);
    setSelectedEdge(null);
    loadedForRef.current = projectId;
    savedJsonRef.current = serialize(parsed, fit);
    setSaveState("idle");
  }, [canvasQuery.data, projectId]);

  // Autosave (debounced) whenever the working copy diverges.
  useEffect(() => {
    if (!doc || projectId === null || loadedForRef.current !== projectId) return;
    const json = serialize(doc, viewport);
    if (json === savedJsonRef.current) return;
    setSaveState("saving");
    const t = setTimeout(() => {
      api
        .saveCanvas({ projectId, canvas: json })
        .then(() => {
          savedJsonRef.current = json;
          setSaveState("saved");
        })
        .catch(() => setSaveState("error"));
    }, 900);
    return () => clearTimeout(t);
  }, [doc, viewport, projectId]);

  const historyQuery = useQuery({
    queryKey: ["history", projectId],
    queryFn: () => api.listHistory({ projectId: projectId as number, limit: 50 }),
    enabled: projectId !== null && showHistory,
  });

  // -------------------------------------------------------- doc mutation
  const updateNode = useCallback((id: string, patch: Partial<NodeData>) => {
    setDoc((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        nodes: prev.nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, ...patch } as NodeData } : n,
        ),
      };
    });
  }, []);

  const removeNode = useCallback((id: string) => {
    setDoc((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        nodes: prev.nodes.filter((n) => n.id !== id),
        edges: prev.edges.filter((e) => e.from !== id && e.to !== id),
      };
    });
    setPendingFrom((p) => (p === id ? null : p));
  }, []);

  const addNode = useCallback(
    (type: NodeType) => {
      const el = containerRef.current;
      const v = viewportRef.current;
      const cx = el ? el.clientWidth / 2 : 300;
      const cy = el ? el.clientHeight / 2 : 300;
      const node: WFNode = {
        id: newId(type),
        type,
        x: (cx - v.x) / v.k - NODE_W[type] / 2 + Math.random() * 30,
        y: (cy - v.y) / v.k - 80 + Math.random() * 30,
        data: defaultData(type),
      };
      setDoc((prev) => (prev ? { ...prev, nodes: [...prev.nodes, node] } : prev));
      setShowAddMenu(false);
    },
    [],
  );

  const canConnect = useCallback((from: WFNode, to: WFNode): string | null => {
    if (from.id === to.id) return "不能连接自己";
    const ok =
      (from.type === "image" && (to.type === "llm" || to.type === "generate")) ||
      (from.type === "llm" && to.type === "generate") ||
      (from.type === "generate" && to.type === "output");
    if (!ok) return `不支持 ${TYPE_LABEL[from.type]} → ${TYPE_LABEL[to.type]}`;
    return null;
  }, []);

  const startConnect = useCallback((id: string) => {
    setPendingFrom((p) => (p === id ? null : id));
    setSelectedEdge(null);
  }, []);

  const endConnect = useCallback(
    (toId: string) => {
      const fromId = pendingFrom;
      setPendingFrom(null);
      if (!fromId || fromId === toId) return;
      const d = docRef.current;
      if (!d) return;
      const from = d.nodes.find((n) => n.id === fromId);
      const to = d.nodes.find((n) => n.id === toId);
      if (!from || !to) return;
      const problem = canConnect(from, to);
      if (problem) {
        setNotice(problem);
        return;
      }
      setDoc((prev) => {
        if (!prev) return prev;
        if (prev.edges.some((e) => e.from === fromId && e.to === toId)) return prev;
        // A generate node accepts only one LLM input.
        const edges =
          from.type === "llm"
            ? prev.edges.filter(
                (e) =>
                  !(
                    e.to === toId &&
                    prev.nodes.find((n) => n.id === e.from)?.type === "llm"
                  ),
              )
            : prev.edges;
        return {
          ...prev,
          edges: [...edges, { id: newId("e"), from: fromId, to: toId }],
        };
      });
    },
    [pendingFrom, canConnect],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setPendingFrom(null);
        setSelectedEdge(null);
        setLightbox(null);
      }
      if ((e.key === "Delete" || e.key === "Backspace") && selectedEdge) {
        const target = e.target as HTMLElement | null;
        if (target && (target.tagName === "TEXTAREA" || target.tagName === "INPUT")) return;
        setDoc((prev) =>
          prev
            ? { ...prev, edges: prev.edges.filter((ed) => ed.id !== selectedEdge) }
            : prev,
        );
        setSelectedEdge(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedEdge]);

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 2600);
    return () => clearTimeout(t);
  }, [notice]);

  // ------------------------------------------------------------ uploads
  const uploadToNode = useCallback(
    async (id: string, file: File) => {
      try {
        const scaled = await fileToScaledBase64(file, 1280, file.type === "image/png");
        const res = await api.uploadImage(scaled);
        updateNode(id, { blobKey: res.blob_key, url: res.url, name: file.name });
      } catch (err) {
        setNotice(`上传失败：${err instanceof Error ? err.message : String(err)}`);
      }
    },
    [updateNode],
  );

  // ------------------------------------------------------------- running
  /** Images connected into a node, in edge order (= 图1, 图2, …). */
  const connectedImages = useCallback((targetId: string) => {
    const d = docRef.current;
    if (!d) return [];
    const out: { node: WFNode; data: ImageData }[] = [];
    for (const e of d.edges) {
      if (e.to !== targetId) continue;
      const n = d.nodes.find((x) => x.id === e.from);
      if (n && n.type === "image") {
        const data = n.data as ImageData;
        if (data.url) out.push({ node: n, data });
      }
    }
    return out;
  }, []);

  const visionPayload = useCallback(
    async (targetId: string) => {
      const imgs = connectedImages(targetId);
      const payload = [];
      for (let i = 0; i < imgs.length; i++) {
        const item = imgs[i];
        if (!item) continue;
        const v = await urlToVisionImage(item.data.url as string);
        payload.push({
          dataBase64: v.dataBase64,
          mimeType: v.mimeType,
          caption: `图${i + 1}（角色：${item.data.role}）`,
        });
      }
      return payload;
    },
    [connectedImages],
  );

  const runLLM = useCallback(
    async (id: string): Promise<string | null> => {
      const d = docRef.current;
      const node = d?.nodes.find((n) => n.id === id);
      if (!node || node.type !== "llm") return null;
      const data = node.data as LlmData;
      updateNode(id, { status: "running", error: undefined });
      try {
        const images = await visionPayload(id);
        const res = await api.runLLM({
          input: data.input,
          system: data.system,
          mode: data.mode,
          images,
        });
        updateNode(id, { status: "done", output: res.text, error: undefined });
        // The original fills the downstream generate node's PROMPTS.
        const dd = docRef.current;
        if (dd) {
          for (const e of dd.edges) {
            if (e.from !== id) continue;
            const target = dd.nodes.find((n) => n.id === e.to);
            if (target && target.type === "generate") {
              updateNode(target.id, { prompts: res.text });
            }
          }
        }
        return res.text;
      } catch (err) {
        updateNode(id, {
          status: "error",
          error: `LLM 运行失败：${err instanceof Error ? err.message : String(err)}`,
        });
        return null;
      }
    },
    [updateNode, visionPayload],
  );

  const runGenerate = useCallback(
    async (id: string, promptOverride?: string) => {
      const d = docRef.current;
      const node = d?.nodes.find((n) => n.id === id);
      if (!node || node.type !== "generate" || projectId === null) return;
      const data = node.data as GenerateData;
      const prompt = (promptOverride ?? data.prompts).trim();
      if (!prompt) {
        updateNode(id, { status: "error", error: "提示词为空，先填写 PROMPTS 或运行 LLM" });
        return;
      }
      updateNode(id, { status: "running", error: undefined });
      try {
        const refImages = await visionPayload(id);
        const res = await api.runGenerate({
          projectId,
          prompt,
          orientation: data.orientation,
          count: data.count,
          refImages,
        });
        updateNode(id, { status: "done", error: undefined });
        const dd = docRef.current;
        if (dd) {
          for (const e of dd.edges) {
            if (e.from !== id) continue;
            const target = dd.nodes.find((n) => n.id === e.to);
            if (target && target.type === "output") {
              const out = target.data as OutputData;
              const mapped = res.images.map((im) => ({
                blobKey: im.blob_key,
                url: im.url,
              }));
              updateNode(target.id, { images: [...out.images, ...mapped] });
            }
          }
        }
        queryClient.invalidateQueries({ queryKey: ["history", projectId] });
      } catch (err) {
        updateNode(id, {
          status: "error",
          error: `生成失败：${err instanceof Error ? err.message : String(err)}`,
        });
      }
    },
    [projectId, updateNode, visionPayload, queryClient],
  );

  const runAll = useCallback(
    async (genId: string) => {
      const d = docRef.current;
      if (!d) return;
      const llmEdge = d.edges.find((e) => {
        if (e.to !== genId) return false;
        return d.nodes.find((n) => n.id === e.from)?.type === "llm";
      });
      if (llmEdge) {
        const text = await runLLM(llmEdge.from);
        if (text === null) return;
        await runGenerate(genId, text);
      } else {
        await runGenerate(genId);
      }
    },
    [runLLM, runGenerate],
  );

  const actions: NodeActions = useMemo(
    () => ({
      update: updateNode,
      remove: removeNode,
      startConnect,
      endConnect,
      upload: (id, file) => void uploadToNode(id, file),
      runLLM: (id) => void runLLM(id),
      runGenerate: (id) => void runGenerate(id),
      runAll: (id) => void runAll(id),
      openLightbox: (url) => setLightbox(url),
    }),
    [updateNode, removeNode, startConnect, endConnect, uploadToNode, runLLM, runGenerate, runAll],
  );

  // ------------------------------------------------------ canvas gestures
  const panRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    vx: number;
    vy: number;
  } | null>(null);
  const pointersRef = useRef(new Map<number, { x: number; y: number }>());
  const pinchRef = useRef<{ dist: number; k: number; cx: number; cy: number } | null>(null);
  const dragRef = useRef<{
    id: string;
    startX: number;
    startY: number;
    nx: number;
    ny: number;
  } | null>(null);

  const onHeaderPointerDown = useCallback(
    (node: WFNode, e: ReactPointerEvent) => {
      if (e.button !== 0 && e.pointerType === "mouse") return;
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      dragRef.current = {
        id: node.id,
        startX: e.clientX,
        startY: e.clientY,
        nx: node.x,
        ny: node.y,
      };
    },
    [],
  );

  const onContainerPointerDown = (e: ReactPointerEvent) => {
    const target = e.target as HTMLElement;
    pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointersRef.current.size === 2) {
      const pts = [...pointersRef.current.values()];
      const a = pts[0];
      const b = pts[1];
      if (a && b) {
        pinchRef.current = {
          dist: Math.hypot(a.x - b.x, a.y - b.y),
          k: viewportRef.current.k,
          cx: (a.x + b.x) / 2,
          cy: (a.y + b.y) / 2,
        };
      }
      panRef.current = null;
      return;
    }
    if (target.closest("[data-node]")) return;
    (containerRef.current as HTMLElement).setPointerCapture(e.pointerId);
    panRef.current = {
      pointerId: e.pointerId,
      startX: e.clientX,
      startY: e.clientY,
      vx: viewportRef.current.x,
      vy: viewportRef.current.y,
    };
    setSelectedEdge(null);
  };

  const onContainerPointerMove = (e: ReactPointerEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) setMouse({ x: e.clientX - rect.left, y: e.clientY - rect.top });

    if (dragRef.current) {
      const drag = dragRef.current;
      const k = viewportRef.current.k;
      const nx = drag.nx + (e.clientX - drag.startX) / k;
      const ny = drag.ny + (e.clientY - drag.startY) / k;
      setDoc((prev) =>
        prev
          ? {
              ...prev,
              nodes: prev.nodes.map((n) =>
                n.id === drag.id ? { ...n, x: nx, y: ny } : n,
              ),
            }
          : prev,
      );
      return;
    }

    if (pointersRef.current.has(e.pointerId)) {
      pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    }
    if (pinchRef.current && pointersRef.current.size === 2 && rect) {
      const pts = [...pointersRef.current.values()];
      const a = pts[0];
      const b = pts[1];
      if (a && b) {
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        const pinch = pinchRef.current;
        const nk = Math.min(2.5, Math.max(0.2, pinch.k * (dist / Math.max(20, pinch.dist))));
        const mx = (a.x + b.x) / 2 - rect.left;
        const my = (a.y + b.y) / 2 - rect.top;
        const v = viewportRef.current;
        const wx = (mx - v.x) / v.k;
        const wy = (my - v.y) / v.k;
        setViewport({ k: nk, x: mx - wx * nk, y: my - wy * nk });
      }
      return;
    }

    if (panRef.current) {
      const pan = panRef.current;
      setViewport((v) => ({
        ...v,
        x: pan.vx + (e.clientX - pan.startX),
        y: pan.vy + (e.clientY - pan.startY),
      }));
    }
  };

  const endPointer = (e: ReactPointerEvent) => {
    pointersRef.current.delete(e.pointerId);
    if (pointersRef.current.size < 2) pinchRef.current = null;
    if (panRef.current && panRef.current.pointerId === e.pointerId) {
      panRef.current = null;
    }
    dragRef.current = null;
  };

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const v = viewportRef.current;
      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      const nk = Math.min(2.5, Math.max(0.2, v.k * factor));
      const wx = (mx - v.x) / v.k;
      const wy = (my - v.y) / v.k;
      setViewport({ k: nk, x: mx - wx * nk, y: my - wy * nk });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const fitView = useCallback(() => {
    const el = containerRef.current;
    const d = docRef.current;
    if (!el || !d) return;
    setViewport(computeFit(d.nodes, el.clientWidth, el.clientHeight));
  }, []);

  const zoomBy = (factor: number) => {
    const el = containerRef.current;
    if (!el) return;
    const mx = el.clientWidth / 2;
    const my = el.clientHeight / 2;
    const v = viewportRef.current;
    const nk = Math.min(2.5, Math.max(0.2, v.k * factor));
    const wx = (mx - v.x) / v.k;
    const wy = (my - v.y) / v.k;
    setViewport({ k: nk, x: mx - wx * nk, y: my - wy * nk });
  };

  // ----------------------------------------------------- project actions
  const createMutation = useMutation({
    mutationFn: (name: string) => api.createProject({ name }),
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      loadedForRef.current = null;
      setProjectId(res.id);
    },
  });
  const renameMutation = useMutation({
    mutationFn: (args: { id: number; name: string }) => api.renameProject(args),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["projects"] }),
  });
  const deleteMutation = useMutation({
    mutationFn: (id: number) => api.deleteProject({ id }),
    onSuccess: () => {
      loadedForRef.current = null;
      setProjectId(null);
      queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });
  const deleteHistoryMutation = useMutation({
    mutationFn: (id: number) => api.deleteHistory({ id }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["history", projectId] }),
  });

  // ------------------------------------------------------- export/import
  const exportWorkflow = async () => {
    const d = docRef.current;
    if (!d || !currentProject) return;
    try {
      const images: Record<string, string> = {};
      const keys = new Set<string>();
      for (const n of d.nodes) {
        if (n.type === "image") {
          const data = n.data as ImageData;
          if (data.blobKey && data.url) keys.add(data.blobKey);
        }
        if (n.type === "output") {
          for (const im of (n.data as OutputData).images) keys.add(im.blobKey);
        }
      }
      const urlByKey = new Map<string, string>();
      for (const n of d.nodes) {
        if (n.type === "image") {
          const data = n.data as ImageData;
          if (data.blobKey && data.url) urlByKey.set(data.blobKey, data.url);
        }
        if (n.type === "output") {
          for (const im of (n.data as OutputData).images) urlByKey.set(im.blobKey, im.url);
        }
      }
      for (const key of keys) {
        const url = urlByKey.get(key);
        if (url) images[key] = await blobUrlToDataUrl(url);
      }
      downloadText(
        `${currentProject.name}.workflow.json`,
        JSON.stringify(
          { name: currentProject.name, nodes: d.nodes, edges: d.edges, viewport: viewportRef.current, images },
          null,
          2,
        ),
      );
      setNotice("工作流已导出（包含参考图，不含任何密钥）");
    } catch {
      setNotice("导出失败");
    }
  };

  const importWorkflow = async (file: File) => {
    try {
      const text = await file.text();
      const parsed: unknown = JSON.parse(text);
      if (typeof parsed !== "object" || parsed === null) throw new Error("格式不对");
      const wf = parsed as {
        name?: string;
        nodes?: WFNode[];
        edges?: WFEdge[];
        viewport?: Viewport;
        images?: Record<string, string>;
      };
      if (!Array.isArray(wf.nodes) || !Array.isArray(wf.edges)) {
        throw new Error("不是有效的工作流文件");
      }
      const keyMap = new Map<string, { blobKey: string; url: string }>();
      for (const [oldKey, dataUrl] of Object.entries(wf.images ?? {})) {
        const m = /^data:([^;]+);base64,(.+)$/s.exec(dataUrl);
        if (!m) continue;
        const res = await api.uploadImage({
          dataBase64: m[2] as string,
          mimeType: m[1] as string,
        });
        keyMap.set(oldKey, { blobKey: res.blob_key, url: res.url });
      }
      const nodes = wf.nodes.map((n) => {
        if (n.type === "image") {
          const data = n.data as ImageData;
          const mapped = data.blobKey ? keyMap.get(data.blobKey) : undefined;
          return {
            ...n,
            data: mapped
              ? { ...data, blobKey: mapped.blobKey, url: mapped.url }
              : { ...data, blobKey: null, url: null },
          };
        }
        if (n.type === "output") {
          const data = n.data as OutputData;
          return {
            ...n,
            data: {
              images: data.images
                .map((im) => keyMap.get(im.blobKey))
                .filter((x): x is { blobKey: string; url: string } => Boolean(x)),
            },
          };
        }
        return n;
      });
      const created = await api.createProject({
        name: `${wf.name ?? "导入的工作流"}（导入）`.slice(0, 80),
      });
      await api.saveCanvas({
        projectId: created.id,
        canvas: JSON.stringify({
          nodes,
          edges: wf.edges,
          viewport: wf.viewport ?? { x: 0, y: 0, k: 1 },
        }),
      });
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      loadedForRef.current = null;
      setProjectId(created.id);
      setNotice("工作流已导入为新项目");
    } catch (err) {
      setNotice(`导入失败：${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // ------------------------------------------------------------ history
  const reuseAsReference = (item: HistoryItem) => {
    const v = viewportRef.current;
    const el = containerRef.current;
    const cx = el ? el.clientWidth / 2 : 300;
    const cy = el ? el.clientHeight / 2 : 300;
    const baseX = (cx - v.x) / v.k;
    const baseY = (cy - v.y) / v.k;
    const newNodes: WFNode[] = item.images.map((im, i) => ({
      id: newId("image"),
      type: "image",
      x: baseX + i * 40,
      y: baseY + i * 40,
      data: { blobKey: im.blob_key, url: im.url, role: "风格", name: "历史生成图" },
    }));
    setDoc((prev) =>
      prev ? { ...prev, nodes: [...prev.nodes, ...newNodes] } : prev,
    );
    setShowHistory(false);
    setNotice("已把历史图片放回画布，可连线继续使用");
  };

  const reuseAsGenerate = (item: HistoryItem) => {
    const v = viewportRef.current;
    const el = containerRef.current;
    const cx = el ? el.clientWidth / 2 : 300;
    const cy = el ? el.clientHeight / 2 : 300;
    const genId = newId("generate");
    const outId = newId("output");
    const gx = (cx - v.x) / v.k - 160;
    const gy = (cy - v.y) / v.k - 100;
    const orientation = (["square", "landscape", "portrait"] as const).includes(
      item.orientation as Orientation,
    )
      ? (item.orientation as Orientation)
      : "square";
    setDoc((prev) =>
      prev
        ? {
            ...prev,
            nodes: [
              ...prev.nodes,
              {
                id: genId,
                type: "generate" as const,
                x: gx,
                y: gy,
                data: {
                  prompts: item.prompt,
                  orientation,
                  count: 1,
                  status: "idle",
                } as GenerateData,
              },
              {
                id: outId,
                type: "output" as const,
                x: gx + 380,
                y: gy,
                data: { images: [] } as OutputData,
              },
            ],
            edges: [...prev.edges, { id: newId("e"), from: genId, to: outId }],
          }
        : prev,
    );
    setShowHistory(false);
  };

  // -------------------------------------------------------------- render
  const nodes = doc?.nodes ?? [];
  const edges = doc?.edges ?? [];
  const nodeById = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

  const anchorOut = (n: WFNode) => ({
    x: (n.x + NODE_W[n.type]) * viewport.k + viewport.x,
    y: (n.y + 19) * viewport.k + viewport.y,
  });
  const anchorIn = (n: WFNode) => ({
    x: n.x * viewport.k + viewport.x,
    y: (n.y + 19) * viewport.k + viewport.y,
  });
  const edgePath = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    `M ${a.x} ${a.y} C ${a.x + 60} ${a.y}, ${b.x - 60} ${b.y}, ${b.x} ${b.y}`;

  const pendingNode = pendingFrom ? nodeById.get(pendingFrom) : undefined;
  const selectedEdgeObj = selectedEdge
    ? edges.find((e) => e.id === selectedEdge)
    : undefined;
  const selectedMid =
    selectedEdgeObj &&
    nodeById.get(selectedEdgeObj.from) &&
    nodeById.get(selectedEdgeObj.to)
      ? (() => {
          const a = anchorOut(nodeById.get(selectedEdgeObj.from) as WFNode);
          const b = anchorIn(nodeById.get(selectedEdgeObj.to) as WFNode);
          return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
        })()
      : null;

  const importRef = useRef<HTMLInputElement>(null);

  const toolbarBtn =
    "rounded-lg bg-[#1d212a] border border-[#2b303c] px-3 text-sm text-[#e8eaf0] whitespace-nowrap";

  return (
    <div className="flex h-dvh flex-col bg-[#101216] text-[#e8eaf0]">
      {/* ---------------------------------------------------------- bar */}
      <header className="z-30 flex flex-wrap items-center gap-2 border-b border-[#262b36] bg-[#14171d] px-3 pb-2 pt-[max(0.5rem,env(safe-area-inset-top))]">
        <select
          aria-label="选择项目"
          className="max-w-44 rounded-lg border border-[#2b303c] bg-[#101216] px-2 text-sm font-medium"
          value={projectId ?? ""}
          onChange={(e) => {
            loadedForRef.current = null;
            setProjectId(Number(e.target.value));
          }}
        >
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className={toolbarBtn}
          onClick={() =>
            createMutation.mutate(`未命名工作流 ${projects.length + 1}`)
          }
        >
          ＋ 新建
        </button>
        {renaming ? (
          <span className="flex items-center gap-1">
            <input
              aria-label="项目新名称"
              className="w-36 rounded-lg border border-[#2b303c] bg-[#101216] px-2 text-sm"
              value={renameText}
              onChange={(e) => setRenameText(e.target.value)}
            />
            <button
              type="button"
              className={toolbarBtn}
              onClick={() => {
                if (projectId !== null && renameText.trim()) {
                  renameMutation.mutate({ id: projectId, name: renameText.trim() });
                }
                setRenaming(false);
              }}
            >
              保存
            </button>
          </span>
        ) : (
          <button
            type="button"
            className={toolbarBtn}
            onClick={() => {
              setRenameText(currentProject?.name ?? "");
              setRenaming(true);
            }}
          >
            重命名
          </button>
        )}
        <button
          type="button"
          className={toolbarBtn}
          onClick={() => {
            if (projectId !== null) deleteMutation.mutate(projectId);
          }}
        >
          删除项目
        </button>
        <span className="mx-1 hidden h-6 w-px bg-[#2b303c] sm:block" />
        <span className="relative">
          <button
            type="button"
            className="rounded-lg bg-[#f5b83d] px-3 text-sm font-semibold text-[#171207]"
            aria-expanded={showAddMenu}
            onClick={() => setShowAddMenu((s) => !s)}
          >
            ＋ 添加节点
          </button>
          {showAddMenu && (
            <span className="absolute left-0 top-full z-40 mt-1 block w-44 rounded-xl border border-[#2b303c] bg-[#1d212a] p-1 shadow-xl">
              {(Object.keys(TYPE_LABEL) as NodeType[]).map((t) => (
                <button
                  key={t}
                  type="button"
                  className="flex w-full items-center gap-2 rounded-lg px-2 text-left text-sm hover:bg-[#262b36]"
                  onClick={() => addNode(t)}
                >
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-sm"
                    style={{ background: TYPE_ACCENT[t] }}
                  />
                  {TYPE_LABEL[t]}
                </button>
              ))}
            </span>
          )}
        </span>
        <button type="button" className={toolbarBtn} onClick={fitView}>
          适应画布
        </button>
        <button
          type="button"
          className={toolbarBtn}
          aria-expanded={showHistory}
          onClick={() => setShowHistory((s) => !s)}
        >
          生成历史
        </button>
        <button type="button" className={toolbarBtn} onClick={() => void exportWorkflow()}>
          导出
        </button>
        <button
          type="button"
          className={toolbarBtn}
          onClick={() => importRef.current?.click()}
        >
          导入
        </button>
        <input
          ref={importRef}
          type="file"
          accept="application/json"
          className="hidden"
          aria-label="导入工作流文件"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void importWorkflow(f);
            e.target.value = "";
          }}
        />
        <span className="ml-auto font-mono text-xs text-[#9aa3b2]" role="status">
          {saveState === "saving"
            ? "保存中…"
            : saveState === "saved"
              ? "✓ 已自动保存"
              : saveState === "error"
                ? "保存失败"
                : ""}
        </span>
      </header>

      {notice && (
        <div className="z-30 bg-[#f5b83d] px-3 py-1.5 text-center text-sm font-medium text-[#171207]">
          {notice}
        </div>
      )}
      {pendingFrom && (
        <div className="z-30 bg-[#1d212a] px-3 py-1.5 text-center font-mono text-xs text-[#f5b83d]">
          连线中：点目标节点左侧端口完成连接，Esc 取消
        </div>
      )}

      {/* ------------------------------------------------------- canvas */}
      <div className="relative flex-1 overflow-hidden">
        <div
          ref={containerRef}
          className="canvas-dots absolute inset-0 cursor-grab active:cursor-grabbing"
          style={{ touchAction: "none" }}
          onPointerDown={onContainerPointerDown}
          onPointerMove={onContainerPointerMove}
          onPointerUp={endPointer}
          onPointerCancel={endPointer}
        >
          {/* edges (screen space) */}
          <svg className="pointer-events-none absolute inset-0 h-full w-full">
            {edges.map((e) => {
              const from = nodeById.get(e.from);
              const to = nodeById.get(e.to);
              if (!from || !to) return null;
              const a = anchorOut(from);
              const b = anchorIn(to);
              const selected = selectedEdge === e.id;
              return (
                <g key={e.id}>
                  <path
                    d={edgePath(a, b)}
                    fill="none"
                    stroke={selected ? "#f5b83d" : "#4b5568"}
                    strokeWidth={selected ? 3 : 2}
                    strokeDasharray="1 0"
                  />
                  <path
                    d={edgePath(a, b)}
                    fill="none"
                    stroke="transparent"
                    strokeWidth={16}
                    className="pointer-events-auto cursor-pointer"
                    onClick={() => setSelectedEdge(e.id)}
                  >
                    <title>选中连线</title>
                  </path>
                </g>
              );
            })}
            {pendingNode && mouse && (
              <path
                d={edgePath(anchorOut(pendingNode), mouse)}
                fill="none"
                stroke="#f5b83d"
                strokeWidth={2}
                strokeDasharray="6 4"
              />
            )}
          </svg>
          {selectedMid && (
            <button
              type="button"
              aria-label="删除连线"
              className="absolute z-30 flex h-9 items-center rounded-full bg-[#fb7185] px-3 text-xs font-semibold text-white shadow-lg"
              style={{
                left: selectedMid.x,
                top: selectedMid.y,
                transform: "translate(-50%, -50%)",
              }}
              onClick={() => {
                setDoc((prev) =>
                  prev
                    ? { ...prev, edges: prev.edges.filter((e) => e.id !== selectedEdge) }
                    : prev,
                );
                setSelectedEdge(null);
              }}
            >
              删除连线
            </button>
          )}

          {/* nodes (world space) */}
          <div
            className="absolute left-0 top-0"
            style={{
              transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.k})`,
              transformOrigin: "0 0",
              width: 1,
              height: 1,
            }}
          >
            {nodes.map((node) => {
              const refThumbs = ((): { url: string; role: string }[] => {
                if (node.type !== "generate") return [];
                const out: { url: string; role: string }[] = [];
                for (const e of edges) {
                  if (e.to !== node.id) continue;
                  const src = nodeById.get(e.from);
                  if (src && src.type === "image") {
                    const d = src.data as ImageData;
                    if (d.url) out.push({ url: d.url, role: d.role });
                  }
                }
                return out;
              })();
              const hasLlm =
                node.type === "generate" &&
                edges.some(
                  (e) =>
                    e.to === node.id &&
                    nodeById.get(e.from)?.type === "llm",
                );
              return (
                <NodeShell
                  key={node.id}
                  node={node}
                  pendingFrom={pendingFrom}
                  actions={actions}
                  onHeaderPointerDown={onHeaderPointerDown}
                >
                  {node.type === "image" && (
                    <ImageNodeBody node={node} data={node.data as ImageData} actions={actions} />
                  )}
                  {node.type === "llm" && (
                    <LlmNodeBody node={node} data={node.data as LlmData} actions={actions} />
                  )}
                  {node.type === "generate" && (
                    <GenerateNodeBody
                      node={node}
                      data={node.data as GenerateData}
                      actions={actions}
                      refThumbs={refThumbs}
                      hasLlm={hasLlm}
                    />
                  )}
                  {node.type === "output" && (
                    <OutputNodeBody node={node} data={node.data as OutputData} actions={actions} />
                  )}
                </NodeShell>
              );
            })}
          </div>

          {doc !== null && nodes.length === 0 && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-6">
              <p className="max-w-sm text-center text-sm leading-6 text-[#9aa3b2]">
                画布是空的。点上方「＋ 添加节点」放几个 IMAGE 参考图，
                连到 LLM 节点整理提示词，再连到「生成」节点出图——
                全程由 Muse 执行，不用任何 API Key。
              </p>
            </div>
          )}
          {doc === null && (
            <div className="absolute inset-0 flex items-center justify-center">
              <p className="font-mono text-sm text-[#9aa3b2]">加载画布中…</p>
            </div>
          )}
        </div>

        {/* zoom controls */}
        <div className="absolute bottom-4 right-3 z-30 flex items-center gap-1 rounded-full border border-[#2b303c] bg-[#1d212a]/95 px-1 py-1 shadow-lg">
          <button
            type="button"
            aria-label="缩小画布"
            className="flex h-9 w-9 items-center justify-center rounded-full text-lg"
            onClick={() => zoomBy(0.85)}
          >
            −
          </button>
          <span className="w-12 text-center font-mono text-xs">
            {Math.round(viewport.k * 100)}%
          </span>
          <button
            type="button"
            aria-label="放大画布"
            className="flex h-9 w-9 items-center justify-center rounded-full text-lg"
            onClick={() => zoomBy(1.18)}
          >
            ＋
          </button>
        </div>

        {/* -------------------------------------------------- history */}
        {showHistory && (
          <aside className="absolute bottom-0 right-0 top-0 z-40 flex w-[min(92vw,380px)] flex-col border-l border-[#2b303c] bg-[#14171d] shadow-2xl">
            <div className="flex items-center gap-2 border-b border-[#262b36] px-3 py-2">
              <h2 className="text-sm font-semibold">生成历史</h2>
              <button
                type="button"
                aria-label="关闭生成历史"
                className="ml-auto flex h-8 w-8 items-center justify-center rounded text-[#9aa3b2] hover:bg-[#262b36]"
                onClick={() => setShowHistory(false)}
              >
                ×
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-3">
              {historyQuery.isPending && (
                <p className="font-mono text-xs text-[#9aa3b2]">加载中…</p>
              )}
              {historyQuery.data && historyQuery.data.items.length === 0 && (
                <p className="text-sm text-[#9aa3b2]">
                  还没有生成记录。在「生成」节点跑一次，结果会自动存进历史。
                </p>
              )}
              <div className="flex flex-col gap-4">
                {(historyQuery.data?.items ?? []).map((item) => (
                  <article
                    key={item.id}
                    className="rounded-xl border border-[#2b303c] bg-[#171a20] p-2"
                  >
                    <div className="flex gap-1.5">
                      {item.images.map((im, i) => (
                        <img
                          key={im.blob_key}
                          src={im.url}
                          alt={`历史生成图 ${i + 1}`}
                          className="h-16 w-16 cursor-zoom-in rounded-lg object-cover"
                          onClick={() => setLightbox(im.url)}
                        />
                      ))}
                    </div>
                    <p className="mt-2 line-clamp-3 text-xs leading-5 text-[#c6ccd8]">
                      {item.prompt}
                    </p>
                    <p className="mt-1 font-mono text-[11px] text-[#5f6879]">
                      {new Date(item.created_at).toLocaleString()} ·{" "}
                      {ORIENTATIONS.find((o) => o.value === item.orientation)?.label ??
                        item.orientation}{" "}
                      · {item.image_count} 张
                      {item.ref_summary ? ` · 参考：${item.ref_summary}` : ""}
                    </p>
                    <div className="mt-2 flex gap-1.5">
                      <button
                        type="button"
                        className="flex-1 rounded-lg bg-[#262b36] px-2 text-xs"
                        onClick={() => reuseAsReference(item)}
                      >
                        用作参考图
                      </button>
                      <button
                        type="button"
                        className="flex-1 rounded-lg bg-[#262b36] px-2 text-xs"
                        onClick={() => reuseAsGenerate(item)}
                      >
                        同提示词再生成
                      </button>
                      <button
                        type="button"
                        aria-label="删除这条历史"
                        className="rounded-lg bg-[#262b36] px-2 text-xs text-[#fb7185]"
                        onClick={() => deleteHistoryMutation.mutate(item.id)}
                      >
                        删除
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </div>
          </aside>
        )}
      </div>

      {/* ----------------------------------------------------- lightbox */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex flex-col bg-black/92 p-4"
          role="dialog"
          aria-label="图片预览"
          onClick={() => setLightbox(null)}
        >
          <div className="flex items-center gap-2 pb-3 pt-safe">
            <a
              href={lightbox}
              download
              target="_blank"
              rel="noreferrer"
              className="rounded-lg bg-[#f5b83d] px-3 text-sm font-semibold text-[#171207]"
              onClick={(e) => e.stopPropagation()}
            >
              下载原图
            </a>
            <button
              type="button"
              aria-label="关闭预览"
              className="ml-auto flex h-9 w-9 items-center justify-center rounded-lg bg-[#262b36] text-lg text-white"
              onClick={() => setLightbox(null)}
            >
              ×
            </button>
          </div>
          <div className="flex flex-1 items-center justify-center overflow-hidden">
            <img
              src={lightbox}
              alt="图片大图预览"
              className="max-h-full max-w-full rounded-lg object-contain"
            />
          </div>
        </div>
      )}
    </div>
  );
}
