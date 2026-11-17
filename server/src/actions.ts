// Actions for 多图工作流画布 — a multi-image workflow canvas modeled on the
// user's openai-canvas-portable repo, but with every model call served by
// Muse's own managed pipeline (ctx.inference for the LLM node,
// ctx.tool.generate_media for image generation). No external API key.
//
// Canvas model: each project stores one JSON document
//   { nodes: [{id, type, x, y, data}], edges: [{id, from, to}],
//     viewport: {x, y, k} }
// Node types: image | llm | generate | output. Edges carry data flow:
// image→llm, image→generate, llm→generate, generate→output. The order of
// edges into a node defines the reference-image numbering (图1, 图2, …).

import { defineAction, z, type ActionsModule, type Ctx } from "@hatch/space-sdk";
import { desc, eq } from "drizzle-orm";
import * as schema from "./schema";

// ---------------------------------------------------------------- helpers

const imageInput = z.object({
  dataBase64: z.string().min(1).max(12_000_000),
  mimeType: z.string(),
  caption: z.string().max(200).default(""),
});

type ImageInput = z.infer<typeof imageInput>;

/** Keep only JPEG/PNG inputs — the only formats ctx.inference accepts. */
function visionImages(images: ImageInput[]) {
  return images
    .filter((im) => im.mimeType === "image/jpeg" || im.mimeType === "image/png")
    .slice(0, 8)
    .map((im) => ({
      dataBase64: im.dataBase64,
      mimeType: im.mimeType as "image/jpeg" | "image/png",
      caption: im.caption || undefined,
    }));
}

const DEFAULT_LLM_SYSTEM =
  "你是一名资深的 AI 生图提示词专家。用户会给你若干张参考图（按图1、图2……编号，并标注每张图的角色）和一段要求。" +
  "请仔细观察每张参考图，只输出一段可直接用于图片生成的完整提示词：明确写出每张参考图贡献的元素（人物五官与发型、服装款式与材质、场景与光线、构图与姿势等）、画面主体、风格、光线和质感。" +
  "不要解释过程，不要输出多余文字，只输出提示词本身。";

const DEFAULT_CHAT_SYSTEM =
  "你是一名看图助手。用户会给你若干张参考图（按图1、图2……编号）和一个问题。请根据图片内容直接、准确地回答问题。";

const STARTER_CANVAS = (() => {
  const roles = ["人物", "人物", "场景", "场景", "服装", "服装"];
  const nodes: unknown[] = [];
  const edges: unknown[] = [];
  roles.forEach((role, i) => {
    const id = `img-${i + 1}`;
    nodes.push({
      id,
      type: "image",
      x: 24,
      y: 24 + i * 196,
      data: { blobKey: null, url: null, role, name: "" },
    });
    edges.push({ id: `e-${id}-llm`, from: id, to: "llm-1" });
    edges.push({ id: `e-${id}-gen`, from: id, to: "gen-1" });
  });
  nodes.push(
    {
      id: "llm-1",
      type: "llm",
      x: 400,
      y: 320,
      data: {
        input:
          "图1是主要人物身份参考，请保留她的五官与发型。\n图3和图4参考场景、光线与构图。\n图5参考服装款式，图6只参考面料纹理。\n请按我的要求整理成一段可直接生图的详细提示词。",
        system: DEFAULT_LLM_SYSTEM,
        mode: "prompt",
        output: "",
        status: "idle",
      },
    },
    {
      id: "gen-1",
      type: "generate",
      x: 790,
      y: 320,
      data: { prompts: "", orientation: "landscape", count: 1, status: "idle" },
    },
    {
      id: "out-1",
      type: "output",
      x: 1180,
      y: 320,
      data: { images: [] },
    },
  );
  edges.push(
    { id: "e-llm-gen", from: "llm-1", to: "gen-1" },
    { id: "e-gen-out", from: "gen-1", to: "out-1" },
  );
  return JSON.stringify({
    nodes,
    edges,
    viewport: { x: 0, y: 0, k: 0.75 },
  });
})();

async function projectExists(ctx: Ctx, projectId: number): Promise<boolean> {
  const db = ctx.db<typeof schema>();
  const rows = await db
    .select({ id: schema.projects.id })
    .from(schema.projects)
    .where(eq(schema.projects.id, projectId))
    .limit(1);
  return rows.length > 0;
}

// ---------------------------------------------------------------- actions

export const Actions = {
  listProjects: defineAction({
    description: "Lists all workflow projects, most recently updated first.",
    request: z.object({}),
    response: z.object({
      projects: z.array(
        z.object({
          id: z.number(),
          name: z.string(),
          updated_at: z.string(),
        }),
      ),
    }),
    async handler(ctx) {
      const db = ctx.db<typeof schema>();
      const rows = await db
        .select({
          id: schema.projects.id,
          name: schema.projects.name,
          updatedAt: schema.projects.updatedAt,
        })
        .from(schema.projects)
        .orderBy(desc(schema.projects.updatedAt));
      return {
        projects: rows.map((r) => ({
          id: r.id,
          name: r.name,
          updated_at: r.updatedAt.toISOString(),
        })),
      };
    },
  }),

  seedStarter: defineAction({
    description:
      "Creates the starter project with the classic pre-wired empty workflow layout when no project exists yet.",
    request: z.object({}),
    response: z.object({
      created: z.boolean(),
      project_id: z.number().nullable(),
    }),
    async handler(ctx) {
      const db = ctx.db<typeof schema>();
      const existing = await db
        .select({ id: schema.projects.id })
        .from(schema.projects)
        .limit(1);
      if (existing.length > 0) {
        return { created: false, project_id: existing[0]?.id ?? null };
      }
      const now = new Date();
      const inserted = await db
        .insert(schema.projects)
        .values({
          name: "我的第一个工作流",
          canvas: STARTER_CANVAS,
          createdAt: now,
          updatedAt: now,
        })
        .returning({ id: schema.projects.id });
      ctx.invalidateQueries();
      return { created: true, project_id: inserted[0]?.id ?? null };
    },
  }),

  createProject: defineAction({
    description: "Creates a new empty workflow project with the given name.",
    request: z.object({ name: z.string().min(1).max(80) }),
    response: z.object({ id: z.number() }),
    async handler(ctx, args) {
      const db = ctx.db<typeof schema>();
      const now = new Date();
      const empty = JSON.stringify({
        nodes: [],
        edges: [],
        viewport: { x: 0, y: 0, k: 1 },
      });
      const inserted = await db
        .insert(schema.projects)
        .values({ name: args.name, canvas: empty, createdAt: now, updatedAt: now })
        .returning({ id: schema.projects.id });
      const row = inserted[0];
      if (!row) throw new Error("createProject: insert returned no rows");
      ctx.invalidateQueries();
      return { id: row.id };
    },
  }),

  renameProject: defineAction({
    description: "Renames a workflow project.",
    request: z.object({ id: z.number().int().positive(), name: z.string().min(1).max(80) }),
    response: z.object({ ok: z.literal(true) }),
    async handler(ctx, args): Promise<{ ok: true }> {
      const db = ctx.db<typeof schema>();
      await db
        .update(schema.projects)
        .set({ name: args.name, updatedAt: new Date() })
        .where(eq(schema.projects.id, args.id));
      ctx.invalidateQueries();
      return { ok: true };
    },
  }),

  deleteProject: defineAction({
    description: "Deletes a workflow project and its generation history.",
    request: z.object({ id: z.number().int().positive() }),
    response: z.object({ ok: z.literal(true) }),
    async handler(ctx, args): Promise<{ ok: true }> {
      const db = ctx.db<typeof schema>();
      await db.delete(schema.history).where(eq(schema.history.projectId, args.id));
      await db.delete(schema.projects).where(eq(schema.projects.id, args.id));
      ctx.invalidateQueries();
      return { ok: true };
    },
  }),

  getCanvas: defineAction({
    description: "Returns the saved canvas JSON document for a project.",
    request: z.object({ projectId: z.number().int().positive() }),
    response: z.object({ canvas: z.string() }),
    async handler(ctx, args) {
      const db = ctx.db<typeof schema>();
      const rows = await db
        .select({ canvas: schema.projects.canvas })
        .from(schema.projects)
        .where(eq(schema.projects.id, args.projectId))
        .limit(1);
      const row = rows[0];
      if (!row) throw new Error("项目不存在");
      return { canvas: row.canvas };
    },
  }),

  saveCanvas: defineAction({
    description: "Saves the canvas JSON document for a project (autosave).",
    request: z.object({
      projectId: z.number().int().positive(),
      canvas: z.string().max(4_000_000),
    }),
    response: z.object({ ok: z.literal(true) }),
    async handler(ctx, args): Promise<{ ok: true }> {
      // Reject malformed documents instead of persisting a broken canvas.
      const parsed: unknown = JSON.parse(args.canvas);
      if (
        typeof parsed !== "object" ||
        parsed === null ||
        !Array.isArray((parsed as { nodes?: unknown }).nodes) ||
        !Array.isArray((parsed as { edges?: unknown }).edges)
      ) {
        throw new Error("画布数据格式不正确");
      }
      const db = ctx.db<typeof schema>();
      await db
        .update(schema.projects)
        .set({ canvas: args.canvas, updatedAt: new Date() })
        .where(eq(schema.projects.id, args.projectId));
      return { ok: true };
    },
  }),

  uploadImage: defineAction({
    description:
      "Stores one uploaded reference image in blob storage and returns its key and renderable URL.",
    request: z.object({
      dataBase64: z.string().min(1).max(16_000_000),
      mimeType: z.string().max(60),
    }),
    response: z.object({ blob_key: z.string(), url: z.string() }),
    async handler(ctx, args) {
      const ext =
        args.mimeType === "image/png"
          ? "png"
          : args.mimeType === "image/webp"
            ? "webp"
            : args.mimeType === "image/gif"
              ? "gif"
              : "jpg";
      const key = `uploads/${Date.now()}-${Math.random().toString(36).slice(2, 10)}.${ext}`;
      const bytes = Buffer.from(args.dataBase64, "base64");
      await ctx.blobs.put(key, bytes, { contentType: args.mimeType });
      const url = await ctx.blobs.getUrl(key);
      return { blob_key: key, url };
    },
  }),

  imageUrl: defineAction({
    description: "Resolves a stored blob key to a renderable URL.",
    request: z.object({ blobKey: z.string().min(1).max(300) }),
    response: z.object({ url: z.string() }),
    async handler(ctx, args) {
      const url = await ctx.blobs.getUrl(args.blobKey);
      return { url };
    },
  }),

  runLLM: defineAction({
    description:
      "Runs the LLM node: Muse analyzes the connected reference images plus the user's input and returns prompt text (or a chat answer in chat mode).",
    request: z.object({
      input: z.string().max(8000),
      system: z.string().max(4000).default(""),
      mode: z.enum(["prompt", "chat"]).default("prompt"),
      images: z.array(imageInput).max(8).default([]),
    }),
    response: z.object({ text: z.string() }),
    async handler(ctx, args) {
      const imgs = visionImages(args.images);
      const system =
        args.system.trim() ||
        (args.mode === "chat" ? DEFAULT_CHAT_SYSTEM : DEFAULT_LLM_SYSTEM);
      const userPrompt =
        imgs.length > 0
          ? `以下是本次调用的参考图说明：共 ${imgs.length} 张，已按图1、图2……的顺序随消息附上，编号与角色见每张图的标注。\n\n用户要求：\n${args.input}`
          : `用户要求：\n${args.input}`;
      const result = await ctx.inference.complete(userPrompt, {
        schema: z.object({ text: z.string() }),
        system,
        images: imgs,
        timeout_secs: 180,
      });
      return { text: result.text };
    },
  }),

  runGenerate: defineAction({
    description:
      "Generates images with Muse's image pipeline. Connected reference images are first analyzed by Muse vision and woven into the prompt; the run is saved to history.",
    request: z.object({
      projectId: z.number().int().positive(),
      prompt: z.string().min(1).max(8000),
      orientation: z.enum(["square", "landscape", "portrait"]).default("square"),
      count: z.number().int().min(1).max(4).default(1),
      refImages: z.array(imageInput).max(8).default([]),
    }),
    response: z.object({
      images: z.array(z.object({ blob_key: z.string(), url: z.string() })),
      effective_prompt: z.string(),
    }),
    async handler(ctx, args) {
      if (!(await projectExists(ctx, args.projectId))) {
        throw new Error("项目不存在");
      }
      let effectivePrompt = args.prompt;
      const refs = visionImages(args.refImages);
      if (refs.length > 0) {
        const vision = await ctx.inference.complete(
          `共 ${refs.length} 张参考图随消息附上，编号与角色见每张图的标注。` +
            "请逐张详细描述每张参考图对最终画面应贡献的元素：人物的五官、发型与体态，服装的款式、颜色与材质纹理，场景的环境、光线与氛围，构图与姿势。" +
            "只输出这些视觉细节描述，按「图1：……；图2：……」组织，不要评论，不要提出建议。",
          {
            schema: z.object({ description: z.string() }),
            system:
              "你是一名视觉分析师，为图片生成整理参考图的视觉细节。只描述图中确实存在的内容，不臆测。",
            images: refs,
            timeout_secs: 180,
          },
        );
        effectivePrompt =
          `${args.prompt}\n\n参考图视觉细节（务必在画面中落实）：\n${vision.description}`;
      }

      const images: { blob_key: string; url: string }[] = [];
      for (let i = 0; i < args.count; i++) {
        const media = await ctx.tool.generate_media(effectivePrompt, {
          orientation: args.orientation,
        });
        images.push({ blob_key: media.blobKey, url: media.url });
      }

      const db = ctx.db<typeof schema>();
      await db.insert(schema.history).values({
        projectId: args.projectId,
        prompt: effectivePrompt,
        orientation: args.orientation,
        imageCount: images.length,
        blobKeys: JSON.stringify(images.map((im) => im.blob_key)),
        refSummary: args.refImages.map((im) => im.caption).filter(Boolean).join("；"),
      });
      ctx.invalidateQueries();
      return { images, effective_prompt: effectivePrompt };
    },
  }),

  listHistory: defineAction({
    description: "Lists a project's generation history, newest first, with image URLs resolved.",
    request: z.object({
      projectId: z.number().int().positive(),
      limit: z.number().int().positive().max(100).default(50),
    }),
    response: z.object({
      items: z.array(
        z.object({
          id: z.number(),
          prompt: z.string(),
          orientation: z.string(),
          image_count: z.number(),
          images: z.array(z.object({ blob_key: z.string(), url: z.string() })),
          ref_summary: z.string(),
          created_at: z.string(),
        }),
      ),
    }),
    async handler(ctx, args) {
      const db = ctx.db<typeof schema>();
      const rows = await db
        .select()
        .from(schema.history)
        .where(eq(schema.history.projectId, args.projectId))
        .orderBy(desc(schema.history.id))
        .limit(args.limit);
      const items = [];
      for (const row of rows) {
        let keys: string[] = [];
        try {
          const parsed: unknown = JSON.parse(row.blobKeys);
          if (Array.isArray(parsed)) {
            keys = parsed.filter((k): k is string => typeof k === "string");
          }
        } catch {
          keys = [];
        }
        const images = [];
        for (const key of keys) {
          images.push({ blob_key: key, url: await ctx.blobs.getUrl(key) });
        }
        items.push({
          id: row.id,
          prompt: row.prompt,
          orientation: row.orientation,
          image_count: row.imageCount,
          images,
          ref_summary: row.refSummary,
          created_at: row.createdAt.toISOString(),
        });
      }
      return { items };
    },
  }),

  deleteHistory: defineAction({
    description: "Deletes one generation history entry.",
    request: z.object({ id: z.number().int().positive() }),
    response: z.object({ ok: z.literal(true) }),
    async handler(ctx, args): Promise<{ ok: true }> {
      const db = ctx.db<typeof schema>();
      await db.delete(schema.history).where(eq(schema.history.id, args.id));
      ctx.invalidateQueries();
      return { ok: true };
    },
  }),
} satisfies ActionsModule;
