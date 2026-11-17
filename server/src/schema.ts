// Drizzle schema for 多图工作流画布.
//
// projects: one row per workflow; the whole canvas (nodes + edges +
// viewport) is stored as one JSON document in `canvas`, mirroring the
// reference tool's workflow.json autosave model.
// history: one row per completed generation run (prompt, params,
// reference summary, generated blob keys), newest first.

import { integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const projects = sqliteTable("projects", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  name: text("name").notNull(),
  canvas: text("canvas").notNull().default("{}"),
  createdAt: integer("created_at", { mode: "timestamp_ms" })
    .notNull()
    .$defaultFn(() => new Date()),
  updatedAt: integer("updated_at", { mode: "timestamp_ms" })
    .notNull()
    .$defaultFn(() => new Date()),
});

export const history = sqliteTable("history", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  projectId: integer("project_id").notNull(),
  prompt: text("prompt").notNull(),
  orientation: text("orientation").notNull().default("square"),
  imageCount: integer("image_count").notNull().default(1),
  blobKeys: text("blob_keys").notNull().default("[]"),
  refSummary: text("ref_summary").notNull().default(""),
  createdAt: integer("created_at", { mode: "timestamp_ms" })
    .notNull()
    .$defaultFn(() => new Date()),
});
