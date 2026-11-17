# Data Plan

## Context provenance
- `参考我GitHub 项目做一个一样的空间，不过不用别人家的api 了，全部由你操作` (verbatim build request; the reference project is the user's own GitHub repo, and all LLM/image generation must run on Muse's managed pipeline — no user API key, no OpenAI calls)
- `你 GitHub 上那个是 openai-canvas-portable` (parent conversation: assistant identified the repo and summarized its feature set; user then asked to replicate it as a space)
- Repo README of bullshitAI52/openai-canvas-portable (read in full via GitHub connector): node types IMAGE / LLM / API生成 / OUTPUT; multi-image references with roles （人物/服装/场景/动作构图/商品/风格）; connection order defines image numbering; LLM node modes （节点模式/聊天模式）, editable System + INPUT + OUTPUT; one-click run （先 LLM 后生图） or step-by-step; generation params （尺寸/画幅/质量/数量）; generation history with one-click reuse as new nodes; workflow import/export JSON; canvas pan/zoom, port-to-port connect, click edge to delete.

## Tested sources
### GitHub connector — bullshitAI52/openai-canvas-portable
**Used by**: design reference only (feature parity checklist); no repo content is copied into the artifact at runtime
**Test command**: `/opt/hatch/bin/github call-read-tool --name search_repositories --arguments-json '{"query":"openai-canvas-portable user:bullshitAI52"}'` then `get_file_contents` for README.md
**Sample output**: repo found (public, Python, description: 多图提示词与图片生成工作流，使用 OpenAI API); README full text returned with the feature table and usage flow quoted in Context provenance
**Processing**: feature list mapped 1:1 to space capabilities; OpenAI-specific parts (API key settings, model names, local portable runtime) replaced by Muse's managed pipeline

### ctx.inference (vision + text) via probe-ctx
**Used by**: runLLM action (LLM node: analyzes connected reference images + INPUT/System, returns prompt text or chat answer); runGenerate action (vision pass over directly-connected reference images to enrich the generation prompt)
**Test command**: `bun /opt/muse/assets/spaces/ts-runtime/dist/probe-ctx.js < /tmp/probe2.ts` (text rewrite) and `< /tmp/probe4.ts` (1×1 PNG image attachment)
**Sample output**: text probe returned a full rewritten image prompt under the zod schema; vision probe accepted `images: [{dataBase64, mimeType: "image/png"}]` and returned a schema-valid color answer — image attachments work
**Processing**: server converts client-supplied data URLs to InferenceImageInput entries (jpeg/png only), captions each with its connection-order number and role （图1=人物…) so numbering semantics match the original

### ctx.tool.generate_media
**Used by**: runGenerate action (one call per requested image, orientation from 画幅）
**Test command**: probe attempt returned `ctx.tool.generate_media is not available in probe mode — requires a built web artifact`
**Sample output**: SDK type contract (node_modules/@hatch/space-sdk/dist/verticals.d.ts): `generate_media(prompt, {orientation?: square|landscape|portrait, key?}) -> {blobKey, url, contentType, bytes}`, image stored in this space's ctx.blobs with a stable own-origin URL
**Processing**: verified instead through the built artifact's own actions during Verify (invoke runGenerate end-to-end). Because the managed generator accepts a text prompt only, reference images condition generation through Muse's vision pass (detailed visual description merged into the prompt) — the same IMAGE→LLM→生成 path the original product is built around; the UI never claims pixel-level image editing

### ctx.blobs (reference + generated image storage)
**Used by**: uploadImage action (client downscales to ≤1280px JPEG/PNG, sends base64, server `blobs.put` + `getUrl`); canvas JSON stores only blobKey + url
**Test command**: SDK contract read (server-contract.d.ts): BlobClient put/getUrl/head/list/delete; note there is no server-side byte read, so at run time the client fetches its own blob URLs and passes base64 in the runLLM/runGenerate request — blobs remain the system of record, bytes transit via the client only during a run
**Processing**: image nodes persist {blobKey, url, role, name}; history rows persist generated blobKeys

## Long-term data behavior
- **Refresh policy**: none — all data is user-created (projects, canvases, uploads, generations, history); no external feed, no cron
- **Growth**: projects table (canvas stored as one JSON document per project: nodes + edges + viewport), history table (one row per generation run). Starter template: one default project seeded once post-submit with the original's classic empty layout (6 empty IMAGE nodes with preset roles wired into LLM + 生成， LLM→生成， 生成→OUTPUT) — structure only, no images or content
- **Ordering**: projects by updated_at desc; history newest first; canvas node/edge order as saved
- **Time semantics**: created_at/updated_at are creation instants rendered viewer-local

## Rejected approaches
- **Tried**: probe ctx.tool.generate_media directly via probe-ctx
  **Why rejected**: not available in probe mode by design; deferred to verifying the built artifact's runGenerate action, which is the real path anyway
- **Tried**: storing uploaded reference images as base64 in the DB / reading blob bytes server-side for the vision pass
  **Why rejected**: BlobClient exposes no byte-read API and DB columns are the wrong home for binary; client fetches its own blob URLs at run time and passes base64 in the run request instead
- **Tried**: exposing the original's model-name/size/quality fields
  **Why rejected**: Muse's managed generator takes prompt + orientation only; fake parameter controls that do nothing would be dishonest. Kept: 画幅 （横/竖/方） and 数量 (1–4), which map to real behavior
