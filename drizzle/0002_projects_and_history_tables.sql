DROP TABLE entries;
--> statement-breakpoint
CREATE TABLE projects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  canvas TEXT NOT NULL DEFAULT '{}',
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
--> statement-breakpoint
CREATE TABLE history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id INTEGER NOT NULL,
  prompt TEXT NOT NULL,
  orientation TEXT NOT NULL DEFAULT 'square',
  image_count INTEGER NOT NULL DEFAULT 1,
  blob_keys TEXT NOT NULL DEFAULT '[]',
  ref_summary TEXT NOT NULL DEFAULT '',
  created_at INTEGER NOT NULL
);
