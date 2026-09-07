create table if not exists page_sources (
  source text primary key,
  language text not null
);

create table if not exists page_chunks (
  id text primary key,
  page_id text not null references page_metadata(id) on delete cascade,
  chunk_index integer not null,
  variant text not null default 'base',
  heading_path text,
  text text not null,
  char_count integer not null,
  start_char integer,
  end_char integer,
  unique(page_id, variant, chunk_index)
);

create table if not exists eval_questions (
  id text primary key,
  question text not null,
  answer text not null,
  question_type text not null,
  question_language text not null,
  approved integer not null default 1
);

create table if not exists eval_relevant_chunks (
  question_id text not null references eval_questions(id) on delete cascade,
  chunk_id text not null references page_chunks(id) on delete cascade,
  primary key (question_id, chunk_id)
);

-- Where in a page the text supporting an answer actually sits.
--
-- Anchors are chunking-independent on purpose: they are page character offsets,
-- not chunk ids, so one extraction pass serves every chunk variant and survives
-- any re-cutting. Labelling a variant is then interval overlap rather than a
-- model's choice among candidate chunks.
create table if not exists eval_answer_anchors (
  question_id text primary key references eval_questions(id) on delete cascade,
  page_id text not null references page_metadata(id) on delete cascade,
  quote text not null,
  start_char integer not null,
  end_char integer not null
);

create index if not exists idx_eval_answer_anchors_page_id
  on eval_answer_anchors(page_id);

create index if not exists idx_page_chunks_page_id
  on page_chunks(page_id);

create index if not exists idx_eval_questions_approved_type
  on eval_questions(approved, question_type);

-- The index over `variant` is created by `initialize_page_artifacts_db` after
-- migrations, because an older database reaches this script before that column
-- exists.

create index if not exists idx_eval_relevant_chunks_chunk_id
  on eval_relevant_chunks(chunk_id);
