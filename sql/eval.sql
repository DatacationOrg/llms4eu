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
  approved integer not null default 1,
  variant text not null default 'base'
);

create table if not exists eval_relevant_chunks (
  question_id text not null references eval_questions(id) on delete cascade,
  chunk_id text not null references page_chunks(id) on delete cascade,
  primary key (question_id, chunk_id)
);

create index if not exists idx_page_chunks_page_id
  on page_chunks(page_id);

create index if not exists idx_eval_questions_approved_type
  on eval_questions(approved, question_type);

-- Indexes over the `variant` columns are created by
-- `initialize_page_artifacts_db` after migrations, because an older database
-- reaches this script before those columns exist.

create index if not exists idx_eval_relevant_chunks_chunk_id
  on eval_relevant_chunks(chunk_id);
