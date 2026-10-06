create table if not exists page_sources (
  source text primary key,
  language text not null
);

-- One row per chunk per variant. `base` is the cut the pipeline runs on; the
-- other variants are alternative cuts of the same pages, kept side by side for
-- chunk-size experiments (src/preprocess/README.md). A database from before
-- this column is rebuilt once with sql/migrate_chunk_variants.sql.
create table if not exists page_chunks (
  id text primary key,
  page_id text not null references page_metadata(id) on delete cascade,
  variant text not null default 'base',
  chunk_index integer not null,
  heading_path text,
  text text not null,
  char_count integer not null,
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

create index if not exists idx_page_chunks_page_id
  on page_chunks(page_id);

create index if not exists idx_page_chunks_variant
  on page_chunks(variant);

create index if not exists idx_eval_questions_approved_type
  on eval_questions(approved, question_type);

create index if not exists idx_eval_relevant_chunks_chunk_id
  on eval_relevant_chunks(chunk_id);

-- A verbatim quote from the page that supports each answer. Labels are
-- projected from it onto whatever chunks exist, so they survive a rechunk.
create table if not exists eval_evidence (
  question_id text primary key references eval_questions(id) on delete cascade,
  page_id text not null references page_metadata(id) on delete cascade,
  quote text not null
);

-- Where a page is about, when Wikidata knows: one point per page.
create table if not exists page_locations (
  page_id text primary key references page_metadata(id) on delete cascade,
  qid text not null,
  latitude real not null,
  longitude real not null
);

-- The place each retrieval query is anchored to; a null point means none.
create table if not exists geo_query_places (
  query text primary key,
  latitude real,
  longitude real,
  decay_km real
);
