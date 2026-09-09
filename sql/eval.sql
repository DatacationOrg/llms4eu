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

-- Where a page is about, as 0..n locations per page.
--
-- Coordinates are the canonical fact; the codes are derived from them against
-- the Eurostat NUTS boundaries (src/shared/nuts.py) so they can be recomputed
-- when NUTS changes. Names are display labels only and are never filtered on.
-- One `primary` row per page is what gets denormalised into chunk metadata; a
-- page with no primary row (a biography, a concept page) has no location and
-- is never dropped by the boost-only path.
create table if not exists page_locations (
  page_id text not null references page_metadata(id) on delete cascade,
  role text not null check (role in ('primary', 'mentioned')),
  -- Wikidata QID when known, else the normalised name, so a re-run replaces
  -- rather than duplicates.
  location_key text not null,
  name text,
  wikidata_qid text,
  latitude real,
  longitude real,
  granularity text not null default 'point'
    check (granularity in ('point', 'municipality', 'region', 'country')),
  country_code text,
  nuts2 text,
  nuts3 text,
  nuts3_name text,
  iso_3166_2 text,
  confidence real,
  -- wikidata | source_default | llm_nominatim | manual
  method text not null,
  primary key (page_id, role, location_key)
);

create unique index if not exists idx_page_locations_one_primary
  on page_locations(page_id) where role = 'primary';

create index if not exists idx_page_locations_codes
  on page_locations(country_code, nuts2, nuts3);

-- Resolved query scopes, so an eval rerun or an agent retry does not call the
-- LLM and the gazetteer again for a question it has already placed.
create table if not exists geo_scope_cache (
  query_norm text primary key,
  scope_json text not null,
  created_at text not null
);
