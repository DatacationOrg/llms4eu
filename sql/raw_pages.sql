create table if not exists page_metadata (
  id text primary key,
  source text not null,
  url text not null unique,
  final_url text,
  fetched_at text not null,
  status_code integer,
  content_type text,
  title text,
  content_hash text,
  raw_bytes integer not null default 0,
  fetch_method text not null default 'httpx',
  extractor text not null default 'trafilatura',
  page_kind text not null default 'prose',
  markdown_chars integer not null default 0,
  language text,
  error text
);

create table if not exists page_markdown_content (
  page_id text primary key references page_metadata(id) on delete cascade,
  markdown text not null
);
