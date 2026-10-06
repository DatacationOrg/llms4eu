-- One-off: give a page_chunks table from before chunk variants its `variant`
-- column and the (page_id, variant, chunk_index) uniqueness. SQLite cannot
-- change a unique constraint in place, so the table is rebuilt. Every chunk id
-- is kept, so the labels in eval_relevant_chunks stay valid. The table shape
-- repeats sql/eval.sql; tests/test_chunk_variants.py checks the two agree.
--
-- Run it once, by hand, from the repository root:
--
--   sqlite3 "$LLMS4EU_DATA/db/pages.db" < sql/migrate_chunk_variants.sql
--
-- Foreign keys stay off and the rename stays legacy, so the labels are neither
-- cascaded away with the old table nor re-pointed at its new name.
pragma foreign_keys = off;
pragma legacy_alter_table = on;

begin;

alter table page_chunks rename to page_chunks_old;
drop index if exists idx_page_chunks_page_id;

create table page_chunks (
  id text primary key,
  page_id text not null references page_metadata(id) on delete cascade,
  variant text not null default 'base',
  chunk_index integer not null,
  heading_path text,
  text text not null,
  char_count integer not null,
  unique(page_id, variant, chunk_index)
);

insert into page_chunks (id, page_id, chunk_index, heading_path, text, char_count)
select id, page_id, chunk_index, heading_path, text, char_count
from page_chunks_old;

drop table page_chunks_old;

create index if not exists idx_page_chunks_page_id
  on page_chunks(page_id);

create index if not exists idx_page_chunks_variant
  on page_chunks(variant);

commit;
