create table if not exists places (
  id text primary key,
  place_description text not null,
  summary text not null
);
