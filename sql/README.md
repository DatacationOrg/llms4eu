# SQL

`init.sql` is the storage schema for the prototype.

SQLite owns full place data for now. The SQL stays portable so moving to
Postgres later should stay small. The vector index stores only the place `id`,
so changing place fields starts here and in `src.shared.schema.Place`.
