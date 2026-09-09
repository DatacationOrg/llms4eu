# Architecture Proposals

To make the LLMs4EU scalable there are several mechanisms we can use to reduce search space and guide the retrieval with personal intent

## Geographic Enrichment (search space reduction)

*Adopted 2026-09-08; see "Geographic Scope" in architecture-decisions.md. The
codes live in `page_locations` and in chunk metadata rather than as columns on
`page_metadata`/`page_chunks`, are derived from coordinates, and NUTS-3 joins
NUTS-2 as a filter level.*

We inject country_codes to page and page_chunk tables in DB.
e.g. a french article gets labeled by ISO norm "FR" or "SI"
we could further refine such geographical searchspace with NUTS-2 codes 
e.g. an article regarding a certain region gets "FRL" = Provence-Alpes-Côte d'Azur

This geographic enrichment could be used to eliminate vast amounts of information from the search space by users. When the retriever is able to discriminate chunks based on locality. Many sources have locality encoded and have bulk extraction methods such as Wikidata.

## Agentic search

