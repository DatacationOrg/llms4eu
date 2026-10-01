# Team sources

These are the team's sources, kept as reference; not yet assessed by me.

**Retrieval and spatial RAG**


| source                                                                          | team read           | claim (one line)                                               |
| ------------------------------------------------------------------------------- | ------------------- | -------------------------------------------------------------- |
| Wang et al. 2025, GeoRAG                                                        | abstract            | Classifier routes queries to geographic dimensions             |
| Chen et al. 2025, GeoRAG urban STKG                                             | abstract            | Retrieval via region hierarchy and coordinates                 |
| Arabov et al. 2026, Tatarstan Toponyms                                          | abstract            | Small bilingual heritage set; dense + distance; recall@1 0.988 |
| Kang et al. 2026, I-GUIDE                                                       | abstract            | In production, spatial metadata quality is the bottleneck      |
| Yang et al. 2026, CubeGraph                                                     | abstract            | Spatial pre-partitioning breaks vector index connectivity      |
| Campo et al. 2026, Real-time Spatial RAG                                        | abstract            | Filter-first spatial RAG, Madrid tourism assistant             |
| Amendola et al. 2025, WalkRAG                                                   | abstract            | Conversational RAG for walkable itineraries                    |
| Li & Lim 2025, RALLM-POI                                                        | abstract            | Distance reranker for next-POI recommendation                  |
| Ni et al. 2025, TP-RAG                                                          | abstract            | Benchmark for travel-planning agents                           |
| Banerjee et al. 2024, Sustainable city trips RAG                                | abstract            | Rerank by popularity × seasonality                            |
| Bruch et al. 2023, Fusion functions                                             | abstract + snippets | Weighted sum of normalised scores beats rank fusion            |
| Shi et al. 2025, Filtered ANN benchmark                                         | abstract            | Recall degrades at low filter selectivity                      |
| Li et al. 2026, Attribute filtering in ANN                                      | abstract            | Benchmark of filtered vector search                            |
| Ingber & Liberty 2025, Pinecone filtering                                       | full intro          | Filtering must be integrated into retrieval                    |
| Ye et al. 2025, Compass                                                         | abstract            | Filtered search across vector and structured data              |
| Wu et al. 2024, STaRK                                                           | abstract            | Retrievers struggle with mixed relational and textual queries  |
| Poliakov & Shvai 2024, Multi-Meta-RAG                                           | abstract            | LLM-extracted hard metadata filters                            |
| Vendor posts (Qdrant, Elastic, Milvus, Alonso, LangChain, LlamaIndex, Weaviate) | mixed               | Industry default is hard geo filters                           |

**Location data and toponym resolution**


| source                                    | team read | claim (one line)                                         |
| ----------------------------------------- | --------- | -------------------------------------------------------- |
| Hu et al. 2024, IJGIS                     | full      | LLM names, gazetteer places: 0.91 vs 0.21–0.77 accuracy |
| Hu et al. 2025, GeoExT                    | full      | Candidate list in prompt: 0.93, 7× faster               |
| Li et al. 2023, GeoLM                     | abstract  | Language model with geographic grounding                 |
| Halterman 2023, Mordecai 3                | abstract  | Neural geoparser over GeoNames                           |
| Bisiani et al. 2025, UK local media       | abstract  | Local LLMs with voting; metadata in prompt hurt          |
| Mioduski 2025, Virginia land grants       | abstract  | LLMs predict coordinates, ~20 km error                   |
| Fernando et al. 2026, Relative localities | abstract  | LLMs georeference "5 km NW of X"                         |
| Cafferata et al. 2026, Crisis geolocation | abstract  | Context-aware disambiguation                             |
| Kopanov 2024, Multilingual geo-entities   | abstract  | No single extractor wins across languages                |
| Masis & O'Connor 2024, Geo-entity linking | abstract  | Cheap multilingual linking with abstention               |
| Ljubešić et al. 2023, CLASSLA-Stanza    | –        | Slovenian NER and lemmatisation                          |

**LLM spatial ability and evaluation**


| source                                    | team read | claim (one line)                                     |
| ----------------------------------------- | --------- | ---------------------------------------------------- |
| Dorobantu & Badea 2026, Systematic review | full      | LLMs cannot compute distance or containment reliably |
| Ji et al. 2025, Topological relations     | abstract  | GPT-4 ~0.66 on spatial predicates                    |
| Han et al. 2025, LLMs and spatial data    | abstract  | Good at human-scale relations, bad at geometry       |
| Truong et al. 2026, GPSBench              | abstract  | Strong at country level, weak at city level          |
| GeoSQL-Bench 2025                         | abstract  | Text-to-spatial-SQL benchmark                        |
| Staniek et al. 2024, Text-to-OverpassQL   | abstract  | Natural language to OSM queries                      |
| Dihan et al. 2025, MapEval                | abstract  | Best model under 67% on map reasoning                |
| Li et al. 2025, MapQA                     | abstract  | Geospatial QA typed by spatial relation              |
| Saeedan et al. 2026, GS-QA                | abstract  | Template spatial QA; drops on complex predicates     |
| Krechetova & Kochedykov 2025, GeoBenchX   | abstract  | Includes unsolvable geo tasks                        |
| Böckling et al. 2026, MultiGlobeQA       | abstract  | 17 languages; country and region level most reliable |
| Xu et al. 2024, Spatial tasks benchmark   | abstract  | Chain-of-thought helps route planning                |
