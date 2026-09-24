# Seed quality against the Brestanica sources, 2026-09-13

Database: `/home/gerson/llms4eu/data/db/pages.db`. Prose is the text left after stripping
links, images, markup and lines that are mostly not letters; a stub has
under 300 prose characters. A source is `thin` when its median
prose is under 50% of its reference, `stubs` when its stub share is
over 2x the reference (and over 10%), `duplicates` likewise for
pages whose content hash repeats within the source, `language` when under
60% of its pages detect as the declared language.

## Reference (gold standard)

| source | kind | pages | median chars | median prose | prose share | stubs | non-prose | duplicates | language match | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| castle_rajhenburg | attraction | 43 | 1,196 | 757 | 87% | 33% | 21% | 16% | 64% | reference |
| brestanica_webpage | town | 31 | 701 | 590 | 86% | 32% | 3% | 0% | 100% | reference |
| svn_biography | biography | 51 | 2,512 | 2,502 | 98% | 0% | 0% | 0% | 100% | reference |
| wikipedia | wikipedia | 51 | 5,390 | 2,908 | 57% | 2% | 0% | 2% | 100% | reference |

## New sources

| source | kind | pages | median chars | median prose | prose share | stubs | non-prose | duplicates | language match | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| bojnice_castle | attraction | 29 | 1,314 | 1,180 | 95% | 0% | 0% | 0% | 100% | ok |
| burghausen_castle | attraction | 42 | 1,380 | 1,234 | 93% | 0% | 0% | 0% | 100% | ok |
| duino_castle | attraction | 2 | 969 | 931 | 96% | 0% | 0% | 0% | 100% | ok |
| jurisics_castle | attraction | 22 | 1,052 | 780 | 83% | 0% | 0% | 0% | 100% | ok |
| lednice_castle | attraction | 37 | 1,763 | 1,297 | 94% | 0% | 8% | 0% | 100% | ok |
| ptuj_museum | attraction | 30 | 1,238 | 1,072 | 82% | 0% | 0% | 0% | 100% | ok |
| riegersburg_castle | attraction | 43 | 704 | 700 | 95% | 0% | 0% | 0% | 100% | ok |
| valtice_castle | attraction | 36 | 1,567 | 1,248 | 89% | 0% | 8% | 0% | 100% | ok |
| veliki_tabor_castle | attraction | 25 | 1,744 | 1,312 | 92% | 0% | 0% | 0% | 100% | ok |
| at_riegersburg_biography | biography | 34 | 1,546 | 1,539 | 99% | 0% | 0% | 0% | 100% | ok |
| cz_lednice_biography | biography | 6 | 4,396 | 4,079 | 90% | 0% | 0% | 0% | 100% | ok |
| de_burghausen_biography | biography | 45 | 1,013 | 753 | 88% | 0% | 0% | 0% | 100% | thin: median prose 753 vs 2502 |
| hr_zagorje_biography | biography | 42 | 6,334 | 6,198 | 98% | 0% | 0% | 0% | 100% | ok |
| hu_koszeg_biography | biography | 50 | 1,363 | 1,363 | 100% | 0% | 0% | 0% | 100% | ok |
| it_trieste_biography | biography | 50 | 13,526 | 13,382 | 99% | 0% | 0% | 0% | 100% | ok |
| si_ptuj_biography | biography | 16 | 1,919 | 1,911 | 99% | 0% | 0% | 0% | 100% | ok |
| bojnice_town | town | 28 | 5,013 | 4,634 | 90% | 0% | 0% | 0% | 100% | ok |
| burghausen_town | town | 32 | 1,363 | 1,170 | 66% | 0% | 3% | 0% | 100% | ok |
| desinic_municipality | town | 17 | 4,518 | 4,444 | 94% | 0% | 6% | 0% | 100% | ok |
| koszeg_town | town | 34 | 1,952 | 1,518 | 86% | 0% | 3% | 0% | 100% | ok |
| lednice_municipality | town | 18 | 694 | 508 | 92% | 0% | 0% | 0% | 100% | ok |
| miramare_castle | town | 25 | 3,792 | 3,531 | 96% | 0% | 4% | 0% | 100% | ok |
| ptuj_tourism | town | 22 | 552 | 449 | 88% | 0% | 0% | 0% | 100% | ok |
| riegersburg_municipality | town | 24 | 1,271 | 1,088 | 88% | 0% | 0% | 0% | 100% | ok |
| valtice_town | town | 26 | 1,509 | 1,138 | 85% | 0% | 0% | 0% | 100% | ok |
| at_riegersburg_wikipedia | wikipedia | 45 | 10,675 | 10,282 | 96% | 0% | 0% | 0% | 100% | ok |
| cz_lednice_wikipedia | wikipedia | 45 | 13,078 | 11,290 | 94% | 0% | 0% | 0% | 100% | ok |
| cz_valtice_wikipedia | wikipedia | 26 | 11,120 | 9,781 | 95% | 0% | 0% | 0% | 100% | ok |
| de_burghausen_wikipedia | wikipedia | 52 | 13,099 | 12,736 | 97% | 0% | 0% | 0% | 100% | ok |
| hr_zagorje_wikipedia | wikipedia | 52 | 2,710 | 2,399 | 92% | 0% | 0% | 0% | 100% | ok |
| hu_koszeg_wikipedia | wikipedia | 52 | 13,272 | 12,584 | 92% | 0% | 0% | 0% | 100% | ok |
| it_trieste_wikipedia | wikipedia | 40 | 13,690 | 12,386 | 93% | 0% | 0% | 0% | 100% | ok |
| si_ptuj_wikipedia | wikipedia | 45 | 5,582 | 5,159 | 94% | 0% | 0% | 0% | 100% | ok |
| sk_bojnice_wikipedia | wikipedia | 65 | 3,760 | 3,282 | 89% | 0% | 0% | 0% | 100% | ok |

Flagged: 1 of 34 sources: de_burghausen_biography
