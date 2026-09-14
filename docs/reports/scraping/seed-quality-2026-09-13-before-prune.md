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
| wikipedia | wikipedia | 49 | 5,003 | 2,908 | 58% | 2% | 0% | 2% | 100% | reference |

## New sources

| source | kind | pages | median chars | median prose | prose share | stubs | non-prose | duplicates | language match | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| bojnice_castle | attraction | 45 | 1,033 | 1,014 | 96% | 0% | 0% | 33% | 100% | duplicates: 33% vs 16% |
| burghausen_castle | attraction | 45 | 1,383 | 1,248 | 93% | 2% | 2% | 2% | 100% | ok |
| duino_castle | attraction | 4 | 371 | 344 | 97% | 50% | 0% | 25% | 100% | thin: median prose 344 vs 757 |
| jurisics_castle | attraction | 45 | 860 | 449 | 78% | 33% | 0% | 9% | 100% | ok |
| lednice_castle | attraction | 45 | 1,251 | 776 | 94% | 0% | 7% | 13% | 100% | ok |
| ptuj_museum | attraction | 43 | 1,002 | 830 | 81% | 26% | 0% | 5% | 100% | ok |
| riegersburg_castle | attraction | 45 | 700 | 652 | 95% | 4% | 0% | 0% | 100% | ok |
| valtice_castle | attraction | 45 | 1,230 | 773 | 88% | 16% | 7% | 2% | 100% | ok |
| veliki_tabor_castle | attraction | 45 | 895 | 873 | 79% | 31% | 0% | 9% | 100% | ok |
| at_riegersburg_biography | biography | 34 | 1,546 | 1,539 | 99% | 0% | 0% | 0% | 100% | ok |
| cz_lednice_biography | biography | 50 | 493 | 143 | 64% | 88% | 0% | 0% | 100% | thin: median prose 143 vs 2502; stubs: 88% vs 0% |
| de_burghausen_biography | biography | 50 | 815 | 516 | 88% | 10% | 0% | 0% | 100% | thin: median prose 516 vs 2502 |
| hr_zagorje_biography | biography | 44 | 6,261 | 6,038 | 98% | 0% | 0% | 2% | 100% | ok |
| hu_koszeg_biography | biography | 50 | 1,363 | 1,363 | 100% | 0% | 0% | 0% | 100% | ok |
| it_trieste_biography | biography | 50 | 13,526 | 13,382 | 99% | 0% | 0% | 0% | 100% | ok |
| si_ptuj_biography | biography | 16 | 1,919 | 1,911 | 99% | 0% | 0% | 0% | 100% | ok |
| bojnice_town | town | 35 | 4,405 | 2,485 | 90% | 20% | 0% | 3% | 100% | ok |
| burghausen_town | town | 35 | 1,288 | 1,125 | 65% | 3% | 3% | 3% | 100% | ok |
| desinic_municipality | town | 34 | 3,098 | 3,088 | 95% | 0% | 3% | 47% | 100% | duplicates: 47% vs 0% |
| koszeg_town | town | 35 | 1,879 | 1,468 | 86% | 3% | 3% | 0% | 100% | ok |
| lednice_municipality | town | 35 | 424 | 301 | 90% | 49% | 0% | 31% | 100% | duplicates: 31% vs 0% |
| miramare_castle | town | 35 | 2,477 | 2,365 | 96% | 0% | 3% | 26% | 100% | duplicates: 26% vs 0% |
| ptuj_tourism | town | 35 | 502 | 488 | 91% | 3% | 0% | 31% | 100% | duplicates: 31% vs 0% |
| riegersburg_municipality | town | 35 | 834 | 540 | 78% | 31% | 0% | 9% | 100% | ok |
| valtice_town | town | 35 | 1,102 | 672 | 83% | 26% | 0% | 3% | 100% | ok |
| at_riegersburg_wikipedia | wikipedia | 47 | 9,663 | 7,319 | 96% | 0% | 0% | 2% | 100% | ok |
| cz_lednice_wikipedia | wikipedia | 45 | 13,078 | 11,290 | 94% | 0% | 0% | 0% | 100% | ok |
| cz_valtice_wikipedia | wikipedia | 26 | 11,120 | 9,781 | 95% | 0% | 0% | 0% | 100% | ok |
| de_burghausen_wikipedia | wikipedia | 52 | 13,099 | 12,736 | 97% | 0% | 0% | 0% | 100% | ok |
| hr_zagorje_wikipedia | wikipedia | 53 | 2,654 | 2,193 | 92% | 2% | 0% | 0% | 100% | ok |
| hu_koszeg_wikipedia | wikipedia | 52 | 13,272 | 12,584 | 92% | 0% | 0% | 0% | 100% | ok |
| it_trieste_wikipedia | wikipedia | 40 | 13,690 | 12,386 | 93% | 0% | 0% | 0% | 100% | ok |
| si_ptuj_wikipedia | wikipedia | 49 | 7,493 | 6,440 | 94% | 0% | 0% | 2% | 100% | ok |
| sk_bojnice_wikipedia | wikipedia | 67 | 3,452 | 2,727 | 89% | 3% | 0% | 0% | 100% | ok |

Flagged: 8 of 34 sources: ptuj_tourism, desinic_municipality, duino_castle, miramare_castle, lednice_municipality, cz_lednice_biography, bojnice_castle, de_burghausen_biography
