# Which method works for which question: examples

*Real questions from the stratified sample (2026-10-08). Rank = position of the right page (1 is best, 101 = not in the top 100). "Anchors" = BM25 fused with a second BM25 search on the question's numbers and capitalised words. Numbers behind the cheat sheet: [README](README.md), [location](../location/README.md).*

## Cheat sheet

| question type | dense | BM25 | BM25 + anchors | location filter (oracle) |
|---|---|---|---|---|
| named place, same language | ✅ | ✅ | ✅ | not needed |
| named place, other language | ✅ | ❌ | ✅ mostly | ✅ |
| described place, same language | ❌ | ✅ | ✅ | ✅✅ |
| described place, other language | ❌ | ❌ | ⚠️ sometimes | ✅ (2% -> 45-70%) |
| German translation | - | ❌ | ⚠️ the heuristic grabs every noun | ✅ |

## Examples

**1. Named place, same language: everything works.** *"Waar in Noord-Holland stond het kasteel te Boekel?"* -> Kasteel te Boekel. Dense 1, BM25 1. The name does all the work; these questions only catch broken indexes.

**2. Named place, other language: dense carries it, BM25 needs help.** *"What can be seen from the viewpoints on Veľký Tribeč mountain?"* (Slovak page) -> Veľký Tribeč. Dense 1, BM25 61, BM25 + anchors 1. BM25 drowns in English pages about viewpoints; the name survives translation, so the anchor query ("Veľký Tribeč") finds it.

**3. Described place, same language: BM25 wins, dense finds the neighbourhood.** *"Welke groep meren in Zuid-Holland, nabij Oud Verlaat, droeg vanaf de 17e eeuw een andere naam en kreeg pas in 1968 de huidige naam op de topografische kaarten?"* -> Rottemeren. BM25 1, dense not in the top 100: it returns other Dutch nature areas (Strand Nijstad, Donkere Duinen). It gets the gist, not the details ("Oud Verlaat", "1968") BM25 matches exactly.

**4. Described place, other language: BM25 is lost, the anchors rescue it.** *"Which Ligurian fortress in the Savona hinterland stands at an altitude of 697 m and had a polygonal plan with two rings of walls?"* (Italian page) -> Castello di Cosseria. BM25 101, dense 101, BM25 + anchors 3. "Ligurian Savona 697" is language-agnostic enough; the full English question returns English pages.

**5. Described place, other language: everything fails.** *"Welk kustbekken in Dalmatië is verbonden met het Velebitkanaal door een smalle zeestraat die steil in kalksteen is ingesneden…?"* (Croatian page) -> Novigradsko more. All methods 101. The anchors are Dutch spellings ("Dalmatië", "Velebitkanaal"; the page says "Dalmacija", "Velebitski kanal"), so even the names do not match; dense returns Dutch pages that look alike on the surface ("Damil", "Damvallei"). This is where location helps most: searching only Croatian pages or the region removes those distractors.

**6. German translation: the heuristic breaks.** *"Welcher innere Kamin in einem Herrenhaus im Département Allier imitiert mittelalterliche Vorbilder mit sehr hohem Mantel?"* (French page). Anchors: "Kamin Herrenhaus Département Allier Vorbilder Mantel", every German noun is capitalised; only "Allier" helps. BM25 + anchors 63. Needs a German rule, or the location filter.

**7. The rare case where dense wins on a described place.** *"Кой приморски лиман в Югоизточна България е най-голямата естествена водна площ в страната от 1963 г. насам?"* -> Бургаско езеро (Lake Burgas). Dense 2, BM25 49. "The largest natural body of water in the country" is a meaning dense captures and BM25 cannot match word for word: dense is strong on what a place *is*, not on its exact specifics.

## Takeaways

- **The name is the strongest signal** and travels across languages: dense always uses it, BM25 only through the anchors.
- **Exact details (years, heights, local names) are BM25's strength**, but only in the page's language.
- **Translated place names break everything lexical** ("Dalmatië" vs "Dalmacija"); location filtering is the remaining lever.
- **Small heuristics have language-specific traps:** German nouns, and "Welk" counted as an anchor when it starts a second sentence.
