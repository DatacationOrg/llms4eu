Split a question about a place into its clues. A clue is one constraint the question puts on the place: one attribute with one value. The question is in $lang; do not translate.

## Rules

- Quote every clue **exactly** as it appears in the question: a contiguous piece of the question, same words, same spelling, same case. Do not paraphrase, shorten inside, or join separate pieces.
- One attribute per clue. Split where the attributes change: "castle ruin in South Tyrol" is two clues, "castle ruin" (type) and "in South Tyrol" (location).
- Each location level is its own clue only when written apart: "in the Bohemian Forest in Bavaria" is two; "near Dundalk" is one.
- Leave out the question words and the asked-for part ("Which", "How high is", "In what year was"), and words that add no constraint ("the", "a", "that").
- If the question names the place it asks about, the name is a clue with attribute `name`.

## Attributes

- `name`: the place's own name, or what its name means or comes from.
- `location`: country, region, municipality, nearby town, river, mountain range, or a relation to another place ("across the railway from X", "south of Delfzijl").
- `type`: what kind of place it is (lake, castle ruin, nature reserve, hill, country house).
- `feature`: something physical you could see: shape, towers, islands, materials, vegetation, animals, what is there to do.
- `quantity`: a size, height, area, depth, length, count, or ranking ("132 hectares", "fifth highest", "three islands").
- `date`: a year, century, period or age ("in 1308", "19th-century", "22,000 years ago").
- `event`: something that happened or a story: history, ownership, legends, records, designations, appearances in media.
- `other`: a constraint that fits none of the above.

## Example

Question (nl): "Welke Nederlandse burcht wordt in 1308 voor het eerst vermeld, toen een graaf haar aan een vertrouwde man in leen gaf?"
clues:
- "Nederlandse" (location)
- "burcht" (type)
- "in 1308 voor het eerst vermeld" (date)
- "een graaf haar aan een vertrouwde man in leen gaf" (event)

## The question

Language: $lang
Question: $question
