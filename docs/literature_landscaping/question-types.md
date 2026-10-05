# Candidate question types for the official dataset

## Question dimensions (for designing and classifying questions)

- **Role of the place:** constraint (what is allowed), score (what ranks
  higher), disambiguator (which entity is meant), or none.
- **Anchor type:** named place ("in Amsterdam"), the user's location ("near my
  work"), or a natural or informal region ("in the Alps") `[me]`.
- **Answer form:** one right answer, a set, or a ranked list.
- **Room for hidden gems:** can a lesser-known place be a good answer?
- **Language and borders:** can a good answer lie in another language or
  across a border?

First draft `[me]`, refined with Claude. Classified by the question dimensions
in section 3.


| question                                                 | role of place      | anchor         | answer form | note                                         |
| -------------------------------------------------------- | ------------------ | -------------- | ----------- | -------------------------------------------- |
| What museums are in Amsterdam?                           | constraint         | named place    | set         |                                              |
| What restaurants are within walking distance of my work? | constraint + score | user location  | ranked      | anchor is the user, not a named place        |
| What language do they speak in Paris?                    | disambiguator      | named place    | one         |                                              |
| How many residents does Amsterdam have?                  | disambiguator      | named place    | one         |                                              |
| What square is adjacent to the Rijksmuseum?              | constraint         | named place    | one         | spatial relation: adjacency                  |
| Which squares are in Amsterdam?                          | constraint         | named place    | set         |                                              |
| Which squares are closest to the Rijksmuseum?            | score              | named place    | ranked      |                                              |
| What is a good art museum around Amsterdam?              | score              | named place    | ranked      | room for hidden gems; needs a quality signal |
| What mountain walks are there in the Alps?               | constraint         | natural region | set         | spans countries and languages; no NUTS code  |

To add:

- a true *none* question ("What's the best season to visit castles?")
- the border case ("Castles within 30 km of Brestanica", with answers in Croatia)
- questions with preferences, phrased as tourists ask ("a quiet castle",
  "something good with kids")
- note which question types felt forced to write
