# Datasets

One row per dataset, same fields for all, for comparison with the official dataset.


| dataset                           | source | size | languages | domain | location coverage | question types  | answer form | gold format | licence / available? |
| :-------------------------------- | ------ | ---- | --------- | ------ | ----------------- | --------------- | ----------- | ----------- | :------------------- |
| TourismQA-NYC | Contractor 2019; travel forums + reviews | large (thousands of POIs) | not stated | hotels, restaurants, attractions | one city | real recommendation questions, full of preferences | a few places | places recommended in forum replies; incomplete | scripts only, no data |
| TourismQA-Miami | same as NYC | small | not stated | hotels, restaurants, attractions | one city | real recommendation questions, full of preferences | a few places | places recommended in forum replies; incomplete | scripts only, no data |
| MapQA-Adjacent | Li et al. 2025; map data | large map, few questions | ? | map POIs | ? | Adjacent-based | a set of places | computed from the map; exact | ? |
| MapQA Amenities\_Around\_Specific | Li et al. 2025; map data | large map, few questions | ? | map POIs | ? | proximity-based | a set of places | computed from the map; exact | ? |
| TravelDest |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |

TravelDest: 50 queries × 774 cities, every pair labelled by three people (1–5), relevant if the average is 3 or more, then checked by two experts. Small but exhaustive: a template for a human-labelled recommendation set, and the human-calibration sample the LLM-judge paper said is needed