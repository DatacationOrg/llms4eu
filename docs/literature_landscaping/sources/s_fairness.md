# Modeling sustainable city trips: integrating CO2e emissions,  popularity, and seasonality into tourism recommender systems
### Banerjee et al. 2025, Modeling sustainable city trips (Information Technology & Tourism)  (https://link.springer.com/article/10.1007/s40558-024-00303-1)
- Problem stated: Tourism recommendation is a multi-stakeholder problem,
  affecting not just the traveller but also society, businesses and the
  environment (over- and undertourism).
- New problem for us? Partly: popularity is already known (Banerjee 2024);
  seasonality (when to visit) and travel emissions are new.
- Goal: technique
- Relates to: (re)rank scores; Banerjee 2024 (this paper defines its S-Fairness
  score)
- Verdict: adds
- The idea in one sentence: Based on starting point and month of travel,
  score destinations with S-Fairness, a weighted sum of CO2e, popularity and
  seasonality, with weights from a user study.
- So what for us: Gives the methods for getting the signals. Popularity =
  number of POIs + number of reviews + Google Trends image searches; Google
  Trends barely correlates with the other two, so popularity isn't one signal.
  POI and review counts are close to documentation volume. Seasonality = Gini
  over monthly arrivals (TourMIS) and Airbnb prices.
- Feasibility: No training. Weights come from a Likert survey (200 people), not
  learned. Most data is scraped or external (Tripadvisor, Google Flights);
  Google Trends, TourMIS and Inside Airbnb are open, but TourMIS covers ~65
  cities and Inside Airbnb 45 ⚑.
- Open question: Do these sources exist for small localities, or is the corpus
  itself the only signal there?

Notes:
- Destinations are the 200 most populated cities with an airport, so
  undertourism places are excluded by design.
- S-Fairness is an interesting evaluation metric, but only validated by user
  perception (Likert), not by whether rankings shift.
- CO2e is an interesting target, but an extension beyond the popularity /
  seasonality basis in our current scope.
- Users know about sustainability but prioritize convenience. The paper
  concludes: better communication. A more direct approach: explain why a place
  was recommended ("avoids crowds this time of year"). Their UI already shows
  low/medium/high popularity and seasonality tags, but only 30% found the
  score helpful (32% didn't).