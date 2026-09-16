# Gazetteer Misses Classified, 2026-09-09

The first action of the geo improvement plan
([../../architecture/geo-improvement-plan.md](../../architecture/geo-improvement-plan.md),
§2a and WP4): keep every place name the enrichment model produced that
Nominatim refused, and classify each by the cheapest fix that resolves it.
Produced by `just locate-pages --rejections PATH` (read-only; nothing was
written to `page_locations`), then reclassified three times with
`--reclassify` as the classifier was tightened; this is the fourth pass.
Code: `src/preprocess/rejections.py`.

## Result

| category | distinct names | page mentions | meaning |
|---|---|---|---|
| corpus | 14 | 16 | the head of the name is a primary location or a NUTS region the corpus already knows; no lookup needed |
| qualifier | 17 | 17 | a type word or trailing qualifier hid the place; stripping it resolves with the country hint |
| inflected | 0 | 0 | a Slovenian oblique form; none occurred |
| outside_hint | 42 | 44 | resolves in Nominatim once `geo_country_hint: si` is lifted |
| wikidata | 51 | 56 | Nominatim has nothing, a Slovenian-language Wikidata search has an item with coordinates (exonyms: Dunaj, Budimpešta, Celovec, Trst; historical: Kraljevina Ogrska, Avstrijsko primorje) |
| unresolved | 27 | 27 | none of the above |
| **total** | **151** | **160** | |

124 of 151 names (82%) resolve with a fix that needs no new model and no new
gazetteer. 86 of those 124 are places abroad. **The country hint, not
Slovenian morphology, is the cause of the misses.**

The dry run itself found 88 primary locations and 366 mentioned rows (the
2026-09-08 apply run: 89 and 375; the model's answers vary a little between
runs) and refused 147 names as "no hit" and 4 as "name mismatch".

## The finding that matters more than the misses

Checking the stored `mentioned` rows for the same names showed that the hint
does not only block places abroad, it places them wrongly. With
`countrycodes=si`, Nominatim returns the Slovenian hamlet that carries the
exonym, and `name_matches` passes because the token is identical:

| stored name | stored coordinates | what the page meant |
|---|---|---|
| Gradec | 45.465, 13.904 (SI) | Graz, Austria |
| Dunaj | 46.319, 14.046 (SI) | Vienna, Austria |
| Rim | 45.540, 15.293 (SI) | Rome, Italy |
| Trst | 45.610, 13.916 (SI) | Trieste, Italy |
| Gorica | 46.298, 15.254 (SI) | Gorizia / Nova Gorica |

374 of the 375 stored mentioned rows are in Slovenia, although the biographies
name Vienna, Graz, Trieste and Rome throughout. These rows are wrong data in
`data/db/pages.db` today; they are not used by any filter yet (only primaries
reach chunk metadata), but WP2 plans to make mentioned rows scoreable, so they
must be re-resolved first.

## What this changes in the plan

1. WP4 order: lift the country hint for tier 3 and use it as a tie-break or
   prior only; use the `sl`-language Wikidata search for exonyms; then the
   saint-name and qualifier stripping. The Slovenian lemmatiser (CLASSLA) is
   demoted to "only if a later run shows oblique forms": this run showed none.
2. Before WP2 promotes mentioned rows to scoreable footprints, re-run tier 3
   for every mentioned row with the hint lifted and compare; expect roughly a
   quarter of the 375 to move abroad.
3. The 27 unresolved are hamlets ("Koprivnica pri Brestanici", "Raztez pri
   Brestanici", "Slom, Brestanica"), Roman-era sites (Siscia) and geological
   features (antiklinala). Hamlets are the case for the Slovenian geographic
   names register (REZI) as a domain gazetteer, if its licence allows.

## Limitations of the classifier

- `corpus` matches the head of the name only, so "Koroška, Avstrija"
  (Carinthia, Austria) matched the Slovenian NUTS-3 region Koroška. One row.
- `qualifier` accepts a hit that names the head; for saint-named churches the
  head is the saint ("Marije Lurške"), so the hit is the right basilica in
  Brestanica by luck of the corpus, not by design.
- `wikidata` in the `sl` search accepts the top item with coordinates without a
  label check, which is what makes exonyms resolve; a reader should scan the
  "Resolves as" column. "Nürnberški grad" resolving to "Nuremberg trials" is
  the kind of error this admits.
- Categories are exclusive and ordered; a name that would resolve two ways is
  counted once under the cheapest.

## Full table

| Name | Reason | Category | Resolves as | Pages | Roles |
|---|---|---|---|---|---|
| Graz, Austria | no hit | corpus | graz | 2 | mentioned |
| Rajhenburg | name mismatch | corpus | grad rajhenburg | 2 | mentioned |
| Berlin | no hit | corpus | berlin | 1 | mentioned |
| Brestanica (stream), Slovenia | no hit | corpus | brestanica | 1 | mentioned |
| Groningen, Netherlands | no hit | corpus | groningen | 1 | mentioned |
| Koroška, Avstrija | no hit | corpus | koroska | 1 | mentioned |
| Neviodunum (Drnovo pri Krškem) | no hit | corpus | neviodunum | 1 | mentioned |
| Pordenone, Italy | no hit | corpus | pordenone | 1 | mentioned |
| Savinjska dolina | no hit | corpus | savinjska | 1 | mentioned |
| Sevnica Castle, Sevnica, Slovenia | no hit | corpus | sevnica | 1 | mentioned |
| Slovenia | name mismatch | corpus | slovenia | 1 | mentioned |
| Trieste, Italy | no hit | corpus | trieste | 1 | mentioned |
| Wittenberg, Germany | no hit | corpus | wittenberg | 1 | mentioned |
| Wittenberg, Nemčija | no hit | corpus | wittenberg | 1 | mentioned |
| Bazilika Lurške Marije, Brestanica, Slovenia | no hit | qualifier | Lurške Marije | 1 | mentioned |
| Bazilika Marije Lurške, Brestanica, Slovenia | no hit | qualifier | Marije Lurške | 1 | mentioned |
| Bazilika Marije Snežne, Rim, Italija | no hit | qualifier | Marije Snežne | 1 | mentioned |
| Bazilika sv. Frančiška Asiškega, Assisi, Italija | no hit | qualifier | Frančiška Asiškega | 1 | mentioned |
| Bazilika sv. Petra, Vatikan | no hit | qualifier | Petra | 1 | mentioned |
| Brežice Castle, Brežice, Slovenia | no hit | qualifier | Brežice Castle, Brežice | 1 | mentioned |
| Cerkev Marije Lurške, Brestanica, Slovenia | no hit | qualifier | Marije Lurške | 1 | mentioned |
| Cerkev sv. Lovrenca, Vojnik, Slovenija | no hit | qualifier | Lovrenca | 1 | mentioned |
| Cerkev sv. Ožbolta, Uniše, Slovenija | no hit | qualifier | Ožbolta | 1 | mentioned |
| Cerkev sv. Trojice, Slovenske gorice, Slovenija | no hit | qualifier | Trojice | 1 | mentioned |
| Hom (Bohor), Slovenia | no hit | qualifier | Hom | 1 | mentioned |
| Kostanjevica na Krki Monastery, Kostanjevica na Krki, Slovenia | no hit | qualifier | Kostanjevica na Krki, Kostanjevica na Krki | 1 | mentioned |
| Otočec (Šempeter), Novo Mesto, Slovenia | no hit | qualifier | Otočec, Novo Mesto | 1 | mentioned |
| Soška dolina, Slovenia | no hit | qualifier | Soška | 1 | mentioned |
| Stara vas, Videm ob Savi, Slovenia | no hit | qualifier | Stara | 1 | mentioned |
| Trška gora pri Krškem, Slovenia | no hit | qualifier | Trška gora pri Krškem | 1 | mentioned |
| Zagorje pri Planini, Slovenia | no hit | qualifier | Zagorje pri Planini | 1 | mentioned |
| Lyon | no hit | outside_hint | Lyon, Métropole de Lyon, Rhône, Auvergne-Rhône-Alpes, France métropolitaine, France | 2 | mentioned |
| Lyon, France | no hit | outside_hint | Lyon, Métropole de Lyon, Rhône, Auvergne-Rhône-Alpes, France métropolitaine, France | 2 | mentioned |
| Accra, Ghana | no hit | outside_hint | Accra, Korle-Klottey Municipal District, Greater Accra Region, Ghana | 1 | mentioned |
| Aix-en-Provence, France | no hit | outside_hint | Aix-en-Provence, Bouches-du-Rhône, Provence-Alpes-Côte d'Azur, France métropolitaine, France | 1 | mentioned |
| Antwerpen | no hit | outside_hint | Antwerpen, Vlaanderen, België / Belgique / Belgien | 1 | mentioned |
| Atrans (Trojane) | no hit | outside_hint | ATRANS, Kozłów Szlachecki, gmina Nowa Sucha, powiat sochaczewski, województwo mazowieckie, Polska | 1 | mentioned |
| Banská Štiavnica, Slovakia | no hit | outside_hint | Banská Štiavnica, okres Banská Štiavnica, Banskobystrický kraj, 969 01, Slovensko | 1 | mentioned |
| Beckov, Slovakia | no hit | outside_hint | Beckov, okres Nové Mesto nad Váhom, Trenčiansky kraj, Slovensko | 1 | mentioned |
| Benediktinski samostan Melk, Melk, Avstrija | no hit | outside_hint | Melk, Bezirk Melk, Niederösterreich, 3390, Österreich | 1 | mentioned |
| Beram, Istria, Croatia | no hit | outside_hint | Beram, Grad Pazin, Istarska županija, Hrvatska | 1 | mentioned |
| Bilina, Czech Republic | no hit | outside_hint | Bílina, okres Teplice, Ústecký kraj, Česko | 1 | mentioned |
| Bleiburg, Austria | no hit | outside_hint | Bleiburg, Bezirk Völkermarkt, Kärnten, 9150, Österreich | 1 | mentioned |
| Bratislava, Slovakia | no hit | outside_hint | Bratislava, Bratislavský kraj, Slovensko | 1 | mentioned |
| Bristol, Združeno kraljestvo | no hit | outside_hint | City of Bristol, West of England, England, United Kingdom | 1 | mentioned |
| Brno, Czech Republic | no hit | outside_hint | Brno, okres Brno-město, Jihomoravský kraj, Česko | 1 | mentioned |
| Brugge | no hit | outside_hint | Brugge, West-Vlaanderen, Vlaanderen, België / Belgique / Belgien | 1 | mentioned |
| Frankfurt na Majni | no hit | outside_hint | Frankfurt am Main, Hessen, Deutschland | 1 | mentioned |
| Gemona, Italy | no hit | outside_hint | Gemona del Friuli / Glemone, Udine, Friuli-Venezia Giulia, 33013, Italia | 1 | mentioned |
| Gonars, Italy | no hit | outside_hint | Gonars / Gonârs, Udine, Friuli-Venezia Giulia, 33050, Italia | 1 | mentioned |
| Grad Burghausen, Burghausen, Nemčija | no hit | outside_hint | Burghausen, Miltitzer Straße, Burghausen-Rückmarsdorf, Altwest, Leipzig, Sachsen, 04178, Deutschland | 1 | mentioned |
| Greenwich, London, Združeno kraljestvo | no hit | outside_hint | Greenwich, Greater London, England, SE10 9HF, United Kingdom | 1 | mentioned |
| Humin/Gemona del Friuli, Italija | no hit | outside_hint | Gemona del Friuli, Piazzale della Stazione, Piovega, Maniaglia, Gemona del Friuli / Glemone, Udine, Friuli-Venezia Giulia, 33013, Italia | 1 | mentioned |
| Jugoslavija | no hit | outside_hint | Jugoslavija, 25, Tannenbergstraße, Kernstadt Aurich, Aurich, Landkreis Aurich, Niedersachsen, 26603, Deutschland | 1 | mentioned |
| Laško, Štajerska | no hit | outside_hint | Lasko, B76, Katastralgemeinde Bachholz, Eibiswald, Bezirk Deutschlandsberg, Steiermark, 8552, Österreich | 1 | mentioned |
| Maglaj, Bosnia and Herzegovina | no hit | outside_hint | Maglaj, Općina Maglaj, Zeničko-dobojski kanton, Federacija Bosne i Hercegovine, 74250, Bosna i Hercegovina / Босна и Херцеговина | 1 | mentioned |
| Modra, Slovakia | no hit | outside_hint | Modra, okres Pezinok, Bratislavský kraj, 900 01, Slovensko | 1 | mentioned |
| Nabrežina, Italy | no hit | outside_hint | Aurisina / Nabrežina, Duino Aurisina / Devin - Nabrežina, Trieste, Friuli-Venezia Giulia, 34011, Italia | 1 | mentioned |
| Nancy, France | no hit | outside_hint | Nancy, Meurthe-et-Moselle, Grand Est, France métropolitaine, 54100, France | 1 | mentioned |
| Nitra, Slovakia | no hit | outside_hint | Nitra, okres Nitra, Nitriansky kraj, Slovensko | 1 | mentioned |
| Opatija, Croatia | no hit | outside_hint | Grad Opatija, Primorsko-goranska županija, Hrvatska | 1 | mentioned |
| Opčine, Italy | no hit | outside_hint | Opicina / Opčine, Trieste, Friuli-Venezia Giulia, 34135, Italia | 1 | mentioned |
| Pazin, Istria, Croatia | no hit | outside_hint | Grad Pazin, Istarska županija, 52000, Hrvatska | 1 | mentioned |
| Pula, Istria, Croatia | no hit | outside_hint | Grad Pula, Istarska županija, Hrvatska | 1 | mentioned |
| Pécs, Hungary | no hit | outside_hint | Pécs, Pécsi járás, Baranya vármegye, Dél-Dunántúl, Dunántúl, Magyarország | 1 | mentioned |
| Romsdalen, Norway | no hit | outside_hint | Romsdalen, Møre og Romsdal, Norge | 1 | mentioned |
| Sarajevo, Bosnia and Herzegovina | no hit | outside_hint | Sarajevo, Mjesna zajednica Trg oslobođenja-Centar, Općina Centar, Grad Sarajevo, Kanton Sarajevo, Federacija Bosne i Hercegovine, 71000, Bosna i Hercegovina / Босна и Херцеговина | 1 | mentioned |
| South Bend, Indiana, USA | no hit | outside_hint | South Bend, Saint Joseph County, Indiana, United States | 1 | mentioned |
| Stralsund, Germany | no hit | outside_hint | Stralsund, Vorpommern-Rügen, Mecklenburg-Vorpommern, Deutschland | 1 | mentioned |
| Traismauer, Austria | no hit | outside_hint | Traismauer, Bezirk St. Pölten, Niederösterreich, Österreich | 1 | mentioned |
| Trviž, Istria, Croatia | no hit | outside_hint | Trviž, Grad Pazin, Istarska županija, Hrvatska | 1 | mentioned |
| Zadar, Croatia | no hit | outside_hint | Zadar, Grad Zadar, Zadarska županija, Hrvatska | 1 | mentioned |
| Zalavár, Hungary | no hit | outside_hint | Zalavár, Keszthelyi járás, Zala vármegye, Nyugat-Dunántúl, Dunántúl, 8392, Magyarország | 1 | mentioned |
| Gradec, Austria | no hit | wikidata | Graz | 3 | mentioned |
| Dunaj (Vienna), Austria | no hit | wikidata | Vienna | 2 | mentioned |
| Prague, Czech Republic | no hit | wikidata | Prague | 2 | mentioned |
| Vienna, Austria | no hit | wikidata | Vienna | 2 | mentioned |
| Avstrija | no hit | wikidata | Austria | 1 | mentioned |
| Avstrijsko primorje | no hit | wikidata | Austrian Littoral | 1 | mentioned |
| Blatenski Kostel, Hungary | no hit | wikidata | Keszthely | 1 | mentioned |
| Boston, ZDA | no hit | wikidata | Boston | 1 | mentioned |
| Brezje, Brestanica, Slovenia | no hit | wikidata | Brezje | 1 | mentioned |
| Budimpešta | no hit | wikidata | Budapest | 1 | mentioned |
| Budimpešta, Hungary | no hit | wikidata | Budapest | 1 | mentioned |
| Celje Regional Museum | no hit | wikidata | Celje Regional Museum | 1 | mentioned |
| Celovec | no hit | wikidata | Klagenfurt am Wörthersee | 1 | mentioned |
| dolina Save Bohinjke, Slovenia | no hit | wikidata | Sava Bohinjka | 1 | mentioned |
| Dunaj, Austria | no hit | wikidata | Vienna | 1 | mentioned |
| Dunaj, Avstrija | no hit | wikidata | Vienna | 1 | mentioned |
| Islandija | no hit | wikidata | Iceland | 1 | mentioned |
| Italija | name mismatch | wikidata | Italy | 1 | mentioned |
| Kraljevina Ogrska | no hit | wikidata | Kingdom of Hungary | 1 | mentioned |
| Lokve, Brestanica, Slovenia | no hit | wikidata | Lokve | 1 | mentioned |
| Lurd, France | no hit | wikidata | Lourdes | 1 | mentioned |
| Lurd, Francija | no hit | wikidata | Lourdes | 1 | mentioned |
| Lviv, Ukraine | no hit | wikidata | Lviv | 1 | mentioned |
| Marienstatt, Hessen Nassau, Germany | no hit | wikidata | Marienstatt Abbey | 1 | mentioned |
| Mogila (Krakow), Poland | no hit | wikidata | Mogila | 1 | mentioned |
| National Gallery of Slovenia | no hit | wikidata | National Gallery of Slovenia | 1 | mentioned |
| National Museum of Slovenia | name mismatch | wikidata | National Museum of Slovenia | 1 | mentioned |
| New York, ZDA | no hit | wikidata | New York City | 1 | mentioned |
| Nova Zelandija | no hit | wikidata | New Zealand | 1 | mentioned |
| Nürnberški grad, Nürnberg, Nemčija | no hit | wikidata | Nuremberg trials | 1 | mentioned |
| Osek, Vipavska dolina, Slovenia | no hit | wikidata | Osek | 1 | mentioned |
| Pariz, France | no hit | wikidata | Paris | 1 | mentioned |
| Pihovec, Brestanica, Slovenia | no hit | wikidata | Pihovec | 1 | mentioned |
| Praga, Czech Republic | no hit | wikidata | Prague | 1 | mentioned |
| Retje nad Trbovljami | no hit | wikidata | Retje nad Trbovljami | 1 | mentioned |
| Sveto rimsko cesarstvo | no hit | wikidata | Holy Roman Empire | 1 | mentioned |
| Trnovski gozd, prepad pri Mrzli Rupi | no hit | wikidata | Trnovo Forest Plateau | 1 | mentioned |
| Trst, Italija | no hit | wikidata | Trieste | 1 | mentioned |
| Trst, Italy | no hit | wikidata | Trieste | 1 | mentioned |
| Videm (Udine), Italy | no hit | wikidata | Udine | 1 | mentioned |
| Videm, Italy | no hit | wikidata | Udine | 1 | mentioned |
| Vrhpolje pri Vipavi | no hit | wikidata | Vrhpolje | 1 | mentioned |
| Združeno kraljestvo | no hit | wikidata | United Kingdom | 1 | mentioned |
| Zemun, Srbija | no hit | wikidata | Zemun | 1 | mentioned |
| Zgornjesavska dolina, Slovenia | no hit | wikidata | Upper Sava Valley | 1 | mentioned |
| Šampanja | no hit | wikidata | Champagne | 1 | mentioned |
| Šmarje, Sevnica, Slovenia | no hit | wikidata | Šmarje | 1 | mentioned |
| Šrajbarski Turn | no hit | wikidata | Šrajbarski Turn Castle | 1 | mentioned |
| Štajerska, Avstrija | no hit | wikidata | Styria | 1 | mentioned |
| Žabnice, Kanalska dolina | no hit | wikidata | Camporosso | 1 | mentioned |
| Ženeva | no hit | wikidata | Geneva | 1 | mentioned |
| Bazilika sv. Janeza v Lateranu, Rim, Italija | no hit | unresolved |  | 1 | mentioned |
| Bazilika sv. Pavla zunaj obzidja, Rim, Italija | no hit | unresolved |  | 1 | mentioned |
| Brestanica stream | no hit | unresolved |  | 1 | mentioned |
| Dumb, France | no hit | unresolved |  | 1 | mentioned |
| Ečka pri Zrenjaninu, Srbija | no hit | unresolved |  | 1 | mentioned |
| Gornja Lokvica pri Metliki, Slovenia | no hit | unresolved |  | 1 | mentioned |
| House of Mozer, Brestanica, Slovenia | no hit | unresolved |  | 1 | mentioned |
| Kapla na Koroškem, Slovenia | no hit | unresolved |  | 1 | mentioned |
| Koprivnica pri Brestanici, Slovenia | no hit | unresolved |  | 1 | mentioned |
| Krško polje, Slovenia | no hit | unresolved |  | 1 | mentioned |
| Lehn na Pohorju, Slovenia | no hit | unresolved |  | 1 | mentioned |
| Litijska antiklinala | no hit | unresolved |  | 1 | mentioned |
| Mačkovci pri Brestanici, Slovenia | no hit | unresolved |  | 1 | mentioned |
| Nemško-Poljsko nižavje, Germany/Poland | no hit | unresolved |  | 1 | mentioned |
| Posavje Museum Brežice | no hit | unresolved |  | 1 | mentioned |
| Raztez pri Brestanici | no hit | unresolved |  | 1 | mentioned |
| Raztez pri Brestanici, Slovenia | no hit | unresolved |  | 1 | mentioned |
| Selišča pri Vidmu ob Ščavnici, Slovenia | no hit | unresolved |  | 1 | mentioned |
| Siscia (Sisak) | no hit | unresolved |  | 1 | mentioned |
| Slom, Brestanica, Slovenia | no hit | unresolved |  | 1 | mentioned |
| Spodnji grad Turn | no hit | unresolved |  | 1 | mentioned |
| Sremič pri Vidmu | no hit | unresolved |  | 1 | mentioned |
| Sveta Marjeta na Dravskem polju, Slovenia | no hit | unresolved |  | 1 | mentioned |
| Sveti Jurij ob južni železnici, Slovenia | no hit | unresolved |  | 1 | mentioned |
| Trojanska antiklinala | no hit | unresolved |  | 1 | mentioned |
| Veliki Tinj, Pohorje, Slovenia | no hit | unresolved |  | 1 | mentioned |
| Visoko pri Novem Marofu | no hit | unresolved |  | 1 | mentioned |
