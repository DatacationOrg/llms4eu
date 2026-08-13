from src.shared.geocode import (
    Coordinates,
    ExtractedLocation,
    NominatimGeocoder,
    extract_location_query,
    locate_text,
)


class _FakeLlm:
    def __init__(self, location_query):
        self.location_query = location_query

    def structured_output(self, prompt, output_schema, *, retries=3):
        return ExtractedLocation(location_query=self.location_query)


class _FakeGeocoder:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.queries = []

    def geocode(self, query):
        self.queries.append(query)
        return self.coordinates


def test_extract_location_query_returns_stripped_string():
    llm = _FakeLlm("  Brestanica Castle, Slovenia  ")
    assert (
        extract_location_query(llm, "some place text") == "Brestanica Castle, Slovenia"
    )


def test_extract_location_query_returns_none_when_llm_finds_no_place():
    llm = _FakeLlm(None)
    assert extract_location_query(llm, "a generic description") is None


def test_locate_text_geocodes_the_extracted_query():
    llm = _FakeLlm("Brestanica Castle, Slovenia")
    geocoder = _FakeGeocoder(Coordinates(45.9, 15.5))

    result = locate_text(llm, geocoder, "text about the castle")

    assert result == Coordinates(45.9, 15.5)
    assert geocoder.queries == ["Brestanica Castle, Slovenia"]


def test_locate_text_skips_geocoding_when_no_place_found():
    llm = _FakeLlm(None)
    geocoder = _FakeGeocoder(Coordinates(45.9, 15.5))

    assert locate_text(llm, geocoder, "text with no place") is None
    assert geocoder.queries == []


def test_nominatim_geocoder_parses_first_result(monkeypatch):
    calls = []

    class _FakeResponse:
        status_code = 200

        def json(self):
            return [{"lat": "45.9", "lon": "15.5"}]

    def _fake_get(url, *, params, headers, timeout):
        calls.append((url, params, headers))
        return _FakeResponse()

    monkeypatch.setattr("src.shared.geocode.httpx.get", _fake_get)
    geocoder = NominatimGeocoder(min_interval_seconds=0)

    result = geocoder.geocode("Brestanica Castle")

    assert result == Coordinates(45.9, 15.5)
    assert calls[0][1]["q"] == "Brestanica Castle"


def test_nominatim_geocoder_returns_none_for_empty_results(monkeypatch):
    class _FakeResponse:
        status_code = 200

        def json(self):
            return []

    monkeypatch.setattr("src.shared.geocode.httpx.get", lambda *a, **k: _FakeResponse())
    geocoder = NominatimGeocoder(min_interval_seconds=0)

    assert geocoder.geocode("nowhere in particular") is None


def test_nominatim_geocoder_returns_none_on_request_error(monkeypatch):
    import httpx

    def _raise(*args, **kwargs):
        raise httpx.RequestError("boom")

    monkeypatch.setattr("src.shared.geocode.httpx.get", _raise)
    geocoder = NominatimGeocoder(min_interval_seconds=0)

    assert geocoder.geocode("anywhere") is None
