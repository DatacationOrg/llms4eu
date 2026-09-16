import json

import pytest

from src.shared import nuts as nuts_module


@pytest.fixture
def tiny_nuts(tmp_path):
    """Two adjacent NUTS-3 squares in one NUTS-2 and one country.

    SI036 (Posavska) spans lon 14.5-15.5, SI037 (Jugovzhodna Slovenija) spans
    15.5-16.5, both between lat 45 and 47. Enough to test point-in-polygon,
    parent codes and name matching without the 16 MB Eurostat file.
    """

    def square(lon0, lon1):
        return [[[lon0, 45.0], [lon1, 45.0], [lon1, 47.0], [lon0, 47.0], [lon0, 45.0]]]

    def feature(code, level, name, geometry):
        return {
            "type": "Feature",
            "properties": {
                "NUTS_ID": code,
                "LEVL_CODE": level,
                "CNTR_CODE": "SI",
                "NAME_LATN": name,
            },
            "geometry": geometry,
        }

    features = [
        feature(
            "SI036",
            3,
            "Posavska",
            {"type": "Polygon", "coordinates": square(14.5, 15.5)},
        ),
        feature(
            "SI037",
            3,
            "Jugovzhodna Slovenija",
            {"type": "Polygon", "coordinates": square(15.5, 16.5)},
        ),
        feature(
            "SI03",
            2,
            "Vzhodna Slovenija",
            {
                "type": "MultiPolygon",
                "coordinates": [square(14.5, 15.5), square(15.5, 16.5)],
            },
        ),
        feature(
            "SI", 0, "Slovenija", {"type": "Polygon", "coordinates": square(14.5, 16.5)}
        ),
    ]
    path = tmp_path / "nuts.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return nuts_module.nuts_index(path)
