"""Synthetic, explicitly offline examples; never represented as live route data."""
from datetime import datetime, timezone

from .models import Fact, RouteCandidate, Source


def fixture_candidates(scenario: str = "complete") -> list[RouteCandidate]:
    now = datetime.now(timezone.utc).isoformat()
    candidates = []
    for cid, name, distance, ascent in [("demo-a", "演示林间环线", 8.0, 300.0),
                                        ("demo-b", "演示山脊路线", 12.0, 650.0)]:
        values = {"distance_km": distance, "ascent_m": ascent, "duration_hours": 4.0,
                  "weather": "晴", "map": "https://example.org/demo/map/" + cid,
                  "trail_status": "open", "risk": "low"}
        if scenario == "severe_weather":
            values["weather"] = "雷暴 / thunderstorm"
        if scenario == "caution":
            values["risk"] = "moderate"
        if scenario == "partial":
            values.pop("weather")
            values.pop("risk")
        if scenario == "empty":
            values = {}
        sid = cid + "-fixture"
        source = Source(id=sid, tool="offline_fixture", url="https://example.org/demo/" + cid,
                        excerpt="仅供离线演示的合成数据，不可用于真实出行。", observed_at=now)
        candidates.append(RouteCandidate(id=cid, name=name, data_mode="offline",
            facts={field: Fact(value=value, source_id=sid, source_field=field)
                   for field, value in values.items()},
            sources=[source] if values else [], evidence_values={sid: values}))
    return candidates
