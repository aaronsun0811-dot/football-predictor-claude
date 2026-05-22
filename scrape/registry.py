from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "config" / "leagues.yaml"


EXTRA_ALIASES = {
    "英超": "premier_league",
    "premier league": "premier_league",
    "england premier league": "premier_league",
    "英冠": "championship",
    "efl championship": "championship",
    "英甲": "league_one",
    "efl league one": "league_one",
    "西甲": "la_liga",
    "la liga": "la_liga",
    "西乙": "segunda",
    "segunda": "segunda",
    "la liga 2": "segunda",
    "意甲": "serie_a",
    "serie a": "serie_a",
    "意乙": "serie_b",
    "serie b": "serie_b",
    "葡超": "primeira",
    "primeira liga": "primeira",
    "葡甲": "liga_portugal_2",
    "liga portugal 2": "liga_portugal_2",
    "荷甲": "eredivisie",
    "eredivisie": "eredivisie",
    "德甲": "bundesliga",
    "bundesliga": "bundesliga",
    "德乙": "bundesliga_2",
    "2. bundesliga": "bundesliga_2",
    "法甲": "ligue_1",
    "ligue 1": "ligue_1",
    "法乙": "ligue_2",
    "ligue 2": "ligue_2",
    "比甲": "belgian_pro",
    "belgian pro league": "belgian_pro",
    "沙特": "saudi_pro",
    "沙特职业足球联赛": "saudi_pro",
    "沙特职业联赛": "saudi_pro",
    "saudi": "saudi_pro",
    "saudi pro league": "saudi_pro",
    "j1": "j1",
    "j1联赛": "j1",
    "j1 league": "j1",
    "韩国k1联赛": "k1",
    "k联赛1": "k1",
    "k league 1": "k1",
    "中超": "chinese_super_league",
    "中超联赛": "chinese_super_league",
    "中国超级联赛": "chinese_super_league",
    "中国足球超级联赛": "chinese_super_league",
    "中国足球协会超级联赛": "chinese_super_league",
    "csl": "chinese_super_league",
    "chinese super league": "chinese_super_league",
    "china super league": "chinese_super_league",
    "美职联": "mls",
    "mls": "mls",
    "major league soccer": "mls",
    "巴甲": "brasileirao",
    "brasileirão": "brasileirao",
    "brasileirao": "brasileirao",
    "阿甲": "primera_argentina",
    "liga profesional": "primera_argentina",
    "墨超": "liga_mx",
    "liga mx": "liga_mx",
    "世界杯": "world_cup",
    "world cup": "world_cup",
    "fifa world cup": "world_cup",
    "2026 world cup": "world_cup",

    # ----- Continental club competitions -----
    "欧冠": "champions_league",
    "欧洲冠军联赛": "champions_league",
    "冠军联赛": "champions_league",
    "champions league": "champions_league",
    "uefa champions league": "champions_league",
    "ucl": "champions_league",

    "欧联": "europa_league",
    "欧罗巴": "europa_league",
    "欧罗巴联赛": "europa_league",
    "europa league": "europa_league",
    "uefa europa league": "europa_league",
    "uel": "europa_league",

    "欧会杯": "conference_league",
    "欧协联": "conference_league",
    "conference league": "conference_league",
    "uefa conference league": "conference_league",
    "uecl": "conference_league",

    "解放者": "copa_libertadores",
    "解放者杯": "copa_libertadores",
    "南美解放者杯": "copa_libertadores",
    "南美自由杯": "copa_libertadores",
    "南美洲冠军联赛": "copa_libertadores",
    "libertadores": "copa_libertadores",
    "copa libertadores": "copa_libertadores",

    "南美杯": "copa_sudamericana",
    "南美俱乐部杯": "copa_sudamericana",
    "sudamericana": "copa_sudamericana",
    "copa sudamericana": "copa_sudamericana",

    "亚冠": "afc_champions_league",
    "亚冠精英联赛": "afc_champions_league",
    "亚洲冠军联赛": "afc_champions_league",
    "afc champions league": "afc_champions_league",
    "afc cl": "afc_champions_league",
    "acl": "afc_champions_league",

    "亚冠二级": "afc_champions_league_two",
    "亚冠精英二级": "afc_champions_league_two",
    "afc champions league two": "afc_champions_league_two",
    "afc cl two": "afc_champions_league_two",
    "亚足联杯": "afc_champions_league_two",
    "afc cup": "afc_champions_league_two",

    "非洲冠军联赛": "caf_champions_league",
    "非冠": "caf_champions_league",
    "caf champions league": "caf_champions_league",
    "caf cl": "caf_champions_league",

    "中北美冠军杯": "concacaf_champions_cup",
    "concacaf champions cup": "concacaf_champions_cup",
    "concacaf cc": "concacaf_champions_cup",
    "中北美冠军联赛": "concacaf_champions_cup",

    "大洋洲冠军联赛": "ofc_champions_league",
    "大洋冠": "ofc_champions_league",
    "ofc champions league": "ofc_champions_league",

    "南美超级杯": "recopa_sudamericana",
    "南美再杯": "recopa_sudamericana",
    "recopa sudamericana": "recopa_sudamericana",
    "recopa": "recopa_sudamericana",

    "欧超杯": "uefa_super_cup",
    "欧洲超级杯": "uefa_super_cup",
    "uefa super cup": "uefa_super_cup",
    "european super cup": "uefa_super_cup",

    "世俱杯": "fifa_club_world_cup",
    "国际足联俱乐部世界杯": "fifa_club_world_cup",
    "fifa club world cup": "fifa_club_world_cup",
    "club world cup": "fifa_club_world_cup",

    "欧国联": "uefa_nations_league",
    "欧洲国家联赛": "uefa_nations_league",
    "uefa nations league": "uefa_nations_league",
    "unl": "uefa_nations_league",

    "中北美国家联赛": "concacaf_nations_league",
    "concacaf nations league": "concacaf_nations_league",

    # ----- Continental national-team competitions -----
    "欧洲杯": "euro",
    "欧锦赛": "euro",
    "欧洲杯足球赛": "euro",
    "欧洲国家杯": "euro",
    "uefa euro": "euro",
    "european championship": "euro",

    "美洲杯": "copa_america",
    "南美国家杯": "copa_america",
    "copa america": "copa_america",
    "copa américa": "copa_america",

    "亚洲杯": "asian_cup",
    "亚洲国家杯": "asian_cup",
    "afc asian cup": "asian_cup",

    "非洲杯": "africa_cup_of_nations",
    "非洲国家杯": "africa_cup_of_nations",
    "afcon": "africa_cup_of_nations",
    "africa cup of nations": "africa_cup_of_nations",
}


@dataclass(frozen=True)
class League:
    key: str
    name: str
    country: str | None = None
    football_data_code: str | None = None
    api_football_id: int | None = None
    clubelo: bool = False
    fbref_id: int | None = None
    tier: int = 3
    note: str | None = None
    # Continental competitions
    continent: str | None = None         # europe / south_america / asia / africa / north_america
    competition_type: str | None = None  # club / national / domestic_league
    knockout: bool = False               # true for cups / continental knockouts

    @classmethod
    def from_config(cls, key: str, payload: dict[str, Any]) -> "League":
        return cls(
            key=key,
            name=str(payload.get("name", key)),
            country=payload.get("country"),
            football_data_code=payload.get("football_data_code"),
            api_football_id=payload.get("api_football_id"),
            clubelo=bool(payload.get("clubelo", False)),
            fbref_id=payload.get("fbref_id"),
            tier=int(payload.get("tier", 3)),
            note=payload.get("note"),
            continent=payload.get("continent"),
            competition_type=payload.get("competition_type"),
            knockout=bool(payload.get("knockout", False)),
        )

    @property
    def is_continental(self) -> bool:
        return self.continent is not None and self.competition_type is not None

    @property
    def is_continental_club(self) -> bool:
        return self.is_continental and self.competition_type == "club"

    @property
    def is_continental_national(self) -> bool:
        return self.is_continental and self.competition_type == "national"


class LeagueRegistry:
    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
        self.path = Path(path)
        self._raw = self._load()
        self.leagues = {
            key: League.from_config(key, payload)
            for key, payload in self._raw.get("leagues", {}).items()
        }
        self.aliases = self._build_aliases()

    def normalize(self, value: str) -> str:
        needle = _alias_key(value)
        if needle in self.leagues:
            return needle
        if needle in self.aliases:
            return self.aliases[needle]
        raise KeyError(f"Unknown league: {value}")

    def get(self, value: str) -> League:
        return self.leagues[self.normalize(value)]

    def covered_by_football_data(self) -> list[League]:
        return [league for league in self.leagues.values() if league.football_data_code]

    def all(self) -> list[League]:
        return list(self.leagues.values())

    @property
    def worldcup(self) -> dict[str, Any]:
        return self._raw.get("worldcup_2026", {})

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"League registry not found: {self.path}")
        return yaml.safe_load(self.path.read_text()) or {}

    def _build_aliases(self) -> dict[str, str]:
        aliases = {_alias_key(alias): key for alias, key in EXTRA_ALIASES.items()}
        for key, league in self.leagues.items():
            aliases[_alias_key(key)] = key
            aliases[_alias_key(league.name)] = key
        return aliases


def normalize_league(value: str, *, registry: LeagueRegistry | None = None) -> str:
    registry = registry or LeagueRegistry()
    return registry.normalize(value)


def _alias_key(value: str) -> str:
    return " ".join(str(value).strip().casefold().replace("_", " ").split())
