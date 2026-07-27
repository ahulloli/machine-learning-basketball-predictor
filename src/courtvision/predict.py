"""Daily prediction pipeline.

Given an upcoming matchup (player vs opponent, home/away, date), reconstruct the
SAME features used in training — but computed only from games strictly BEFORE
the target date — and run the trained XGBoost model to project the player's
production. This mirrors real deployment: at tip-off we only know the past.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from xgboost import XGBRegressor

from .features import _team_clusters, load_stats
from .train import MODEL_DIR

logger = logging.getLogger(__name__)

# ---- caches so repeated predictions don't reload/recompute -------------------
_STATS: pd.DataFrame | None = None
_CLUSTERS: dict | None = None
_ABBR2ID: dict[str, int] | None = None
_ID2ABBR: dict[int, str] | None = None


def _stats() -> pd.DataFrame:
    global _STATS
    if _STATS is None:
        df = load_stats()
        df["game_date"] = pd.to_datetime(df["game_date"])
        _STATS = df
    return _STATS


def _clusters() -> dict:
    global _CLUSTERS
    if _CLUSTERS is None:
        _CLUSTERS = _team_clusters(_stats())
    return _CLUSTERS


def _abbr_maps() -> tuple[dict[str, int], dict[int, str]]:
    """Return (abbreviation -> team_id, team_id -> abbreviation) from teams."""
    global _ABBR2ID, _ID2ABBR
    if _ABBR2ID is None:
        from sqlalchemy import text

        from .db import engine

        with engine.connect() as conn:
            t = pd.read_sql(text("SELECT team_id, abbreviation FROM teams"), conn)
        _ID2ABBR = dict(zip(t["team_id"].astype(int), t["abbreviation"]))
        _ABBR2ID = {a: int(tid) for tid, a in _ID2ABBR.items()}
    return _ABBR2ID, _ID2ABBR


def season_for_date(d: date) -> str:
    """NBA season string for a calendar date (season spans Oct->Jun)."""
    y = d.year
    if d.month >= 10:
        return f"{y}-{str(y + 1)[2:]}"
    return f"{y - 1}-{str(y)[2:]}"


@dataclass
class Prediction:
    player_name: str
    player_id: int
    opponent_abbr: str
    home: bool
    on_date: str
    target: str
    projection: float
    baseline_last5: float
    games_of_history: int


def _opponent_cluster(opp_team_id: int, season: str) -> int:
    clusters = _clusters()
    if (season, opp_team_id) in clusters:
        return clusters[(season, opp_team_id)]
    # Fallback: latest season we have a cluster for this team.
    cand = [(s, c) for (s, t), c in clusters.items() if t == opp_team_id]
    if cand:
        return sorted(cand)[-1][1]
    return -1


def build_live_features(
    player_id: int, opponent_abbr: str, home: bool, on_date: date
) -> dict:
    """Compute the model's feature vector for an upcoming game."""
    df = _stats()
    abbr2id, _ = _abbr_maps()
    clusters = _clusters()
    on_ts = pd.Timestamp(on_date)
    season = season_for_date(on_date)

    prior = df[(df["player_id"] == player_id) & (df["game_date"] < on_ts)].sort_values(
        "game_date"
    )

    feat: dict = {"home": int(home)}
    roll5 = {"pts": "pts", "reb": "reb", "ast": "ast", "min": "min",
             "fg3m": "fg3m", "fga": "fga", "stl": "stl", "blk": "blk", "tov": "tov"}
    last5 = prior.tail(5)
    last10 = prior.tail(10)
    for col in roll5:
        feat[f"{col}_last5"] = float(last5[col].mean()) if len(last5) else None
    for col in ["pts", "reb", "ast", "min"]:
        feat[f"{col}_last10"] = float(last10[col].mean()) if len(last10) else None

    m5 = feat.get("min_last5")
    for stat in ("pts", "reb", "ast"):
        v = feat.get(f"{stat}_last5")
        feat[f"{stat}_per_min_last5"] = round(v / m5, 4) if m5 and v is not None else None

    feat["games_played_so_far"] = int(len(prior))
    feat["rest_days"] = (
        float((on_ts - prior["game_date"].max()).days) if len(prior) else None
    )

    # Opponent defensive rating (points allowed) from opponent's prior games.
    opp_id = abbr2id.get(opponent_abbr)
    feat["opp_def_rating"], feat["opp_def_rating_last10"] = _opp_defense(opp_id, on_ts)

    # Player vs this exact opponent (per target stat).
    vs = prior[prior["opponent_abbr"] == opponent_abbr]
    for stat in ("pts", "reb", "ast"):
        feat[f"{stat}_vs_opp"] = float(vs[stat].mean()) if len(vs) else None
    feat["games_vs_opp"] = int(len(vs))

    # Player vs teams that play like this opponent (same cluster), per stat.
    opp_cluster = _opponent_cluster(opp_id, season) if opp_id is not None else -1
    feat["opp_cluster"] = opp_cluster
    if len(prior):
        prior_opp_ids = prior["opponent_abbr"].map(abbr2id)
        prior_seasons = prior["season"]
        prior_clusters = [
            clusters.get((s, int(t)) if pd.notna(t) else None, -2)
            for s, t in zip(prior_seasons, prior_opp_ids)
        ]
        mask = (pd.Series(prior_clusters, index=prior.index) == opp_cluster).values
        vs_clu = prior[mask]
        for stat in ("pts", "reb", "ast"):
            feat[f"{stat}_vs_opp_cluster"] = (
                float(vs_clu[stat].mean()) if len(vs_clu) else None
            )
        feat["games_vs_opp_cluster"] = int(len(vs_clu))
    else:
        for stat in ("pts", "reb", "ast"):
            feat[f"{stat}_vs_opp_cluster"] = None
        feat["games_vs_opp_cluster"] = 0

    return feat


def _opp_defense(opp_team_id: int | None, on_ts: pd.Timestamp) -> tuple[float | None, float | None]:
    if opp_team_id is None:
        return None, None
    df = _stats()
    tg = df.groupby(["game_id", "team_id"], as_index=False).agg(
        team_pts=("pts", "sum"), game_date=("game_date", "first")
    )
    pair = tg.merge(tg, on="game_id", suffixes=("", "_opp"))
    pair = pair[pair["team_id"] != pair["team_id_opp"]]
    opp = pair[(pair["team_id"] == opp_team_id) & (pair["game_date"] < on_ts)].sort_values(
        "game_date"
    )
    if opp.empty:
        return None, None
    allowed = opp["team_pts_opp"]
    return round(float(allowed.mean()), 3), round(float(allowed.tail(10).mean()), 3)


def _load_model(target: str) -> tuple[XGBRegressor, list[str]]:
    tag = target.replace("target_", "")
    model_path = MODEL_DIR / f"xgb_{tag}.json"
    meta_path = MODEL_DIR / f"xgb_{tag}_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run scripts/train_model.py first."
        )
    model = XGBRegressor()
    model.load_model(model_path)
    feature_names = json.loads(meta_path.read_text())["feature_names"]
    return model, feature_names


def _resolve_player(name_or_id: str | int) -> tuple[int, str]:
    df = _stats()
    if isinstance(name_or_id, int) or str(name_or_id).isdigit():
        pid = int(name_or_id)
        row = df[df["player_id"] == pid]
    else:
        row = df[df["player_name"].str.lower() == str(name_or_id).lower()]
        if row.empty:
            row = df[df["player_name"].str.contains(str(name_or_id), case=False, na=False)]
    if row.empty:
        raise ValueError(f"Player not found: {name_or_id}")
    return int(row.iloc[0]["player_id"]), str(row.iloc[0]["player_name"])


def predict_player(
    name_or_id: str | int,
    opponent_abbr: str,
    home: bool,
    on_date: date | None = None,
    target: str = "target_pts",
) -> Prediction:
    on_date = on_date or date.today()
    player_id, player_name = _resolve_player(name_or_id)
    model, feature_names = _load_model(target)

    feat = build_live_features(player_id, opponent_abbr, home, on_date)
    row = pd.Series(feat)
    # One-hot the cluster to match training columns (cluster_<k>).
    row_dict = {k: v for k, v in feat.items() if k != "opp_cluster"}
    row_dict[f"cluster_{feat['opp_cluster']}"] = 1
    X = pd.DataFrame([row_dict]).reindex(columns=feature_names, fill_value=0)
    # Preserve NaNs (XGBoost treats them as missing) for numeric features.
    for c in feature_names:
        if c in feat and feat[c] is None:
            X[c] = pd.NA
    X = X.apply(pd.to_numeric, errors="coerce")

    projection = float(model.predict(X)[0])
    # Target-aware naive baseline (last-5 average of the SAME stat).
    base_col = f"{target.replace('target_', '')}_last5"
    base_val = feat.get(base_col)
    return Prediction(
        player_name=player_name,
        player_id=player_id,
        opponent_abbr=opponent_abbr,
        home=home,
        on_date=str(on_date),
        target=target,
        projection=round(projection, 2),
        baseline_last5=round(base_val, 2) if base_val is not None else float("nan"),
        games_of_history=feat["games_played_so_far"],
    )


def fetch_schedule(on_date: date) -> list[dict]:
    """Return [{home_abbr, away_abbr, home_id, away_id}] for a date via nba_api."""
    from nba_api.stats.endpoints import scoreboardv2

    _, id2abbr = _abbr_maps()
    sb = scoreboardv2.ScoreboardV2(game_date=on_date.strftime("%Y-%m-%d"), timeout=60)
    header = sb.game_header.get_data_frame()
    games = []
    for r in header.itertuples(index=False):
        row = r._asdict()
        hid, vid = int(row["HOME_TEAM_ID"]), int(row["VISITOR_TEAM_ID"])
        games.append({
            "home_id": hid, "away_id": vid,
            "home_abbr": id2abbr.get(hid, str(hid)),
            "away_abbr": id2abbr.get(vid, str(vid)),
        })
    return games


def roster_proxy(team_id: int, on_date: date, top_n: int = 8) -> list[int]:
    """Players whose most recent game before on_date was with this team,
    ranked by recent minutes (a stand-in for tonight's rotation)."""
    df = _stats()
    on_ts = pd.Timestamp(on_date)
    prior = df[df["game_date"] < on_ts]
    if prior.empty:
        return []
    latest = prior.sort_values("game_date").groupby("player_id").tail(1)
    on_team = latest[latest["team_id"] == team_id]
    # rank by minutes in their most recent game
    on_team = on_team.sort_values("min", ascending=False)
    return on_team["player_id"].head(top_n).astype(int).tolist()


def predict_slate(on_date: date, target: str = "target_pts", top_n: int = 6) -> list[Prediction]:
    """Predict for every player on both teams of every game on the slate."""
    games = fetch_schedule(on_date)
    preds: list[Prediction] = []
    for g in games:
        for team_id, opp_abbr, home in [
            (g["home_id"], g["away_abbr"], True),
            (g["away_id"], g["home_abbr"], False),
        ]:
            for pid in roster_proxy(team_id, on_date, top_n):
                try:
                    preds.append(predict_player(pid, opp_abbr, home, on_date, target))
                except Exception as e:
                    logger.warning(
                        "Prediction failed for player %s vs %s on %s (%s): %s",
                        pid, opp_abbr, on_date, target, e,
                    )
    return preds
