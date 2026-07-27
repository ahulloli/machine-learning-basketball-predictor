"""Build leakage-safe rolling 5-game features from player_game_stats.

For each player, ordered by date, every ``*_last5`` feature is the mean of the
PREVIOUS (up to) 5 games — computed with ``shift(1)`` so the current game's
outcome never leaks into its own features. The current game's actuals become
the supervised-learning targets.
"""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from .db import SessionLocal, engine
from .models import Base, PlayerGameFeature

ROLL_COLS = ["pts", "reb", "ast", "min", "fg3m", "fga", "stl", "blk", "tov"]
WINDOW = 5
LONG_WINDOW = 10
LONG_COLS = ["pts", "reb", "ast", "min"]
N_CLUSTERS = 6
REST_CAP = 7           # cap rest so an offseason gap != in-season rest
LONG_BREAK_DAYS = 14   # gaps beyond this flag a season opener / long absence


def load_stats(season: str | None = None) -> pd.DataFrame:
    query = """
        SELECT s.game_id, s.player_id, p.name AS player_name, s.season,
               s.game_date, s.team_id, s.opponent_abbr, s.home,
               s.pts, s.reb, s.oreb, s.ast, s.min, s.fg3m, s.fg3a, s.fga, s.fta,
               s.stl, s.blk, s.tov
        FROM player_game_stats s
        JOIN players p ON p.player_id = s.player_id
    """
    params: dict = {}
    if season:
        query += " WHERE s.season = :season"
        params["season"] = season
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _opponent_defense(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe opponent points-allowed (a simple defensive proxy).

    Aggregate box scores to team-game totals, derive points allowed (the
    opposing team's total), then for each team compute the mean points allowed
    over PRIOR games only (``shift(1)``). Returns, per (game_id, team_id), the
    points-allowed the opponent team carries INTO that game.

    NOTE: this is raw points allowed per game, NOT possession-adjusted
    defensive rating, hence the honest ``opp_pts_allowed`` naming.
    """
    tg = df.groupby(["game_id", "team_id"], as_index=False).agg(
        team_pts=("pts", "sum"), game_date=("game_date", "first")
    )
    # Pair the two teams within each game to get points allowed.
    merged = tg.merge(tg, on="game_id", suffixes=("", "_opp"))
    merged = merged[merged["team_id"] != merged["team_id_opp"]].copy()
    merged["pts_allowed"] = merged["team_pts_opp"]

    merged = merged.sort_values(["team_id", "game_date", "game_id"])
    g = merged.groupby("team_id", sort=False)["pts_allowed"]
    merged["opp_pts_allowed"] = g.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    ).round(3)
    merged["opp_pts_allowed_last10"] = g.transform(
        lambda s: s.shift(1).rolling(LONG_WINDOW, min_periods=1).mean()
    ).round(3)
    return merged[[
        "game_id", "team_id", "team_id_opp",
        "opp_pts_allowed", "opp_pts_allowed_last10",
    ]]


def _season_start_year(season: str) -> int:
    """'2023-24' -> 2023."""
    return int(season.split("-")[0])


def _team_clusters(df: pd.DataFrame) -> dict[tuple[str, int], int]:
    """Cluster teams by play style with NO future information and STABLE ids.

    Profile per (season, team): points for/allowed, 3-point rate, assists,
    rebounds, turnovers and an estimated pace (possessions/game).

    Two properties matter:

    1. **No leakage.** A team's cluster for season S is determined by its
       *previous* season's (S-1) profile only — never by any S games. So a
       prediction for an October S game does not depend on that team's
       January-April S statistics.
    2. **Stable cluster meanings.** A single StandardScaler + KMeans is fit
       once on the earliest (context) season and reused for every season.
       Thus ``cluster_0`` denotes the same play-style archetype every year,
       which is required because the model one-hot encodes the cluster id.

    Returns {(season, team_id): cluster_id}. The earliest season has no prior
    season and is context-only (never modeled), so it is labelled from its own
    profile purely to seed the vs-cluster history of the first modeled season.
    """
    tg = df.groupby(["season", "game_id", "team_id"], as_index=False).agg(
        pts=("pts", "sum"), fga=("fga", "sum"), fg3a=("fg3a", "sum"),
        fta=("fta", "sum"), ast=("ast", "sum"), reb=("reb", "sum"),
        oreb=("oreb", "sum"), tov=("tov", "sum"),
    )
    # Points allowed = opponent's total within the same game.
    pair = tg.merge(tg[["game_id", "team_id", "pts"]], on="game_id", suffixes=("", "_opp"))
    pair = pair[pair["team_id"] != pair["team_id_opp"]]
    pair["pts_allowed"] = pair["pts_opp"]
    # Standard possession estimate (offensive rebounds extend a possession).
    pair["poss"] = pair["fga"] - pair["oreb"] + pair["tov"] + 0.44 * pair["fta"]
    pair["fg3_rate"] = (pair["fg3a"] / pair["fga"]).fillna(0)

    profile = pair.groupby(["season", "team_id"], as_index=False).agg(
        pts=("pts", "mean"), pts_allowed=("pts_allowed", "mean"),
        fg3_rate=("fg3_rate", "mean"), ast=("ast", "mean"),
        reb=("reb", "mean"), tov=("tov", "mean"), poss=("poss", "mean"),
    )

    feats = ["pts", "pts_allowed", "fg3_rate", "ast", "reb", "tov", "poss"]
    seasons = sorted(profile["season"].unique(), key=_season_start_year)
    start_year_to_season = {_season_start_year(s): s for s in seasons}

    # Fit ONE reference model on the earliest season's profiles.
    ref_grp = profile[profile["season"] == seasons[0]]
    scaler = StandardScaler().fit(ref_grp[feats].values)
    k = min(N_CLUSTERS, len(ref_grp))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(scaler.transform(ref_grp[feats].values))

    def _labels_for(src_season: str) -> dict[int, int]:
        grp = profile[profile["season"] == src_season]
        labs = km.predict(scaler.transform(grp[feats].values))
        return {int(t): int(l) for t, l in zip(grp["team_id"], labs)}

    mapping: dict[tuple[str, int], int] = {}
    for season in seasons:
        prev_season = start_year_to_season.get(_season_start_year(season) - 1)
        # Classify from the PREVIOUS season; earliest season uses itself.
        src_season = prev_season if prev_season is not None else season
        for team_id, lab in _labels_for(src_season).items():
            mapping[(season, team_id)] = lab
    return mapping


def build_features(season: str | None = None) -> pd.DataFrame:
    df = load_stats(season)
    if df.empty:
        raise RuntimeError("No rows in player_game_stats. Run ingestion first.")

    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    grp = df.groupby("player_id", sort=False)

    out = pd.DataFrame({
        "game_id": df["game_id"],
        "player_id": df["player_id"],
        "player_name": df["player_name"],
        "game_date": df["game_date"].dt.date,
        "season": df["season"],
        "opponent_abbr": df["opponent_abbr"],
        "home": df["home"].astype(bool),
    })

    # Rolling last-5 mean of previous games only (exclude current via shift).
    for col in ROLL_COLS:
        out[f"{col}_last5"] = (
            grp[col]
            .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())
            .round(3)
        )

    # Longer-window (last-10) minutes/production stability.
    for col in LONG_COLS:
        out[f"{col}_last10"] = (
            grp[col]
            .transform(lambda s: s.shift(1).rolling(LONG_WINDOW, min_periods=1).mean())
            .round(3)
        )

    # Per-minute efficiency over the last 5 games, for each target stat.
    min5 = out["min_last5"].where(out["min_last5"] > 0)
    out["pts_per_min_last5"] = (out["pts_last5"] / min5).round(4)
    out["reb_per_min_last5"] = (out["reb_last5"] / min5).round(4)
    out["ast_per_min_last5"] = (out["ast_last5"] / min5).round(4)

    # Days since the player's previous game. Raw gaps can be ~190 days across an
    # offseason, which is a different concept from a 1-3 day in-season rest, so
    # cap at a week and add an explicit long-break flag (season opener / return
    # from injury). The SAME transform is mirrored in predict.py.
    raw_gap = grp["game_date"].transform(lambda s: s.diff().dt.days)
    out["rest_days"] = raw_gap.clip(lower=0, upper=REST_CAP)
    out["long_break"] = (raw_gap > LONG_BREAK_DAYS).astype("Int64")
    out["games_played_so_far"] = grp.cumcount()

    # Opponent defense proxy: join the OPPONENT team's prior points-allowed.
    defense = _opponent_defense(df)
    # 1) attach each player-game's opponent team id
    key = df[["game_id", "team_id"]].reset_index(drop=True)
    key = key.merge(
        defense[["game_id", "team_id", "team_id_opp"]],
        on=["game_id", "team_id"], how="left",
    )
    # 2) look up that opponent team's points-allowed going into the game
    opp_def = defense[[
        "game_id", "team_id", "opp_pts_allowed", "opp_pts_allowed_last10",
    ]].rename(columns={"team_id": "team_id_opp"})
    key = key.merge(opp_def, on=["game_id", "team_id_opp"], how="left")
    out["opp_pts_allowed"] = key["opp_pts_allowed"].values
    out["opp_pts_allowed_last10"] = key["opp_pts_allowed_last10"].values

    # Attach opponent team id + opponent play-style cluster onto df rows.
    df = df.copy()
    df["team_id_opp"] = key["team_id_opp"].values
    clusters = _team_clusters(df)
    df["opp_cluster"] = [
        clusters.get((s, int(t)) if pd.notna(t) else None, -1)
        for s, t in zip(df["season"], df["team_id_opp"])
    ]
    out["opp_cluster"] = df["opp_cluster"].values

    # Player-vs-opponent history: expanding mean of each target stat vs this
    # exact team, prior meetings only (shift(1)). Kept in df's player/date order.
    opp_grp = df.groupby(["player_id", "opponent_abbr"], sort=False)
    for stat in ("pts", "reb", "ast"):
        out[f"{stat}_vs_opp"] = opp_grp[stat].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        ).round(3)
    out["games_vs_opp"] = opp_grp.cumcount()

    # Player-vs-similar-team history: expanding mean of each target stat vs
    # opponents that play like this opponent (same cluster), prior games only.
    # Generalizes the head-to-head signal and reduces exact-matchup sparsity.
    clu_grp = df.groupby(["player_id", "opp_cluster"], sort=False)
    for stat in ("pts", "reb", "ast"):
        out[f"{stat}_vs_opp_cluster"] = clu_grp[stat].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        ).round(3)
    out["games_vs_opp_cluster"] = clu_grp.cumcount()

    out["target_pts"] = df["pts"]
    out["target_reb"] = df["reb"]
    out["target_ast"] = df["ast"]
    return out


def _to_none(v):
    return None if pd.isna(v) else v


def persist_features(df: pd.DataFrame, chunk: int = 1000) -> int:
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    table = PlayerGameFeature.__table__
    # Recreate so newly added feature columns are applied on re-runs.
    table.drop(engine, checkfirst=True)
    table.create(engine, checkfirst=True)
    records = df.to_dict(orient="records")
    records = [{k: _to_none(v) for k, v in r.items()} for r in records]

    with SessionLocal() as session:
        for i in range(0, len(records), chunk):
            batch = records[i : i + chunk]
            stmt = pg_insert(table).values(batch).on_conflict_do_nothing(
                index_elements=["game_id", "player_id"]
            )
            session.execute(stmt)
        session.commit()
    return len(records)
