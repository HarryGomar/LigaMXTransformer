from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import os
# Check if running in an interactive environment (like a notebook) vs. a script
try:
    __file__
except NameError:
    print("Running in interactive mode. Setting CWD.")
    os.chdir(os.path.abspath(os.path.curdir)) 
else:
    # If __file__ is defined, we're running as a script.
    print("Running as script. Setting CWD to script directory.")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ==============================
# Configuration
# ==============================
INPUT_PARQUETS = [
    "./Raw/events_comp73_season108.parquet",
    "./Raw/events_comp73_season235.parquet",
    "./Raw/events_comp73_season281.parquet",
    "./Raw/events_comp73_season317.parquet",
]

OUT_DIR = Path("./artifacts/data")
VOCAB_DIR = OUT_DIR / "vocabs"
SEQ_DIR = OUT_DIR / "sequences"
STATS_DIR = OUT_DIR / "stats"
MAPS_DIR = OUT_DIR / "maps"

# Sequence caps
MAX_T_POSSESSION = 40
MAX_T_PLAYER = 64
PLAYER_WINDOW_STRIDE = 32  # when slicing long player sequences

# Grid / pitch
GRID_W, GRID_H = 12, 8
PITCH_X, PITCH_Y = 120.0, 80.0
GOAL_CENTER = (PITCH_X, PITCH_Y / 2.0)
FINAL_THIRD_X = PITCH_X * (2.0 / 3.0)  # 80m
PROGRESS_MIN_DX = 10.0  # progressive threshold in meters

# Special IDs
PAD_ID = 0
UNK_ID = 1

# Random
RANDOM_STATE = 73
VERBOSE = True

# Robustness thresholds
MIN_EVENTS_PER_PS = 150  # min total events for a player-season to be kept
MIN_MINUTES_PER_PS = 400.0  # minutes threshold (best-effort from subs)
DEDUP_KEYS = ["match_id", "index", "timestamp"]  # fallbacks if no unique id

# Attack-sign cache (normalized orientation)
_ATTACK_SIGN: Dict[Tuple[Any, Any, int], float] = {}

# ==============================
# Utilities
# ==============================
def log(msg: str):
    if VERBOSE:
        print(msg)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parquet_read(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as e1:
        log(f"[warn] pyarrow failed for {path} ({e1}). Trying fastparquet…")
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as e2:
            raise RuntimeError(
                f"Failed to read {path} with both pyarrow and fastparquet.\n"
                f"pyarrow -> {e1}\nfastparquet -> {e2}\n"
                f"Install one engine: pip install pyarrow OR pip install fastparquet"
            )

def parse_season_id_from_path(path: str | Path) -> Optional[int]:
    m = re.search(r"season(\d+)", str(path))
    return int(m.group(1)) if m else None

def safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def as_py_obj(x: Any) -> Any:
    try:
        if hasattr(x, "as_py"):
            return x.as_py()
    except Exception:
        pass
    return x

def jsonish_to_list(s: str) -> Optional[List[float]]:
    try:
        s2 = s.strip()
        if s2.startswith("[") and s2.endswith("]"):
            arr = json.loads(s2)
            if isinstance(arr, list) and len(arr) >= 2:
                return [float(arr[0]), float(arr[1])]
        parts = s2.split(",")
        if len(parts) >= 2:
            return [float(parts[0]), float(parts[1])]
    except Exception:
        return None
    return None

def parse_location(loc: Any) -> Optional[Tuple[float, float]]:
    if loc is None:
        return None
    loc = as_py_obj(loc)
    if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2:
        try:
            x, y = float(loc[0]), float(loc[1])
            if math.isnan(x) or math.isnan(y):
                return None
            return (x, y)
        except Exception:
            return None
    if isinstance(loc, dict):
        keys = {k.lower(): k for k in loc.keys()}
        if "x" in keys and "y" in keys:
            try:
                x = float(loc[keys["x"]]); y = float(loc[keys["y"]])
                if math.isnan(x) or math.isnan(y):
                    return None
                return (x, y)
            except Exception:
                return None
        if "coordinates" in keys and isinstance(loc[keys["coordinates"]], dict):
            d2 = loc[keys["coordinates"]]
            keys2 = {k.lower(): k for k in d2.keys()}
            if "x" in keys2 and "y" in keys2:
                try:
                    x = float(d2[keys2["x"]]); y = float(d2[keys2["y"]])
                    if math.isnan(x) or math.isnan(y):
                        return None
                    return (x, y)
                except Exception:
                    return None
    if isinstance(loc, str):
        arr = jsonish_to_list(loc)
        if arr and len(arr) >= 2:
            try:
                x, y = float(arr[0]), float(arr[1])
                if math.isnan(x) or math.isnan(y):
                    return None
                return (x, y)
            except Exception:
                return None
    return None

def xy_to_cell(x: float, y: float) -> Optional[int]:
    if x is None or y is None:
        return None
    # Accept [0..1] or [0..120/80]
    if 0 <= x <= 1.5 and 0 <= y <= 1.5:
        nx, ny = min(max(x, 0.0), 0.999999), min(max(y, 0.0), 0.999999)
    else:
        if not (0 <= x <= PITCH_X and 0 <= y <= PITCH_Y):
            return None
        nx, ny = x / PITCH_X, y / PITCH_Y
        nx, ny = min(max(nx, 0.0), 0.999999), min(max(ny, 0.0), 0.999999)
    xb, yb = int(nx * GRID_W), int(ny * GRID_H)
    return yb * GRID_W + xb

def angle_sin_cos(x0: float, y0: float, x1: float, y1: float) -> Tuple[float, float]:
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return (0.0, 1.0)
    theta = math.atan2(dy, dx)
    return (math.sin(theta), math.cos(theta))

def compute_distance(x0: float, y0: float, x1: float, y1: float) -> float:
    return float(math.hypot(x1 - x0, y1 - y0))

def minute_bucket(minute: Any) -> str:
    try:
        m = int(minute)
    except Exception:
        return "m_unk"
    if m < 0: return "m_unk"
    bins = [(0,15),(16,30),(31,45),(46,60),(61,75),(76,90),(91,105),(106,120)]
    for a,b in bins:
        if a <= m <= b:
            return f"{a:02d}-{b:02d}"
    return "m_>120"

def home_flag(team: Any, home: Any, away: Any) -> str:
    if pd.isna(team) or pd.isna(home) or pd.isna(away):
        return "away"  # default
    ts, hs, as_ = str(team), str(home), str(away)
    if ts == hs: return "home"
    if ts == as_: return "away"
    return "away"

def coarse_pos(position: Any) -> str:
    if position is None or (isinstance(position, float) and math.isnan(position)):
        return "POS_UNK"
    p = str(position).lower()
    if any(tok in p for tok in ["goalkeeper", "keeper", "gk"]): return "GK"
    if any(tok in p for tok in ["back", "wing back", "full back", "centre back", "center back"]): return "DEF"
    if any(tok in p for tok in ["midfield"]): return "MID"
    if any(tok in p for tok in ["forward", "wing", "striker", "centre forward", "center forward"]): return "FWD"
    return "POS_UNK"

def distance_bin(d: Optional[float]) -> str:
    if d is None: return "dist_unk"
    if d < 10: return "dist_short"
    if d < 25: return "dist_med"
    return "dist_long"

def safe_str(x: Any) -> Optional[str]:
    if x is None: return None
    if isinstance(x, float) and math.isnan(x): return None
    s = str(x).strip()
    return s if s else None

def minute_to_weight(minute_val: Any) -> float:
    """Logistic minute weight centered around 60' (grows late)."""
    try:
        m = float(minute_val)
    except Exception:
        return 0.5
    return 1.0 / (1.0 + math.exp(-(m - 60.0) / 10.0))

# ==============================
# Orientation normalization
# ==============================
def compute_attack_sign(df: pd.DataFrame) -> Dict[tuple, float]:
    tmp = df.copy()

    def x_of(loc):
        p = parse_location(loc)
        return p[0] if p else np.nan

    tmp["sx"] = tmp["location"].apply(x_of)
    tmp["ex"] = np.where(
        tmp.get("pass_end_location", pd.Series([np.nan]*len(tmp))).notna(),
        tmp["pass_end_location"].apply(x_of),
        np.where(
            tmp.get("carry_end_location", pd.Series([np.nan]*len(tmp))).notna(),
            tmp["carry_end_location"].apply(x_of),
            np.nan
        )
    )
    tmp["dx"] = tmp["ex"] - tmp["sx"]
    key = ["match_id", "team_id", "period"]
    sgn = (
        tmp.dropna(subset=["dx"])
           .groupby(key, dropna=False)["dx"]
           .median()
           .apply(lambda v: 1.0 if v >= 0 else -1.0)
    )
    return {tuple(k): float(v) for k, v in sgn.items()}

def flip_if_needed(loc: Optional[Tuple[float, float]], sgn: Optional[float]):
    if (loc is None) or (sgn is None) or (sgn >= 0):
        return loc
    x, y = loc
    return (PITCH_X - x, y)

# ==============================
# Action / Family inference
# ==============================
PRIMARY_TYPE_MAP = {
    "Pass": "pass",
    "Shot": "shot",
    "Dribble": "dribble",
    "Carry": "carry",
    "Ball Recovery": "ball_recovery",
    "Ball Receipt*": "ball_receipt",
    "Duel": "duel",
    "Interception": "interception",
    "Foul Won": "foul_won",
    "Foul Committed": "foul_committed",
    "Clearance": "clearance",
    "Block": "block",
    "Goal Keeper": "goalkeeper",
    "Miscontrol": "miscontrol",
    "Injury Stoppage": "injury_stoppage",
    "Substitution": "substitution",
    "Offside": "offside",
    "Half Start": "half_start",
    "Half End": "half_end",
    "50/50": "50_50",
    "50_50": "50_50",
}

def infer_action_family(row: pd.Series) -> Optional[str]:
    t = safe_str(row.get("type"))
    if t and t in PRIMARY_TYPE_MAP:
        return PRIMARY_TYPE_MAP[t]

    # Core heuristics
    if pd.notna(row.get("shot_statsbomb_xg")) or pd.notna(row.get("shot_outcome")):
        return "shot"
    pass_ind = ["pass_length","pass_outcome","pass_end_location","pass_type","pass_goal_assist","pass_shot_assist","pass_cross","pass_through_ball"]
    if any(pd.notna(row.get(c)) for c in pass_ind):
        return "pass"
    if pd.notna(row.get("dribble_outcome")) or pd.notna(row.get("dribble_no_touch")) or pd.notna(row.get("dribble_overrun")):
        return "dribble"
    if pd.notna(row.get("carry_end_location")):
        return "carry"
    if pd.notna(row.get("interception_outcome")):
        return "interception"
    if any(pd.notna(row.get(c)) for c in ["block_deflection","block_offensive","block_save_block"]):
        return "block"
    if any(pd.notna(row.get(c)) for c in ["clearance_head","clearance_left_foot","clearance_right_foot","clearance_aerial_won","clearance_other"]):
        return "clearance"
    if pd.notna(row.get("ball_recovery_recovery_failure")) or pd.notna(row.get("ball_recovery_offensive")):
        return "ball_recovery"
    if pd.notna(row.get("duel_outcome")) or pd.notna(row.get("duel_type")):
        return "duel"
    if pd.notna(row.get("foul_won_advantage")) or pd.notna(row.get("foul_won_defensive")) or pd.notna(row.get("foul_won_penalty")):
        return "foul_won"
    if pd.notna(row.get("foul_committed_type")) or pd.notna(row.get("foul_committed_card")) or pd.notna(row.get("foul_committed_penalty")):
        return "foul_committed"

    # STRICT GK detection — only explicit GK events
    if t == "Goal Keeper":
        return "goalkeeper"
    if pd.notna(row.get("goalkeeper_type")) or pd.notna(row.get("goalkeeper_outcome")):
        pos_raw = safe_str(row.get("position"))
        pos = pos_raw if pos_raw else coarse_pos(row.get("position"))
        if pos and "gk" in pos.lower():
            return "goalkeeper"

    if pd.notna(row.get("substitution_replacement_id")) or pd.notna(row.get("substitution_outcome")):
        return "substitution"
    if pd.notna(row.get("injury_stoppage_in_chain")):
        return "injury_stoppage"
    return None

# ==============================
# Factor specs
# ==============================
# Categorical factors (single ID per event)
CAT_FACTORS = [
    "action_family", "pass_height", "pass_technique", "pass_type",
    "pass_outcome", "shot_type", "shot_outcome", "shot_technique", "shot_body_part",
    "dribble_outcome", "duel_type", "duel_outcome", "interception_outcome",
    "clearance_body_part", "goalkeeper_type", "goalkeeper_outcome", "play_pattern",
    "start_cell", "end_cell", "distance_bin",
    "minute_bucket", "period", "home_away", "position_hint",
    "pass_cluster_label",
    "game_state", "corridor", "third",
]

# Numeric features (floats per event)
NUM_FEATURES = [
    "log_pass_length", "sin_angle", "cos_angle", "duration",
    "shot_statsbomb_xg", "shot_exec_xg", "shot_exec_uplift",
    "pass_expected_success", "pass_cluster_prob",
    "obv_total_net_event",
    "score_diff", "leverage", "red_card_diff", "opp_def_strength",
    "dt_prev_s", "dt_prev_by_player_s",
    "v_m_s", "goal_dist", "goal_angle_cos",
    "pos_idx_norm", "pos_len", "pos_time_from_start_s", "pos_obv_cum", "pos_obv_share",
]

# Qualifiers (multi-hot per event)
QUALIFIERS = [
    "under_pressure", "counterpress",
    "pass_cross", "pass_switch", "pass_through_ball", "pass_cut_back",
    "pass_inswinging", "pass_outswinging", "pass_no_touch", "pass_deflected", "pass_aerial_won",
    "shot_first_time", "shot_follows_dribble", "shot_one_on_one", "shot_open_goal", "shot_aerial_won",
    "miscontrol_aerial_won", "clearance_aerial_won",
    "block_deflection", "block_offensive", "block_save_block",
    "goalkeeper_punched_out", "goalkeeper_shot_saved_off_target", "goalkeeper_shot_saved_to_post",
    "goalkeeper_success_in_play", "goalkeeper_success_out",
    "shot_saved_off_target",
    "ball_recovery_offensive",
    # NEW qualifiers
    "pass_shot_assist", "pass_goal_assist",
    "progressive", "final_third_entry",
]

# ==============================
# Vocab builders
# ==============================
@dataclass
class Vocab:
    token2id: Dict[str, int]
    id2token: Dict[int, str]
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"

    @classmethod
    def build(cls, items: Iterable[Optional[str]]) -> "Vocab":
        uniq: List[str] = []
        seen = set()
        for x in items:
            if x is None: continue
            s = str(x)
            if not s: continue
            if s not in seen:
                seen.add(s); uniq.append(s)
        uniq.sort()
        token2id = {cls.pad_token: PAD_ID, cls.unk_token: UNK_ID}
        for i, s in enumerate(uniq, start=2):
            token2id[s] = i
        id2token = {i: s for s, i in token2id.items()}
        return cls(token2id, id2token)

    def encode(self, x: Optional[str]) -> int:
        if x is None: return UNK_ID
        return self.token2id.get(str(x), UNK_ID)

    def to_json(self) -> Dict[str, Any]:
        return {"token2id": self.token2id, "id2token": {str(k): v for k, v in self.id2token.items()},
                "pad_token": self.pad_token, "unk_token": self.unk_token}

# ==============================
# Precompute stage (opponent, time, score, cards, leverage, possession flow, opp strength)
# ==============================
def _match_time_seconds(row: pd.Series) -> float:
    m = safe_float(row.get("minute")) or 0.0
    s = safe_float(row.get("second")) or 0.0
    return float(m * 60.0 + s)

def _detect_red_card_str(val: Any) -> bool:
    s = str(val).lower() if val is not None and not (isinstance(val, float) and math.isnan(val)) else ""
    return ("red" in s) or ("second yellow" in s) or ("straight red" in s)

def precompute_opponent(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["opp_team_id"] = np.nan
    for mid, g in df.groupby("match_id", sort=False):
        teams = [int(t) for t in pd.Series(g["team_id"]).dropna().unique().tolist()]
        if len(teams) != 2:
            continue
        t1, t2 = teams[0], teams[1]
        mask1 = (df["match_id"]==mid) & (df["team_id"]==t1)
        mask2 = (df["match_id"]==mid) & (df["team_id"]==t2)
        df.loc[mask1, "opp_team_id"] = t2
        df.loc[mask2, "opp_team_id"] = t1
    return df

def precompute_score_cards_and_tempo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["match_time_s"] = df.apply(_match_time_seconds, axis=1)
    df["dt_prev_s"] = np.nan
    df["dt_prev_by_player_s"] = np.nan
    df["score_for"] = 0
    df["score_against"] = 0
    df["score_diff"] = 0
    df["game_state"] = "tied"
    df["red_card_diff"] = 0  # opp_reds - team_reds (positive => numerical advantage)
    df["leverage"] = 0.0

    # iterate per match in time order
    order_cols = ["period","minute","second","timestamp","index"]
    for c in order_cols:
        if c not in df.columns:
            df[c] = np.nan
    for mid, gidx in df.sort_values(by=["match_id"]+order_cols, kind="mergesort").groupby("match_id").groups.items():
        idxs = df.loc[gidx].index.tolist()
        team_goals: Dict[int,int] = {}
        team_reds: Dict[int,int] = {}
        last_time = None
        last_time_by_player: Dict[int,float] = {}

        for i in idxs:
            row = df.loc[i]
            team = row.get("team_id")
            opp = row.get("opp_team_id")
            pid = row.get("player_id")
            t = float(row["match_time_s"])

            # dt_prev
            if last_time is None:
                df.at[i, "dt_prev_s"] = np.nan
            else:
                df.at[i, "dt_prev_s"] = max(0.0, t - last_time)
            last_time = t

            # dt_prev_by_player
            if pd.notna(pid):
                pid_i = int(pid)
                if pid_i in last_time_by_player:
                    df.at[i, "dt_prev_by_player_s"] = max(0.0, t - last_time_by_player[pid_i])
                else:
                    df.at[i, "dt_prev_by_player_s"] = np.nan
                last_time_by_player[pid_i] = t

            # scores BEFORE this event
            tg = int(team_goals.get(int(team), 0)) if pd.notna(team) else 0
            og = int(team_goals.get(int(opp), 0)) if pd.notna(opp) else 0
            df.at[i, "score_for"] = tg
            df.at[i, "score_against"] = og
            diff = tg - og
            df.at[i, "score_diff"] = diff
            df.at[i, "game_state"] = "leading" if diff > 0 else ("trailing" if diff < 0 else "tied")

            # red card differential BEFORE this event (opp_red - team_red)
            tr = int(team_reds.get(int(team), 0)) if pd.notna(team) else 0
            orr = int(team_reds.get(int(opp), 0)) if pd.notna(opp) else 0
            df.at[i, "red_card_diff"] = orr - tr

            # leverage
            df.at[i, "leverage"] = math.exp(-abs(diff)) * minute_to_weight(row.get("minute"))

            # update counters AFTER event
            # goals
            is_goal = str(row.get("shot_outcome")).lower().find("goal") >= 0 if row.get("shot_outcome") is not None else False
            if is_goal and pd.notna(team):
                team_goals[int(team)] = team_goals.get(int(team), 0) + 1

            # reds
            red_flag = _detect_red_card_str(row.get("bad_behaviour_card")) or \
                       _detect_red_card_str(row.get("foul_committed_card")) or \
                       bool(row.get("player_off_permanent") == True)
            if red_flag and pd.notna(team):
                team_reds[int(team)] = team_reds.get(int(team), 0) + 1

    return df

def _possession_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pos_idx_norm"] = 0.0
    df["pos_len"] = 0.0
    df["pos_time_from_start_s"] = 0.0
    df["pos_obv_cum"] = 0.0
    df["pos_obv_share"] = 0.0

    # ensure columns exist
    obv = df.get("obv_total_net", pd.Series([0.0]*len(df)))
    dur = df.get("duration", pd.Series([0.0]*len(df))).fillna(0.0).astype(float)

    # group
    order_cols = ["period","minute","second","timestamp","index"]
    for c in order_cols:
        if c not in df.columns:
            df[c] = np.nan

    for (mid, poss), g in df.sort_values(by=["match_id","possession"]+order_cols, kind="mergesort").groupby(["match_id","possession"], sort=False):
        idx = g.index
        n = len(idx)
        if n == 0:
            continue
        df.loc[idx, "pos_len"] = float(n)

        # normalized index
        if n > 1:
            norm = np.arange(n, dtype=float) / float(n - 1)
        else:
            norm = np.zeros(n, dtype=float)
        df.loc[idx, "pos_idx_norm"] = norm

        # time from start: cumulative duration (shifted)
        gdur = dur.loc[idx].values.astype(float)
        cum_t = np.cumsum(np.concatenate([[0.0], gdur[:-1]]))
        df.loc[idx, "pos_time_from_start_s"] = cum_t

        # OBV cumulative and share
        gobv = obv.loc[idx].fillna(0.0).astype(float).values
        df.loc[idx, "pos_obv_cum"] = np.cumsum(gobv)
        denom = float(np.sum(np.abs(gobv))) + 1e-6
        df.loc[idx, "pos_obv_share"] = np.abs(gobv) / denom

    return df

def precompute_opponent_def_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Opponent defensive strength = 1 - percentile(xGA_per_match) for team-season."""
    df = df.copy()

    # Matches per team-season
    mts = (
        df[["match_id","team_id","season_id"]].dropna().drop_duplicates()
          .groupby(["team_id","season_id"])["match_id"].nunique()
          .rename("matches")
          .reset_index()
    )

    # xGA conceded: for each shot by team T, the defending team is opponent
    shots = df[df["shot_statsbomb_xg"].notna()][
        ["match_id","season_id","team_id","opp_team_id","shot_statsbomb_xg"]
    ].dropna(subset=["team_id","opp_team_id"])

    shots["def_team_id"] = shots["opp_team_id"].astype("int64")

    # Sum xG conceded by the defending team per season
    xga = (
        shots.groupby(["def_team_id","season_id"])["shot_statsbomb_xg"].sum()
             .rename("xga_total")
             .reset_index()
    )

    # Join matches played for that team-season to get per-match xGA
    xga = xga.merge(
        mts, left_on=["def_team_id","season_id"], right_on=["team_id","season_id"], how="left"
    )
    xga["matches"] = xga["matches"].fillna(1).astype(int)

    # per-match xGA
    xga["xga_per_match"] = xga["xga_total"] / xga["matches"]

    # Percentile rank (ascending => worse = high xGA; strength = 1 - pct)
    xga["pct"] = xga["xga_per_match"].rank(pct=True, ascending=True)
    xga["def_strength"] = 1.0 - xga["pct"]
    xga = xga[["def_team_id","season_id","def_strength"]]

    # Map back to events
    df["opp_def_strength"] = 0.5
    df = df.merge(
        xga, left_on=["opp_team_id","season_id"], right_on=["def_team_id","season_id"], how="left"
    )
    df["opp_def_strength"] = df["def_strength"].fillna(df["opp_def_strength"])
    df = df.drop(columns=[c for c in ["def_team_id","def_strength"] if c in df.columns])
    return df

# ==============================
# Enrichment: per-event feature extraction
# ==============================
def _corridor_from_y(y: Optional[float]) -> Optional[str]:
    if y is None: return None
    w = PITCH_Y / 5.0  # 5 lanes
    # [0,w) LW, [w,2w) LHS, [2w,3w) C, [3w,4w) RHS, [4w,5w] RW
    if y < 0 or y > PITCH_Y:
        return None
    if y < w: return "LW"
    if y < 2*w: return "LHS"
    if y < 3*w: return "C"
    if y < 4*w: return "RHS"
    return "RW"

def _third_from_x(x: Optional[float]) -> Optional[str]:
    if x is None: return None
    if x < 0 or x > PITCH_X:
        return None
    if x < FINAL_THIRD_X - (PITCH_X/3.0):  # 0..40
        return "def"
    if x < FINAL_THIRD_X:                      # 40..80
        return "mid"
    return "att"                               # 80..120

def enrich_event_row(row: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Action family
    fam = infer_action_family(row)
    out["action_family"] = fam or "other"

    # Raw locations
    loc0_raw = parse_location(row.get("location"))
    loc1_raw = None
    if fam == "pass":
        loc1_raw = parse_location(row.get("pass_end_location"))
    elif fam == "shot":
        loc1_raw = parse_location(row.get("shot_end_location"))
    elif fam == "carry":
        loc1_raw = parse_location(row.get("carry_end_location"))
    elif fam == "goalkeeper":
        loc1_raw = parse_location(row.get("goalkeeper_end_location"))

    # Orientation normalization
    try:
        per = int(row.get("period")) if pd.notna(row.get("period")) else -1
    except Exception:
        per = -1
    sgn = _ATTACK_SIGN.get((row.get("match_id"), row.get("team_id"), per), None)
    loc0 = flip_if_needed(loc0_raw, sgn)
    loc1 = flip_if_needed(loc1_raw, sgn)

    # Cells
    start_cell = xy_to_cell(*(loc0 or (None, None))) if loc0 else None
    end_cell = xy_to_cell(*(loc1 or (None, None))) if loc1 else None
    out["start_cell"] = str(start_cell) if start_cell is not None else None
    out["end_cell"] = str(end_cell) if end_cell is not None else None

    # Angle & distance (use flipped coords)
    sin_a = cos_a = 0.0
    dist = 0.0
    if loc0 and loc1:
        sin_a, cos_a = angle_sin_cos(loc0[0], loc0[1], loc1[0], loc1[1])
        dist = compute_distance(loc0[0], loc0[1], loc1[0], loc1[1])
    else:
        ang = safe_float(row.get("pass_angle"))
        if ang is not None:
            if abs(ang) > math.pi * 1.5:
                ang = math.radians(ang)
            sin_a, cos_a = math.sin(ang), math.cos(ang)
        dist = safe_float(row.get("pass_length")) or 0.0
    out["sin_angle"] = float(sin_a)
    out["cos_angle"] = float(cos_a)
    out["log_pass_length"] = float(math.log(max(dist, 0.0) + 1.0))
    out["distance_bin"] = distance_bin(dist)

    # Micro-dynamics extras
    duration = float(safe_float(row.get("duration")) or 0.0)
    v_m_s = float(dist / duration) if duration > 0 else 0.0
    out["duration"] = duration
    out["v_m_s"] = v_m_s

    # Goal geometry (from start to goal center, in normalized frame)
    if loc0:
        dxg = GOAL_CENTER[0] - loc0[0]
        dyg = GOAL_CENTER[1] - loc0[1]
        gd = math.hypot(dxg, dyg)
        out["goal_dist"] = float(gd)
        out["goal_angle_cos"] = float(dxg / (gd + 1e-6))
    else:
        out["goal_dist"] = 0.0
        out["goal_angle_cos"] = 0.0

    # Progression & final-third entry (normalized)
    progressive = False
    final_third_entry = False
    if fam in ("pass","carry") and (loc0 and loc1):
        dx = (loc1[0] - loc0[0])
        progressive = dx >= PROGRESS_MIN_DX
        final_third_entry = (loc0[0] < FINAL_THIRD_X) and (loc1[0] >= FINAL_THIRD_X)

    # Corridor & third (from start point)
    corr = _corridor_from_y(loc0[1] if loc0 else None)
    thr = _third_from_x(loc0[0] if loc0 else None)
    out["corridor"] = corr
    out["third"] = thr

    # Pass fields
    out["pass_height"] = safe_str(row.get("pass_height"))
    out["pass_technique"] = safe_str(row.get("pass_technique"))
    out["pass_type"] = safe_str(row.get("pass_type"))
    out["pass_outcome"] = safe_str(row.get("pass_outcome"))
    out["pass_cluster_label"] = safe_str(row.get("pass_pass_cluster_label"))
    out["pass_cluster_prob"] = safe_float(row.get("pass_pass_cluster_probability"))
    out["pass_expected_success"] = safe_float(row.get("pass_pass_success_probability"))

    # Shot fields
    out["shot_type"] = safe_str(row.get("shot_type"))
    out["shot_outcome"] = safe_str(row.get("shot_outcome"))
    out["shot_technique"] = safe_str(row.get("shot_technique"))
    out["shot_body_part"] = safe_str(row.get("shot_body_part"))
    out["shot_statsbomb_xg"] = safe_float(row.get("shot_statsbomb_xg"))
    out["shot_exec_xg"] = safe_float(row.get("shot_shot_execution_xg"))
    out["shot_exec_uplift"] = safe_float(row.get("shot_shot_execution_xg_uplift"))

    # Dribble / Duel / Interception / Clearance
    out["dribble_outcome"] = safe_str(row.get("dribble_outcome"))
    out["duel_type"] = safe_str(row.get("duel_type"))
    out["duel_outcome"] = safe_str(row.get("duel_outcome"))
    out["interception_outcome"] = safe_str(row.get("interception_outcome"))

    # Clearance body part from flags
    cbp = None
    if row.get("clearance_head") == True: cbp = "head"
    elif row.get("clearance_left_foot") == True: cbp = "left_foot"
    elif row.get("clearance_right_foot") == True: cbp = "right_foot"
    elif row.get("clearance_other") == True: cbp = "other"
    out["clearance_body_part"] = cbp

    # GK limited fields
    out["goalkeeper_type"] = safe_str(row.get("goalkeeper_type"))
    out["goalkeeper_outcome"] = safe_str(row.get("goalkeeper_outcome"))

    # Play pattern
    out["play_pattern"] = safe_str(row.get("play_pattern"))

    # Time & context
    out["minute_bucket"] = minute_bucket(row.get("minute"))
    out["period"] = str(int(row.get("period"))) if pd.notna(row.get("period")) else "unk"
    out["home_away"] = home_flag(row.get("team"), row.get("home_team"), row.get("away_team"))
    out["position_hint"] = safe_str(row.get("position")) or coarse_pos(row.get("position"))

    # Importance & context from precompute
    out["score_diff"] = float(safe_float(row.get("score_diff")) or 0.0)
    out["leverage"] = float(safe_float(row.get("leverage")) or 0.0)
    out["red_card_diff"] = float(safe_float(row.get("red_card_diff")) or 0.0)
    out["opp_def_strength"] = float(safe_float(row.get("opp_def_strength")) or 0.5)
    out["game_state"] = safe_str(row.get("game_state")) or "tied"
    out["dt_prev_s"] = float(safe_float(row.get("dt_prev_s")) or 0.0)
    out["dt_prev_by_player_s"] = float(safe_float(row.get("dt_prev_by_player_s")) or 0.0)

    # Possession context (precomputed)
    out["pos_idx_norm"] = float(safe_float(row.get("pos_idx_norm")) or 0.0)
    out["pos_len"] = float(safe_float(row.get("pos_len")) or 0.0)
    out["pos_time_from_start_s"] = float(safe_float(row.get("pos_time_from_start_s")) or 0.0)
    out["pos_obv_cum"] = float(safe_float(row.get("pos_obv_cum")) or 0.0)
    out["pos_obv_share"] = float(safe_float(row.get("pos_obv_share")) or 0.0)

    # Qualifiers (multi-hot flags)
    quals = {}
    for q in QUALIFIERS:
        quals[q] = bool(row.get(q) == True)
    # inject computed flags
    quals["progressive"] = bool(progressive)
    quals["final_third_entry"] = bool(final_third_entry)
    out["qualifiers"] = quals

    # OBV event value (optional numeric)
    out["obv_total_net_event"] = float(safe_float(row.get("obv_total_net")) or 0.0)

    return out

# ==============================
# Vocab building / encoding
# ==============================
def build_vocabs(enriched_rows: Iterable[Dict[str, Any]]) -> Dict[str, Vocab]:
    pools: Dict[str, List[Optional[str]]] = {k: [] for k in CAT_FACTORS}
    for ev in enriched_rows:
        for k in CAT_FACTORS:
            pools[k].append(ev.get(k))
    vocabs = {k: Vocab.build(pools[k]) for k in CAT_FACTORS}
    return vocabs

def encode_event(ev: Dict[str, Any], vocabs: Dict[str, Vocab]) -> Tuple[Dict[str,int], List[float], List[int]]:
    cat_ids: Dict[str,int] = {}
    for k in CAT_FACTORS:
        cat_ids[k] = vocabs[k].encode(ev.get(k))
    nums: List[float] = []
    for k in NUM_FEATURES:
        v = ev.get(k)
        if v is None:
            nums.append(0.0)
        else:
            try:
                nums.append(float(v))
            except Exception:
                nums.append(0.0)
    qual_vec: List[int] = [1 if ev.get("qualifiers", {}).get(q, False) else 0 for q in QUALIFIERS]
    return cat_ids, nums, qual_vec

# ==============================
# Index maps (player-season / team-season / seasons)
# ==============================
@dataclass(frozen=True)
class KeyMap:
    key2id: Dict[Tuple[int,int], int]
    id2key: Dict[int, Tuple[int,int]]

def build_player_season_map(df: pd.DataFrame) -> KeyMap:
    keys = []
    for _, r in df[["player_id","season_id"]].dropna().iterrows():
        try:
            keys.append((int(r["player_id"]), int(r["season_id"])))
        except Exception:
            pass
    uniq = sorted(set(keys))
    k2i = {k: i+2 for i, k in enumerate(uniq)}
    i2k = {i: k for k, i in k2i.items()}
    return KeyMap(k2i, i2k)

def build_team_season_map(df: pd.DataFrame) -> KeyMap:
    keys = []
    for _, r in df[["team_id","season_id"]].dropna().iterrows():
        try:
            keys.append((int(r["team_id"]), int(r["season_id"])))
        except Exception:
            pass
    uniq = sorted(set(keys))
    k2i = {k: i+2 for i, k in enumerate(uniq)}
    i2k = {i: k for k, i in k2i.items()}
    return KeyMap(k2i, i2k)

# ==============================
# Sequence builders
# ==============================
def sort_events(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["period","minute","second","timestamp","index"]:
        if c not in df.columns:
            df[c] = np.nan
    return df.sort_values(
        by=["match_id","period","minute","second","timestamp","index"],
        kind="mergesort"
    ).reset_index(drop=True)

def df_to_enriched(df: pd.DataFrame) -> List[Dict[str,Any]]:
    out = []
    for _, row in df.iterrows():
        out.append(enrich_event_row(row))
    return out

def pad2d(ids_list: List[List[int]], pad_val: int, T: int) -> np.ndarray:
    N = len(ids_list)
    arr = np.full((N, T), pad_val, dtype=np.int32)
    for i, seq in enumerate(ids_list):
        L = min(len(seq), T)
        arr[i, :L] = seq[:L]
    return arr

def pad3d_num(nums_list: List[List[List[float]]], T: int, K: int) -> np.ndarray:
    N = len(nums_list)
    arr = np.zeros((N, T, K), dtype=np.float32)
    for i, seq in enumerate(nums_list):
        L = min(len(seq), T)
        if L > 0:
            arr[i, :L, :] = np.asarray(seq[:L], dtype=np.float32)
    return arr

def pad3d_qual(qual_list: List[List[List[int]]], T: int, Q: int) -> np.ndarray:
    N = len(qual_list)
    arr = np.zeros((N, T, Q), dtype=np.int8)
    for i, seq in enumerate(qual_list):
        L = min(len(seq), T)
        if L > 0:
            arr[i, :L, :] = np.asarray(seq[:L], dtype=np.int8)
    return arr

def build_possession_sequences(
    df: pd.DataFrame,
    vocabs: Dict[str,Vocab],
    team_season_map: KeyMap,
    player_season_map: KeyMap,
    max_T: int = MAX_T_POSSESSION
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """
    Build possession sequences for the possession team (normalized orientation),
    emitting:
      - Packed NPZ arrays (including headers H_match & H_pos_idx)
      - pos_meta (row-per-segment metadata aligned by segment_row)
      - pos_tokens (flattened token table)

    The returned DataFrames can be written directly to Parquet.
    """
    cat_per_factor: Dict[str, List[List[int]]] = {f: [] for f in CAT_FACTORS}
    num_list:  List[List[List[float]]] = []
    qual_list: List[List[List[int]]] = []
    mask_list: List[List[int]] = []

    header_team:   List[int] = []
    header_opp:    List[int] = []
    header_season: List[int] = []
    header_seqtype: List[int] = []
    header_match:  List[int] = []       # NEW
    header_pos_idx: List[int] = []      # NEW

    actor_ps_list: List[List[int]] = []

    # To build meta + tokens
    meta_rows: List[Dict[str, Any]] = []
    token_rows: List[Dict[str, Any]] = []

    grouped = df.groupby(["match_id","possession"], sort=False)
    segment_row = 0
    for (mid, poss), g in grouped:
        # possession team id
        if "possession_team_id" not in g.columns:
            poss_team = g["team_id"].dropna().astype("Int64")
            if len(poss_team) == 0:
                continue
            poss_team_id = int(poss_team.iloc[0])
        else:
            v = g["possession_team_id"].dropna().astype("Int64")
            if len(v) == 0:
                continue
            poss_team_id = int(v.iloc[0])

        season_id = int(g["season_id"].iloc[0])

        # opp team id in this possession
        teams_in_group = list(g["team_id"].dropna().unique())
        opp_id = None
        for t in teams_in_group:
            if int(t) != poss_team_id:
                opp_id = int(t); break

        # keep only events by possession team
        gp = g[g["team_id"] == poss_team_id]
        if gp.empty:
            continue

        # Accumulators for this possession (before windowing)
        cat_rows = {f: [] for f in CAT_FACTORS}
        nums_rows:  List[List[float]] = []
        qual_rows:  List[List[int]] = []
        msk_rows:   List[int] = []
        act_rows:   List[int] = []
        ev_rows:    List[Dict[str, Any]] = []  # to derive meta/token tables quickly

        for _, row in gp.iterrows():
            ev = row["_enrich"]
            cat_ids, nums, quals = encode_event(ev, vocabs)
            for f in CAT_FACTORS:
                cat_rows[f].append(cat_ids[f])
            nums_rows.append(nums)
            qual_rows.append(quals)
            msk_rows.append(1)
            ev_rows.append(ev)

            # Resolve actor PS id
            pid = row.get("player_id")
            if pd.notna(pid):
                try:
                    pid_i = int(pid)
                    ps_id = player_season_map.key2id.get((pid_i, season_id), -1)
                except Exception:
                    ps_id = -1
            else:
                ps_id = -1
            act_rows.append(int(ps_id))

        # slice into segments of at most max_T
        i0 = 0
        pos_idx_counter = 0  # running index within this match possession (window slices)
        while i0 < len(msk_rows):
            i1 = min(i0 + max_T, len(msk_rows))
            seg_len = i1 - i0
            if seg_len <= 1:
                break

            # Append packed slices
            for f in CAT_FACTORS:
                cat_per_factor[f].append(cat_rows[f][i0:i1])
            num_list.append(nums_rows[i0:i1])
            qual_list.append(qual_rows[i0:i1])
            mask_list.append(msk_rows[i0:i1])
            actor_ps_list.append(act_rows[i0:i1])

            header_team.append(team_season_map.key2id.get((poss_team_id, season_id), UNK_ID))
            header_opp.append(team_season_map.key2id.get((opp_id, season_id), UNK_ID) if opp_id is not None else UNK_ID)
            header_season.append(season_id)
            header_seqtype.append(1)  # POS
            header_match.append(int(mid))        # NEW
            header_pos_idx.append(int(poss))     # NEW: index within match

            # ---- Build meta row for this segment ----
            ev_slice = ev_rows[i0:i1]
            act_slice = act_rows[i0:i1]

            # duration_s & minutes_range from per-event enrich (or original cols if needed)
            duration_s = float(np.sum([safe_float(ev.get("duration")) or 0.0 for ev in ev_slice]))
            # minutes_range from df window (fallback to enriched context not stored -> use original gp)
            minutes_vals = gp.iloc[i0:i1]["minute"].dropna().astype(float).values
            if minutes_vals.size:
                minutes_range = [float(np.min(minutes_vals)), float(np.max(minutes_vals))]
            else:
                minutes_range = [None, None]

            # facets
            has_shot = any((ev.get("action_family") == "shot") for ev in ev_slice)
            shot_outcomes = sorted({str(ev.get("shot_outcome")) for ev in ev_slice if ev.get("shot_outcome") is not None})
            has_goal = any((str(ev.get("shot_outcome")).lower().find("goal") >= 0) for ev in ev_slice)
            restart_source = ev_slice[0].get("play_pattern") if ev_slice else None
            players_involved_ps = sorted(list({ps for ps in act_slice if ps is not None and ps >= 0}))
            xg_vals = [safe_float(ev.get("shot_statsbomb_xg")) for ev in ev_slice if ev.get("shot_statsbomb_xg") is not None]
            xg_vals = [v for v in xg_vals if v is not None]
            xg_sum = float(np.sum(xg_vals)) if xg_vals else 0.0
            xg_max = float(np.max(xg_vals)) if xg_vals else 0.0

            meta_rows.append({
                "segment_row": segment_row,
                "match_id": int(mid),
                "possession": int(poss),
                "team_season_id": header_team[-1],
                "opp_team_season_id": header_opp[-1],
                "season_id": int(season_id),
                "seq_len": int(seg_len),
                "duration_s": duration_s,
                "minutes_range": minutes_range,  # list[low, high]
                "has_shot": bool(has_shot),
                "has_goal": bool(has_goal),
                "shot_outcomes": shot_outcomes,   # list[str]
                "restart_source": restart_source,
                "players_involved_ps": players_involved_ps,  # list[int]
                "xg_sum": xg_sum,
                "xg_max": xg_max,
            })

            # ---- Flattened token table for this segment ----
            for t, (ps_id, ev) in enumerate(zip(act_slice, ev_slice)):
                token_rows.append({
                    "segment_row": segment_row,
                    "t": int(t),
                    "actor_ps": int(ps_id),
                    "action_family": ev.get("action_family"),
                    "play_pattern": ev.get("play_pattern"),
                    "shot_outcome": ev.get("shot_outcome"),
                })

            # advance
            segment_row += 1
            pos_idx_counter += 1
            i0 = i1

    N = len(mask_list)
    if N == 0:
        # Return empty packed + empty DFs
        return {}, pd.DataFrame(), pd.DataFrame()

    Knum = len(NUM_FEATURES)
    Q = len(QUALIFIERS)

    def pad2d_local(idss: List[List[int]], pad_val: int, T: int) -> np.ndarray:
        N = len(idss)
        out = np.full((N, T), pad_val, dtype=np.int32)
        for i, seq in enumerate(idss):
            L = min(len(seq), T)
            out[i, :L] = np.asarray(seq[:L], dtype=np.int32)
        return out

    packed: Dict[str,np.ndarray] = {}
    for f in CAT_FACTORS:
        packed[f] = pad2d_local(cat_per_factor[f], PAD_ID, max_T)
    packed["X_num"]  = pad3d_num(num_list,  max_T, Knum)
    packed["X_qual"] = pad3d_qual(qual_list, max_T, Q)
    packed["mask"]   = pad2d_local(mask_list, 0, max_T)

    # headers (including NEW)
    packed["H_team"]    = np.asarray(header_team,   dtype=np.int32)
    packed["H_opp"]     = np.asarray(header_opp,    dtype=np.int32)
    packed["H_season"]  = np.asarray(header_season, dtype=np.int32)
    packed["H_seqtype"] = np.asarray(header_seqtype, dtype=np.int8)
    packed["H_match"]   = np.asarray(header_match,  dtype=np.int64)    # NEW
    packed["H_pos_idx"] = np.asarray(header_pos_idx, dtype=np.int32)  # NEW

    # token-level actor PS id
    packed["actor_ps"]  = pad2d_local(actor_ps_list, -1, max_T)

    # Next-action label
    Y = np.full((N, max_T), PAD_ID, dtype=np.int32)
    af = packed["action_family"]
    m  = packed["mask"]
    for i in range(N):
        for t in range(max_T-1):
            if m[i, t] == 1 and m[i, t+1] == 1:
                Y[i, t] = af[i, t+1]
    packed["Y_next_action_family"] = Y

    # Build DataFrames for meta & tokens
    pos_meta_df = pd.DataFrame(meta_rows)
    pos_tokens_df = pd.DataFrame(token_rows)

    return packed, pos_meta_df, pos_tokens_df

def build_player_sequences(
    df: pd.DataFrame,
    vocabs: Dict[str,Vocab],
    player_season_map: KeyMap,
    team_season_map: KeyMap,
    max_T: int = MAX_T_PLAYER,
    stride: int = PLAYER_WINDOW_STRIDE,
    eligible_ps: Optional[set] = None,
) -> Dict[str, np.ndarray]:
    dfp = df[df["player_id"].notna()].copy()
    if dfp.empty:
        return {}

    cat_per_factor: Dict[str, List[List[int]]] = {f: [] for f in CAT_FACTORS}
    num_list: List[List[List[float]]] = []
    qual_list: List[List[List[int]]] = []
    mask_list: List[List[int]] = []

    header_player: List[int] = []
    header_team: List[int] = []
    header_season: List[int] = []
    header_poshint: List[int] = []
    header_seqtype: List[int] = []

    for (pid, sid), g in dfp.groupby(["player_id","season_id"], sort=False):
        pid = int(pid); sid = int(sid)
        if eligible_ps is not None and (pid, sid) not in eligible_ps:
            continue

        cat_rows = {f: [] for f in CAT_FACTORS}
        nums_rows: List[List[float]] = []
        qual_rows: List[List[int]] = []
        msk_rows: List[int] = []
        poshint_last = None
        team_last = None

        enrich_idx = g.columns.get_loc("_enrich")
        team_idx = g.columns.get_loc("team_id")

        for row in g.itertuples(index=False, name=None):
            ev = row[enrich_idx]
            cat_ids, nums, quals = encode_event(ev, vocabs)
            for f in CAT_FACTORS:
                cat_rows[f].append(cat_ids[f])
            nums_rows.append(nums)
            qual_rows.append(quals)
            msk_rows.append(1)
            poshint_last = ev.get("position_hint") or poshint_last
            team_val = row[team_idx]
            team_last = team_val if team_val is not None and not (isinstance(team_val, float) and math.isnan(team_val)) else team_last

        if len(msk_rows) == 0:
            continue

        i0 = 0
        while i0 < len(msk_rows):
            i1 = min(i0 + max_T, len(msk_rows))
            seg_len = i1 - i0
            if seg_len <= 1:
                break
            for f in CAT_FACTORS:
                cat_per_factor[f].append(cat_rows[f][i0:i1])
            num_list.append(nums_rows[i0:i1])
            qual_list.append(qual_rows[i0:i1])
            mask_list.append(msk_rows[i0:i1])

            header_player.append(player_season_map.key2id.get((pid, sid), UNK_ID))
            header_team.append(team_season_map.key2id.get((int(team_last) if team_last is not None else -1, sid), UNK_ID))
            header_season.append(sid)
            header_poshint.append(vocabs["position_hint"].encode(poshint_last))
            header_seqtype.append(2)  # PLY

            if i1 == len(msk_rows): break
            i0 += stride

    N = len(mask_list)
    if N == 0:
        return {}
    Knum = len(NUM_FEATURES)
    Q = len(QUALIFIERS)

    packed: Dict[str,np.ndarray] = {}
    for f in CAT_FACTORS:
        packed[f] = pad2d(cat_per_factor[f], PAD_ID, max_T)
    packed["X_num"] = pad3d_num(num_list, max_T, Knum)
    packed["X_qual"] = pad3d_qual(qual_list, max_T, Q)
    packed["mask"] = pad2d(mask_list, 0, max_T)

    packed["H_player"] = np.asarray(header_player, dtype=np.int32)
    packed["H_team"] = np.asarray(header_team, dtype=np.int32)
    packed["H_season"] = np.asarray(header_season, dtype=np.int32)
    packed["H_poshint"] = np.asarray(header_poshint, dtype=np.int32)
    packed["H_seqtype"] = np.asarray(header_seqtype, dtype=np.int8)

    Y = np.full((N, max_T), PAD_ID, dtype=np.int32)
    af = packed["action_family"]
    m = packed["mask"]
    for i in range(N):
        for t in range(max_T-1):
            if m[i, t] == 1 and m[i, t+1] == 1:
                Y[i, t] = af[i, t+1]
    packed["Y_next_action_family"] = Y
    return packed

# ==============================
# Player-season stats (per-90 with OBV + GK)
# ==============================
def estimate_minutes(df: pd.DataFrame) -> pd.DataFrame:
    match_max_min = df.groupby("match_id")["minute"].max().fillna(90).clip(lower=0, upper=130)
    subs = df[df["substitution_outcome"].notna() | df["substitution_replacement_id"].notna()]
    intervals = []

    for (mid, pid), g in df[df["player_id"].notna()].groupby(["match_id","player_id"]):
        mmax = float(match_max_min.get(mid, 90.0))
        start = 0.0
        end = mmax
        intervals.append((mid, int(pid), start, end))

    for _, r in subs.iterrows():
        mid = r.get("match_id")
        m = float(safe_float(r.get("minute")) or 0.0)
        off_pid = r.get("player_id")
        on_pid = r.get("substitution_replacement_id")
        if pd.notna(off_pid):
            off_pid = int(off_pid)
            for i, tup in enumerate(intervals):
                if tup[0] == mid and tup[1] == off_pid:
                    intervals[i] = (tup[0], tup[1], tup[2], min(m, tup[3]))
        if pd.notna(on_pid):
            on_pid = int(on_pid)
            found = False
            for i, tup in enumerate(intervals):
                if tup[0] == mid and tup[1] == on_pid:
                    intervals[i] = (tup[0], tup[1], max(m, tup[2]), tup[3])
                    found = True
                    break
            if not found:
                mmax = float(match_max_min.get(mid, 90.0))
                intervals.append((mid, on_pid, m, mmax))

    out = []
    pid_sid = df[["match_id","player_id","season_id"]].dropna().drop_duplicates()
    for (mid, pid, start, end) in intervals:
        sid = pid_sid[(pid_sid["match_id"]==mid) & (pid_sid["player_id"]==pid)]
        if sid.empty: continue
        season = int(sid["season_id"].iloc[0])
        minutes = max(0.0, float(end - start))
        out.append((pid, season, minutes))
    if not out:
        return pd.DataFrame(columns=["player_id","season_id","minutes"])
    mm = (pd.DataFrame(out, columns=["player_id","season_id","minutes"])
          .groupby(["player_id","season_id"], as_index=False)["minutes"].sum())
    return mm

def build_player_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    minutes_df = estimate_minutes(df)

    is_shot = df["shot_statsbomb_xg"].notna() | df["shot_outcome"].notna()
    is_pass = df["pass_length"].notna() | df["pass_outcome"].notna() | df["pass_end_location"].notna()
    is_carry = df["carry_end_location"].notna()
    is_dribble = df["dribble_outcome"].notna() | df["dribble_no_touch"].notna() | df["dribble_overrun"].notna()
    is_intercept = df["interception_outcome"].notna()
    is_block = df["block_deflection"].notna() | df["block_offensive"].notna() | df["block_save_block"].notna()
    is_clear = (df["clearance_head"].notna() | df["clearance_left_foot"].notna() |
                df["clearance_right_foot"].notna() | df["clearance_aerial_won"].notna() | df["clearance_other"].notna())
    is_duel = df["duel_outcome"].notna() | df["duel_type"].notna()
    is_recovery = df["ball_recovery_recovery_failure"].notna() | df["ball_recovery_offensive"].notna()

    # Strict GK mask for stats (align with inference policy)
    strict_gk_col_mask = df["goalkeeper_type"].notna() | df["goalkeeper_outcome"].notna()
    gk_pos_mask = df["position"].apply(coarse_pos) == "GK"
    is_gk_event = (df["type"].astype(str) == "Goal Keeper") | (strict_gk_col_mask & gk_pos_mask)

    goal_mask = df["shot_outcome"].astype(str).str.contains("Goal", na=False)

    obv_event = df["obv_total_net"].fillna(0.0).astype(float)
    obv_for = df["obv_for_net"].fillna(0.0).astype(float)
    obv_against = df["obv_against_net"].fillna(0.0).astype(float)

    gby = df.dropna(subset=["player_id","season_id"]).groupby(["player_id","season_id"], dropna=True)

    rows = []
    for (pid, sid), g in gby:
        pid = int(pid); sid = int(sid)
        mins = float(minutes_df[(minutes_df["player_id"]==pid) & (minutes_df["season_id"]==sid)]["minutes"].sum())
        if mins <= 0:
            matches_played = g["match_id"].nunique()
            mins = float(matches_played * 90.0)

        idx = g.index

        shots = int(is_shot[idx].sum())
        goals = int(goal_mask[idx].sum())
        passes = int(is_pass[idx].sum())
        carries = int(is_carry[idx].sum())
        dribbles = int(is_dribble[idx].sum())
        interceptions = int(is_intercept[idx].sum())
        blocks = int(is_block[idx].sum())
        clearances = int(is_clear[idx].sum())
        duels = int(is_duel[idx].sum())
        recoveries = int(is_recovery[idx].sum())

        pass_len = df.loc[idx, "pass_length"].dropna().astype(float)
        pass_comp = df.loc[idx, "pass_outcome"].astype(str).str.contains("Complete|Completed|Success", na=False).sum()
        pass_outcomes_total = df.loc[idx, "pass_outcome"].notna().sum()
        pass_comp_rate = float(pass_comp) / float(pass_outcomes_total) if pass_outcomes_total > 0 else np.nan
        pass_exp = df.loc[idx, "pass_pass_success_probability"].dropna().astype(float)
        pass_exp_mean = pass_exp.mean() if len(pass_exp)>0 else np.nan

        xg = df.loc[idx, "shot_statsbomb_xg"].dropna().astype(float).sum()
        xa_proxy = float(df.loc[idx, "pass_shot_assist"].fillna(False).astype(bool).sum()) * 0.1

        obv_ev = obv_event[idx].sum()
        obv_for_sum = obv_for[idx].sum()
        obv_against_sum = obv_against[idx].sum()

        # Goalkeeper stats
        gk_idx = g.index[is_gk_event[idx]]
        gk_actions = int(len(gk_idx))
        gk_success_in_play = int(df.loc[gk_idx, "goalkeeper_success_in_play"].fillna(False).astype(bool).sum()) if len(gk_idx)>0 else 0
        gk_success_out = int(df.loc[gk_idx, "goalkeeper_success_out"].fillna(False).astype(bool).sum()) if len(gk_idx)>0 else 0
        gk_saved_to_post = int(df.loc[gk_idx, "goalkeeper_shot_saved_to_post"].fillna(False).astype(bool).sum()) if len(gk_idx)>0 else 0
        gk_saved_off_tgt = int(df.loc[gk_idx, "goalkeeper_shot_saved_off_target"].fillna(False).astype(bool).sum()) if len(gk_idx)>0 else 0
        gk_punched = int(df.loc[gk_idx, "goalkeeper_punched_out"].fillna(False).astype(bool).sum()) if len(gk_idx)>0 else 0

        per90 = lambda v: (float(v) / mins) * 90.0 if mins > 0 else np.nan
        gk_actions_per90 = per90(gk_actions)
        gk_success_total = gk_success_in_play + gk_success_out
        gk_success_rate = (gk_success_total / gk_actions) if gk_actions > 0 else np.nan
        gk_in_play_rate = (gk_success_in_play / gk_actions) if gk_actions > 0 else np.nan
        gk_out_rate = (gk_success_out / gk_actions) if gk_actions > 0 else np.nan

        rows.append({
            "player_id": pid, "season_id": sid, "minutes": mins,
            "shots_per90": per90(shots),
            "goals_per90": per90(goals),
            "xg_total": float(xg),
            "xg_per90": per90(xg),
            "passes_per90": per90(passes),
            "carries_per90": per90(carries),
            "dribbles_per90": per90(dribbles),
            "interceptions_per90": per90(interceptions),
            "blocks_per90": per90(blocks),
            "clearances_per90": per90(clearances),
            "duels_per90": per90(duels),
            "recoveries_per90": per90(recoveries),
            "pass_len_avg": float(pass_len.mean()) if len(pass_len)>0 else np.nan,
            "pass_len_p25": float(pass_len.quantile(0.25)) if len(pass_len)>0 else np.nan,
            "pass_len_p50": float(pass_len.quantile(0.50)) if len(pass_len)>0 else np.nan,
            "pass_len_p75": float(pass_len.quantile(0.75)) if len(pass_len)>0 else np.nan,
            "pass_comp_rate": float(pass_comp_rate) if not pd.isna(pass_comp_rate) else np.nan,
            "pass_expected_success_mean": float(pass_exp_mean) if not pd.isna(pass_exp_mean) else np.nan,
            "xa_proxy": xa_proxy,
            "obv_total_net_sum": float(obv_ev),
            "obv_for_net_sum": float(obv_for_sum),
            "obv_against_net_sum": float(obv_against_sum),
            "obv_total_net_per90": per90(obv_ev),
            "obv_for_net_per90": per90(obv_for_sum),
            "obv_against_net_per90": per90(obv_against_sum),

            # GK additions
            "gk_actions": gk_actions,
            "gk_actions_per90": gk_actions_per90,
            "gk_success_total": gk_success_total,
            "gk_success_rate": gk_success_rate,
            "gk_success_in_play": gk_success_in_play,
            "gk_success_out": gk_success_out,
            "gk_in_play_rate": gk_in_play_rate,
            "gk_out_rate": gk_out_rate,
            "gk_saved_to_post": gk_saved_to_post,
            "gk_saved_off_target": gk_saved_off_tgt,
            "gk_punched_out": gk_punched,
        })

    return pd.DataFrame(rows)

# ==============================
# NEW: Team-season stats
# ==============================
def build_team_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Get denominator: matches_played
    matches_df = (
        df[["match_id", "team_id", "season_id"]].dropna().drop_duplicates()
        .groupby(["team_id", "season_id"])["match_id"].nunique()
        .reset_index(name="matches_played")
    )
    matches_df["team_id"] = matches_df["team_id"].astype(int)
    matches_df["season_id"] = matches_df["season_id"].astype(int)
    
    if matches_df.empty:
        return pd.DataFrame()

    # 2. Define masks (copied from build_player_season_stats)
    is_shot = df["shot_statsbomb_xg"].notna() | df["shot_outcome"].notna()
    is_pass = df["pass_length"].notna() | df["pass_outcome"].notna() | df["pass_end_location"].notna()
    is_carry = df["carry_end_location"].notna()
    is_dribble = df["dribble_outcome"].notna() | df["dribble_no_touch"].notna() | df["dribble_overrun"].notna()
    is_intercept = df["interception_outcome"].notna()
    is_block = df["block_deflection"].notna() | df["block_offensive"].notna() | df["block_save_block"].notna()
    is_clear = (df["clearance_head"].notna() | df["clearance_left_foot"].notna() |
                df["clearance_right_foot"].notna() | df["clearance_aerial_won"].notna() | df["clearance_other"].notna())
    is_duel = df["duel_outcome"].notna() | df["duel_type"].notna()
    is_recovery = df["ball_recovery_recovery_failure"].notna() | df["ball_recovery_offensive"].notna()
    goal_mask = df["shot_outcome"].astype(str).str.contains("Goal", na=False)

    obv_event = df["obv_total_net"].fillna(0.0).astype(float)
    obv_for = df["obv_for_net"].fillna(0.0).astype(float)
    obv_against = df["obv_against_net"].fillna(0.0).astype(float)

    # 3. Stats FOR (group by team_id)
    gby_team = df.dropna(subset=["team_id","season_id"]).groupby(["team_id","season_id"])
    
    stats_for = pd.DataFrame({
        "shots_for": gby_team.apply(lambda g: is_shot[g.index].sum()),
        "goals_for": gby_team.apply(lambda g: goal_mask[g.index].sum()),
        "xg_for": gby_team["shot_statsbomb_xg"].sum(),
        "passes_for": gby_team.apply(lambda g: is_pass[g.index].sum()),
        "pass_comp_for": gby_team.apply(lambda g: df.loc[g.index, "pass_outcome"].astype(str).str.contains("Complete|Completed|Success", na=False).sum()),
        "pass_outcomes_total_for": gby_team.apply(lambda g: df.loc[g.index, "pass_outcome"].notna().sum()),
        "carries_for": gby_team.apply(lambda g: is_carry[g.index].sum()),
        "dribbles_for": gby_team.apply(lambda g: is_dribble[g.index].sum()),
        "interceptions_for": gby_team.apply(lambda g: is_intercept[g.index].sum()),
        "blocks_for": gby_team.apply(lambda g: is_block[g.index].sum()),
        "clearances_for": gby_team.apply(lambda g: is_clear[g.index].sum()),
        "duels_for": gby_team.apply(lambda g: is_duel[g.index].sum()),
        "recoveries_for": gby_team.apply(lambda g: is_recovery[g.index].sum()),
        "obv_for_net_sum": gby_team["obv_for_net"].sum(),
        "obv_against_net_sum": gby_team["obv_against_net"].sum(), # Team's defensive OBV
        "obv_total_net_sum": gby_team["obv_total_net"].sum(),
    }).reset_index()
    
    # 4. Stats AGAINST (group by opp_team_id)
    # Note: opp_team_id is the team being attacked (i.e., the defensive team)
    gby_opp = df.dropna(subset=["opp_team_id","season_id"]).groupby(["opp_team_id","season_id"])
    
    stats_against = pd.DataFrame({
        "shots_against": gby_opp.apply(lambda g: is_shot[g.index].sum()),
        "goals_against": gby_opp.apply(lambda g: goal_mask[g.index].sum()),
        "xg_against": gby_opp["shot_statsbomb_xg"].sum(),
    }).reset_index()
    
    stats_against = stats_against.rename(columns={"opp_team_id": "team_id"})
    stats_against["team_id"] = stats_against["team_id"].astype(int)
    stats_against["season_id"] = stats_against["season_id"].astype(int)

    # 5. Merge all
    stats = matches_df.merge(stats_for, on=["team_id","season_id"], how="left")
    stats = stats.merge(stats_against, on=["team_id","season_id"], how="left")
    
    # 6. Calculate per-match and rates
    stats = stats.fillna(0)
    m = stats["matches_played"]
    
    # Helper for safe division
    def per_match(col):
        return (stats[col] / m).where(m > 0, 0)

    stats["g_per_match"] = per_match("goals_for")
    stats["xg_per_match"] = per_match("xg_for")
    stats["ga_per_match"] = per_match("goals_against")
    stats["xga_per_match"] = per_match("xg_against")
    stats["shots_per_match"] = per_match("shots_for")
    stats["shots_a_per_match"] = per_match("shots_against")
    
    stats["pass_comp_rate"] = (stats["pass_comp_for"] / stats["pass_outcomes_total_for"]).where(stats["pass_outcomes_total_for"] > 0, 0)
    
    stats["obv_for_pm"] = per_match("obv_for_net_sum")
    stats["obv_against_pm"] = per_match("obv_against_net_sum")
    stats["obv_total_pm"] = per_match("obv_total_net_sum")
    
    # Differentials
    stats["gd_per_match"] = stats["g_per_match"] - stats["ga_per_match"]
    stats["xgd_per_match"] = stats["xg_per_match"] - stats["xga_per_match"]

    return stats


# ==============================
# Eligibility + Meta
# ==============================
def build_ps_eligibility(df: pd.DataFrame) -> set[Tuple[int,int]]:
    ev_counts = (
        df[df["player_id"].notna()]
        .groupby(["player_id","season_id"], dropna=True)["player_id"]
        .count()
        .rename("n_events")
    )
    minutes_df = estimate_minutes(df)  # player_id, season_id, minutes
    base = (
        minutes_df.merge(ev_counts.reset_index(), on=["player_id","season_id"], how="left")
                  .fillna({"n_events":0})
    )
    elig = base[(base["n_events"] >= MIN_EVENTS_PER_PS) & (base["minutes"] >= MIN_MINUTES_PER_PS)]
    return set((int(r.player_id), int(r.season_id)) for r in elig.itertuples(index=False))

def build_player_season_meta(df: pd.DataFrame) -> pd.DataFrame:
    name_col = "player_name" if "player_name" in df.columns else ("player" if "player" in df.columns else None)

    if name_col:
        names = (
            df.dropna(subset=["player_id","season_id"])
              .groupby(["player_id","season_id"])[name_col]
              .agg(lambda s: safe_str(s.dropna().iloc[0]) if s.dropna().size else None)
              .rename("player_name")
        )
    else:
        names = None

    pos_mode = (
        df.dropna(subset=["player_id","season_id"])
          .groupby(["player_id","season_id"])["position"]
          .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().size else None)
          .rename("position_mode")
    )

    ev_counts = (
        df[df["player_id"].notna()]
        .groupby(["player_id","season_id"], dropna=True)["player_id"]
        .count()
        .rename("events_count")
        .reset_index()
    )
    minutes_df = estimate_minutes(df)

    meta = ev_counts.merge(minutes_df, on=["player_id","season_id"], how="left")
    meta = meta.merge(pos_mode.reset_index(), on=["player_id","season_id"], how="left")
    if names is not None:
        meta = meta.merge(names.reset_index(), on=["player_id","season_id"], how="left")

    meta["position_mode_coarse"] = meta["position_mode"].apply(coarse_pos)
    return meta

# ==============================
# NEW: Team-season meta
# ==============================
def build_team_season_meta(df: pd.DataFrame) -> pd.DataFrame:
    name_col = "team_name" if "team_name" in df.columns else ("team" if "team" in df.columns else None)
    
    if not name_col:
        log("[warn] No 'team' or 'team_name' column found for team meta.")
        return pd.DataFrame(columns=["team_id", "season_id", "team_name"])

    names = (
        df.dropna(subset=["team_id","season_id"])
          .groupby(["team_id","season_id"])[name_col]
          .agg(lambda s: safe_str(s.dropna().iloc[0]) if s.dropna().size else None)
          .rename("team_name")
          .reset_index()
    )
    names["team_id"] = names["team_id"].astype(int)
    names["season_id"] = names["season_id"].astype(int)
    return names

# ==============================
# Main Orchestration
# ==============================
def run_step1(
    parquet_paths: Sequence[str | Path],
    out_dir: Path = OUT_DIR,
):
    ensure_dir(out_dir); ensure_dir(VOCAB_DIR); ensure_dir(SEQ_DIR); ensure_dir(STATS_DIR); ensure_dir(MAPS_DIR)

    # 1) Load & add season_id
    frames = []
    for p in parquet_paths:
        df = parquet_read(p)
        sid = parse_season_id_from_path(p)
        df["season_id"] = sid
        frames.append(df)
        log(f"[load] {p}: rows={len(df):,} season={sid}")
    df_all = pd.concat(frames, ignore_index=True)
    log(f"[concat] total rows={len(df_all):,}")

    # Minimal required columns
    for c in ["match_id","team_id","team","home_team","away_team","player_id","position","possession","possession_team_id"]:
        if c not in df_all.columns:
            df_all[c] = np.nan

    # 2) Sort events globally
    df_all = sort_events(df_all)

    # Deduplicate early
    if "id" in df_all.columns:
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=["id"])
        log(f"[dedup] by id: {before:,} -> {len(df_all):,}")
    else:
        keys = [c for c in DEDUP_KEYS if c in df_all.columns]
        if keys:
            before = len(df_all)
            df_all = df_all.drop_duplicates(subset=keys)
            log(f"[dedup] by {keys}: {before:,} -> {len(df_all):,}")
        else:
            log("[dedup] skipped (no suitable keys)")

    # 2.5) Precompute opponent mapping
    df_all = precompute_opponent(df_all)

    # Orientation: compute attack directions
    global _ATTACK_SIGN
    log("[orient] computing attack directions…")
    _ATTACK_SIGN = compute_attack_sign(df_all)

    # 3) Precompute score/cards/tempo + possession context + opp def strength
    log("[pre] computing score/cards/tempo/leverage…")
    df_all = precompute_score_cards_and_tempo(df_all)
    log("[pre] computing possession context (idx/time/obv)…")
    df_all = _possession_context(df_all)
    log("[pre] computing opponent defensive strength…")
    df_all = precompute_opponent_def_strength(df_all)

    # 4) Enrich each row with tokenizable features (with orientation normalized)
    log("[enrich] building per-event features…")
    enr = []
    for _, row in df_all.iterrows():
        enr.append(enrich_event_row(row))
    df_all["_enrich"] = enr

    # 5) Build vocabs
    log("[vocab] building categorical vocabs…")
    vocabs = build_vocabs(df_all["_enrich"].tolist())
    for k, v in vocabs.items():
        (VOCAB_DIR / f"{k}.json").write_text(json.dumps({
            "token2id": v.token2id, "id2token": {str(i): tok for i, tok in v.id2token.items()},
            "pad_token": v.pad_token, "unk_token": v.unk_token
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    (VOCAB_DIR / "qualifiers.json").write_text(json.dumps({"qualifiers": QUALIFIERS}, indent=2), encoding="utf-8")
    (VOCAB_DIR / "num_features.json").write_text(json.dumps({"num_features": NUM_FEATURES}, indent=2), encoding="utf-8")

    # 6) Maps
    log("[maps] building player-season and team-season maps…")
    ps_map = build_player_season_map(df_all)
    ts_map = build_team_season_map(df_all)
    (MAPS_DIR / "player_season_map.json").write_text(json.dumps({
        "key2id": {f"{k[0]}|{k[1]}": v for k,v in ps_map.key2id.items()},
        "id2key": {str(i): [int(k[0]), int(k[1])] for i, k in ps_map.id2key.items()}
    }, indent=2), encoding="utf-8")
    (MAPS_DIR / "team_season_map.json").write_text(json.dumps({
        "key2id": {f"{k[0]}|{k[1]}": v for k,v in ts_map.key2id.items()},
        "id2key": {str(i): [int(k[0]), int(k[1])] for i, k in ts_map.id2key.items()}
    }, indent=2), encoding="utf-8")

    # Eligibility
    log("[elig] computing player-season eligibility…")
    eligible_ps = build_ps_eligibility(df_all)
    log(f"[elig] eligible player-seasons: {len(eligible_ps):,}")

    # 7) Sequences
    log("[seq] building possession sequences…")
    pos_pack, pos_meta_df, pos_tokens_df = build_possession_sequences(
        df_all, vocabs, ts_map, ps_map, max_T=MAX_T_POSSESSION
    )
    if pos_pack:
        np.savez_compressed(SEQ_DIR / "possession_sequences.npz", **pos_pack)
        log(f"[seq] possession saved: N={len(pos_pack['mask'])} T={pos_pack['mask'].shape[1]}")

        # Persist POS meta and token tables
        pos_meta_path = OUT_DIR / "pos_meta.parquet"
        pos_tokens_path = OUT_DIR / "pos_tokens.parquet"
        # Ensure lists are preserved (requires pyarrow)
        pos_meta_df.to_parquet(pos_meta_path, index=False)
        pos_tokens_df.to_parquet(pos_tokens_path, index=False)
        log(f"[meta] saved: {pos_meta_path} rows={len(pos_meta_df):,}")
        log(f"[tokens] saved: {pos_tokens_path} rows={len(pos_tokens_df):,}")
    else:
        log("[seq] no possession sequences produced.")

    log("[seq] building player sequences…")
    ply_pack = build_player_sequences(
        df_all, vocabs, ps_map, ts_map,
        max_T=MAX_T_PLAYER, stride=PLAYER_WINDOW_STRIDE,
        eligible_ps=eligible_ps
    )
    if ply_pack:
        np.savez_compressed(SEQ_DIR / "player_sequences.npz", **ply_pack)
        log(f"[seq] player saved: N={len(ply_pack['mask'])} T={ply_pack['mask'].shape[1]}")
    else:
        log("[seq] no player sequences produced.")

    # 8) Player-season stats (with OBV + GK)
    log("[stats] aggregating player-season stats…")
    stats_df = build_player_season_stats(df_all)
    stats_path = STATS_DIR / "player_season_stats.parquet"
    stats_df.to_parquet(stats_path, index=False)
    log(f"[stats] saved: {stats_path} rows={len(stats_df):,}")

    # 9) NEW: Team-season meta and stats
    log("[meta] building team-season meta (name)…")
    team_meta_df = build_team_season_meta(df_all)
    team_meta_path = STATS_DIR / "team_season_meta.parquet"
    team_meta_df.to_parquet(team_meta_path, index=False)
    log(f"[meta] saved: {team_meta_path} rows={len(team_meta_df):,}")

    log("[stats] aggregating team-season stats…")
    team_stats_df = build_team_season_stats(df_all)
    
    # Merge name into stats as requested by user
    team_stats_with_name_df = team_meta_df.merge(team_stats_df, on=["team_id", "season_id"], how="right")
    
    team_stats_path = STATS_DIR / "team_season_stats.parquet"
    team_stats_with_name_df.to_parquet(team_stats_path, index=False)
    log(f"[stats] saved: {team_stats_path} rows={len(team_stats_with_name_df):,}")

    # 10) Player Meta
    log("[meta] building player-season meta (name, position mode)…")
    meta_df = build_player_season_meta(df_all)
    meta_path = STATS_DIR / "player_season_meta.parquet"
    meta_df.to_parquet(meta_path, index=False)
    log(f"[meta] saved: {meta_path} rows={len(meta_df):,}")

    # 11) Diagnostics
    diag = {
        "rows_total": int(len(df_all)),
        "grid": {"width": GRID_W, "height": GRID_H, "pitch_x": PITCH_X, "pitch_y": PITCH_Y},
        "cat_vocab_sizes": {k: len(v.token2id) for k, v in vocabs.items()},
        "num_features": NUM_FEATURES,
        "qualifiers_count": len(QUALIFIERS),
        "possession_sequences": int(len(pos_pack["mask"])) if pos_pack else 0,
        "player_sequences": int(len(ply_pack["mask"])) if ply_pack else 0,
        "eligibility": {
            "min_events": MIN_EVENTS_PER_PS,
            "min_minutes": MIN_MINUTES_PER_PS,
            "eligible_player_seasons": len(eligible_ps)
        }
    }
    (OUT_DIR / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    log("[done] Step 1 preprocessing complete.")

# ==============================
# Entry
# ==============================
if __name__ == "__main__":
    run_step1(INPUT_PARQUETS, OUT_DIR)