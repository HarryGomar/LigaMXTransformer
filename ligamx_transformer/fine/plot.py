#!/usr/bin/env python3
# step3_plot.py (Corrected and Verified, with user additions)
"""
Generates UMAP plots from the embeddings exported by step2_train.py.

This script reads data from:
- ./exports/ (Embeddings)
- ./artifacts/data/stats/ (Player & Team metadata)
- ./artifacts/data/sequences/ (Possession metadata & headers)
- ./artifacts/data/maps/ (Team/Player maps)
- ./artifacts/data/vocabs/ (Vocab maps)

It saves plots to:
- ./plots/
"""

import json
from pathlib import Path
import warnings
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import os

# Check if running in an interactive environment (like a notebook) vs. a script
try:
    __file__
except NameError:
    print("Running in interactive mode. Setting CWD.")
    # In interactive mode (like Jupyter), we assume the CWD is already correct.
else:
    # If __file__ is defined, we're running as a script.
    print("Running as script. Setting CWD to script directory.")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ==============================
# Configuration
# ==============================

# --- Paths from step 1 & 2 ---
STEP1_DIR = Path("./artifacts/data")
EXPORT_DIR = Path("./exports")

STATS_DIR = STEP1_DIR / "stats"
SEQ_DIR = STEP1_DIR / "sequences"
MAPS_DIR = STEP1_DIR / "maps"
VOCAB_DIR = STEP1_DIR / "vocabs"

# --- Source Data Files ---
PLAYER_META_PATH = STATS_DIR / "player_season_meta.parquet"
TEAM_META_PATH     = STATS_DIR / "team_season_meta.parquet"
POSSESSION_META_PATH = SEQ_DIR / "pos_meta.parquet"

# --- New output directory ---
PLOT_DIR = Path("./plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- UMAP Parameters ---
UMAP_REDUCER = UMAP(
    n_neighbors=15, # Reduced for smaller subset
    min_dist=0.1,
    n_components=2,
    random_state=42,
    metric='cosine', 
    verbose=True
)

# --- Metadata Column ---
PLAYER_POSITION_COLUMN = "position_mode_coarse"


# ==============================
# Utilities
# ==============================

def log(msg: str):
    """Prints a message to the console."""
    print(msg, flush=True)

def load_json(path: Path) -> dict:
    """Loads a JSON file, returning empty dict if not found."""
    if not path.exists():
        log(f"Warning: JSON file not found at {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def plot_umap(data_2d: np.ndarray,
              labels: pd.Series,
              title: str,
              save_path: Path,
              palette: Optional[str | Dict] = None,
              s: int = 8,
              alpha: float = 0.7):
    """
    Creates and saves a styled UMAP scatter plot using Seaborn.
    """
    plt.figure(figsize=(14, 10))
    
    if palette is None:
        is_numeric = pd.api.types.is_numeric_dtype(labels)
        n_unique = labels.nunique()
        if is_numeric and n_unique > 1000:
            palette = "flare"
        elif not is_numeric:
            palette = "tab20" if n_unique <= 20 else "husl"
        else:
            palette = "viridis"

    ax = sns.scatterplot(
        x=data_2d[:, 0], y=data_2d[:, 1], hue=labels, palette=palette,
        s=s, alpha=alpha, linewidth=0, legend="auto"
    )
    
    ax.set_title(title, fontsize=18, pad=20, weight="bold")
    ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    sns.despine(left=True, bottom=True, trim=True)
    
    if ax.get_legend() is not None:
        if labels.nunique() > 30:
            ax.get_legend().remove()
            log(f"  > Plot '{title}': Hiding legend ({labels.nunique()} unique categories).")
        else:
            leg_title = labels.name if labels.name else "Category"
            ax.get_legend().set_title(leg_title)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0))

    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    log(f"✅ Saved plot: {save_path}")

def plot_umap_highlight(data_2d: np.ndarray,
                        highlight_mask: pd.Series,
                        title: str,
                        save_path: Path,
                        palette: Dict[str, str],
                        s_bg: int = 8, s_fg: int = 15,
                        alpha_bg: float = 0.5, alpha_fg: float = 0.9):
    """
    Creates a UMAP plot highlighting a boolean mask (e.g., 'has_shot').
    """
    plt.figure(figsize=(14, 10))
    
    non_highlight_mask = ~highlight_mask
    
    ax = sns.scatterplot(
        x=data_2d[non_highlight_mask, 0], y=data_2d[non_highlight_mask, 1],
        color=palette['bg'], s=s_bg, alpha=alpha_bg, linewidth=0,
        label=f"{highlight_mask.name} (False)"
    )
    
    sns.scatterplot(
        ax=ax, x=data_2d[highlight_mask, 0], y=data_2d[highlight_mask, 1],
        color=palette['fg'], s=s_fg, alpha=alpha_fg, linewidth=0,
        label=f"{highlight_mask.name} (True)"
    )
    
    ax.set_title(title, fontsize=18, pad=20, weight="bold")
    ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    sns.despine(left=True, bottom=True, trim=True)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0))
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    log(f"✅ Saved plot: {save_path}")

def plot_umap_styled(data_2d: np.ndarray,
                     color_labels: pd.Series,
                     style_labels: pd.Series,
                     title: str,
                     save_path: Path,
                     palette: Optional[Dict] = None,
                     markers: Optional[Dict | List] = None,
                     s: int = 40,
                     alpha: float = 0.8):
    """
    Creates a UMAP plot with distinct encodings for color (hue) and shape (style).
    """
    plt.figure(figsize=(14, 10))
    
    ax = sns.scatterplot(
        x=data_2d[:, 0],
        y=data_2d[:, 1],
        hue=color_labels,
        style=style_labels,
        palette=palette,
        markers=markers,
        s=s,
        alpha=alpha,
        linewidth=0,
        legend="full"
    )
    
    ax.set_title(title, fontsize=18, pad=20, weight="bold")
    ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    sns.despine(left=True, bottom=True, trim=True)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0))
    
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    log(f"✅ Saved plot: {save_path}")

# ==============================
# Data Loading Functions
# ==============================

def load_team_name_map() -> Dict[int, str]:
    """Loads a mapping from H_team (int) -> team_name (str)."""
    log("...loading team name map")
    ts_map_data = load_json(MAPS_DIR / "team_season_map.json")
    if not TEAM_META_PATH.exists() or not ts_map_data:
        log("Warning: Could not load team metadata. Team plots will use IDs.")
        return {}

    H_team_to_key = {int(k): v for k, v in ts_map_data.get("id2key", {}).items()}
    df_ts_keys = pd.DataFrame.from_dict(
        H_team_to_key, orient="index", columns=["team_id", "season_id"]
    )
    df_ts_keys.index.name = "H_team"
    
    df_ts_meta = pd.read_parquet(TEAM_META_PATH)
    
    team_map_full = df_ts_keys.reset_index().merge(
        df_ts_meta[["team_id", "season_id", "team_name"]],
        on=["team_id", "season_id"], how="left"
    )
    
    team_map_full["team_name"] = team_map_full["team_name"].str.lower()
    
    return pd.Series(
        team_map_full["team_name"].values, index=team_map_full["H_team"]
    ).to_dict()

def load_player_metadata() -> Optional[pd.DataFrame]:
    """Loads player metadata and aligns it with embedding indices."""
    log("...loading player metadata")
    if not PLAYER_META_PATH.exists():
        log(f"Error: Player metadata file not found: {PLAYER_META_PATH}")
        return None
    
    tsv_path = EXPORT_DIR / "player_style_embeddings.tsv"
    if not tsv_path.exists():
        log(f"Error: Player metadata TSV not found: {tsv_path}")
        return None

    player_meta_df = pd.read_parquet(PLAYER_META_PATH)
    export_meta_df = pd.read_csv(tsv_path, sep="\t")
    
    merged = export_meta_df.merge(
        player_meta_df, on=["player_id", "season_id"], how="left"
    )
    
    if PLAYER_POSITION_COLUMN not in merged.columns:
        log(f"Error: Column '{PLAYER_POSITION_COLUMN}' not in {PLAYER_META_PATH.name}.")
        return None
    
    merged[PLAYER_POSITION_COLUMN] = merged[PLAYER_POSITION_COLUMN].fillna("Unknown")
    return merged.set_index("player_season_index")

def load_possession_data() -> Optional[tuple[np.ndarray, pd.DataFrame]]:
    """Loads possession embeddings and all related metadata."""
    log("...loading possession data and metadata")
    
    emb_npz_path = EXPORT_DIR / "pos_embeddings.npz"
    seq_npz_path = SEQ_DIR / "possession_sequences.npz"
    
    if not all([emb_npz_path.exists(), seq_npz_path.exists(), POSSESSION_META_PATH.exists()]):
        log(f"Error: One or more required files are missing.")
        return None

    with np.load(emb_npz_path) as z:
        seq_vecs = z["Z_possession"]
    with np.load(seq_npz_path) as z:
        H_team = z["H_team"]
    
    header_df = pd.DataFrame({'segment_row': np.arange(len(H_team)), 'H_team': H_team})
    outcome_df = pd.read_parquet(POSSESSION_META_PATH)
    
    if 'segment_row' not in outcome_df.columns:
        log(f"Error: 'segment_row' key not found in {POSSESSION_META_PATH.name}.")
        return None
        
    meta_df = pd.merge(header_df, outcome_df, on='segment_row', how='inner')

    if len(meta_df) != len(seq_vecs):
        log("Error: Data misalignment! Embedding and metadata row counts differ.")
        return None
        
    team_name_map = load_team_name_map()
    meta_df["team_name"] = meta_df["H_team"].map(team_name_map).fillna("Unknown")
    
    if "has_shot" not in meta_df.columns: meta_df["has_shot"] = False
    if "has_goal" not in meta_df.columns: meta_df["has_goal"] = False
    meta_df["has_shot"] = meta_df["has_shot"].fillna(False).astype(bool)
    meta_df["has_goal"] = meta_df["has_goal"].fillna(False).astype(bool)
    
    return seq_vecs, meta_df

# ==============================
# Main Execution
# ==============================

def main():
    """Main function to load all embeddings, run UMAP, and plot."""
    log("Starting UMAP plot generation...")
    warnings.filterwarnings("ignore", category=UserWarning)

    # --- Plots for Player Embeddings (if available) ---
    player_meta = load_player_metadata()
    if player_meta is not None:
        style_npy_path = EXPORT_DIR / "player_style_embeddings.npy"
        if style_npy_path.exists():
            log(f"\nProcessing Player Style Embeddings...")
            style_vecs = np.load(style_npy_path)
            aligned_labels = player_meta[PLAYER_POSITION_COLUMN].reindex(player_meta.index).fillna("Unknown")
            style_2d = UMAP(n_neighbors=30).fit_transform(style_vecs)
            plot_umap(
                style_2d, aligned_labels,
                title=f"Player Style Embeddings (by {PLAYER_POSITION_COLUMN})",
                save_path=PLOT_DIR / "umap_player_style_by_position.png"
            )

        id_npy_path = EXPORT_DIR / "player_identity_embeddings.npy"
        if id_npy_path.exists():
            log(f"\nProcessing Player Identity Embeddings...")
            id_vecs = np.load(id_npy_path)
            valid_indices = [i for i in range(id_vecs.shape[0]) if i in player_meta.index]
            vecs_subset = id_vecs[valid_indices]
            labels_subset = player_meta.loc[valid_indices, PLAYER_POSITION_COLUMN]
            id_2d = UMAP(n_neighbors=30).fit_transform(vecs_subset)
            plot_umap(
                id_2d, labels_subset,
                title=f"Player Identity Embeddings (by {PLAYER_POSITION_COLUMN})",
                save_path=PLOT_DIR / "umap_player_identity_by_position.png"
            )

    # --- Plots for Possession Embeddings ---
    pos_data = load_possession_data()
    if pos_data is not None:
        seq_vecs, seq_meta = pos_data
        
        # --- Plot 4 (Subset Reduction): Necaxa & América shots ---
        log("\nProcessing SUBSET: Necaxa & América shots...")
        target_teams = ['necaxa', 'américa']
        filter_mask = (
            seq_meta["team_name"].isin(target_teams) &
            seq_meta["has_shot"]
        )
        
        if filter_mask.any():
            # 1. Filter the ORIGINAL high-dimensional vectors
            seq_vecs_subset = seq_vecs[filter_mask]
            
            # 2. Filter the metadata to align with the vector subset
            meta_subset = seq_meta[filter_mask].copy()
            
            log(f"  > Found {len(seq_vecs_subset)} shot sequences for {', '.join(target_teams)}.")
            log(f"  > Running UMAP reduction on this subset...")
            
            # 3. Run UMAP *only* on the filtered high-dimensional data
            seq_2d_subset = UMAP_REDUCER.fit_transform(seq_vecs_subset)
            
            # 4. Define custom color and marker mappings
            team_palette = {"necaxa": "#e74c3c", "américa": "#f1c40f"} # Red, Yellow
            goal_markers = {True: "X", False: "o"} # Goal: X, No Goal: Circle

            # 5. Plot the newly reduced 2D data
            plot_umap_styled(
                seq_2d_subset,
                color_labels=meta_subset["team_name"],
                style_labels=meta_subset["has_goal"],
                title="UMAP of Necaxa & America Shots (Color: Team, Shape: Goal)",
                save_path=PLOT_DIR / "umap_subset_shots_necaxa_america.png",
                palette=team_palette,
                markers=goal_markers,
                s=50 
            )
        else:
            log(f"  > Skipping: No shot sequences found for {', '.join(target_teams)}.")
        

    log("\nAll plotting complete. Check the 'plots' directory.")


if __name__ == "__main__":
    main()

