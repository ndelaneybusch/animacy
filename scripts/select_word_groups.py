"""Select word groups based on animacy dimensions while controlling for confounds.

This module provides functionality to select matched word groups that differ on
animacy dimensions (mental and physical) while being approximately matched on
potential confounding variables (word frequency, concreteness, valence).

Usage:
    python select_word_groups.py --n <number_of_words_per_group> [--output <output_file>]

Example:
    python select_word_groups.py --n 20 --output data/my_words.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load the Words sheet from the Excel file.

    Args:
        filepath: Path to the Excel file containing word norms data.

    Returns:
        DataFrame containing all words and their associated measurements from
        the "Words" sheet.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the "Words" sheet is not found in the Excel file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_excel(path, sheet_name="Words")
    return df.loc[df["remove"] != 1].reset_index(drop=True)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DataFrame columns to more descriptive, snake_case names.

    Args:
        df: DataFrame with original column names from the Excel file.

    Returns:
        DataFrame with renamed columns using descriptive, snake_case naming
        convention (e.g., 'AnimMental' -> 'anim_mental').
    """
    column_mapping = {
        "Word": "word",
        "Category": "category",
        "LivingN": "living_score",
        "LivingM": "living_mean",
        "ThoughtN": "thought_score",
        "Thought": "thought_mean",
        "ReproN": "repro_score",
        "Repro": "repro_mean",
        "PersonN": "person_score",
        "Person": "person_mean",
        "GoalsN": "goals_score",
        "Goals": "goals_mean",
        "MoveN": "move_score",
        "Move": "move_mean",
        "CNC": "concreteness",
        "FAM": "familiarity",
        "AVAIL": "availability",
        "MNG": "meaningfulness",
        "VAL": "valence",
        "ARO": "arousal",
        "DOM": "dominance",
        "AoA": "age_of_acquisition",
        "LEN": "word_length",
        "ORTHON": "orthographic_n",
        "PhononN": "phonographic_n",
        "NSyll": "n_syllables",
        "SUBTLWF": "word_frequency",
        "AnimMental": "anim_mental",
        "AnimPhysical": "anim_physical",
    }

    # Only rename columns that exist in the dataframe
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    return df


def categorize_broad_category(category: Any) -> str:
    """Categorize words into broad semantic categories.

    Maps fine-grained category codes to three broad categories: Animals, People,
    or Objects. Categories are identified by their first letter:
    - 'A': Animals (including all living things)
    - 'H': People (humans and human-related)
    - 'B': People (body parts)
    - 'O', 'E', 'V', 'P', 'C', 'W', 'S': Objects (including environments,
      vehicles, plants, collectives, weather, and supernatural)

    Args:
        category: Fine-grained category code from the dataset (e.g., 'A F',
            'H A', 'O C'). May be None or NaN.

    Returns:
        Broad category label: 'Animal', 'People', 'Object', 'Unknown', or 'Other'.
    """
    if pd.isna(category):
        return "Unknown"

    cat_str = str(category).strip().upper()
    if not cat_str:
        return "Unknown"

    first_char = cat_str[0]

    # Animals (including living things)
    if first_char == "A":
        return "Animal"

    # People/Humans
    if first_char in ("H"):
        return "People"

    # Objects (including vehicles, environments, etc.)
    if first_char in ("O", "E", "V", "P", "C", "W", "S"):
        return "Object"

    return "Other"


def select_matched_groups(
    df: pd.DataFrame, n: int, balance_categories: bool = True
) -> Dict[int, pd.DataFrame]:
    """Select matched word groups based on animacy dimensions.

    Selects n words for each of three groups defined by animacy dimensions:
    1. High AnimMental, High AnimPhysical (e.g., people)
    2. Low AnimMental, High AnimPhysical (e.g., animals)
    3. Low AnimMental, Low AnimPhysical (e.g., objects)

    Groups are matched on word frequency, concreteness, and valence to minimize
    confounds. Uses fixed thresholds for high (>= 300) and low (< 250) values.

    Args:
        df: DataFrame containing word data with animacy and matching variables.
        n: Number of words to select per group.
        balance_categories: If True, attempts to maintain equal distribution of
            Animals, People, and Objects within each group. Default is True.

    Returns:
        Dictionary mapping group IDs (1, 2, 3) to DataFrames containing the
        selected words for each group.
    """

    # Define thresholds for high/low
    high_threshold = 300
    low_threshold = 250

    logger.info(f"Using thresholds: High >= {high_threshold}, Low < {low_threshold}")

    # Define the three groups
    group1 = df[
        (df["anim_mental"] >= high_threshold) & (df["anim_physical"] >= high_threshold)
    ].copy()
    group2 = df[
        (df["anim_mental"] < low_threshold) & (df["anim_physical"] >= high_threshold)
    ].copy()
    group3 = df[
        (df["anim_mental"] < low_threshold) & (df["anim_physical"] < low_threshold)
    ].copy()

    logger.info("\nAvailable candidates:")
    logger.info(f"  Group 1 (High Mental, High Physical): {len(group1)} words")
    logger.info(f"  Group 2 (Low Mental, High Physical): {len(group2)} words")
    logger.info(f"  Group 3 (Low Mental, Low Physical): {len(group3)} words")

    # Check if we have enough words
    min_available = min(len(group1), len(group2), len(group3))
    if min_available < n:
        logger.warning("\nWARNING: Not enough words in all groups!")
        logger.warning(
            f"Requested {n} words per group, but some groups have fewer candidates."
        )
        n = min_available
        logger.warning(f"Reducing to {n} words per group.")

    # Add broad category
    for grp in [group1, group2, group3]:
        grp["broad_category"] = grp["category"].apply(categorize_broad_category)

    # Features to match on
    match_features = ["word_frequency", "concreteness", "valence"]

    # Remove rows with missing matching features
    group1_clean = group1.dropna(subset=match_features)
    group2_clean = group2.dropna(subset=match_features)
    group3_clean = group3.dropna(subset=match_features)

    logger.info("\nAfter removing rows with missing matching features:")
    logger.info(f"  Group 1: {len(group1_clean)} words")
    logger.info(f"  Group 2: {len(group2_clean)} words")
    logger.info(f"  Group 3: {len(group3_clean)} words")

    pools = {1: group1_clean.copy(), 2: group2_clean.copy(), 3: group3_clean.copy()}

    # If balancing categories, try to maintain equal distribution
    if balance_categories:
        selected = select_with_category_balance(pools, n, match_features)
    else:
        selected = select_with_matching(pools, n, match_features)

    return selected


def select_with_matching(
    pools: dict[int, pd.DataFrame],
    n: int,
    match_features: list[str],
    bias_animacy: bool = True,
) -> dict[int, pd.DataFrame]:
    """Select words from pools to match feature means across groups.

    Uses an iterative greedy algorithm to select words that minimize differences
    in feature means across groups. On each iteration, selects the word from
    each pool that brings the group's feature means closest to the overall
    target means.

    Optimized using vectorized numpy operations for performance.

    Args:
        pools: Dictionary mapping group IDs to DataFrames of candidate words.
        n: Number of words to select from each pool.
        match_features: List of column names to match across groups.
        bias_animacy: If True, biases selection towards words with animacy
            scores furthest from 350.

    Returns:
        Dictionary mapping group IDs to DataFrames of selected words.
    """
    # Constants for animacy bias
    ANIMACY_TARGET = 350
    ANIMACY_WEIGHT = 0.05  # Weight for the animacy bias term

    # Initialize containers
    selected_dfs = {gid: [] for gid in pools}

    # Create working copies to track available candidates
    # We keep them as DataFrames to return the final rows easily
    working_pools = {gid: df.copy() for gid, df in pools.items()}

    # Pre-calculate global stats for normalization
    all_data = pd.concat(working_pools.values())
    if all_data.empty:
        return {gid: pd.DataFrame() for gid in pools}

    feature_stds = all_data[match_features].std().values
    # Handle zero std to avoid division by zero
    feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)

    # Pre-calculate normalized animacy scores for bias
    # Score = (|Mental - 350| + |Physical - 350|) / Scale
    # We want to MAXIMIZE this score.
    animacy_scores_map = {}
    for gid, pool in working_pools.items():
        # Calculate raw distance from target
        dist = np.abs(pool["anim_mental"] - ANIMACY_TARGET) + np.abs(
            pool["anim_physical"] - ANIMACY_TARGET
        )
        # Normalize roughly to 0-1 range (max dist is approx 700)
        animacy_scores_map[gid] = dist / 700.0

    # Track current sums and counts for efficient mean calculation
    # Using numpy arrays for speed
    grp_sums = {gid: np.zeros(len(match_features)) for gid in pools}
    grp_ns = {gid: 0 for gid in pools}

    for i in range(n):
        # Calculate target mean (grand mean of all selected so far)
        total_sum = sum(grp_sums.values())  # type: ignore
        total_n = sum(grp_ns.values())

        if total_n == 0:
            # First iteration
            for gid in pools:
                pool = working_pools[gid]
                if len(pool) > 0:
                    if bias_animacy:
                        # Pick word with max animacy distance
                        # We need to look up scores for current pool indices
                        current_scores = animacy_scores_map[gid].loc[pool.index]
                        best_idx_label = current_scores.idxmax()
                    else:
                        # Random selection
                        best_idx_label = pool.sample(1).index[0]

                    selected_row = pool.loc[[best_idx_label]]
                    vals = selected_row[match_features].values[0]

                    selected_dfs[gid].append(selected_row)
                    grp_sums[gid] += vals
                    grp_ns[gid] += 1

                    working_pools[gid] = working_pools[gid].drop(best_idx_label)
            continue

        # Calculate target mean
        target_mean = total_sum / total_n

        # Greedy selection for each group
        for gid in pools:
            pool = working_pools[gid]
            if len(pool) == 0:
                continue

            # Get candidates as numpy array for vectorized calc
            candidates_vals = pool[match_features].values

            # Calculate what the new group mean WOULD be for each candidate
            # Broadcasting: (n_features,) + (n_candidates, n_features) -> (n_candidates, n_features)
            new_grp_means = (grp_sums[gid] + candidates_vals) / (grp_ns[gid] + 1)

            # Calculate distance to target_mean
            # Normalize by standard deviation
            diffs = (new_grp_means - target_mean) / feature_stds
            dists = np.sum(diffs**2, axis=1)

            # Combine with animacy bias if requested
            if bias_animacy:
                # Get scores aligned with current pool
                # Note: pool.index matches the order of candidates_vals?
                # Yes, providing we don't sort/filter in between.
                # To be safe, we use loc with index.
                current_anim_scores = animacy_scores_map[gid].loc[pool.index].values

                # We want to minimize cost.
                # High animacy score is good -> subtract it.
                final_cost = dists - (ANIMACY_WEIGHT * current_anim_scores)
            else:
                final_cost = dists

            # Find best candidate
            best_idx_in_pool = np.argmin(final_cost)
            best_original_idx = pool.index[best_idx_in_pool]
            best_vals = candidates_vals[best_idx_in_pool]

            # Add to selected
            selected_row = pool.loc[[best_original_idx]]
            selected_dfs[gid].append(selected_row)

            # Update state
            grp_sums[gid] += best_vals
            grp_ns[gid] += 1

            # Remove from pool
            working_pools[gid] = working_pools[gid].drop(best_original_idx)

    # Convert to dataframes
    result = {}
    for gid, dfs in selected_dfs.items():
        if dfs:
            result[gid] = pd.concat(dfs)
        else:
            result[gid] = pd.DataFrame()

    return result


def select_with_category_balance(
    pools: dict[int, pd.DataFrame], n: int, match_features: list[str]
) -> dict[int, pd.DataFrame]:
    """Select words while balancing semantic categories and matching features.

    Args:
        pools: Dictionary mapping group IDs to DataFrames of candidate words.
        n: Total number of words to select per group.
        match_features: List of column names to match across groups.

    Returns:
        Dictionary mapping group IDs to DataFrames of selected words.
    """
    selected = {1: [], 2: [], 3: []}

    # Calculate target counts per category (as equal as possible)
    n_per_category = n // 3
    remainder = n % 3

    categories = ["Animal", "People", "Object"]
    target_counts = {cat: n_per_category for cat in categories}

    # Distribute remainder
    for i in range(remainder):
        target_counts[categories[i]] += 1

    logger.info(f"\nTarget category distribution per group: {target_counts}")

    # For each category, select words that match across groups
    for category in categories:
        target = target_counts[category]

        # Filter pools to this category
        cat_pools = {
            grp_id: pool[pool["broad_category"] == category].copy()
            for grp_id, pool in pools.items()
        }

        # Check availability
        min_available = min(len(p) for p in cat_pools.values())

        if min_available < target:
            logger.warning(
                f"\nWARNING: Only {min_available} {category} words available (target: {target})"
            )
            target = min_available

        if target > 0:
            cat_selected = select_with_matching(cat_pools, target, match_features)

            for grp_id in [1, 2, 3]:
                if not cat_selected[grp_id].empty:
                    selected[grp_id].append(cat_selected[grp_id])

    # If we still need more words (because of category imbalance), fill in
    # Calculate current counts
    current_counts = {
        grp_id: sum(len(df) for df in dfs) for grp_id, dfs in selected.items()
    }
    needed = {grp_id: n - count for grp_id, count in current_counts.items()}

    if any(count > 0 for count in needed.values()):
        logger.info(f"\nFilling remaining slots: {needed}")

        # Remove already selected words from pools
        for grp_id, pool in pools.items():
            if selected[grp_id]:
                # Combine current selection to get indices
                current_selection = pd.concat(selected[grp_id])
                pools[grp_id] = pool[~pool.index.isin(current_selection.index)]

        # Select remaining words
        min_needed = min(needed.values())
        if min_needed > 0:
            additional = select_with_matching(pools, min_needed, match_features)
            for grp_id in [1, 2, 3]:
                if not additional[grp_id].empty:
                    selected[grp_id].append(additional[grp_id])

    # Convert to dataframes
    result = {}
    for grp_id, word_lists in selected.items():
        if word_lists:
            result[grp_id] = pd.concat(word_lists, ignore_index=False)
        else:
            result[grp_id] = pd.DataFrame()

    return result


def print_group_statistics(selected_groups: Dict[int, pd.DataFrame]) -> None:
    """Print descriptive statistics for selected word groups."""
    print("\n" + "=" * 80)
    print("GROUP STATISTICS")
    print("=" * 80)

    for grp_id, df in selected_groups.items():
        if len(df) == 0:
            continue

        print(f"\nGroup {grp_id}:")
        print(f"  N = {len(df)}")

        # Animacy dimensions
        print(
            f"  AnimMental: M = {df['anim_mental'].mean():.3f}, SD = {df['anim_mental'].std():.3f}"
        )
        print(
            f"  AnimPhysical: M = {df['anim_physical'].mean():.3f}, SD = {df['anim_physical'].std():.3f}"
        )

        # Matching variables
        print(
            f"  Word Frequency: M = {df['word_frequency'].mean():.3f}, SD = {df['word_frequency'].std():.3f}"
        )
        print(
            f"  Concreteness: M = {df['concreteness'].mean():.3f}, SD = {df['concreteness'].std():.3f}"
        )
        print(
            f"  Valence: M = {df['valence'].mean():.3f}, SD = {df['valence'].std():.3f}"
        )

        # Category distribution
        if "broad_category" in df.columns:
            cat_counts = df["broad_category"].value_counts()
            print(f"  Categories: {dict(cat_counts)}")


def main() -> None:
    """Execute the word group selection workflow."""
    parser = argparse.ArgumentParser(
        description="Select word groups based on animacy dimensions"
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of words per group"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/VanArsdall_Blunt_NormData.xlsx",
        help="Input Excel file path (default: data/VanArsdall_Blunt_NormData.xlsx)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/selected_words.csv",
        help="Output CSV file path (default: data/selected_words.csv)",
    )
    parser.add_argument(
        "--no-category-balance",
        action="store_true",
        help="Disable category balancing (Animals, People, Objects)",
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.input}...")
    try:
        df = load_data(args.input)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    logger.info(f"Loaded {len(df)} words")

    # Clean column names
    df = clean_column_names(df)

    # Select groups
    logger.info(f"\nSelecting {args.n} words per group...")
    selected_groups = select_matched_groups(
        df, args.n, balance_categories=not args.no_category_balance
    )

    # Print statistics
    print_group_statistics(selected_groups)

    # Combine all groups and add group label
    all_selected = []
    group_names = {
        1: "High Mental, High Physical",
        2: "Low Mental, High Physical",
        3: "Low Mental, Low Physical",
    }

    for grp_id, df_grp in selected_groups.items():
        if df_grp.empty:
            continue
        df_grp = df_grp.copy()
        df_grp["group"] = group_names.get(grp_id, f"Group {grp_id}")
        df_grp["group_id"] = grp_id
        all_selected.append(df_grp)

    if all_selected:
        result_df = pd.concat(all_selected, ignore_index=True)

        # Save to CSV
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)

        print(f"\n" + "=" * 80)
        print(f"Saved {len(result_df)} words to {args.output}")
        print("=" * 80)
    else:
        logger.warning("No words selected.")


if __name__ == "__main__":
    main()
