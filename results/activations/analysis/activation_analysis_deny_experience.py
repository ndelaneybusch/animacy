import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
base_dir = Path("/home/nate/repos/animacy/results/activations/analysis")
summaries_dir = Path(
    "/home/nate/repos/animacy/results/activations/data/Qwen3-30B-A3B-Instruct-2507/with_sys/summaries"
)
ratings_path = Path(
    "/home/nate/repos/animacy/results/ratings/data/Qwen3-30B-A3B-Instruct-2507/response_ratings.csv"
)
selected_words_path = Path("/home/nate/repos/animacy/data/selected_words.csv")
output_dir = base_dir / "plots"
output_dir.mkdir(exist_ok=True)

# 1. Data Loading & Filtering
print("Loading data...")
ratings_df = pd.read_csv(ratings_path)
selected_words = pd.read_csv(selected_words_path)

# Filter out role deviations
bool_cols = [
    "assistant_refusal",
    "role_refusal",
    "identify_as_assistant",
    "deny_internal_experience",
]
filtered_ratings = ratings_df.copy()
print(f"Responses after filtering other deviations: {len(filtered_ratings)}")

# Filter for assistant-like roles
assistant_like_roles = [
    "lawyer",
    "engineer",
    "referee",
    "professor",
    "physician",
    "scientist",
]
print(f"Assistant-like roles: {assistant_like_roles}")

# Identify top 3 tasks with balanced deny_internal_experience
analysis_df = filtered_ratings[
    filtered_ratings["role_name"].isin(assistant_like_roles)
].copy()
task_counts = (
    analysis_df.groupby(["task_name", "deny_internal_experience"])
    .size()
    .unstack(fill_value=0)
)

# Just hardcode the tasks for now
selected_tasks = ["dreams", "past_self", "remembered"]
print(f"Selected tasks: {selected_tasks}")


# 2. Vector Calculation
# Load avg_response_first_10_tokens
layers = [26, 30, 34, 38]
activation_key = "avg_response_first_10_tokens"

# Identify Assistant Group roles (Group ID 4)
assistant_group_roles = selected_words[selected_words["group_id"] == 4]["word"].tolist()
print(f"Assistant group roles: {assistant_group_roles}")


def load_vectors(roles, tasks, layer, check_ratings=True):
    vectors = []
    metadata = []

    for summary_file in summaries_dir.glob(f"*_layer{layer}.json"):
        parts = summary_file.stem.split("_")
        # Filename: {role}_{task}_{sample_idx}_layer{layer}

        # Robust parsing:
        # 1. Try to match role from the provided list
        matched_role = None
        for r in roles:
            # Check if filename starts with role + underscore
            if summary_file.name.startswith(r + "_"):
                matched_role = r
                break

        if not matched_role:
            continue

        # 2. Parse task and sample_idx from the remainder
        # Remainder after role: {task}_{sample_idx}_layer{layer}.json
        remainder = summary_file.name[len(matched_role) + 1 :]  # +1 for underscore
        rem_parts = remainder.split("_")

        try:
            # sample_idx is the second to last part (before layer part)
            # layer part is the last part
            # task is everything before sample_idx
            if len(rem_parts) < 3:  # minimal: task_idx_layer.json
                continue

            sample_idx = int(rem_parts[-2])
            task = "_".join(rem_parts[:-2])
        except (ValueError, IndexError):
            continue

        if task not in tasks:
            continue

        if check_ratings:
            # Check ratings filter
            match = filtered_ratings[
                (filtered_ratings["role_name"] == matched_role)
                & (filtered_ratings["task_name"] == task)
                & (filtered_ratings["sample_idx"] == sample_idx)
            ]
            if len(match) == 0:
                continue
            meta = match.iloc[0].to_dict()
        else:
            # For assistant group, we don't have ratings metadata, just create dummy
            meta = {
                "role_name": matched_role,
                "task_name": task,
                "sample_idx": sample_idx,
            }

        try:
            with open(summary_file, "r") as f:
                data = json.load(f)

            if activation_key in data:
                vec = np.array(data[activation_key])
                vectors.append(vec)
                meta["vector"] = vec  # Store for distance calc
                metadata.append(meta)
            else:
                if len(vectors) < 5:
                    print(
                        f"DEBUG: Key '{activation_key}' not found in {summary_file.name}. Keys: {list(data.keys())}"
                    )
        except Exception as e:
            print(f"Error loading {summary_file}: {e}")

    return vectors, pd.DataFrame(metadata)


# 3. Analysis per Layer
for layer in layers:
    print(f"\nProcessing Layer {layer}...")

    # A. Calculate Assistant Mean Vector for this layer
    # Filter: Assistant Group Roles + Selected Tasks
    print("  Computing Assistant Mean Vector...")
    asst_vectors, _ = load_vectors(
        assistant_group_roles, selected_tasks, layer, check_ratings=False
    )

    if not asst_vectors:
        print(f"  Warning: No assistant vectors found for layer {layer}")
        continue

    assistant_mean_vector = np.mean(asst_vectors, axis=0)

    # B. Load Target Data
    # Filter: Assistant-like Roles + Selected Tasks
    print("  Loading Target Data...")
    _, target_df = load_vectors(
        assistant_like_roles, selected_tasks, layer, check_ratings=True
    )

    if target_df.empty:
        print(f"  Warning: No target data found for layer {layer}")
        continue

    # C. Calculate Distances
    print("  Calculating Distances...")
    # Euclidean distance between each vector and the mean
    # distance = sqrt(sum((x - mean)^2))
    target_vectors = np.stack(target_df["vector"].values)
    distances = np.linalg.norm(target_vectors - assistant_mean_vector, axis=1)
    target_df["distance"] = distances

    # D. Visualization
    print("  Generating Plots...")

    # Figure 1: Faceted by Task
    plt.figure(figsize=(15, 6))
    sns.boxplot(
        data=target_df,
        x="task_name",
        y="distance",
        hue="deny_internal_experience",
        palette={True: "#e41a1c", False: "#377eb8"},
        showfliers=False,
        boxprops={"alpha": 0.4},
    )
    sns.stripplot(
        data=target_df,
        x="task_name",
        y="distance",
        hue="deny_internal_experience",
        palette={True: "#e41a1c", False: "#377eb8"},
        dodge=True,
        alpha=0.8,
        jitter=0.2,
    )
    plt.title(
        f"Layer {layer}: Distance from Assistant Mean when Denying Internal Experience, by Task"
    )
    plt.ylabel("Distance to Assistant Mean for these tasks")
    plt.xlabel("Task")
    plt.legend(title="Deny Internal Exp")
    plt.tight_layout()
    plt.savefig(output_dir / f"layer_{layer}_distance_by_task.png")
    plt.close()

    # Figure 2: Faceted by Role
    plt.figure(figsize=(15, 6))
    sns.boxplot(
        data=target_df,
        x="role_name",
        y="distance",
        hue="deny_internal_experience",
        palette={True: "#e41a1c", False: "#377eb8"},
        showfliers=False,
        boxprops={"alpha": 0.4},
    )
    sns.stripplot(
        data=target_df,
        x="role_name",
        y="distance",
        hue="deny_internal_experience",
        palette={True: "#e41a1c", False: "#377eb8"},
        dodge=True,
        alpha=0.6,
        jitter=0.2,
    )
    plt.title(
        f"Layer {layer}: Distance from Assistant Mean when Denying Internal Experience, by Role"
    )
    plt.ylabel("Euclidean Distance")
    plt.xlabel("Role")
    plt.legend(title="Deny Internal Exp")
    plt.tight_layout()
    plt.savefig(output_dir / f"layer_{layer}_distance_by_role.png")
    plt.close()

print("\nAnalysis Complete.")
