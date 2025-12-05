library(tidyverse)
library(lme4)
library(lmerTest)
library(easystats)

# Load data
data_path <- "results/steering/data/gemma-3-27b-it/word_guess_steered/"

df_at_role_period <- read_csv(paste0(data_path, "at_role_period_response.csv"))
df_at_role_period['role_vector'] <- "at_role_period"

df_at_role <- read_csv(paste0(data_path, "at_role_response.csv"))
df_at_role['role_vector'] <- "at_role_word"

df_avg_response <- read_csv(paste0(data_path, "avg_response.csv"))
df_avg_response['role_vector'] <- "avg_response"

df_avg_response_sys_diff <- read_csv(paste0(data_path, "avg_response_sys_diff.csv"))
df_avg_response_sys_diff['role_vector'] <- "sys_diff"

df_avg_response_first_10 <- read_csv(paste0(data_path, "avg_response_first_10.csv"))
df_avg_response_first_10['role_vector'] <- "avg_response_first_10"

df_avg_response_first_10_sys_diff <- read_csv(paste0(data_path, "avg_response_first_10_sys_diff.csv"))
df_avg_response_first_10_sys_diff['role_vector'] <- "sys_diff_first_10"

df <- bind_rows(df_at_role, df_at_role_period,
                  df_avg_response, df_avg_response_first_10,
                  df_avg_response_sys_diff, df_avg_response_first_10_sys_diff)

rm(df_at_role, df_at_role_period, df_avg_response, df_avg_response_first_10, df_avg_response_sys_diff, df_avg_response_first_10_sys_diff)

words_path <- "data/selected_words.csv"
words_df <- read_csv(words_path)

# Get word groups
df <- df %>%
  left_join(words_df, by = c("role_name" = "word"))

df$animacy_group = df$group
df$group <- NULL  # Remove 'group' column to avoid conflicts with marginaleffects

# plot traces
df_avg <- df[df_avg$animacy_group != "Assistant",] %>%
  group_by(role_vector, animacy_group, steering_magnitude) %>%
  summarise(
    across(where(is.numeric), mean, na.rm = TRUE),
    .groups = "drop"
  )

df_avg_vectors <- df_avg[df_avg$animacy_group != "Assistant",] %>%
  group_by(role_vector, steering_magnitude) %>%
  summarise(
    across(where(is.numeric), mean, na.rm = TRUE),
    .groups = "drop"
  )

# steer overall
ggplot(df_avg_vectors, aes(x=steering_magnitude, y=average_log_probs , color=role_vector)) +
  geom_line(size=1.2) +
  theme_minimal() +
  coord_cartesian(xlim=c(0,8000)) +
  ggtitle("Steering efficiency of different steering vector computations",
          subtitle = "Average log probs of role guess")

# groups per steer
ggplot(df_avg[df_avg$animacy_group != "Assistant",], aes(x=steering_magnitude, y=average_log_probs, color=animacy_group)) +
  geom_line(size=1.2) +
  theme_minimal() +
  facet_wrap(vars(role_vector)) +
  coord_cartesian(xlim=c(0,8000)) +
  ggtitle("Steering efficiency of different animacy groups under different steering vectors.",
          subtitle = "Average log probs of role guess")

# Word properties vs steerability.
df_role_means <- df[(df$role_vector=='avg_response_first_10') & (df$group_id!=4),] %>%
  group_by(role_name, steering_magnitude) %>%
  summarise(
    across(where(is.numeric), mean, na.rm = TRUE),
    .groups = "drop"
  )

#' Plot Steering Trajectories with Binned Averages
#'
#' Creates a visualization showing individual role trajectories (faint lines)
#' overlaid with binned average trajectories (thick lines). The binned averages
#' are colored by the mean value of the feature within each bin.
#'
#' @param df A dataframe containing the data to plot.
#' @param feature_col Name of the feature column to bin and color by (unquoted).
#' @param outcome_col Name of the outcome column to plot on y-axis (unquoted).
#' @param bin_breaks Numeric vector defining bin boundaries. Default: c(100, 300, 500, 700).
#' @param bin_labels Character vector of bin labels. If NULL, auto-generated from breaks.
#' @param group_col Name of the grouping column for individual trajectories. Default: role_name.
#' @param steering_col Name of the steering magnitude column. Default: steering_magnitude.
#' @param feature_display_name Display name for the feature in legend and subtitle.
#' @param outcome_display_name Display name for the outcome in y-axis label.
#'
#' @return A ggplot object.
plot_steering_trajectories <- function(df,
                                       feature_col,
                                       outcome_col,
                                       bin_breaks = c(100, 300, 500, 700),
                                       bin_labels = NULL,
                                       group_col = role_name,
                                       steering_col = steering_magnitude,
                                       feature_display_name = NULL,
                                       outcome_display_name = NULL) {

  # Capture column names as strings for programmatic use
  feature_col_str <- rlang::as_name(rlang::enquo(feature_col))
  outcome_col_str <- rlang::as_name(rlang::enquo(outcome_col))
  group_col_str <- rlang::as_name(rlang::enquo(group_col))
  steering_col_str <- rlang::as_name(rlang::enquo(steering_col))

  # Set display names if not provided
  if (is.null(feature_display_name)) {
    feature_display_name <- feature_col_str
  }
  if (is.null(outcome_display_name)) {
    outcome_display_name <- outcome_col_str
  }

  # Auto-generate bin labels if not provided
  if (is.null(bin_labels)) {
    bin_labels <- paste0("(", bin_breaks[-length(bin_breaks)], "-", bin_breaks[-1], "]")
  }

  # Get the min and max of bin_breaks for filtering and color scaling
  feature_min <- min(bin_breaks)
  feature_max <- max(bin_breaks)

  # ==============================================================================
  # DATA PREPARATION FOR BINNED LAYER
  # ==============================================================================
  df_binned_summary <- df %>%
    filter(
      .data[[feature_col_str]] >= feature_min,
      .data[[feature_col_str]] <= feature_max
    ) %>%
    mutate(
      bin_range = cut(
        .data[[feature_col_str]],
        breaks = bin_breaks,
        labels = bin_labels,
        include.lowest = FALSE
      )
    ) %>%
    group_by(bin_range, .data[[steering_col_str]]) %>%
    summarise(
      avg_outcome = mean(.data[[outcome_col_str]], na.rm = TRUE),
      avg_bin_feature = mean(.data[[feature_col_str]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    filter(!is.na(bin_range))

  # Create labels for end of lines
  df_bin_labels_text <- df_binned_summary %>%
    filter(.data[[steering_col_str]] == max(.data[[steering_col_str]]))

  # ==============================================================================
  # PLOTTING
  # ==============================================================================
  p <- ggplot() +
    # Layer 1: Background Spaghetti (Individual Roles)
    geom_line(
      data = df,
      aes(
        x = .data[[steering_col_str]],
        y = .data[[outcome_col_str]],
        group = .data[[group_col_str]],
        color = .data[[feature_col_str]]
      ),
      alpha = 0.3,
      linewidth = 0.6
    ) +

    # Layer 2: Binned Averages (Thick lines)
    geom_line(
      data = df_binned_summary,
      aes(
        x = .data[[steering_col_str]],
        y = avg_outcome,
        group = bin_range,
        color = avg_bin_feature
      ),
      linewidth = 2.5
    ) +

    # Layer 3: Direct Labels for the Bins
    geom_text(
      data = df_bin_labels_text,
      aes(
        x = .data[[steering_col_str]],
        y = avg_outcome,
        label = bin_range,
        color = avg_bin_feature
      ),
      hjust = -0.1,
      fontface = "bold",
      size = 4.5,
      show.legend = FALSE
    ) +

    # Scale Definition
    scale_color_viridis_c(
      name = paste0(feature_display_name, "\n(Continuous)"),
      limits = c(feature_min, feature_max),
      oob = scales::squish
    ) +

    # Theme and Layout
    scale_x_continuous(expand = expansion(mult = c(0.05, 0.15))) +
    theme_modern() +
    theme(
      legend.position = "right",
      plot.subtitle = element_text(size = 11, color = "grey30")
    ) +
    labs(
      title = paste0("Steerability by ", feature_display_name),
      subtitle = paste0(
        "Background: Individual roles. Thick lines: Averages for specific ranges, ",
        "colored by the mean ", tolower(feature_display_name), " of that range."
      ),
      y = outcome_display_name,
      x = "Steering Magnitude"
    )

  return(p)
}

plot_steering_trajectories(df_role_means, anim_mental, average_log_probs, feature_display_name = "Mental Animacy")
plot_steering_trajectories(df_role_means, anim_physical, average_log_probs, feature_display_name = "Physical Animacy")
plot_steering_trajectories(df_role_means, thought_mean, average_log_probs, feature_display_name = "Thoughts Rating")
plot_steering_trajectories(df_role_means, repro_mean, average_log_probs, feature_display_name = "Reproduces Rating")
plot_steering_trajectories(df_role_means, person_mean, average_log_probs, feature_display_name = "Person-related Rating")
plot_steering_trajectories(df_role_means, goals_mean, average_log_probs, feature_display_name = "Has Goals Rating")
plot_steering_trajectories(df_role_means, move_mean, average_log_probs, feature_display_name = "Moves Rating")

df_role_means$log_freq <- log10(df_role_means$word_frequency)
plot_steering_trajectories(df_role_means, log_freq, average_log_probs,
                           bin_breaks = c(-0.5, 0.5, 1, 1.5, 2.5), feature_display_name = "Log10(Frequency)")


mental_plot <- plot_steering_trajectories(df_role_means, anim_mental, average_log_probs, feature_display_name = "Mental Animacy")
phys_plot <- plot_steering_trajectories(df_role_means, anim_physical, average_log_probs, feature_display_name = "Physical Animacy")

library(patchwork)
mental_plot + phys_plot


# Modeling
model_df <- df[(df$role_vector=='avg_response_first_10') & (df$group_id!=4),]
model_df <- model_df %>%
  mutate(
    # Center animacy so 0 = Average Animacy
    # We DO NOT divide by SD, preserving the unit interpretation
    anim_centered = anim_mental - mean(anim_mental),
    thought_centered = thought_mean - mean(thought_mean),
    person_centered = person_mean - mean(person_mean),
    goals_centered = goals_mean - mean(goals_mean),

  )

# goals
model_lme <- lmer(
  average_log_probs ~ steering_magnitude * goals_centered +
    (1 | role_name),
  data = model_df
)

model_parameters(model_lme) %>% print_md()

role_slopes <- model_df %>%
  group_by(role_name, goals_centered) %>%
  summarise(
    # Extract the slope (coefficient of steering_magnitude)
    steering_efficiency = coef(lm(average_log_probs ~ steering_magnitude))[2],
    baseline_perf = mean(average_log_probs[steering_magnitude == 500]),
    .groups = "drop"
  )

model_slopes <- lm(steering_efficiency ~ goals_centered + baseline_perf, data = role_slopes)
model_parameters(model_slopes) %>%
  print_md()

role_slopes %>%
  ggplot(aes(x = goals_centered + mean(model_df$goals_mean), y = steering_efficiency)) +
  geom_point(alpha = 0.6, size = 3) +
  geom_smooth(method = "lm", color = "firebrick") +
  theme_modern() +
  labs(
    title = "Role-wise Steering Efficiency vs. Goals-having",
    subtitle = "Each point is a distinct Role",
    x = "Ratings for 'has goals'",
    y = "Steering Efficiency (Slope)"
  )


# mental animacy
model_lme <- lmer(
  avg_first_50_log_probs ~ steering_magnitude * anim_centered +
    (1 | role_name),
  data = model_df
)

model_parameters(model_lme) %>% print_md()

role_slopes <- model_df %>%
  group_by(role_name, anim_centered) %>%
  summarise(
    # Extract the slope (coefficient of steering_magnitude)
    steering_efficiency = coef(lm(avg_first_50_log_probs ~ steering_magnitude))[2],
    baseline_perf = mean(avg_first_50_log_probs[steering_magnitude == 500]),
    .groups = "drop"
  )

model_slopes <- lm(steering_efficiency ~ anim_centered + baseline_perf, data = role_slopes)
model_parameters(model_slopes) %>%
  print_md()

role_slopes %>%
  ggplot(aes(x = anim_centered + mean(model_df$anim_mental), y = steering_efficiency)) +
  geom_point(alpha = 0.6, size = 3) +
  geom_smooth(method = "lm", color = "firebrick") +
  theme_modern() +
  labs(
    title = "Role-wise Steering Efficiency vs. Mental Animacy",
    subtitle = "Each point is a distinct Role",
    x = "Mental Animacy",
    y = "Steering Efficiency (Slope)"
  )
