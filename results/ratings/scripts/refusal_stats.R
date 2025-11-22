library(tidyverse)
library(lme4)
library(lmerTest)
library(easystats)

# Load data
data_path <- "results/ratings/data/Qwen3-30B-A3B-Instruct-2507/response_ratings.csv"
df <- read_csv(data_path)

# Load word characteristics
words_path <- "data/selected_words.csv"
words_df <- read_csv(words_path)

# Preprocessing
df <- df %>%
  mutate(
    across(c(assistant_refusal, role_refusal, identify_as_assistant, deny_internal_experience), as.logical),
    task_name = as.factor(task_name),
    role_name = as.factor(role_name)
  )

# Merge in word characteristics
df <- df %>%
  left_join(words_df, by = c("role_name" = "word"))

df$animacy_group = df$group
df$group <- NULL  # Remove 'group' column to avoid conflicts with marginaleffects

# Rescale animacy predictors (z-score standardization)
df <- df %>%
  mutate(
    anim_mental = scale(anim_mental)[,1],
    anim_physical = scale(anim_physical)[,1]
  )

df_role_agg <- df %>%
  group_by(role_name, broad_category) %>%
  summarize(
    anim_mental = mean(anim_mental),
    anim_physical = mean(anim_physical),
    across(c(assistant_refusal, role_refusal, identify_as_assistant, deny_internal_experience), mean)
  )

# Define outcomes
outcomes <- c("assistant_refusal", "role_refusal", "identify_as_assistant", "deny_internal_experience")

# Loop through outcomes and fit models
results_list <- list()

for (outcome in outcomes) {
  cat(paste0("\n\nAnalyzing outcome: ", outcome, "\n"))

  f <- as.formula(paste(outcome, "~ anim_mental + anim_physical + (1|task_name) + (1|role_name)"))

  # Fit GLMM
  # Using nAGQ=0 for speed if needed, but default is usually fine for this size unless it fails to converge
  model <- glmer(f, data = df, family = binomial)

  # Store model
  results_list[[outcome]] <- model

  # 1. Model Summary (Parameters)
  cat("\n--- Model Parameters (Odds Ratios) ---\n")
  print(model_parameters(model, exponentiate = TRUE))

  # 2. Model Performance (R2, ICC)
  cat("\n--- Model Performance ---\n")
  print(model_performance(model))

  # 3. plot mental animacy
  cat("\n--- Estimated Probabilities per Task ---\n")
  means <- estimate_means(model, by = "anim_mental")
  print(means)

  p <- plot(means) +
    labs(title = paste("Predicted Probability of", outcome, "by mental animacy"),
         y = "Probability of role departure", x = "Mental Animacy") +
    geom_point(data=df_role_agg, aes(x=anim_mental, y=.data[[outcome]], color = broad_category)) +
    theme_modern()

  print(p)

  # 4. plot physical animacy
  means <- estimate_means(model, by = "anim_physical")
  print(means)

  p <- plot(means) +
    labs(title = paste("Predicted Probability of", outcome, "by phsyical animacy"),
         y = "Probability of role departure", x = "Physical Animacy") +
    geom_point(data=df_role_agg, aes(x=anim_physical, y=.data[[outcome]], color = broad_category)) +
    theme_modern()

  print(p)
}


for (outcome in outcomes) {
  cat(paste0("\n\nAnalyzing outcome: ", outcome, "\n"))

  # Formula: outcome ~ task_name + (1|role_name)
  f <- as.formula(paste(outcome, "~ anim_mental  + (anim_mental |task_name) + (1|role_name)"))

  # Fit GLMM
  # Using nAGQ=0 for speed if needed, but default is usually fine for this size unless it fails to converge
  model <- glmer(f, data = df, family = binomial)

  # Store model
  results_list[[outcome]] <- model

  # 1. Model Summary (Parameters)
  cat("\n--- Model Parameters (Odds Ratios) ---\n")
  print(model_parameters(model, exponentiate = TRUE))

  # 2. Model Performance (R2, ICC)
  cat("\n--- Model Performance ---\n")
  print(model_performance(model))

  # 4. plot mental animacy by broad category
  means <- estimate_means(model, by = c("anim_mental"))
  print(means)

  p <- plot(means) +
    geom_point(data=df_role_agg, aes(x=anim_mental, y=.data[[outcome]], color = broad_category)) +
    labs(title = paste("Predicted Probability of", outcome, "by mental animacy"),
         y = "Probability of role departure", x = "Mental Animacy") +
    theme_modern()

  print(p)

  # 5. plot effects within each task_name
  cat("\n--- Estimated Probabilities by Task ---\n")
  means_by_task <- estimate_means(model, by = c("task_name", "anim_mental"))
  print(means_by_task)

  p_task <- ggplot(means_by_task, aes(x = anim_mental, y = Probability, color = task_name, fill = task_name)) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(ymin = CI_low, ymax = CI_high), alpha = 0.2, color = NA) +
    facet_wrap(~ task_name) +
    labs(title = paste("Predicted Probability of", outcome, "by task and mental animacy"),
         y = "Probability of role departure", x = "Mental Animacy",
         color = "Category", fill = "Category") +
    theme_modern()

  print(p_task)
}


for (outcome in outcomes) {
  cat(paste0("\n\nAnalyzing outcome: ", outcome, "\n"))

  # Formula: outcome ~ task_name + (1|role_name)
  f <- as.formula(paste(outcome, "~ anim_physical  + (anim_physical |task_name) + (1|role_name)"))

  # Fit GLMM
  # Using nAGQ=0 for speed if needed, but default is usually fine for this size unless it fails to converge
  model <- glmer(f, data = df, family = binomial)

  # Store model
  results_list[[outcome]] <- model

  # 1. Model Summary (Parameters)
  cat("\n--- Model Parameters (Odds Ratios) ---\n")
  print(model_parameters(model, exponentiate = TRUE))

  # 2. Model Performance (R2, ICC)
  cat("\n--- Model Performance ---\n")
  print(model_performance(model))

  # 4. plot physical animacy
  means <- estimate_means(model, by = c("anim_physical"))
  print(means)

  p <- plot(means) +
    labs(title = paste("Predicted Probability of", outcome, "by phsyical animacy"),
         y = "Probability of role departure", x = "Physical Animacy") +
    geom_point(data=df_role_agg, aes(x=anim_physical, y=.data[[outcome]], color = broad_category)) +
    theme_modern()

  print(p)

  # 5. plot effects within each task_name
  cat("\n--- Estimated Probabilities by Task ---\n")
  means_by_task <- estimate_means(model, by = c("task_name", "anim_physical"))
  print(means_by_task)

  p_task <- ggplot(means_by_task, aes(x = anim_physical, y = Probability, color = task_name, fill = task_name)) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(ymin = CI_low, ymax = CI_high), alpha = 0.2, color = NA) +
    facet_wrap(~ task_name) +
    labs(title = paste("Predicted Probability of", outcome, "by task and physical animacy"),
         y = "Probability of role departure", x = "Physical Animacy",
         color = "Category", fill = "Category") +
    theme_modern()

  print(p_task)
}

df_role_agg %>%
  ggplot(aes(x=identify_as_assistant, y=deny_internal_experience, color = broad_category)) +
    geom_point() +
    geom_text(data = . %>% filter(deny_internal_experience > 0.3),
              aes(label = role_name),
              vjust = -0.5, hjust = 0.5, size = 4) +
    labs(title = "Denial of Internal Experience and Assistant Identification",
         x = "Identifies as Assistant",
         y = "Deny Internal Experience",
         color = "Broad Category") +
    theme_minimal()
