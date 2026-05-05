# =============================================================================
# 04_ab_testing.R -- A/B Testing Simulation
# =============================================================================
#
# Research Question:
#   "Does targeting high-risk customers using ML reduce churn more effectively
#    than a random retention strategy?"
#
# Experimental Design:
#   We simulate a retention campaign where the company can afford to offer
#   incentives to a fixed budget of N customers. We compare:
#
#   GROUP A (ML-Targeted):  Customers ranked by predicted churn probability
#                           — the model selects the top-risk individuals.
#
#   GROUP B (Random):       The same number of customers chosen at random,
#                           regardless of predicted churn risk.
#
#   Outcome:  Of those contacted, how many were TRUE churners?
#             (Churn rate among targeted vs. randomly selected)
#
# NOTE: In a real experiment, the "outcome" (actual churn) would be observed
# after the campaign runs. Here we use the TEST SET (held out, never seen by
# the model during training) to simulate observed outcomes.
# =============================================================================

# == 0. Dependencies ===========================================================
required_packages <- c("tidyverse", "ggplot2", "scales", "patchwork", "boot")
invisible(lapply(required_packages, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}))

library(tidyverse)
library(ggplot2)
library(scales)
library(patchwork)
library(boot)

set.seed(12)

plots_path <- "plots/"
dir.create(plots_path, showWarnings = FALSE)

# == 1. Load Model Results =====================================================
comparison <- readRDS("data/model_comparison.rds")
test <- readRDS("data/test_processed.rds")

# Use the best model's predicted probabilities
all_aucs <- sapply(list(
  comparison$results_lr,
  comparison$results_rf,
  comparison$results_svm
), `[[`, "auc")

best_results <- list(
  comparison$results_lr,
  comparison$results_rf,
  comparison$results_svm
)[[which.max(all_aucs)]]

cat("Using model:", best_results$model, "\n")

# Build the experimental dataset
experiment_df <- tibble(
  customer_id = seq_len(nrow(test)),
  actual_churn = test$Churn,
  churn_prob = best_results$probs,
  monthly_charges = test$MonthlyCharges
)

cat(sprintf("Test set: %d customers | Actual churn rate: %.1f%%\n",
            nrow(experiment_df),
            mean(experiment_df$actual_churn == "Yes") * 100))

# == 2. Define Campaign Parameters =============================================
# Suppose marketing can reach 30% of the test cohort (budget constraint).
# In practice this would be set based on cost per outreach vs expected LTV.
CAMPAIGN_BUDGET_PCT <- 0.30
n_targeted <- floor(nrow(experiment_df) * CAMPAIGN_BUDGET_PCT)

cat(sprintf("\nCampaign budget: %d customers (%.0f%% of test set)\n",
            n_targeted, CAMPAIGN_BUDGET_PCT * 100))

# == 3. Assign Group A (ML-Targeted) ==========================================
# Sort by descending churn probability - contact the highest-risk customers
group_a_ids <- experiment_df %>%
  arrange(desc(churn_prob)) %>%
  slice_head(n = n_targeted) %>%
  pull(customer_id)

# == 4. Assign Group B (Random) ================================================
# Sample uniformly at random - this is the baseline "no model"
group_b_ids <- sample(
  experiment_df$customer_id[!experiment_df$customer_id %in% group_a_ids],
  size = n_targeted,
  replace = FALSE
)

# == 5. Tag Groups & Compute Observed Outcomes =================================
experiment_df <- experiment_df %>%
  mutate(
    group = case_when(
      customer_id %in% group_a_ids ~ "A_ML_Targeted",
      customer_id %in% group_b_ids ~ "B_Random",
      TRUE ~ "Control"   # not contacted
    )
  )

groups_ab <- experiment_df %>%
  filter(group %in% c("A_ML_Targeted", "B_Random"))

# Churn counts per group
group_summary <- groups_ab %>%
  group_by(group) %>%
  summarise(
    n = n(),
    churners = sum(actual_churn == "Yes"),
    churn_rate = mean(actual_churn == "Yes"),
    avg_prob = mean(churn_prob)
  )

print(group_summary)

# == 6. Statistical Tests =====================================================
n_A <- group_summary$n[group_summary$group == "A_ML_Targeted"]
n_B <- group_summary$n[group_summary$group == "B_Random"]
hit_A <- group_summary$churners[group_summary$group == "A_ML_Targeted"]
hit_B <- group_summary$churners[group_summary$group == "B_Random"]
p_A <- hit_A / n_A   # churn rate in ML group
p_B <- hit_B / n_B   # churn rate in random group

cat("\n--- Churn Rates ---\n",
    sprintf("Group A (ML): %.1f%% (%d / %d)\n", p_A * 100, hit_A, n_A),
    sprintf("Group B (Random): %.1f%% (%d / %d)\n", p_B * 100, hit_B, n_B),
    sprintf("Absolute diff: %+.1f pp\n", (p_A - p_B) * 100)
    )

# 6a. Two-proportion Z-test
# H0: p_A = p_B   |   H1: p_A > p_B (ML finds more churners)
prop_test <- prop.test(
  x = c(hit_A, hit_B),
  n = c(n_A,   n_B),
  alternative = "greater",
  correct = FALSE
)

cat("\n--- Two-Proportion Z-test ---\n",
    sprintf("Z-statistic (approx): %.3f\n",
            sqrt(prop_test$statistic)),
    sprintf("p-value: %.4f\n", prop_test$p.value),
    sprintf("95%% CI on diff: [%.4f, %.4f]\n",
            prop_test$conf.int[1], prop_test$conf.int[2])
    )

# 6b. Chi-square test of independence
contingency_tbl <- matrix(
  c(hit_A, n_A - hit_A,
    hit_B, n_B - hit_B),
  nrow = 2, byrow = TRUE,
  dimnames = list(Group = c("ML", "Random"),
                  Outcome = c("Churned", "Stayed"))
)

chi_test <- chisq.test(contingency_tbl, correct = FALSE)

cat("\n--- Chi-Square Test ---\n",
    sprintf("Chi-sq = %.3f, df = %d, p = %.4f\n",
            chi_test$statistic, chi_test$parameter, chi_test$p.value)
    )

# 6c. Fisher's Exact Test (robust for any cell sizes)
fisher_test <- fisher.test(contingency_tbl, alternative = "greater")

cat("\n--- Fisher's Exact Test ---\n",
    sprintf("Odds Ratio: %.3f\n", fisher_test$estimate),
    sprintf("p-value: %.4f\n", fisher_test$p.value)
    )

# 6d. Relative Risk & Number Needed to Target (NNT)
# RR > 1 means ML group contacts proportionally more true churners
rr <- p_A / p_B
ard <- p_A - p_B # Absolute Risk Difference
nnt <- 1 / ard # Number Needed to Target (to find 1 extra churner)

cat("\n--- Effect Size Measures ---\n",
    sprintf("Relative Risk (RR): %.3f\n", rr),
    sprintf("Absolute Risk Diff (ARD): %.3f (%.1f pp)\n", ard, ard * 100),
    sprintf("Number Needed to Target: %.1f\n", nnt)
    )

# 6e. Bootstrap CI on Absolute Risk Difference
boot_ard <- function(data, i) {
  d <- data[i, ]
  mean(d$actual_churn[d$group == "A_ML_Targeted"] == "Yes") -
    mean(d$actual_churn[d$group == "B_Random"] == "Yes")
}

boot_result <- boot(data = groups_ab, statistic = boot_ard, R = 2000)
boot_ci <- boot.ci(boot_result, type = "perc", conf = 0.95)

cat("\n--- Bootstrap 95% CI on ARD (2000 resamples) ---\n",
    sprintf("ARD = %.3f, 95%% CI: [%.3f, %.3f]\n",
            boot_result$t0,
            boot_ci$percent[4],
            boot_ci$percent[5])
    )

# == 7. Lift Analysis ==========================================================
# Lift = how many times better is the ML strategy vs. random at identifying
# true churners at each percentile of the ranked list.

lift_df <- experiment_df %>%
  arrange(desc(churn_prob)) %>%
  mutate(
    rank_pct = row_number() / n() * 100,
    is_churner = as.integer(actual_churn == "Yes")
  ) %>%
  mutate(
    cum_churners = cumsum(is_churner),
    cum_pct_reached = row_number() / n(),
    lift = (cumsum(is_churner) / row_number()) / mean(actual_churn == "Yes")
  )

# == 8. Visualizations ========================================================

# 8a. Churn rate comparison bar chart
plot_df <- group_summary %>%
  filter(group != "Control") %>%
  mutate(group_label = if_else(group == "A_ML_Targeted",
                               "Group A\n(ML-Targeted)",
                               "Group B\n(Random)"))

p_bar <- ggplot(plot_df, aes(x = group_label, y = churn_rate, fill = group)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = percent(churn_rate, 0.1)),
            vjust = -0.5, size = 5.5, fontface = "bold") +
  scale_fill_manual(values = c("A_ML_Targeted" = "#E85D5D",
                               "B_Random" = "#4C9BE8")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
  labs(title = "Churn Rate: ML-Targeted vs. Random",
       subtitle = sprintf("Campaign reach = %.0f%% of customers | n = %d per group",
                          CAMPAIGN_BUDGET_PCT * 100, n_targeted),
       x = NULL, y = "Churn Rate Among Contacted") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold"))

ggsave(paste0(plots_path, "13_ab_churn_rate.png"),
       p_bar, width = 6, height = 5, dpi = 150)
print(p_bar)

# 8b. Predicted probability distribution by group
p_dist <- ggplot(groups_ab,
                 aes(x = churn_prob, fill = group, color = group)) +
  geom_density(alpha = 0.4, size = 1) +
  scale_fill_manual(values = c("A_ML_Targeted" = "#E85D5D",
                                "B_Random"       = "#4C9BE8"),
                    labels = c("Group A (ML)", "Group B (Random)")) +
  scale_color_manual(values = c("A_ML_Targeted" = "#C02020",
                                "B_Random"       = "#1A5DB5"),
                     labels = c("Group A (ML)", "Group B (Random)")) +
  labs(title = "Predicted Churn Probability Distribution",
       subtitle = "ML group is concentrated in the high-risk tail",
       x = "Predicted P(Churn)", y = "Density",
       fill = NULL, color = NULL) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "top")

ggsave(paste0(plots_path, "14_ab_prob_distribution.png"),
       p_dist, width = 7, height = 5, dpi = 150)
print(p_dist)

# 8c. Lift Curve
p_lift <- ggplot(lift_df, aes(x = rank_pct, y = lift)) +
  geom_line(color = "#E85D5D", size = 1.2) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey50") +
  annotate("rect",
           xmin = 0, xmax = CAMPAIGN_BUDGET_PCT * 100,
           ymin = -Inf, ymax = Inf,
           fill = "#E85D5D", alpha = 0.08) +
  annotate("text",
           x = CAMPAIGN_BUDGET_PCT * 50,
           y = max(lift_df$lift) * 0.9,
           label = sprintf("Campaign\nwindow\n(top %.0f%%)",
                           CAMPAIGN_BUDGET_PCT * 100),
           color = "#C02020", size = 3.5, fontface = "italic") +
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  labs(title = "Cumulative Lift Curve",
       subtitle = "Lift > 1 means model outperforms random targeting",
       x = "% of Customers Contacted (ranked by risk)",
       y = "Lift over Random") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(paste0(plots_path, "15_lift_curve.png"),
       p_lift, width = 7, height = 5, dpi = 150)
print(p_lift)

# 8d. Cumulative Gains Chart
total_churners <- sum(experiment_df$actual_churn == "Yes")

gains_df <- lift_df %>%
  mutate(
    cum_pct_churners = cum_churners / total_churners * 100,
    random_line = rank_pct  # random = diagonal
  )

p_gains <- ggplot(gains_df, aes(x = rank_pct)) +
  geom_line(aes(y = cum_pct_churners, color = "ML Model"), size = 1.2) +
  geom_line(aes(y = random_line, color = "Random"),
            linetype = "dashed", size = 0.9) +
  scale_color_manual(values = c("ML Model" = "#E85D5D", "Random" = "grey60")) +
  annotate("rect",
           xmin = 0, xmax = CAMPAIGN_BUDGET_PCT * 100,
           ymin = -Inf, ymax = Inf,
           fill = "#E85D5D", alpha = 0.07) +
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_y_continuous(labels = function(y) paste0(y, "%")) +
  labs(title = "Cumulative Gains Chart",
       subtitle = "What % of all churners are captured by targeting top N%?",
       x = "% of Customers Contacted", y = "% of All Churners Captured",
       color = NULL) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "top")

ggsave(paste0(plots_path, "16_gains_chart.png"),
       p_gains, width = 7, height = 5, dpi = 150)
print(p_gains)

# == 9. Final Summary Report ===================================================
cat("============ A/B Testing Summary ============\n",
    sprintf("Model used:              %s\n",       best_results$model),
    sprintf("Campaign budget:         %.0f%%  (%d customers)\n",
            CAMPAIGN_BUDGET_PCT * 100, n_targeted),
    sprintf("Group A churn rate (ML): %.1f%%\n",   p_A * 100),
    sprintf("Group B churn rate (RNG):%.1f%%\n",   p_B * 100),
    sprintf("Absolute improvement:    +%.1f pp\n", (p_A - p_B) * 100),
    sprintf("Relative Risk:           %.2fx\n",    rr),
    sprintf("Number Needed to Target: %.1f\n",     nnt),
    sprintf("Z-test p-value:          %.4f %s\n",  prop_test$p.value,
            if (prop_test$p.value < 0.05) "*** SIGNIFICANT" else "(not significant)"),
    sprintf("Chi-square p-value:      %.4f\n",     chi_test$p.value),
    sprintf("Bootstrap 95%% CI (ARD):  [%.3f, %.3f]\n",
            boot_ci$percent[4], boot_ci$percent[5])
    )

lift_at_budget <- lift_df %>%
  filter(rank_pct <= CAMPAIGN_BUDGET_PCT * 100) %>%
  slice_tail(n = 1) %>%
  pull(lift)

cat(sprintf("Lift at %.0f%% depth: %.2fx\n",
            CAMPAIGN_BUDGET_PCT * 100, lift_at_budget))
cat("\nAll A/B plots saved to plots/\n")
