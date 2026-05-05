# =============================================================================
# 01_eda.R — Exploratory Data Analysis
# =============================================================================
# Dataset: IBM Telco Customer Churn
# Source:  https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data
#
# This script loads the raw data, inspects its structure, and produces
# visualizations that characterize churn patterns across customer segments.
# All plots are saved to plots/ for use in the final presentation.
# =============================================================================

# ============== 0. dependencies ==============
# install missing packages before loading

required_packages <- c("tidyverse", "ggplot2", "scales", "gridExtra",
                       "RColorBrewer", "patchwork", "ggcorrplot")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}

invisible(lapply(required_packages, install_if_missing))

library(tidyverse)
library(ggplot2)
library(ggcorrplot)
library(scales)
library(gridExtra)
library(corrplot)
library(RColorBrewer)
library(patchwork)

# ============== 1. Load data ==============
# place .csv from Kaggle into the data/ folder

data_path <- "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
plots_path <- "plots/"
dir.create(plots_path, showWarnings = FALSE)

df_raw <- read_csv(data_path, show_col_types = FALSE)

cat("Dataset dimensions: ", nrow(df_raw), "rows x", ncol(df_raw), "columns\n")
glimpse(df_raw)

# ============== 2. Data QC ==============
missing_summary <- df_raw %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "Column", values_to = "Missing") %>%
  filter(Missing > 0)
print(missing_summary)

# TotalCharges has blank entries ( new customers with 0 tenure)
blank_total_charges <- sum(is.na(df_raw$TotalCharges))
cat("\nBlank TotalCharges entries (new customers): ", blank_total_charges, "\n")

#convert TotalCharges to numeric
df <- df_raw %>%
  mutate(
    TotalCharges = as.numeric(TotalCharges),
    Churn = factor(Churn, levels = c("No", "Yes"))
  )

# ============== 3. Overall Churn rate ==============
churn_counts <- df %>% count(Churn)
churn_rate <- mean(df$Churn == "Yes", na.rm = TRUE)
cat(sprintf("\nOverall churn rate: %.1f%%\n", churn_rate*100))

p_churn_bar <- ggplot(churn_counts, aes(x = Churn, y = n, fill = Churn)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = paste0(n, "\n(", percent(n / sum(n), 1), ")")),
            vjust = -0.3, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = c("No" = "#4C9BE8", "Yes" = "#E85D5D")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Customer Churn Distribution",
       subtitle = sprintf("Overall churn rate: %.1f%%", churn_rate * 100),
       x = "Churned?",
       y = "Number of Customers") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(paste0(plots_path, "01_churn_distribution.png"),
       p_churn_bar, width = 6, height = 5, dpi = 150)
print(p_churn_bar)

# ============== 4. Churn by contract type ==============
p_contract <- df %>%
  count(Contract, Churn) %>%
  group_by(Contract) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = Contract, y = pct, fill = Churn)) +
  geom_col(position = "fill", width = 0.6) +
  geom_text(aes(label = ifelse(Churn == "Yes", percent(pct, 1), "")),
            position = position_fill(vjust = 0.5), size = 4, color = "white",
            fontface = "bold") +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(values = c("No" = "#4C9BE8", "Yes" = "#E85D5D")) +
  labs(title = "Churn Rate by Contract Type",
       x = "Contract Type", y = "Proportion", fill = "Churned") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(paste0(plots_path, "02_churn_by_contract.png"),
       p_contract, width = 7, height = 5, dpi = 150)
print(p_contract)

# ============== 5. Churn by tenure ==============
p_tenure <- ggplot(df, aes(x = tenure, fill = Churn)) +
  geom_histogram(binwidth = 3, position = "identity", alpha = 0.65, color = "white") +
  scale_fill_manual(values = c("No" = "#4C9BE8", "Yes" = "#E85D5D")) +
  labs(title = "Tenure Distribution by Churn Status",
       subtitle = "Churned customers tend to leave early",
       x = "Tenure (months)", y = "Count", fill = "Churned") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(paste0(plots_path, "03_churn_by_tenure.png"),
       p_tenure, width = 8, height = 5, dpi = 150)
print(p_tenure)

# ============== 6. Churn by monthly charges ==============
p_monthly <- ggplot(df, aes(x = Churn, y = MonthlyCharges, fill = Churn)) +
  geom_violin(trim = FALSE, alpha = 0.7, show.legend = FALSE) +
  geom_boxplot(width = 0.1, fill = "white", outlier.size = 1, show.legend = FALSE) +
  scale_fill_manual(values = c("No" = "#4C9BE8", "Yes" = "#E85D5D")) +
  labs(title = "Monthly Charges by Churn Status",
       subtitle = "Higher charges are associated with higher churn",
       x = "Churned?", y = "Monthly Charges ($)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(paste0(plots_path, "04_churn_by_monthly_charges.png"),
       p_monthly, width = 6, height = 5, dpi = 150)
print(p_monthly)

# ============== 7. Churn by internet service ==============
p_internet <- df %>%
  count(InternetService, Churn) %>%
  group_by(InternetService) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = InternetService, y = pct, fill = Churn)) +
  geom_col(position = "fill", width = 0.6) +
  geom_text(aes(label = ifelse(Churn == "Yes", percent(pct, 1), "")),
            position = position_fill(vjust = 0.5), size = 4, color = "white",
            fontface = "bold") +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(values = c("No" = "#4C9BE8", "Yes" = "#E85D5D")) +
  labs(title = "Churn Rate by Internet Service Type",
       x = "Internet Service", y = "Proportion", fill = "Churned") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(paste0(plots_path, "05_churn_by_internet.png"),
       p_internet, width = 7, height = 5, dpi = 150)
print(p_internet)

# ============== 8. Churn by payment method ==============
p_payment <- df %>%
  count(PaymentMethod, Churn) %>%
  group_by(PaymentMethod) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = reorder(PaymentMethod, -pct * (Churn == "Yes")),
             y = pct, fill = Churn)) +
  geom_col(position = "fill", width = 0.6) +
  geom_text(aes(label = ifelse(Churn == "Yes", percent(pct, 1), "")),
            position = position_fill(vjust = 0.5), size = 4, color = "white",
            fontface = "bold") +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(values = c("No" = "#4C9BE8", "Yes" = "#E85D5D")) +
  labs(title = "Churn Rate by Payment Method",
       x = NULL, y = "Proportion", fill = "Churned") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 15, hjust = 1),
        plot.title = element_text(face = "bold"))

ggsave(paste0(plots_path, "06_churn_by_payment.png"),
       p_payment, width = 8, height = 5, dpi = 150)
print(p_payment)

# ============== 9. Numeric feature correlations ==============
cor_matrix <- cor(num_vars)

p_corr <- ggcorrplot(cor_matrix, lab = TRUE,
                     colors = c("#E85D5D", "white", "#4C9BE8")) +
  labs(title = "Numeric Feature Correlations") +
  theme(plot.title = element_text(face = "bold"))

ggsave(paste0(plots_path, "07_correlation_matrix.png"),
       p_corr, width = 6, height = 5, dpi = 150)
print(p_corr)

# ============== 10. multi-panel service features v. churn ==============
service_cols <- c("PhoneService", "MultipleLines", "OnlineSecurity",
                  "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies")

service_churn <- df %>%
  select(Churn, all_of(service_cols)) %>%
  pivot_longer(-Churn, names_to = "Service", values_to = "Status") %>%
  count(Service, Status, Churn) %>%
  group_by(Service, Status) %>%
  mutate(pct = n / sum(n)) %>%
  filter(Churn == "Yes", Status == "Yes")

p_services <- ggplot(service_churn,
                     aes(x = reorder(Service, pct), y = pct, fill = pct)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = percent(pct, 1)), hjust = -0.1, size = 3.5) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 0.55)) +
  scale_fill_gradient(low = "#4C9BE8", high = "#E85D5D") +
  coord_flip() +
  labs(title = "Churn Rate Among Subscribers of Each Service",
       subtitle = "Customers WITH the service who churned",
       x = NULL, y = "Churn Rate") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(paste0(plots_path, "08_churn_by_services.png"),
       p_services, width = 8, height = 5, dpi = 150)
print(p_services)

# ============== 11. summary stats table ==============
df %>%
  group_by(Contract) %>%
  summarise(churn_rate = mean(Churn == "Yes"),
            n = n()) %>%
  arrange(desc(churn_rate)) %>%
  print()

df %>%
  group_by(InternetService) %>%
  summarise(churn_rate = mean(Churn == "Yes"),
            n = n()) %>%
  arrange(desc(churn_rate)) %>%
  print()

cat("\nEDA complete. Plots saved to", plots_path, "\n")
cat("\nNext: run 02_preprocessing.R\n")

