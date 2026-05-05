# =============================================================================
# 02_preprocessing.R — Feature Engineering & Train/Test Split
# =============================================================================
# This script cleans raw data, encodes categorical variables, scales numeric
# features, and produces train/test splits used by all model scripts.
#
# Outputs (saved to data/):
#   - train_processed.rds
#   — training set (80%)
#   - test_processed.rds
#   — test set (20%)
#   - preprocessor.rds
#   — scaling parameters (for inverse transforms)
# =============================================================================

# == 0. Dependencies ===========================================================
required_packages <- c("tidyverse", "caret", "janitor")
invisible(lapply(required_packages, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}))

library(tidyverse)
library(caret)

set.seed(12)

# == 1. Load Raw Data ==========================================================
df_raw <- read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
                   show_col_types = FALSE)

# == 2. Cleaning ===============================================================
df <- df_raw %>%
  # Remove customer ID
  select(-customerID) %>%

  # Fix TotalCharges: blank entries are new customers (tenure = 0);
  mutate(TotalCharges = replace_na(TotalCharges, 0)) %>%

  # Convert target variable to factor (Yes = churned)
  mutate(Churn = factor(Churn, levels = c("No", "Yes")))

cat("Cleaned dimensions:", nrow(df), "rows x", ncol(df), "cols\n")

# == 3. Identify Variable Types ================================================
# numeric features that will be standardized
numeric_features <- c("tenure", "MonthlyCharges", "TotalCharges")

# binary categorical features encode as 0/1
binary_features <- c("gender", "SeniorCitizen", "Partner", "Dependents",
                     "PhoneService", "PaperlessBilling")

# multi-level categorical features one-hot encode
multi_features  <- c("MultipleLines", "InternetService", "OnlineSecurity",
                     "OnlineBackup", "DeviceProtection", "TechSupport",
                     "StreamingTV", "StreamingMovies", "Contract",
                     "PaymentMethod")

# == 4. Encode Binary Features (0 / 1 ) ========================================
# gender: female = 0, male = 1
df <- df %>%
  mutate(
    gender = if_else(gender == "Male", 1L, 0L),
    SeniorCitizen = if_else(SeniorCitizen == "Yes", 1L, 0L),
    Partner = if_else(Partner == "Yes", 1L, 0L),
    Dependents = if_else(Dependents == "Yes", 1L, 0L),
    PhoneService = if_else(PhoneService == "Yes", 1L, 0L),
    PaperlessBilling = if_else(PaperlessBilling == "Yes", 1L, 0L)
  )

# == 5. One-Hot Encode =========================================================
# using model.matrix drops the reference level and avoids the dummy variable trap
df_encoded <- df %>%
  select(-Churn) %>%  # temporarily remove target
  {
    dummies <- model.matrix(~ . - 1, data = select(., all_of(multi_features)))
    # Clean up column names
    colnames(dummies) <- janitor::make_clean_names(colnames(dummies),
                                                   allow_dupes = FALSE)
    bind_cols(
      select(., -all_of(multi_features)),
      as_tibble(dummies)
    )
  }

# Re-attach target
df_encoded <- bind_cols(df_encoded, Churn = df$Churn)


# == 6. Train / Test Split (80 / 20, stratified by Churn) ======================
train_idx <- createDataPartition(df_encoded$Churn,
                                 p    = 0.80,
                                 list = FALSE)

train_raw <- df_encoded[ train_idx, ]
test_raw <- df_encoded[-train_idx, ]

cat(
  sprintf("Training set: %d rows\n", nrow(train_raw)),
  sprintf("Test set: %d rows\n", nrow(test_raw)),
  sprintf("Train churn %%: %.1f%%\n",
          mean(train_raw$Churn == "Yes") * 100),
  sprintf("Test churn %%: %.1f%%\n",
          mean(test_raw$Churn == "Yes") * 100)
)

# == 7. Scale Numeric Features =================================================
# fit scaling parameters on training data only to prevent data leakage.
pre_proc <- preProcess(train_raw[, numeric_features],
                       method = c("center", "scale"))

train_processed <- train_raw
test_processed  <- test_raw

train_processed[, numeric_features] <- predict(pre_proc,
                                               train_raw[, numeric_features])
test_processed[, numeric_features] <- predict(pre_proc,
                                               test_raw[, numeric_features])

# == 8. Save Outputs ===========================================================
saveRDS(train_processed, "data/train_processed.rds")
saveRDS(test_processed, "data/test_processed.rds")
saveRDS(pre_proc, "data/preprocessor.rds")

cat(
  "\nPreprocessing complete.\n",
  "Feature count:", ncol(train_processed) - 1, "\n",
  "Outputs saved to data/\n\n",
  "Next: run 03_models.R\n"
)
