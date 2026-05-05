# =============================================================================
# 03_models.R — Model Training, Evaluation & Comparison
# =============================================================================
# Models trained:
#   1. Logistic Regression (baseline)
#   2. Random Forest
#   3. Support Vector Machine (RBF kernel)
#
# All models are tuned via 5-fold cross-validation on the training set.
# Evaluation uses: Accuracy, AUC-ROC, Precision, Recall, F1, Brier Score.
#
# Outputs saved to:
#   data/model_lr.rds, data/model_rf.rds, data/model_svm.rds
#   data/model_comparison.rds -- summary table + predictions
#   plots/09_roc_curves.png
#   plots/10_confusion_matrices.png
#   plots/11_feature_importance.png
#   plots/12_model_comparison.png
# =============================================================================

# == 0. Dependencies ==========================================================
required_packages <- c("tidyverse", "caret", "randomForest", "e1071",
                       "pROC", "ggplot2", "patchwork", "scales", "kernlab")
invisible(lapply(required_packages, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}))

library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(ggplot2)
library(patchwork)
library(scales)
library(kernlab)

set.seed(12)

plots_path <- "plots/"
dir.create(plots_path, showWarnings = FALSE)

# == 1. Load Preprocessed Data  ===============================================
train <- readRDS("data/train_processed.rds")
test <- readRDS("data/test_processed.rds")

X_train <- select(train, -Churn)
y_train <- train$Churn
X_test <- select(test,  -Churn)
y_test <- test$Churn

cat("Training samples:", nrow(train), " Test samples:", nrow(test), "\n")

# == 2. Cross-Validation Control  =============================================
# 5-fold CV, optimizing for ROC (AUC).
ctrl <- trainControl(
  method = "cv", number = 5, classProbs = TRUE,
  summaryFunction = twoClassSummary, savePredictions = "final"
)

# == 3. Logistic Regression  ==================================================

model_lr <- train(
  x = X_train, y = y_train,
  method = "glm", family = "binomial",
  trControl = ctrl, metric = "ROC"
)

cat("LR CV AUC:", round(max(model_lr$results$ROC), 4), "\n")
saveRDS(model_lr, "data/model_lr.rds")

# == 4. Random Forest  ========================================================
# Tune mtry (# vars sampled at each split) over a small grid.
rf_grid <- expand.grid(mtry = c(3, 5, 7, 10))

model_rf <- train(
  x = X_train, y = y_train, method = "rf",
  ntree = 500, tuneGrid = rf_grid, trControl = ctrl,
  metric = "ROC", importance = TRUE
)

cat("RF CV AUC (best mtry =", model_rf$bestTune$mtry, "):",
    round(max(model_rf$results$ROC), 4), "\n")
saveRDS(model_rf, "data/model_rf.rds")

# == 5. Support Vector Machine (Radial Basis Function Kernel) =================
# Tune C (regularization) and sigma (kernel width).
# using a small grid to keep runtime reasonable.

svm_grid <- expand.grid(
  C = c(0.1, 1, 10),
  sigma = c(0.01, 0.05, 0.1)
)

model_svm <- train(
  x = X_train, y = y_train, method = "svmRadial",
  tuneGrid = svm_grid, trControl = ctrl, metric = "ROC",
  preProcess = NULL
)

cat("SVM CV AUC (best C =", model_svm$bestTune$C,
    ", sigma =", model_svm$bestTune$sigma, "):",
    round(max(model_svm$results$ROC), 4), "\n")
saveRDS(model_svm, "data/model_svm.rds")

# == 6. Evaluation function  ===================================================
# returns a named list of metrics given predicted probs and true labels
evaluate_model <- function(model, X, y, model_name) {
  probs <- predict(model, X, type = "prob")[, "Yes"]
  preds <- predict(model, X)

  cm <- confusionMatrix(preds, y, positive = "Yes")
  roc <- roc(as.numeric(y == "Yes"), probs, quiet = TRUE)

  # Brier Score
  brier <- mean((probs - as.numeric(y == "Yes"))^2)

  list(
    model = model_name,
    accuracy = cm$overall["Accuracy"],
    auc = as.numeric(auc(roc)),
    precision = cm$byClass["Precision"],
    recall = cm$byClass["Recall"],
    f1 = cm$byClass["F1"],
    brier = brier,
    confusion = cm$table,
    roc_obj = roc,
    probs = probs,
    preds = preds
  )
}

# == 7. Evaluate All Models on Test Set  ======================================
results_lr <- evaluate_model(model_lr, X_test, y_test, "Logistic Regression")
results_rf <- evaluate_model(model_rf, X_test, y_test, "Random Forest")
results_svm <- evaluate_model(model_svm, X_test, y_test, "SVM (RBF)")

# summary comparison table
comparison_tbl <- map_dfr(
  list(results_lr, results_rf, results_svm),
  ~tibble(
    Model = .x$model,
    Accuracy = round(.x$accuracy, 4),
    AUC = round(.x$auc, 4),
    Precision = round(.x$precision, 4),
    Recall = round(.x$recall, 4),
    F1 = round(.x$f1, 4),
    Brier = round(.x$brier, 4)
  )
)

print(comparison_tbl)

saveRDS(list(
  comparison  = comparison_tbl,
  results_lr  = results_lr,
  results_rf  = results_rf,
  results_svm = results_svm
), "data/model_comparison.rds")

# == 8. Plot: ROC Curves  =====================================================
roc_data <- bind_rows(
  tibble(model = "Logistic Regression",
         fpr = 1 - results_lr$roc_obj$specificities,
         tpr = results_lr$roc_obj$sensitivities),
  tibble(model = "Random Forest",
         fpr = 1 - results_rf$roc_obj$specificities,
         tpr = results_rf$roc_obj$sensitivities),
  tibble(model = "SVM (RBF)",
         fpr = 1 - results_svm$roc_obj$specificities,
         tpr = results_svm$roc_obj$sensitivities)
)

auc_labels <- comparison_tbl %>%
  transmute(label = sprintf("%s (AUC = %.3f)", Model, AUC)) %>%
  pull(label)

p_roc <- ggplot(roc_data, aes(x = fpr, y = tpr, color = model)) +
  geom_line(size = 1.1) +
  geom_abline(linetype = "dashed", color = "grey60") +
  scale_color_manual(values = c("#E85D5D", "#4C9BE8", "#5DB85D"),
                     labels = auc_labels) +
  labs(title = "ROC Curves -- Test Set",
       subtitle = "All models vs. random classifier (dashed)",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = NULL) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "bottom",
        legend.text = element_text(size = 10))

ggsave(paste0(plots_path, "09_roc_curves.png"),
       p_roc, width = 7, height = 6, dpi = 150)
print(p_roc)

# == 9. Confusion Matrices  ====================================================
plot_confusion <- function(cm_table, title) {
  as_tibble(cm_table) %>%
    rename(Predicted = Prediction, Actual = Reference) %>%
    ggplot(aes(x = Actual, y = Predicted, fill = n)) +
    geom_tile(color = "white") +
    geom_text(aes(label = n), size = 7, fontface = "bold") +
    scale_fill_gradient(low = "#EEF4FF", high = "#4C9BE8") +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal(base_size = 12) +
    theme(plot.title    = element_text(face = "bold", size = 11),
          legend.position = "none")
}

p_cm <- plot_confusion(results_lr$confusion, "Logistic Regression") +
  plot_confusion(results_rf$confusion, "Random Forest") +
  plot_confusion(results_svm$confusion, "SVM (RBF)")

ggsave(paste0(plots_path, "10_confusion_matrices.png"),
       p_cm, width = 12, height = 4.5, dpi = 150)
print(p_cm)

# == 10. Plot: Feature Importance (Random Forest) =============================
importance_df <- varImp(model_rf)$importance %>%
  rownames_to_column("Feature") %>%
  rename(Importance = Yes) %>%
  select(Feature, Importance) %>%
  arrange(desc(Importance)) %>%
  head(15)

p_importance <- ggplot(importance_df,
                       aes(x = reorder(Feature, Importance),
                           y = Importance, fill = Importance)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_gradient(low = "#B0D4F1", high = "#1A6DB5") +
  labs(title = "Top 15 Features -- Random Forest Importance",
       subtitle = "Mean Decrease in Gini across 500 trees",
       x = NULL, y = "Importance Score") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(paste0(plots_path, "11_feature_importance.png"),
       p_importance, width = 8, height = 6, dpi = 150)
print(p_importance)

# == 11. Plot: Model Comparison Bar Chart  ====================================
comparison_long <- comparison_tbl %>%
  select(Model, AUC, F1, Recall) %>%
  pivot_longer(-Model, names_to = "Metric", values_to = "Value")

p_comparison <- ggplot(comparison_long,
                       aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = "dodge", width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", Value)),
            position = position_dodge(width = 0.65),
            vjust = -0.4, size = 3.5) +
  scale_fill_manual(values = c("AUC" = "#4C9BE8", "F1" = "#E8A044",
                               "Recall" = "#5DB85D")) +
  scale_y_continuous(limits = c(0, 1.05)) +
  labs(title = "Model Performance Comparison",
       x = NULL, y = "Score", fill = "Metric") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        axis.text.x = element_text(size = 11))

ggsave(paste0(plots_path, "12_model_comparison.png"),
       p_comparison, width = 8, height = 5, dpi = 150)
print(p_comparison)

# == 12. Identify Best Model for A/B Testing  =================================
best_model_name <- comparison_tbl %>%
  slice_max(AUC, n = 1) %>%
  pull(Model)

cat(sprintf("\nBest model by AUC: %s\n", best_model_name),
    "\nNext: run 04_ab_testing.R\n")
