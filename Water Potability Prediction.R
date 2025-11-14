# ============================================================
# WATER POTABILITY PREDICTION PROJECT - CLEAN FULL CODE
# ============================================================

library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(corrplot)
library(GGally)

set.seed(42)

# ============================================================
# 1 — LOAD DATA
# ============================================================

df <- read_csv("water_potability.csv")
df <- df %>% rename_all(tolower)
glimpse(df)

# ============================================================
# 2 — CLEANING & IMPUTATION
# ============================================================

num_cols <- df %>% select(-potability) %>% names()

for (col in num_cols) {
  med <- median(df[[col]], na.rm = TRUE)
  df[[col]][is.na(df[[col]])] <- med
}

sum(is.na(df))   # Should be 0 now

# ============================================================
# 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# =======================
# Density plot (KEEPING)
# =======================
df %>%
  pivot_longer(cols = -potability, names_to = "feature", values_to = "value") %>%
  ggplot(aes(value, fill = factor(potability))) +
  geom_density(alpha = 0.4) +
  facet_wrap(~feature, scales = "free") +
  labs(title="Density Plot of Features", fill="Potable") +
  theme_minimal()

# Boxplots
df %>%
  pivot_longer(cols = -potability, names_to = "feature", values_to = "value") %>%
  ggplot(aes(factor(potability), value, fill=factor(potability))) +
  geom_boxplot() +
  facet_wrap(~feature, scales="free") +
  labs(title="Boxplots of Water Features by Potability", x="Potability")


# ======================
# CORRELATION HEATMAP
# ======================
corr_matrix <- cor(df %>% select(-potability))
corrplot(corr_matrix, method="color", tl.cex=0.7)

# ============================================================
# 4 — TRAIN / TEST SPLIT
# ============================================================

train_index <- createDataPartition(df$potability, p=0.8, list=FALSE)
train <- df[train_index, ]
test  <- df[-train_index, ]

# ============================================================
# 5 — LOGISTIC REGRESSION MODEL
# ============================================================

logit_model <- glm(potability ~ ., data=train, family=binomial)
summary(logit_model)

# Predict
test$logit_prob <- predict(logit_model, newdata=test, type="response")
test$logit_pred <- ifelse(test$logit_prob >= 0.5, 1, 0)

# Evaluation
cat("\n===== LOGISTIC REGRESSION RESULTS =====\n")
print(confusionMatrix(factor(test$logit_pred), factor(test$potability), positive="1"))

roc_logit <- roc(test$potability, test$logit_prob)
cat("AUC (Logistic Regression): ", auc(roc_logit), "\n")
plot(roc_logit, col="blue", main="ROC Curve Comparison")

# ============================================================
# 6 — RANDOM FOREST MODEL
# ============================================================

rf_model <- randomForest(as.factor(potability) ~ ., data=train, ntree=200, importance=TRUE)

# Predict
test$rf_prob <- predict(rf_model, test, type="prob")[, "1"]
test$rf_pred <- predict(rf_model, test, type="response")

# Evaluation
cat("\n===== RANDOM FOREST RESULTS =====\n")
print(confusionMatrix(test$rf_pred, factor(test$potability), positive="1"))

roc_rf <- roc(test$potability, test$rf_prob)
cat("AUC (Random Forest): ", auc(roc_rf), "\n")

# Add RF to ROC plot
plot(roc_rf, col="red", add=TRUE)
legend("bottomright", legend=c("Logistic", "Random Forest"), col=c("blue","red"), lty=1)

# ============================================================
# SAFE VARIABLE IMPORTANCE PLOT (NO ERRORS)
# ============================================================

imp <- importance(rf_model)
imp <- imp[apply(imp, 1, function(x) all(is.finite(x))), , drop=FALSE]

imp_df <- data.frame(
  Feature = rownames(imp),
  Importance = imp[, 1]
)

ggplot(
  imp_df %>% arrange(desc(Importance)) %>% head(10),
  aes(x=reorder(Feature, Importance), y=Importance)
) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title="Random Forest Variable Importance",
       x="Feature",
       y="Importance Score") +
  theme_minimal()
# ============================================================
# EXTRA SUMMARY PLOTS
# ============================================================




# ------------------------------------------------------------
# 2) Average Value of Each Feature by Potability
# ------------------------------------------------------------
mean_values <- df %>%
  group_by(potability) %>%
  summarise(across(everything(), mean)) %>%
  pivot_longer(-potability, names_to="feature", values_to="avg")

ggplot(mean_values, aes(x=feature, y=avg, fill=factor(potability))) +
  geom_col(position="dodge") +
  scale_fill_manual(values=c("red","blue"), labels=c("Not Potable","Potable")) +
  labs(
    title = "Average Feature Values by Potability",
    x = "Feature",
    y = "Average Value",
    fill = "Category"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1))



# ============================================================
# 7 — FINAL OUTPUT
# ============================================================

cat("\n===== PROJECT COMPLETED SUCCESSFULLY =====\n")
cat("Only ONE boxplot generated.\n")
cat("All previous errors removed.\n")