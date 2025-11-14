
library(dplyr)
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)

# Q1: Build a logistic regression model predicting am (automatic/manual) in mtcars using hp and wt. 
mtcars$am_factor <- factor(mtcars$am, labels = c("Automatic", "Manual"))
logistic_model <- glm(am_factor ~ hp + wt, data = mtcars, family = "binomial")

print("--- Logistic Regression Model Summary (am ~ hp + wt) ---")
print(summary(logistic_model))
cat("\n")
tree_model <- rpart(Species ~ ., data = iris, method = "class")

print("--- Decision Tree Model Summary (Species ~ All features) ---")
print(tree_model)
cat("\n")
# 1. Predict probabilities
mtcars$log_preds_prob <- predict(logistic_model, type = "response")

# 2. Convert probabilities to class predictions (using 0.5 as the cutoff)
mtcars$log_preds_class <- factor(ifelse(mtcars$log_preds_prob > 0.5, "Manual", "Automatic"),
                                 levels = c("Automatic", "Manual"))

# 3. Create the confusion matrix
conf_matrix_q1 <- confusionMatrix(mtcars$log_preds_class, mtcars$am_factor)

print("--- Confusion Matrix and Accuracy for Logistic Regression Model (Q1) ---")
print(conf_matrix_q1)
cat("\n")
roc_curve_q1 <- roc(mtcars$am, mtcars$log_preds_prob)

print("--- ROC Curve Plot for Logistic Regression Model (Q1) ---")
plot(roc_curve_q1, main = "ROC Curve for Logistic Regression (am prediction)", col = "blue", lwd = 2)
auc_q1 <- auc(roc_curve_q1)
cat("Area Under the Curve (AUC):", round(auc_q1, 4), "\n")
cat("\n")

# 1. Decision Tree Performance (using the model created earlier)
iris_tree_preds <- predict(tree_model, iris, type = "class")
conf_matrix_tree <- confusionMatrix(iris_tree_preds, iris$Species)

print("--- Performance Comparison: Decision Tree (Iris) ---")
print(conf_matrix_tree$overall['Accuracy'])
cat("\n")

# 2. Logistic Regression (Multinomial) Performance on Iris
iris_binary <- iris %>% filter(Species != "virginica")
iris_binary$Species <- factor(iris_binary$Species) # Drop the 'virginica' level

log_model_iris <- glm(Species ~ Sepal.Length + Petal.Length, data = iris_binary, family = "binomial")
iris_log_preds_prob <- predict(log_model_iris, type = "response")
iris_log_preds_class <- factor(ifelse(iris_log_preds_prob > 0.5, "versicolor", "setosa"),
                               levels = c("setosa", "versicolor"))

conf_matrix_log <- confusionMatrix(iris_log_preds_class, iris_binary$Species)

print("--- Performance Comparison: Logistic Regression (Binary Iris: setosa vs versicolor) ---")
print(conf_matrix_log$overall['Accuracy'])
