
# 1. Build a regression model predicting Petal.Length from Sepal.Length in iris. 
model_q1 <- lm(Petal.Length ~ Sepal.Length, data = iris)

print("--- Simple Regression Model Summary (Petal.Length ~ Sepal.Length) ---")
print(summary(model_q1))
cat("\n")

# 2. Create a multiple regression model predicting mpg using wt, hp, and drat in mtcars.
model_q2 <- lm(mpg ~ wt + hp + drat, data = mtcars)

print("--- Multiple Regression Model Summary (mpg ~ wt + hp + drat) ---")
print(summary(model_q2))
cat("\n")

# 3. Check the R^2 and Adjusted R^2 of the model in Q2. 
summary_q2 <- summary(model_q2)

print("--- R-squared and Adjusted R-squared for Q2 Model ---")
cat("R-squared:", round(summary_q2$r.squared, 4), "\n")
cat("Adjusted R-squared:", round(summary_q2$adj.r.squared, 4), "\n")
cat("\n")

#  4. Plot residuals of your regression model in Q1 and interpret the result. 

print("--- Residual Plots for Q1 Model (Petal.Length ~ Sepal.Length) ---")
par(mfrow = c(2, 2))
plot(model_q1)
par(mfrow = c(1, 1))

cat("\nInterpretation:\n")
cat("The 'Residuals vs Fitted' plot shows a slight U-shape, suggesting that a simple linear relationship might not fully capture the variance, and a non-linear model might fit the data better. The points are not perfectly randomly scattered around the horizontal line at 0.\n")
cat("\n")

# 5. Build a regression model predicting Ozone using Temp from the airquality dataset.

model_q5 <- lm(Ozone ~ Temp, data = airquality, na.action = na.omit)

print("--- Simple Regression Model Summary (Ozone ~ Temp) ---")
print(summary(model_q5))
