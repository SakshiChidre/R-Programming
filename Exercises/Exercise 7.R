
# 1. Perform a one-sample t-test to check if the mean mpg in mtcars differs from 20. 
t_test_result <- t.test(mtcars$mpg, mu = 20)

print("--- One-Sample t-test Result (mean mpg vs 20) ---")
print(t_test_result)
cat("\n")

#2. Compare Petal.Width across species in iris using ANOVA.

anova_model <- aov(Petal.Width ~ Species, data = iris)

print("--- ANOVA Result (Petal.Width by Species) ---")
print(summary(anova_model))
cat("\n")

#  3. Find the correlation between mpg and hp in mtcars. 
correlation_mpg_hp <- cor(mtcars$mpg, mtcars$hp)

print("--- Correlation between mpg and hp in mtcars ---")
cat("Correlation Coefficient:", round(correlation_mpg_hp, 4), "\n")
cat("\n")

# 4. Perform a paired t-test on Sepal.Length and Sepal.Width in iris. 
paired_t_test_result <- t.test(iris$Sepal.Length, iris$Sepal.Width, paired = TRUE)

print("--- Paired t-test Result (Sepal.Length vs Sepal.Width) ---")
print(paired_t_test_result)
cat("\n")

# 5. Check if Ozone and Temp in airquality are significantly correlated. 
correlation_test_ozone_temp <- cor.test(airquality$Ozone, airquality$Temp, use = "complete.obs")

print("--- Correlation Test Result (Ozone vs Temp in airquality) ---")
print(correlation_test_ozone_temp)
