
library(DescTools)

# 1. Calculate the mean, median, mode, and standard deviation of Petal.Width from iris.
print("1. Summary statistics for Petal.Width in iris:")
mean_pw <- mean(iris$Petal.Width)
median_pw <- median(iris$Petal.Width)
mode_pw <- Mode(iris$Petal.Width)
sd_pw <- sd(iris$Petal.Width)

print(paste("Mean:", mean_pw))
print(paste("Median:", median_pw))
# Mode() can return multiple modes if they have the same frequency
print(paste("Mode:", mode_pw[1])) 
print(paste("Standard Deviation:", sd_pw))

# 2. Draw a histogram of mpg from the mtcars dataset.
print("2. Histogram of mpg (plotting...):")
hist(mtcars$mpg, 
     main = "Histogram of MPG in mtcars",
     xlab = "Miles Per Gallon (mpg)",
     col = "lightblue",
     border = "black")

# 3. Create a scatterplot of hp vs mpg from mtcars.
print("3. Scatterplot of hp vs mpg (plotting...):")
plot(mtcars$hp, mtcars$mpg, 
     main = "Scatterplot of Horsepower vs. MPG",
     xlab = "Horsepower (hp)",
     ylab = "Miles Per Gallon (mpg)",
     col = "red",
     pch = 19)

# 4. Generate a boxplot for Ozone in the airquality dataset.
print("4. Boxplot for Ozone (plotting...):")
boxplot(airquality$Ozone, 
        main = "Boxplot of Ozone Levels",
        ylab = "Ozone (parts per billion)",
        col = "lightgreen")

# 5. Find the variance of Sepal.Width in the iris dataset.
print("5. Variance of Sepal.Width in iris:")
variance_sw <- var(iris$Sepal.Width)
print(variance_sw)
