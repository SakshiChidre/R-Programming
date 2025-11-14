
#1. Create a histogram of mpg from the mtcars dataset. 
print("1. Histogram of mpg (plotting...):")
hist(mtcars$mpg,
     main = "Histogram of Miles Per Gallon (MPG)",
     xlab = "MPG",
     col = "darkblue",
     border = "white")

# 2. Generate a scatterplot of hp vs wt from mtcars. 
print("2. Scatterplot of Horsepower (hp) vs Weight (wt) (plotting...):")
plot(mtcars$wt, mtcars$hp,
     main = "Scatterplot: Horsepower vs. Weight",
     xlab = "Weight (wt)",
     ylab = "Horsepower (hp)",
     pch = 19,
     col = "red")

#  3. Draw a boxplot of Ozone values from the airquality dataset.
print("3. Boxplot of Ozone values (plotting...):")
boxplot(airquality$Ozone,
        main = "Boxplot of Ozone Levels",
        ylab = "Ozone (parts per billion)",
        col = "lightgreen",
        na.rm = TRUE) # na.rm=TRUE handles missing (NA) values in the Ozone column

#  4. Find the correlation between Sepal.Length and Sepal.Width in iris. 
print("4. Correlation between Sepal.Length and Sepal.Width:")
correlation_value <- cor(iris$Sepal.Length, iris$Sepal.Width)
print(correlation_value)

#  5. Create a pair plot of the first four columns of mtcars. 
print("5. Pair plot of the first four mtcars columns (plotting...):")
pairs(mtcars[, 1:4],
      main = "Pair Plot of mtcars: mpg, cyl, disp, hp",
      col = "blue")
