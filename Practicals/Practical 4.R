
# Lab 4
library(dplyr)
library(ggplot2)
library(modeest) # for mode
# Load dataset
data("iris")
# Descriptive statistics
mean(iris$Sepal.Length) # mean
median(iris$Sepal.Length) # median
mlv(iris$Sepal.Length, method="mfv") # mode
sd(iris$Sepal.Length) # standard deviation
range(iris$Sepal.Length) # min and max
summary(iris$Sepal.Length) # summary
# Histogram
hist(iris$Sepal.Length,
     main="Histogram of Sepal Length",
     xlab="Sepal Length", col="lightblue", border="black")
# Scatterplot
plot(iris$Sepal.Length, iris$Petal.Length,
     main="Scatterplot of Sepal vs Petal Length",
     xlab="Sepal Length", ylab="Petal Length",
     col=c("blue", pch=19))
# Boxplot
boxplot(Sepal.Length ~ Species, data=iris,
        main="Boxplot of Sepal Length by Species",
        xlab="Species", ylab="Sepal Length",
        col=c("lightgreen", "lightblue", "pink"))
