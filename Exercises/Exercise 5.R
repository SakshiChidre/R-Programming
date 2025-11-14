
library(dplyr)
library(caret)

# 1. Perform binning on Sepal.Length from iris into categories: Short, Medium, Long.
print("1. Performing binning on Sepal.Length:")
breaks <- c(4.2, 5.5, 6.5, 8.0) # Define breakpoints
labels <- c("Short", "Medium", "Long") # Define category labels
iris$Sepal.Length.Category <- cut(iris$Sepal.Length, 
                                  breaks = breaks,
                                  labels = labels,
                                  include.lowest = TRUE)

print("Frequency of the new categories:")
print(table(iris$Sepal.Length.Category))

# 2. Convert Species column in iris into dummy variables (One-Hot Encoding).
print("2. Converting Species column to dummy variables (One-Hot Encoding):")
dummy_model <- dummyVars("~ Species", data = iris)
dummy_variables <- predict(dummy_model, newdata = iris)
iris_ohe <- cbind(iris, dummy_variables)
print("First 5 rows with OHE columns:")
print(head(iris_ohe[, c("Species", "Species.setosa", "Species.versicolor", "Species.virginica")]))

# 3. Normalize the mpg column in mtcars dataset (Min-Max Scaling).
print("3. Normalizing the mpg column (Min-Max Scaling):")
min_mpg <- min(mtcars$mpg)
max_mpg <- max(mtcars$mpg)

mtcars$mpg_normalized <- (mtcars$mpg - min_mpg) / (max_mpg - min_mpg)

print("Range check (should be between 0 and 1):")
print(range(mtcars$mpg_normalized))

# 4. Standardize the Sepal.Width column in iris (Z-Score Standardization).
print("4. Standardizing the Sepal.Width column (Z-Score Standardization):")

iris$Sepal.Width_standardized <- scale(iris$Sepal.Width)

print("Mean and Standard Deviation check (should be close to 0 and 1):")
print(paste("Mean:", round(mean(iris$Sepal.Width_standardized), 4)))
print(paste("SD:", round(sd(iris$Sepal.Width_standardized), 4)))

# 5. Create a new feature in mtcars: efficiency = mpg / hp.
print("5. Creating the new 'efficiency' feature:")

mtcars$efficiency <- mtcars$mpg / mtcars$hp

print("First 5 values of the new efficiency feature:")
print(head(mtcars$efficiency))
