
# Exercise 2


# 1. Import the airquality dataset and check its structure using str().
print("1. Structure of airquality dataset:")
str(airquality)

# 2. Find the number of missing values in airquality.
print("2. Total number of missing values (NA) in airquality:")
total_na <- sum(is.na(airquality))
print(total_na)

# 3. Use summary() on the mtcars dataset. (Interpretation follows the code block)
print("3. Summary of mtcars dataset:")
summary(mtcars)

# 4. Find the mean horsepower (hp) in the mtcars dataset.
print("4. Mean horsepower (hp) in mtcars:")
mean_hp <- mean(mtcars$hp)
print(mean_hp)

# 5. Group the iris dataset by species and calculate the average Sepal.Width.
print("5. Average Sepal.Width grouped by Species (using the 'dplyr' package):")

library(dplyr)
average_sepal_width <- iris %>%
  group_by(Species) %>%
  summarise(Average_Sepal_Width = mean(Sepal.Width))
print(average_sepal_width)
