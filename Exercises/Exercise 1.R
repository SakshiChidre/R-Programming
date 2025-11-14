
# Exercise 1
# 1. Display the first 15 rows of the mtcars dataset.
print("1. First 15 rows of mtcars:")
head(mtcars, 15)

# 2. Find the maximum and minimum value of Sepal.Length from iris.
print("2. Max and Min Sepal.Length from iris:")
max_sepal <- max(iris$Sepal.Length)
min_sepal <- min(iris$Sepal.Length)
print(paste("Maximum Sepal.Length:", max_sepal))
print(paste("Minimum Sepal.Length:", min_sepal))

# 3. Calculate the mean of the variable mpg in mtcars.
print("3. Mean of mpg in mtcars:")
mean_mpg <- mean(mtcars$mpg)
print(paste("Mean MPG:", mean_mpg))

# 4. Display the structure of the airquality dataset.
print("4. Structure of airquality dataset:")
str(airquality)

# 5. Check whether the number 100 is greater than 50 in R.
print("5. Check if 100 is greater than 50:")
is_greater <- 100 > 50
print(is_greater)
