
# Lab 3
library(dplyr)
# Load dataset
data("airquality")
# Display the first few rows of the original dataset
print("Original Airquality Dataset Head:")
head(airquality)
# 1. Check for missing values
print("--- Missing Value Check (airquality) ---")
# Total NA count across the entire dataset
print(paste("Total NA count:", sum(is.na(airquality))))
# Missing values per column
print("Missing values per column:")
colSums(is.na(airquality))
# 2. Handle missing values
print("--- Handling Missing Values (airquality) ---")
# Replace missing Ozone values with the mean of the column
# 'na.rm = TRUE' ensures the mean is calculated by ignoring existing NA values
airquality$Ozone[is.na(airquality$Ozone)] <- mean(airquality$Ozone, na.rm = TRUE)
# Replace missing Solar.R values with the median of the column
airquality$Solar.R[is.na(airquality$Solar.R)] <- median(airquality$Solar.R, na.rm = TRUE)
# Verify the changes
print("Missing values per column after imputation:")
colSums(is.na(airquality))
# 3. Remove duplicates (using iris dataset as an example)
data("iris")
print("--- Duplicates Removal (iris) ---")
# Creates a clean dataset by selecting rows that are NOT duplicates.
# 'duplicated()' returns TRUE for the second and subsequent occurrences of duplicate rows.
iris_clean <- iris[!duplicated(iris), ]
# Display the number of rows before and after cleaning
print(paste("Original rows in iris:", nrow(iris)))
print(paste("Rows after removing duplicates:", nrow(iris_clean)))
# 4. Standardize and use formats
print("--- Data Standardization (iris) ---")
# Convert the Species factor to character, then to lowercase for standardization
iris_clean$Species <- tolower(as.character(iris_clean$Species))
# Convert the standardized character column back to a factor
iris_clean$Species <- as.factor(iris_clean$Species)
# View the cleaned dataset head
print("Cleaned Iris Dataset Head:")
head(iris_clean)
