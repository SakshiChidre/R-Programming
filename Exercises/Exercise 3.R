
# 1. Count how many missing values are present in the airquality dataset.
print("1. Total number of missing values (NA) in airquality:")
total_missing <- sum(is.na(airquality))
print(total_missing)

# 2. Replace missing values in Ozone with the median instead of mean.
print("2. Replacing NA values in Ozone with the median:")

# Calculate the median of the Ozone column, ignoring NAs
ozone_median <- median(airquality$Ozone, na.rm = TRUE)

# Create a new dataset 'airquality_imputed' for demonstration
airquality_imputed <- airquality

# Impute NA values in the Ozone column
airquality_imputed$Ozone[is.na(airquality_imputed$Ozone)] <- ozone_median

# Check the number of NAs in the imputed column (should be 0)
print(paste("NAs in Ozone column after imputation:", sum(is.na(airquality_imputed$Ozone))))

# 3. Create a duplicate dataset from mtcars and remove the duplicates.
print("3. Duplicating mtcars and removing duplicates:")

# Create a duplicate dataset (we'll manually add a duplicate row for demonstration)
mtcars_dup <- rbind(mtcars, mtcars[1, ]) # Adds the first row as a duplicate
print(paste("Total rows before removing duplicates:", nrow(mtcars_dup)))

# Remove duplicate rows (based on all columns)
mtcars_unique <- unique(mtcars_dup)

print(paste("Total rows after removing duplicates:", nrow(mtcars_unique)))
# The number of rows should revert back to the original 32

# 4. Convert all car names in mtcars row names to uppercase.
print("4. Converting mtcars row names to uppercase:")

# Row names are accessed using rownames()
rownames(mtcars) <- toupper(rownames(mtcars))

# Display the first few rows to show the change
print("First 5 rows with uppercase names:")
head(mtcars)

# 5. Convert the Species column in iris dataset into numeric codes (1, 2, 3).
print("5. Converting Species column to numeric codes:")

# The as.numeric() function automatically converts factors to their underlying numeric codes.
iris$Species_Code <- as.numeric(iris$Species)

# Display the new column with the codes
print("First 5 rows showing Species and its numeric code:")
head(iris[, c("Species", "Species_Code")])
