
# Lab 1
# Print a message 
print('Hello, Welcome to R Programming')
#Assign variables
x <- 25
y <- 5
#Arithmetic operations
sum_val <- x + y
diff_val<- x - y
prod_val <- x * y
div_val<- x / y

sum_val
diff_val
prod_val
div_val
# Relational operations
greater_check <- x > y
equal_check <- x == y
# Load and explore iris dataset
data("iris")  #load dataset
head(iris)    #first 6 rows
str(iris)     #structure of dataset
summary(iris) #descriptive statistics
