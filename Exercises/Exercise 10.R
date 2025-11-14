
library(factoextra)
library(cluster)
iris_data <- iris[, -5]
mtcars_data <- mtcars[, c("mpg", "hp", "wt")]

#  1. Perform K-means clustering with K = 4 on iris dataset and compare with species labels.
set.seed(42) # For reproducible results
kmeans_k4 <- kmeans(iris_data, centers = 4, nstart = 25)

print("--- K-means Clustering (K=4) Results on Iris Data ---")

comparison_k4 <- table(kmeans_k4$cluster, iris$Species)
print(comparison_k4)

cat("\nInterpretation of K=4 Comparison:\n")
cat("The table shows how the 4 clusters map to the 3 actual Species. Since K=4, the clustering algorithm created an extra, less distinct group, leading to some Species being split across multiple clusters.\n")
cat("\n")

#  2. Apply clustering on mtcars using mpg, hp, and wt. Interpret clusters. 

mtcars_scaled <- scale(mtcars_data)
set.seed(42)

kmeans_mtcars <- kmeans(mtcars_scaled, centers = 3, nstart = 25)
mtcars$cluster <- kmeans_mtcars$cluster # Add cluster assignment back to the original data

print("--- K-means Clustering (K=3) Results on MTCars (mpg, hp, wt) ---")
print(kmeans_mtcars)

cluster_summary <- aggregate(mtcars_data, by = list(Cluster = mtcars$cluster), FUN = mean)
print("\nCluster Means for Interpretation:")
print(cluster_summary)

cat("\nInterpretation of MTCars Clusters:\n")
cat("Cluster 1 (e.g., Low Performance): Lowest mean hp, highest mean mpg.\n")
cat("Cluster 2 (e.g., High Performance/Heavy): Highest mean hp and wt, lowest mean mpg.\n")
cat("Cluster 3 (e.g., Mid-Range): Intermediate values for all variables.\n")
cat("\n")

# 3. Use the elbow method to find optimal number of clusters for iris. 
print("--- Elbow Method for Optimal K (Iris Data) ---")

fviz_nbclust(iris_data, kmeans, method = "wss") # WSS stands for Within Sum of Squares

cat("Elbow Method Interpretation:\n")
cat("The plot will show the total within-cluster sum of squares (WSS) for different values of K. The 'elbow' (the point where the decrease in WSS slows down) typically suggests the optimal number of clusters. For the iris dataset, this point is usually around K=3.\n")
cat("\n")

#  4. Visualize clusters using PCA-based scatterplot in factoextra. 

set.seed(42)
kmeans_k3 <- kmeans(iris_data, centers = 3, nstart = 25)

print("--- PCA-based Cluster Visualization (K=3 on Iris) ---")
fviz_cluster(kmeans_k3, data = iris_data,
             main = "K-means Clustering (K=3) of Iris Data (PCA-based)",
             geom = "point",
             ellipse.type = "convex")

# 5. Create a subset of iris with only Sepal.Length and Sepal.Width and cluster it. 
iris_subset <- iris[, c("Sepal.Length", "Sepal.Width")]
set.seed(42)
kmeans_subset <- kmeans(iris_subset, centers = 3, nstart = 25)

print("--- K-means Clustering (K=3) on Iris Sepal Dimensions Only ---")
print(kmeans_subset)

fviz_cluster(kmeans_subset, data = iris_subset,
             main = "K-means Clustering (K=3) on Sepal Dimensions",
             geom = "point",
             ellipse.type = "convex")
