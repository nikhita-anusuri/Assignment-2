from sklearn import datasets
iris = datasets.load_iris()
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)

iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Q1
print("Q1")
print(" ")

# a. Make a histogram of the variable Sepal.Width.
print(iris.head())
print(iris.count())
plt.hist(iris['sepal width (cm)'], bins=10, edgecolor='black')
plt.title("Sepal Width")
plt.show()
print("-----------------------------------------------------------------------------------")

# b. Based on the histogram from #1a, which would you expect to be higher, the mean or the median? Why?
# Since the histogram looks symmetrical, we can expect the mean and median to be close in value.

# c. Confirm your answer to #1b by actually finding these values.
print("c")
sepal_width_mean = iris["sepal width (cm)"].mean()
print('Sepal width mean : ',sepal_width_mean)
sepal_width_median = iris["sepal width (cm)"].median()
print('Sepal width median : ', sepal_width_median)
print("-----------------------------------------------------------------------------------")

# d. Only 27% of the flowers have a Sepal.Width higher than ________ cm
print("d")
sepal_width = iris["sepal width (cm)"]
sepal_width_73rd_percentile = np.percentile(sepal_width, 73)
print("27 percent of flowers have a sepal width higher than :", sepal_width_73rd_percentile)
print("-----------------------------------------------------------------------------------")

# e. Make scatterplots of each pair of the numerical variables in iris (There should be 6 pairs/plots).
pd.plotting.scatter_matrix(iris, figsize=(10, 10))
plt.suptitle("Scatter Matrix of Iris Dataset")    
plt.show()
print("-----------------------------------------------------------------------------------")

# f. Based on #1e, which two variables appear to have the strongest relationship? 
#    And which two appear to have the weakest relationship?
# Strongest relationships: Petal length and petal width.
# Weakest relationships: Sepal length and sepal width.

# Q2
print("Q2")
print(" ")
# a. Make a histogram of the variable weight with breakpoints (bin edges) at every 0.3 units, starting at 3.3.
print(PlantGrowth.head(30))
print(PlantGrowth.count())
print(PlantGrowth['weight'].min(), PlantGrowth['weight'].max())
plt.hist(PlantGrowth['weight'], bins=np.arange(3.3, PlantGrowth["weight"].max() + 0.3, 0.3), edgecolor='black')
plt.title("Weight Histogram")
plt.show()
print("-----------------------------------------------------------------------------------")    

# b. Make boxplots of weight separated by group in a single graph.
PlantGrowth.boxplot(column='weight', by='group')
plt.title("Boxplot of Weight by Group")
plt.xlabel("Group")
plt.ylabel("Weight")
plt.show()


# c. Based on the boxplots in #2b, approximately what percentage of the "trt1" weights are below the minimum "trt2" weight?
# Approximately 80% of the trt1 weights are below the minimum trt2 weight. There are 2 points above the minimum on the plot. 

# d. Find the exact percentage of the "trt1" weights that are below the minimum "trt2" weight.
print("d")
trt2_min_weight = PlantGrowth[PlantGrowth['group'] == 'trt2']['weight'].min()
trt1_less_than_trt2_min =PlantGrowth[(PlantGrowth['group'] == 'trt1') & (PlantGrowth['weight'] < trt2_min_weight)]
percentage_below = (len(trt1_less_than_trt2_min) / len(PlantGrowth[PlantGrowth['group'] == 'trt1'])) * 100
print(f"Percentage of trt1 weights below minimum trt2 weight: {percentage_below:.2f}%")
print("-----------------------------------------------------------------------------------")

# Only including plants with a weight above 5.5, make a barplot of the variable group. 
# Make the barplot colorful using some color palette (in R, try running ?heat.colors and/or check out https://www.r-bloggers.com/palettes-in-r/).
PlantGrowth_above_55 = PlantGrowth[PlantGrowth['weight'] > 5.5]
group_counts = PlantGrowth_above_55['group'].value_counts()
colors = ['#42b6f5', '#f542a1', '#44c983']
plt.bar(group_counts.index, group_counts.values, color=colors)
plt.title("Barplot of Group (Weight > 5.5)")
plt.xlabel("Group")
plt.ylabel("Count")
plt.show()