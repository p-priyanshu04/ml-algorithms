# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from collections import Counter
import math

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import r2_score,roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA

# %% [markdown]
# ## 1. Data Analysis

# %% [markdown]
# ### 1.1 Loading the Dataset

# %%
from sklearn.datasets import fetch_openml
boston = fetch_openml(data_id=531,as_frame=True)

x = boston.data
y = boston.target

df = pd.DataFrame(pd.concat([x,y],axis=1))

# Renaming the column "MEDV" to "Target"
df = df.rename(columns={"MEDV":"Target"})

# %% [markdown]
# ### 1.2 Dropping Categorical Features

# %%
x = x.drop(columns=['CHAS','RAD'])
df = df.drop(columns=['CHAS','RAD'])
x.head()

# %% [markdown]
# ### 1.3 Correlation HeatMap

# %%
corr_mat = df.corr()
print("Correlation Matrix:",corr_mat)
# Create the heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(corr_mat, annot=True, cmap='Pastel2', fmt='.2f', linewidths=1)
# Add titles and labels
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# ### 1.4 Pairplot of Selected Features

# %%
# Seaborn style
sns.set(style="ticks")

# Pairplot
pairplot = sns.pairplot(
    df,
    hue="Target",
    diag_kind="kde",
    palette="Set1",
    corner=False,
    height=3
)

pairplot.fig.suptitle("Pairwise Plot of Features", y=1.03)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.5 Feature vs. Target (Scatter Plots)

# %%
for col in x.columns:
  plt.figure(figsize=(5,4))
  plt.scatter(df[col],df["Target"],alpha=0.6,color="blue")
  plt.title(f'{col} vs Target')
  plt.xlabel(col)
  plt.ylabel("Target")
  plt.tight_layout()
  plt.show()

# %% [markdown]
# ### 1.6 Distribution of Target (MEDV)

# %%
df["Target"].hist(bins=20,color="blue",edgecolor="black")
plt.title("Histogram of Target")
plt.xlabel("Values")
plt.ylabel("Target")
plt.show()

# %% [markdown]
# ## 2. Scaling and Splitting the data

# %% [markdown]
# ### 2.1 Scaling the Features using Min-Max Scaling

# %%
def min_max_scaling(col):
  return (col - col.min()) / (col.max() - col.min())
x = x.apply(min_max_scaling)

# %% [markdown]
# ### 2.2 Splitting the dataset into training and testing sets, with 80% training and 20% testing

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## 3. Implement KNN Regressor from Scratch

# %% [markdown]
# ### 3.1 Implementing a KNN regressor from scratch- Simple Average

# %%
class KNN:
  def __init__(self,k,distance_metric):
    self.distance_metric=distance_metric
    self.k=k

  def fit(self,x,y):
    self.x_train=np.array(x)
    self.y_train=np.array(y)

  def euclidean_distance(self,x,y):
    return np.sqrt(np.sum((x - y) ** 2))

  def manhattan_distance(self,x,y):
    dist = np.sum(np.abs(x-y))
    return dist

  def cosine_distance(self,x,y):
    dist = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return 1-dist

  def compute_distance(self,x):
    distances=[]
    for x1 in self.x_train:
      if self.distance_metric == 'euclidean':
         distances.append(self.euclidean_distance(x1,x))
      elif self.distance_metric == 'manhattan':
        distances.append(self.manhattan_distance(x1,x))
      elif self.distance_metric == 'cosine':
        distances.append(self.cosine_distance(x1,x))
    return np.array(distances)

  def predict(self,x_test):
    results=[]
    x_test = np.array(x_test, dtype=float)
    for x in x_test:
      distances = self.compute_distance(x)
      k_nearest = np.argsort(distances)[:self.k]
      results.append(np.mean(self.y_train[k_nearest]))
    return results

# %% [markdown]
# ### 3.2 Creating a table summarizing the R² scores for each (k, distance metric) pair

# %%
res_custom_simple=[]
k_values=[3,5,7,9,11]
for i in k_values:
  temp=[]
  for j in ["euclidean","manhattan","cosine"]:
    knn = KNN(k=i, distance_metric=j)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    temp.append(r2)
  res_custom_simple.append(temp)
print(f"{'k':<10} {'euclidean':<20} {'manhattan':<20} {'cosine':<20}")
print()
for i in range(len(res_custom_simple)):
    print(f"{k_values[i]:<5} {res_custom_simple[i][0]:<20} {res_custom_simple[i][1]:<20} {res_custom_simple[i][2]:<20}")

# %% [markdown]
# The best R2 score is given by the combination (k=3,distance metric=manhattan)

# %% [markdown]
# ## 4. Compare with scikit-learn Simple KNN

# %% [markdown]
# ### 4.1 Initialising and Making Predictions using scikit-learn's KNeighborsRegressor

# %%
# Storing the Results using scikit-learn's KNeighborsRegressor
res_library_simple=[]
k_values=[3,5,7,9,11]
for i in k_values:
  temp=[]
  for j in ["euclidean","manhattan","cosine"]:
    knn = KNeighborsRegressor(n_neighbors=i, weights='uniform', metric=j,algorithm='brute')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    temp.append(r2)
  res_library_simple.append(temp)
print(f"{'k':<10} {'euclidean':<20} {'manhattan':<20} {'cosine':<20}")
print()
for i in range(len(res_library_simple)):
    print(f"{k_values[i]:<5} {res_library_simple[i][0]:<20} {res_library_simple[i][1]:<20} {res_library_simple[i][2]:<20}")

# %% [markdown]
# ### 4.2 Comparing the R² score of the scikit-learn model with the custom implementation for that best configuration

# %%
#The best configuration
r2_custom = res_custom_simple[0][1]

r2_library = res_library_simple[0][1]

if(r2_library>r2_custom):
  print("The R2 score of the scikit-learn model is greater than the custom implementation")
elif(r2_custom>r2_library):
  print("The R2 score of the custom implementation is greater than the scikit-learn model")
else:
  print("The R2 scores of both models are equal")

# %% [markdown]
# ## 5. Implement Weighted KNN Regressor from Scratch

# %% [markdown]
# ### 5.1 Implementing a weighted average KNN regressor from scratch, where weights are the inverse of the distance to neighbors.

# %%
class Weighted_KNN:
  def __init__(self,k,distance_metric):
    self.distance_metric=distance_metric
    self.k=k

  def fit(self,x,y):
    self.x_train=np.array(x)
    self.y_train=np.array(y)

  def euclidean_distance(self,x,y):
    return np.sqrt(np.sum((x - y) ** 2))

  def manhattan_distance(self,x,y):
    dist = np.sum(np.abs(x-y))
    return dist

  def cosine_distance(self,x,y):
    dist = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return 1-dist

  def compute_distance(self,x):
    distances=[]
    for x1 in self.x_train:
      if self.distance_metric == 'euclidean':
         distances.append(self.euclidean_distance(x1,x))
      elif self.distance_metric == 'manhattan':
        distances.append(self.manhattan_distance(x1,x))
      elif self.distance_metric == 'cosine':
        distances.append(self.cosine_distance(x1,x))
    return np.array(distances)

  def predict(self,x_test):
    results=[]
    x_test = np.array(x_test, dtype=float)
    for x in x_test:
      if(self.distance_metric!='cosine') :
        distances = self.compute_distance(x)
        k_nearest_id = np.argsort(distances)[:self.k]
      else:
        distances = self.compute_distance(x)
        k_nearest_id = np.argsort(distances)[:self.k]
      k_nearest=distances[k_nearest_id]
      for i in range(0,len(k_nearest)):
        if(k_nearest[i]==0):
          k_nearest[i]=1e9
        else :
          k_nearest[i]=1/k_nearest[i]
      total = np.sum(k_nearest)
      results.append(np.dot(k_nearest,self.y_train[k_nearest_id])/total)
    return results

# %% [markdown]
# ### 5.2 Creating a table summarizing the R² scores for each (k, distance metric) pair.

# %%
# Storing the Results using scikit-learn's KNeighborsRegressor
res_custom_weighted=[]
k_values=[3,5,7,9,11]
for i in k_values:
  temp=[]
  for j in ["euclidean","manhattan","cosine"]:
    knn = Weighted_KNN(k=i, distance_metric=j)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    temp.append(r2)
  res_custom_weighted.append(temp)

print(f"{'k':<10} {'euclidean':<20} {'manhattan':<20} {'cosine':<20}")
print()
for i in range(len(res_custom_weighted)):
    print(f"{k_values[i]:<5} {res_custom_weighted[i][0]:<20} {res_custom_weighted[i][1]:<20} {res_custom_weighted[i][2]:<20}")

# %% [markdown]
# The best R2 score is given by the combination (k=7,distance metric=cosine)

# %% [markdown]
# ## 6. Compare with scikit-learn Weighted KNN

# %% [markdown]
# ### 6.1 Initialising and Making Predictions using scikit-learn's KNeighborsRegressor

# %%
res_library_weighted=[]
k_values=[3,5,7,9,11]
for i in k_values:
  temp=[]
  for j in ["euclidean","manhattan","cosine"]:
    knn = KNeighborsRegressor(n_neighbors=i, weights='distance', metric=j,algorithm='brute')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    temp.append(r2)
  res_library_weighted.append(temp)

print(f"{'k':<10} {'euclidean':<20} {'manhattan':<20} {'cosine':<20}")
print()
for i in range(len(res_library_weighted)):
    print(f"{k_values[i]:<5} {res_library_weighted[i][0]:<20} {res_library_weighted[i][1]:<20} {res_library_weighted[i][2]:<20}")

# %% [markdown]
# ### 6.2 Comparing the R² score of the scikit-learn model with the custom implementation for that best configuration

# %%
#The best configuration
r2_custom = res_custom_weighted[2][2]

r2_library = res_library_weighted[2][2]

if(r2_library>r2_custom):
  print("The R2 score of the scikit-learn model is greater than the custom implementation by: ",r2_library-r2_custom)
elif(r2_custom>r2_library):
  print("The R2 score of the custom implementation is greater than the scikit-learn model by: ",r2_custom-r2_library)
else:
  print("The R2 scores of both models are equal")

# %% [markdown]
# ## 7. Reporting and Visualization

# %% [markdown]
# ### 7.1 Plot the R² scores for both simple and weighted KNN from the implementation across different k and distance metrics

# %%
distance_metrics=["Euclidean","Manhattan","Cosine"]
colors=["red","green","blue"]

plt.figure(figsize=(20,12))

for i in range(3):
  a1=[]
  for j in range(5):
    a1.append(res_custom_simple[j][i])

  plt.plot(k_values,a1,label=distance_metrics[i],color=colors[i],linestyle='--')


plt.title("R2 Scores of Simple KNN")
plt.xlabel("k")
plt.ylabel("R2 Score")
plt.xticks(k_values)
plt.legend()
plt.tight_layout()
plt.show()



# %%
distance_metrics=["Euclidean","Manhattan","Cosine"]
colors=["red","green","blue"]

plt.figure(figsize=(20,12))

for i in range(3):
  a2=[]
  for j in range(5):
    a2.append(res_custom_weighted[j][i])

  plt.plot(k_values,a2,label=distance_metrics[i],color=colors[i],linestyle='--')


plt.title("R2 Scores of Weighted KNN")
plt.xlabel("k")
plt.ylabel("R2 Score")
plt.xticks(k_values)
plt.legend()
plt.tight_layout()
plt.show()



# %% [markdown]
# ### 7.2 Plot corresponding R² scores from scikit-learn for the same parameters

# %%
distance_metrics=["Euclidean","Manhattan","Cosine"]
colors=["red","green","blue"]

plt.figure(figsize=(20,12))

for i in range(3):
  a1=[]
  for j in range(5):
    a1.append(res_library_simple[j][i])

  plt.plot(k_values,a1,label=distance_metrics[i],color=colors[i],linestyle='--')


plt.title("R2 Scores of Simple KNN")
plt.xlabel("k")
plt.ylabel("R2 Score")
plt.xticks(k_values)
plt.legend()
plt.tight_layout()
plt.show()



# %%
distance_metrics=["Euclidean","Manhattan","Cosine"]
colors=["red","green","blue"]

plt.figure(figsize=(20,12))

for i in range(3):
  a2=[]
  for j in range(5):
    a2.append(res_library_weighted[j][i])

  plt.plot(k_values,a2,label=distance_metrics[i],color=colors[i],linestyle='--')


plt.title("R2 Scores of Weighted KNN")
plt.xlabel("k")
plt.ylabel("R2 Score")
plt.xticks(k_values)
plt.legend()
plt.tight_layout()
plt.show()



# %% [markdown]
# ### 7.3 Comparing the R² scores of scikit-learn weighted KNN with the custom weighted KNN for the best combination.

# %%
#The best configuration
r2_custom = res_custom_weighted[2][2]

r2_library = res_library_weighted[2][2]

if(r2_library>r2_custom):
  print("The R2 score of the scikit-learn model is greater than the custom implementation by: ",r21-r2)
elif(r2_custom>r2_library):
  print("The R2 score of the custom implementation is greater than the scikit-learn model by: ",r2-r21)
else:
  print("The R2 scores of both models are equal")

# %% [markdown]
# ### 7.4 Plotting the Scatter Plots of the actual and predicted datapoints

# %%
#Plot for the best configuration in simple knn
knn = KNN(k=3, distance_metric="manhattan")
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

plt.figure(figsize=(10,8))
plt.scatter(y_test,y_pred,color="red",alpha=0.6,label="Predicted Vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scatter Plot of Actual Vs Predicted Values")
plt.legend()
plt.show()

# %%
#Plot for the best configuration in weighted knn
knn = Weighted_KNN(k=3, distance_metric="manhattan")
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

plt.figure(figsize=(10,8))
plt.scatter(y_test,y_pred,color="red",alpha=0.6,label="Predicted Vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scatter Plot of Actual Vs Predicted Values")
plt.legend()
plt.show()

# %% [markdown]
# ## 8. Summarizing the findings

# %% [markdown]
# *   Which (k, distance metric) performed best?
# 
# 1.   For Simple KNN the combination of k=3 and manhattan distance metric performed the best.
# 2.   For Weighted KNN the combination of k=7 and cosine distance metric performed the best.
# 
# *   How does your implementation compare with scikit-learn?
# 
# My implementation of KNN for both simple and weighted gives nearly the same R2 score for all the pairs of k and distance metric.
# 
# *   Discuss any observations regarding distance metrics or weighting.
# 
# 1. Simple KNN
# *   Manhattan distance yields the best result as compared to the other distance metrics.
# *   As the value of the k becomes large(like 9,11) the R2 score for each distance metric decreases.
# 
# 2. Weighted KNN
# *   In Weighted KNN cosine distance metric outperfroms other distance metrics for nearly all values of k.
# *  Large values of k yield lower R2 score for Weighted KNN also.
# *   The R2 scores calculated for this is higher than Simple KNN, which shows that weighing the values of y with the inverse of distance gives better performance than given by simply taking the average.
# *   If distance of any test data point is zero then a very large value of `1e9` has been taken as the weight.
# 
# 


