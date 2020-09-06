import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#plots the scatter plot given originally
data = np.genfromtxt("ClusterPlot.csv", delimiter=",", names=["V1", "V2"], usecols=(1, -1))
plt.scatter(data["V1"], data["V2"])
plt.show() #displays the original plot
dataArray = pd.read_csv("ClusterPlot.csv", encoding="utf-8", sep=",", usecols=[1, 2]) #gets the data in the form of a matrix with labels
SSEFunction = [] #an empty array that will store the values of the silhoutte score used to judge the amount of clusters that are best for the program
index = -1 #temp variables in order to store the num of clusters that is the best to use
comparator = -1
for c in range(2, 20): #for loop goes through 2 to 19 clusters to pick which one is the best
    kmeans = KMeans(n_clusters=c)
    prediction = kmeans.fit_predict(dataArray) #use the kmeans algorithm to compute the clustering of the data with the amount of clusters specified in the run of the loop
    center = kmeans.cluster_centers_
    score = silhouette_score(dataArray, prediction) #use the silhouette score method to score the amount of clusters that make sense for this dataset
    if(comparator < score): #get the highest silhouette score value as it is
        index = c
        comparator = score
    SSEFunction.append(score) #add to the array in order to plot later
plt.plot(range(2, 20), SSEFunction) #plot the score compared to the clustering
plt.xticks(range(2, 20))
plt.xlabel("Num of Clusters")
plt.ylabel("Silhouette Score")
plt.show()
print("Optimal Number of clusters using the Silhouette method is: " + str(index)) #the best is supposed to be 2 for this dataset