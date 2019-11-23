'''
Using K-means and the Iris or Wine data set, create a graph that visually displays how the total 
squared distance decreases as the number of clusters increases. Then, use the elbow heuristic to 
confirm or reject the assumed number of clusters on the data.
'''

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans_iris():
    
    # load data to 'iris' variable
    iris = load_iris()
    
    # feature names in boston dataset
    print(iris.feature_names)
    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    
    # finding inertia (total squared distance inside each cluster)
    total_sq_dist = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i).fit(iris.data)
        total_sq_dist.append(kmeans.inertia_) 
    
    plt.plot([i for i in range(1, 11)], total_sq_dist, '-o')
    plt.xlim([0, 11])
    plt.xticks([i for i in range(1, 11)])
    plt.suptitle('Total Squared Distance for Each Cluster in Iris Data Set \n vs. \n Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Total Squared Distance (inertia)')
    plt.axvline(x = 3, linestyle = 'dotted', color = 'red')
    plt.savefig('kmeans_iris.png')
    plt.show()
    
    # Elbow heuristic confirms the ideal number of clusters is the same as the assumed number of clusters in Iris
    # dataset i.e. three.
    
if __name__ == '__main__':
    kmeans_iris()