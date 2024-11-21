import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
from sklearn.metrics import silhouette_score
from cluster import (KMeans)
from cluster.visualization import plot_3d_clusters


def main(): 
    # Set random seed for reproducibility
    np.random.seed(42)
    # Using sklearn iris dataset to train model
    og_iris = np.array(load_iris().data)
    
    # Initialize your KMeans algorithm
    kmeans = KMeans(3, 'euclidean', 100, 0.01)
    
    # Fit model
    kmeans.fit(og_iris)

    # Load new dataset
    df = np.array(pd.read_csv('data/iris_extended.csv', 
                usecols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']))

    # Predict based on this new dataset
    pred = kmeans.predict(df)
    # true  = 
    # You can choose which scoring method you'd like to use here:
    score = silhouette_score(df, pred)
    # Plot your data using plot_3d_clusters in visualization.py
    plot_3d_clusters(df, pred, kmeans, score)
    
    # Try different numbers of clusters
    errors = []
    kvalues = [1, 2, 3, 4, 5, 6, 7]
    for k in kvalues:
        curr_kmeans = KMeans(k, 'euclidean', 10, 0.01)
        curr_kmeans.fit(og_iris)
        errors.append(curr_kmeans.get_error())
    
    # Plot the elbow plot
    plt.figure(figsize=(8, 6))
    plt.plot(kvalues, errors, marker='o', linestyle='-', color='b', label='Error (Inertia)')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Error (Inertia)', fontsize=12)
    plt.title('Elbow Plot', fontsize=14)
    plt.xticks(kvalues)  # Show all k values on the x-axis
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.show()

    
    # Question: 
    # Please answer in the following docstring how many species of flowers (K) you think there are.
    # Provide a reasoning based on what you have done above: 
    
    """
    How many species of flowers are there: 3
    
    Reasoning: I believe there are 3 species of flowers as indicated by the elbow plot. 3 is the elbow of the cruve because when moving past k=3 the error values only
               slightly decrease (the rate of decrease is is slowing down) meaning as more clusters are being added the fit is not improving. 
    """

    
if __name__ == "__main__":
    main()