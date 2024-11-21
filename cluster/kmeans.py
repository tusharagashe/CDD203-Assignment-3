import numpy as np
from scipy.spatial.distance import cdist

class KMeans():
    def __init__(self, k: int, metric:str, max_iter: int, tol: float):
        """
        Args:
            k (int): 
                The number of centroids you have specified. 
                (number of centroids you think there are)
                
            metric (str): 
                The way you are evaluating the distance between the centroid and data points.
                (euclidean)
                
            max_iter (int): 
                The number of times you want your KMeans algorithm to loop and tune itself.
                
            tol (float): 
                The value you specify to stop iterating because the update to the centroid 
                position is too small (below your "tolerance").
        """ 
        # In the following 4 lines, please initialize your arguments
        self.k = k 
        self.metric = metric
        self.iterations = max_iter
        self.tolerance = tol
       
        # In the following 2 lines, you will need to initialize 1) centroid, 2) error (set as numpy infinity)
        self.centroids = None
        self.error = np.inf
    
    def fit(self, matrix: np.ndarray):
        """
        This method takes in a matrix of features and attempts to fit your created KMeans algorithm
        onto the provided data 
        
        Args:
            matrix (np.ndarray): 
                This will be a 2D matrix where rows are your observation and columns are features.
                
                Observation meaning, the specific flower observed.
                Features meaning, the features the flower has: Petal width, length and sepal length width 
        """
        
        # In the line below, you need to randomly select where the centroid's positions will be.
        # Also set your initialized centroid to be your random centroid position
        self.centroids = matrix[np.random.choice(len(matrix), self.k, replace=False)]
        
        # In the line below, calculate the first distance between your randomly selected centroid positions
        # and the data points
        distances = cdist(matrix, self.centroids, self.metric)
        
        # In the lines below, Create a for loop to keep assigning data points to clusters, updating centroids, 
        # calculating distance and error until the iteration limit you set is reached
        for i in range(self.iterations):
            # Within the loop, find the each data point's closest centroid
            closest_centroids = np.argmin(distances, axis=1)
        
        
            # Within the loop, go through each centroid and update the position.
            # Essentially, you calculate the mean of all data points assigned to a specific cluster. This becomes the new position for the centroid
            
            #for each centroid find the mean of all datapoints assigned and calcualte new centroid  
            new_centroids = []
            inertia = 0
            for index in range(len(self.centroids)):
                datapoint_indexes = np.where(closest_centroids==index)
                datapoints_mean = np.mean(matrix[datapoint_indexes], axis=0)
                inertia += np.sum((matrix[datapoint_indexes] - datapoints_mean) ** 2)
                new_centroids.append(datapoints_mean)
            
            # Within the loop, calculate distance of data point to centroid then calculate MSE or SSE (inertia)
            new_centroids = np.array(new_centroids)
            distances = cdist(matrix, new_centroids, self.metric)
        
            self.centroids = new_centroids
            # Within the loop, compare your previous error and the current error
            # Break if the error is less than the tolerance you've set 
            if (inertia < self.tolerance):   
                # Set your error as calculated inertia here before you break!
                self.error = inertia
                break
            # Set error as calculated inertia
            
            self.error = inertia

        
            
    
    
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """
        Predicts which cluster each observation belongs to.
        Args:
            matrix (np.ndarray): 
                

        Returns:
            np.ndarray: 
                An array/list of predictions will be returned.
        """
        # In the line below, return data point's assignment 
        return np.argmin(cdist(matrix, self.centroids, self.metric), axis=1)
    
    def get_error(self) -> float:
        """
        The inertia of your KMeans model will be returned

        Returns:
            float: 
                inertia of your fit

        """
        return self.error
    
    
    def get_centroids(self) -> np.ndarray:
    
        """
        Your centroid positions will be returned. 
        """
        # In the line below, return centroid location
        return self.centroids
        
    