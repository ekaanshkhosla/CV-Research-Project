import cv2
import numpy as np
import pickle
import os
import random
from scipy import spatial
from collections import Counter

# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.dim_embeddings = 128
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=50, max_distance=0.8, min_prob=0.5):
        
        ########################ToDo: Prepare FaceNet and set all parameters for kNN.#############################
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob
        
        
        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)


    ################################################## ToDo #####################################################
    def update(self, face, label):
        embedding = self.facenet.predict(face)
        self.labels.append(label)
        self.embeddings = np.vstack([self.embeddings, embedding])
        return None



    ################################################# ToDo ######################################################
    def predict(self, face):
        # Extract the embedding for the given face using the FaceNet model.
        embedding = self.facenet.predict(face)
    
        # Calculate distances from the input face to all faces in the gallery.
        distances = np.linalg.norm(self.embeddings - embedding, axis=1)
    
        # Find the indices of the k smallest distances (k nearest neighbors).
        k_nearest_indices = np.argsort(distances)[:self.num_neighbours]
    
        # Retrieve the labels for the k nearest neighbors.
        nearest_labels = [self.labels[i] for i in k_nearest_indices]
    
        # Count the frequency of each label in the nearest neighbors.
        label_count = Counter(nearest_labels)
    
        # If no nearest neighbors, return None or a default value.
        if not label_count:
            return None, 0, float('inf')
    
        # Calculate the posterior probability for each class.
        posterior_probabilities = {label: count / self.num_neighbours for label, count in label_count.items()}
    
        # Find the label with the highest posterior probability.
        majority_label, majority_prob = max(posterior_probabilities.items(), key=lambda item: item[1])
    
        # Compute the distance of the face to the predicted class C_i.
        # It's the minimum distance among the k_i nearest neighbors that belong to class C_i.
        distances_to_majority_label = [distances[idx] for idx in k_nearest_indices if self.labels[idx] == majority_label]
        min_distance_to_majority_label = min(distances_to_majority_label) if distances_to_majority_label else float('inf')
    
        return majority_label, majority_prob, min_distance_to_majority_label



# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self,num_clusters=5, max_iter=250):
        
        ############################################ ToDo: Prepare FaceNet.########################################
        self.facenet = FaceNet()
        

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        if os.path.getsize("clustering_gallery.pkl") > 0:  # Check if the file is non-empty
            with open("clustering_gallery.pkl", 'rb') as f:
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)
        else:
            print("The pickle file is empty.")

    ###################################################### ToDo ##################################################
    def update(self, face):
        embedding = self.facenet.predict(face)
        self.embeddings = np.vstack([self.embeddings, embedding])
        return None
    
    ##################################################### ToDo ####################################################
    def fit(self):
        # Randomly initialize the cluster centers.
        indices = np.random.choice(self.embeddings.shape[0], self.num_clusters, replace=False)
        self.cluster_center = self.embeddings[indices]
    
        for _ in range(self.max_iter):
            # Assign each embedding to the nearest cluster center.
            distances = np.sqrt(((self.embeddings[:, np.newaxis, :] - self.cluster_center[np.newaxis, :, :]) ** 2).sum(axis=2))
            self.cluster_membership = np.argmin(distances, axis=1)
    
            # Update cluster centers.
            for i in range(self.num_clusters):
                if np.any(self.cluster_membership == i):
                    self.cluster_center[i] = np.mean(self.embeddings[self.cluster_membership == i], axis=0)


    #################################################### ToDo ######################################################
    def predict(self, face):
        # Get the embedding for the face.
        embedding = self.facenet.predict(face)
        
        # Calculate the distances from this embedding to each cluster center.
        distances = np.linalg.norm(self.cluster_center - embedding, axis=1)
        
        # Find the index of the nearest cluster center.
        best_matching_cluster = np.argmin(distances)
        
        return best_matching_cluster, distances
    
    
    
    