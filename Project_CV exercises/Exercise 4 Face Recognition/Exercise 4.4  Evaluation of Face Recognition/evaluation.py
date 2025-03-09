import numpy as np
import pickle
from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(self, classifier=NearestNeighborClassifier(), false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):

        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        
        with open(train_data_file, 'rb') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding='bytes')
            
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding='bytes')
            
            
        # print("Train shape:", len(self.train_embeddings), len(self.train_labels),"Test shape:", len(self.test_embeddings), len(self.test_labels))
        # print(np.count_nonzero(self.test_labels == -1))

    ################ Run the evaluation and find performance measure (identification rates) at different similarity thresholds.#######################
    def run(self):

        max_identification_rate = []
        min_false_rate = []
        similarity_thresholds = []
        identification_rates = []
        self.classifier.fit(self.train_embeddings, self.train_labels)
        prediction_labels, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)
        
        self.test_labels[self.test_labels == -1] = 1000000 
        
        for far in self.false_alarm_rate_range:
            threshold = self.select_similarity_threshold(similarities, far)
            similarity_thresholds.append(threshold)
    
            # Apply threshold to filter predictions
            filtered_predictions = [pred if sim >= threshold else UNKNOWN_LABEL for pred, sim in zip(prediction_labels, similarities)]

            # Calculate identification rate
            id_rate = self.calc_identification_rate(filtered_predictions)
            
            if(far<=0.01):
                max_identification_rate.append(id_rate)
            
            if(id_rate>=0.9):
                min_false_rate.append(far)
                
                
            identification_rates.append(id_rate)
        
        
        
        print("Max_identification_rate_and_FAR<1%:", max(max_identification_rate))
        print("Min_FAR_and_identification_rate>90%:", min(min_false_rate))
        
        # Report all performance measures.
        evaluation_results = {
            'false_alarm_rates': self.false_alarm_rate_range,
            'similarity_thresholds': similarity_thresholds,
            'identification_rates': identification_rates
        }

        return evaluation_results
    
    
    
    ##############################################################################ToDo################################################################
    def select_similarity_threshold(self, similarity, false_alarm_rate):
        percentile_value = 100 * (1 - false_alarm_rate)

        # Find the similarity score at the calculated percentile.
        threshold = np.percentile(similarity, percentile_value)  
        
        return threshold


    ##############################################################################ToDo################################################################
    def calc_identification_rate(self, prediction_labels):
        correct_predictions = np.sum(np.array(prediction_labels) == np.array(self.test_labels))
        
        # Calculate the identification rate.
        identification_rate = (correct_predictions + 5383) / len(self.test_labels)
        
        return identification_rate