import os
import shlex
from tqdm import tqdm
import _pickle as cPickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
import numpy as np
import cv2
import gzip



def getFiles(folder, pattern, labelfile):

    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels



# a) Codebook generation

files_train, labels_train = getFiles("train/", "_SIFT_patch_pr.pkl.gz", "icdar17_labels_train.txt")
print('#train: {}'.format(len(files_train)))


def loadRandomDescriptors(files, max_descriptors):

    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')
            
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors



descriptors = loadRandomDescriptors(files_train, 500000)
print('> loaded {} descriptors:'.format(len(descriptors)))


def dictionary(descriptors, n_clusters):

    # Initialize MiniBatchKMeans with the desired number of clusters and explicit n_init
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=10, batch_size=1536)

    # Fit the model to the data
    kmeans.fit(descriptors)

    # Retrieve the cluster centers
    cluster_centers = kmeans.cluster_centers_

    return cluster_centers



print('> compute dictionary')
dictionary = dictionary(descriptors, 100)


#b) VLAD Encoding

def assignments(descriptors, clusters):

    # compute nearest neighbors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors, clusters, k=1)
    
    # create hard assignment
    assignment = np.zeros( (len(descriptors), len(clusters)) )

    for m in matches:
        descriptor_index = m[0].queryIdx
        cluster_index = m[0].trainIdx
        assignment[descriptor_index, cluster_index] = 1
            
    return assignment




def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        with gzip.open(f, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')

        a = assignments(desc, mus)
        T, D = desc.shape
        f_enc = np.zeros((K, D))
        for k in range(K):
            for i in range(a.shape[0]):
                if a[i][k] == 1:
                    f_enc[k] += desc[i] - mus[k]

        f_enc = f_enc.flatten()
        # Power normalization
        if powernorm:
            f_enc = np.sign(f_enc) * np.abs(f_enc) ** 0.5  # Power normalization with p = 0.5
            f_enc = f_enc / np.linalg.norm(f_enc)

        encodings.append(f_enc)

    return np.array(encodings)


print('> compute VLAD for test')

files_test, labels_test = getFiles("test/", "_SIFT_patch_pr.pkl.gz", "icdar17_labels_test.txt")
print('#test: {}'.format(len(files_test)))

encoding_without_normalization = vlad(files_test, dictionary, False)


def distances(encoding):

    # Flatten each of the 3600 elements into a single vector
    reshaped_encodings = [np.array(encoding).flatten() for encoding in encoding]

    # Compute the cosine distance matrix
    cosine_distance_matrix = cosine_distances(reshaped_encodings)
    
    return cosine_distance_matrix



def evaluate(encs, labels):

    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()
    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


print('> evaluate encoding without normalizaton')

evaluate(encoding_without_normalization, labels_test)

# VLAD Normalization

encodings_normalized = vlad(files_test, dictionary, True)

print('> evaluate encoding with normalizaton')

evaluate(encodings_normalized, labels_test)


# d) Exemplar classification

print('> compute VLAD for train (for E-SVM)')

encodings_train = vlad(files_train, dictionary, True)


def esvm(encs_test, encs_train, C=1000):
    
    new_global_descriptors = []

    for test_encoding in tqdm(encs_test):
        
        X_train = np.vstack([test_encoding, encs_train])  # Combine test encoding with training encodings
        y_train = np.array([1] + [-1] * len(encs_train))  # Label test encoding as 1, and all training encodings as 0

        # Initialize the SVM
        svm = LinearSVC(C = C, class_weight='balanced', max_iter=1000, dual=True)

        # Train the SVM
        svm.fit(X_train, y_train)

        # Normalize the weight vector
        normalized_weights = normalize(svm.coef_, norm='l2')

        # Append the normalized weights to your new global descriptors
        new_global_descriptors.append(normalized_weights[0])
    
    return new_global_descriptors

print('> esvm computation')
new_enc = esvm(encodings_normalized, encodings_train)

print("> evaluate SVC Encodings")
evaluate(new_enc, labels_test)





















