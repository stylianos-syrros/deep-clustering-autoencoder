from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.stats import mode
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "train.csv")
train_df = pd.read_csv(train_path)

X_train = train_df.iloc[:, :20].values
Y_train = train_df.iloc[:, -1].values

k_values = [2, 4, 6, 8, 10]


def train_autoencoder(x_train, M):
    input_dim = x_train.shape[1]
    
    input_layer = Input(shape=(input_dim,))

    encoder_layer1 = Dense(100, activation='relu')(input_layer)
    encoder_output = Dense(M, activation='relu')(encoder_layer1)

    decoder_layer1 = Dense(100, activation='relu')(encoder_output)
    decoder_output = Dense(input_dim, activation='sigmoid')(decoder_layer1)

    autoencoder = Model(input_layer, decoder_output)

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True)

    return autoencoder


def calculate_f_measure(cluster_labels, true_labels):
    f_measure = 0

    for i in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_labels_subset = true_labels[cluster_indices]

        label_counts = np.bincount(cluster_labels_subset)
        majority_label = np.argmax(label_counts)

        TP = np.sum(cluster_labels_subset == majority_label)
        FP = np.sum(cluster_labels_subset != majority_label)
        FN = np.sum((true_labels == majority_label) & (cluster_labels != i))

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0

        if precision + recall > 0:
            f_measure += 2 * (precision * recall) / (precision + recall)

    f_measure = f_measure / len(np.unique(cluster_labels))

    return f_measure


def calculate_purity(cluster_labels, true_labels):
    N = len(true_labels)
    purity = 0

    for i in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_labels_subset = true_labels[cluster_indices]

        if len(cluster_labels_subset) == 0:
            continue

        label_counts = np.bincount(cluster_labels_subset)
        majority_label = np.argmax(label_counts)

        num_same_category = np.sum(cluster_labels_subset == majority_label)

        purity += num_same_category

    purity /= N

    return purity


def kmeans(X_train, Y_train):
    global k_values
    results = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(X_train)

        cluster_labels_train = kmeans.labels_

        purity = calculate_purity(cluster_labels_train,Y_train)
        f_measure = calculate_f_measure(cluster_labels_train,Y_train)

        results.append((k, purity,f_measure))

    return results

def agglomerative(X_train, Y_train):
    global k_values
    results = []

    for k in k_values:
        agglomerative = AgglomerativeClustering(n_clusters=k)
        agglomerative.fit(X_train)

        cluster_labels_train = agglomerative.labels_

        purity = calculate_purity(cluster_labels_train,Y_train)
        f_measure = calculate_f_measure(cluster_labels_train,Y_train)

        results.append((k, purity, f_measure))

    return results



def apply_clustering(X_train, Y_train, M):
    autoencoder = train_autoencoder(X_train, M)
    
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-3].output)

    encoded_train = encoder.predict(X_train)

    kmeans_results = []
    agglomerative_results = []
        
    for k in k_values:
        sum_kmeans_purity = 0
        sum_kmeans_f_measure = 0
        sum_agglomerative_purity = 0
        sum_agglomerative_f_measure = 0
        for _ in range(10):
            kmeans = KMeans(n_clusters=k, n_init=10)
            agglomerative = AgglomerativeClustering(n_clusters=k)

            kmeans_cluster_labels = kmeans.fit_predict(encoded_train)
            agglomerative_cluster_labels = agglomerative.fit_predict(encoded_train)

            kmeans_purity = calculate_purity(kmeans_cluster_labels, Y_train)
            agglomerative_purity = calculate_purity(agglomerative_cluster_labels, Y_train)

            kmeans_f_measure = calculate_f_measure(kmeans_cluster_labels, Y_train)
            agglomerative_f_measure = calculate_f_measure(agglomerative_cluster_labels, Y_train)

            sum_kmeans_purity += kmeans_purity
            sum_kmeans_f_measure += kmeans_f_measure
            sum_agglomerative_purity += agglomerative_purity
            sum_agglomerative_f_measure += agglomerative_f_measure

        kmeans_results.append((k, sum_kmeans_purity/10, sum_kmeans_f_measure/10))
        agglomerative_results.append((k, sum_agglomerative_purity/10, sum_agglomerative_f_measure/10))

    if M == 2:
        plt.scatter(encoded_train[:, 0],encoded_train[:,1],c=Y_train, cmap="jet")
        plt.title("M=2 diagram")
        plt.show()
    
    return kmeans_results, agglomerative_results


def run_clustering_method(func, X_train, Y_train, iterations=10):
    results_list = []
    for i in range(iterations):
        results = func(X_train, Y_train)
        results_list.append(results)

    averages = []
    for k in k_values:
        total_purity = 0
        total_f_measure = 0
        for results in results_list:
            for k_result, purity, f_measure in results:
                if k_result == k:
                    total_purity += purity
                    total_f_measure += f_measure
                    break
        averages.append((k, total_purity/iterations, total_f_measure/iterations))

    return averages


results_kmeans = run_clustering_method(kmeans, X_train, Y_train)
results_agglomerative = agglomerative(X_train, Y_train)

print("K-means results:")
for k, purity, f_measure in results_kmeans:
    print(f"For k={k}:")
    print(f"Purity: {purity:.2f}")
    print(f"F-measure: {f_measure:.2f}")
    print()

print("Agglomerative results:")
for k, purity, f_measure in results_agglomerative:
    print(f"For k={k}:")
    print(f"Purity: {purity:.2f}")
    print(f"F-measure: {f_measure:.2f}")
    print()


M_values = [2, 10, 50]

for M in M_values:
    print(f"Running for M = {M}")
    
    kmeans_results, agglomerative_results = apply_clustering(X_train, Y_train, M)
    
    print("K-means results:")
    for k, purity, f_measure in kmeans_results:
        print(f"For k={k}:")
        print(f"Purity: {purity:.2f}")
        print(f"F-measure: {f_measure:.2f}")
        print()

    print("Agglomerative results:")
    for k, purity, f_measure in agglomerative_results:
        print(f"For k={k}:")
        print(f"Purity: {purity:.2f}")
        print(f"F-measure: {f_measure:.2f}")
        print()
