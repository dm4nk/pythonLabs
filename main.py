import operator
from calendar import timegm
from time import gmtime

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer


def test_data(n: int = 300, centers: int = 5):
    X, y_true = make_blobs(n_samples=n, centers=centers, cluster_std=.6, random_state=timegm(gmtime()))
    return X, y_true


def plot_scatter(x, y):
    plt.scatter(x, y)
    plt.show()


def plot_line(x, y):
    plt.plot(x, y)
    plt.show()


def define_best_number_of_classes_by_elbow(X):
    kmeans = KMeans(n_init='auto')
    # calinski_harabasz score computes the ratio of dispersion between and within clusters
    visualizer = KElbowVisualizer(kmeans, metric='calinski_harabasz', timings=False)
    visualizer.fit(X)
    visualizer.show()
    return visualizer.elbow_value_


def define_best_number_of_classes_by_silhouette(X, min_value=2, max_value=10):
    # scores = [
    #     SilhouetteVisualizer(
    #         KMeans(n_clusters=hypothetical_clusters_num, n_init='auto', random_state=1)
    #     )
    #     .fit(X)
    #     .silhouette_score_ for hypothetical_clusters_num in range(min, max)
    # ]
    scores = [
        silhouette_score(X=X, labels=KMeans(n_clusters=hypothetical_clusters_num, n_init='auto').fit_predict(X)) for
        hypothetical_clusters_num in range(min_value, max_value)
    ]
    plot_line([i for i in range(min_value, max_value)], scores)

    index, value = max(enumerate(scores), key=operator.itemgetter(1))
    return min_value + index


def show_centers(X, num_of_clusters):
    kmeans = KMeans(n_clusters=num_of_clusters, random_state=10)
    y_kmeans = kmeans.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
    plt.show()


def main():
    X, y = test_data()
    plot_scatter(X[:, 0], X[:, 1])
    e = define_best_number_of_classes_by_elbow(X)
    s = define_best_number_of_classes_by_silhouette(X)
    print(f'Clusters by Elbow: {e}; by Silhouette: {s}')
    show_centers(X, e)


if __name__ == "__main__":
    main()
