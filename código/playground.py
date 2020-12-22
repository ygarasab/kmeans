import numpy as np
import seaborn as sns

from sklearn import datasets

from kmeans import KMeans
from kmeans.caixinha import rotula_dados
from matplotlib import pyplot as plt


# noinspection SpellCheckingInspection
def gerar_grafico_de_dispersao(centroides, dados):
    rotulos = rotula_dados(dados=dados, centroides=centroides)

    # calculated_centroid = centralize_clustering(a_clustering, labels, data)

    figure, axis = plt.subplots(1, 1)

    sns.scatterplot(x=dados[:, 0], y=dados[:, 1],
                    palette=sns.color_palette("bright", np.unique(rotulos).size), hue=rotulos, ax=axis)

    axis.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    axis.scatter(x=centroides[:, 0], y=centroides[:, 1], c='k', marker='s')
    # axis.scatter(x=calculated_centroid[:, 0], y=calculated_centroid[:, 1], c='k', marker='x')

    figure.show()


X, _ = datasets.load_iris(return_X_y=True)

while True:
    k_means = KMeans(numero_de_centroides=3)

    # noinspection SpellCheckingInspection
    centroides_encontrados = k_means.clusteriza(dados=X, centroides_fixos=None)

    gerar_grafico_de_dispersao(centroides_encontrados, X)

