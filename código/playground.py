import numpy as np
import pandas as pd
import seaborn as sns

from kmeans import KMeans
from kmeans.caixinha import rotula_dados
from matplotlib import pyplot as plt
from time import sleep


# noinspection SpellCheckingInspection
def gerar_grafico_de_dispersao(centroides, dados):
    rotulos = rotula_dados(dados=dados, centroides=centroides)

    figura, eixo = plt.subplots(1, 1)

    sns.scatterplot(x=dados[:, 0], y=dados[:, 1],
                    palette=sns.color_palette("bright", np.unique(rotulos).size), hue=rotulos, ax=eixo)

    eixo.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    eixo.scatter(x=centroides[:, 0], y=centroides[:, 1], c='k', marker='s')

    return figura, eixo


# noinspection SpellCheckingInspection
caminho_para_os_dados = "../dados/coordenadas_bairros.csv"

# noinspection SpellCheckingInspection
dados = pd.read_csv(caminho_para_os_dados, index_col=0)

# noinspection SpellCheckingInspection
dados["Latitude"] = dados["Coordenadas"].apply(lambda x: x.split(", ")[0])
# noinspection SpellCheckingInspection
dados["Longitude"] = dados["Coordenadas"].apply(lambda x: x.split(", ")[1])

# noinspection SpellCheckingInspection
dados.drop("Coordenadas", axis=1, inplace=True)
dados.infer_objects()

dados["Latitude"] = pd.to_numeric(dados["Latitude"])
dados["Longitude"] = pd.to_numeric(dados["Longitude"])

# noinspection SpellCheckingInspection
aeroporto = dados.loc["Aeroporto", ["Latitude", "Longitude"]].to_numpy().reshape(1, -1)
# noinspection SpellCheckingInspection
bairros = dados.drop("Aeroporto").to_numpy()

while True:
    print("Começou:")
    k_means = KMeans(numero_de_centroides=4)
    # noinspection SpellCheckingInspection
    centroides = k_means.clusteriza(dados=bairros, centroides_fixos=aeroporto)
    figura, eixo = gerar_grafico_de_dispersao(centroides, bairros)

    eixo.scatter(aeroporto[0, 0], aeroporto[0, 1], s=150, c="indigo")
    figura.show()
