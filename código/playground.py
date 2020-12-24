import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from kmedias import KMedias, operadores


from dask.distributed import Client, as_completed


# noinspection SpellCheckingInspection
def gera_grafico_de_dispersao(centroides, dados):
    rotulos = operadores.rotula_dados(dados=dados, centroides=centroides)

    figura, eixo = plt.subplots(1, 1)

    sns.scatterplot(
        x=dados[:, 0], y=dados[:, 1], palette=sns.color_palette("bright", np.unique(rotulos).size), hue=rotulos, ax=eixo
    )

    eixo.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    eixo.scatter(x=centroides[:, 0], y=centroides[:, 1], c="k", marker="s")

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

# k_medias = KMedias(numero_de_centroides=4)
# noinspection SpellCheckingInspection
# solucoes = k_medias.clusteriza_dados(dados=bairros, centroides_fixos=aeroporto).sort_values(by="Erro")
# noinspection SpellCheckingInspection
# centroides = solucoes.iloc[0, 0]
# noinspection SpellCheckingInspection
# figura, eixo = gera_grafico_de_dispersao(centroides, bairros)

# eixo.scatter(aeroporto[0, 0], aeroporto[0, 1], s=150, c="indigo")
# figura.show()


# noinspection SpellCheckingInspection
def clusteriza_em_paralelo(*, dados, numero_de_centroides, centroide_fixo, cliente, numero_de_execucoes=1000):
    solucoes = pd.DataFrame(columns=["Solução", "Erro"])
    futuros = [cliente.submit(
        clusteriza, dados=dados, numero_de_centroides=numero_de_centroides, centroide_fixo=centroide_fixo, pure=False
    ) for _ in range(numero_de_execucoes)]

    for futuro in as_completed(futuros):
        indice_do_futuro = futuros.index(futuro)
        solucoes.iloc[indice_do_futuro, :] = np.array(futuro.result(), dtype=object)

    solucoes = solucoes.sort_values(by="Erro").infer_objects()

    return solucoes


# noinspection SpellCheckingInspection
def clusteriza(*, dados, numero_de_centroides, centroide_fixo):
    k_medias = KMedias(numero_de_centroides=numero_de_centroides)
    centroides = k_medias._clusteriza_dados(dados=dados, centroides_fixos=centroide_fixo)
    erro = operadores.calcula_erro_da_solucao(dados, centroides)

    return centroides, erro


if __name__ == "__main__":
    cliente = Client(threads_per_worker=1)

    solucoes = clusteriza_em_paralelo(dados=bairros, numero_de_centroides=4, centroide_fixo=aeroporto, cliente=cliente)
    centroides = solucoes.iloc[0, 0]
    figura, eixo = gera_grafico_de_dispersao(centroides, bairros)

    eixo.scatter(aeroporto[0, 0], aeroporto[0, 1], s=150, c="indigo")
    figura.show()
