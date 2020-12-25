import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from kmedias import KMedias, operadores


from dask import distributed


# noinspection PyProtectedMember,SpellCheckingInspection
def clusteriza(*, dados, numero_de_centroides, centroides_fixos):
    k_medias = KMedias(numero_de_centroides=numero_de_centroides)
    centroides = k_medias._clusteriza_dados(dados=dados, centroides_fixos=centroides_fixos)
    erro = operadores.calcula_erro_da_solucao(dados, centroides)

    return centroides, erro


# noinspection SpellCheckingInspection
def clusteriza_em_paralelo(*, dados, numero_de_centroides, centroides_fixos, cliente, numero_de_execucoes):
    solucoes = pd.DataFrame(index=range(numero_de_execucoes), columns=["Solução", "Erro"])
    futuros = [cliente.submit(
        clusteriza, dados=dados, numero_de_centroides=numero_de_centroides, centroides_fixos=centroides_fixos, pure=False
    ) for _ in range(numero_de_execucoes)]

    for futuro in distributed.as_completed(futuros):
        indice_do_futuro = futuros.index(futuro)
        solucoes.iloc[indice_do_futuro, :] = np.array(futuro.result(), dtype=object)

    solucoes = solucoes.sort_values(by="Erro", ignore_index=True).infer_objects()

    return solucoes


# noinspection SpellCheckingInspection
def gera_grafico_de_dispersao(*, dados, centroides=None):
    if centroides is None:
        rotulos = None
    else:
        rotulos = operadores.rotula_dados(dados=dados, centroides=centroides)

    figura, eixo = plt.subplots(1, 1, squeeze=True)
    figura.tight_layout()

    sns.scatterplot(
        x=dados[:, 1], y=dados[:, 0], palette=sns.color_palette("bright", np.unique(rotulos).size), hue=rotulos, ax=eixo
    )

    eixo.legend(bbox_to_anchor=(1, 1.05), loc=2, borderaxespad=0.0)
    eixo.scatter(x=centroides[:, 1], y=centroides[:, 0], c="k", marker="s")

    return figura, eixo


# noinspection PyTypeChecker,SpellCheckingInspection
def gera_graficos_de_dispersao(*, dados, melhor_solucao, pior_solucao):
    rotulos_da_melhor_solucao = operadores.rotula_dados(dados=dados, centroides=melhor_solucao)
    rotulos_da_pior_solucao = operadores.rotula_dados(dados=dados, centroides=pior_solucao)

    figura, eixos = plt.subplots(1, 2, squeeze=True, sharey=True, figsize=(10, 5))
    figura.tight_layout()

    sns.scatterplot(
        x=dados[:, 0],
        y=dados[:, 1],
        palette=sns.color_palette("bright", np.unique(rotulos_da_melhor_solucao).size),
        hue=rotulos_da_melhor_solucao,
        ax=eixos[0],
        legend=False
    )

    eixos[0].scatter(x=melhor_solucao[:, 1], y=melhor_solucao[:, 0], c="k", marker="s")
    eixos[0].set_title("Melhor solução")

    sns.scatterplot(
        x=dados[:, 0],
        y=dados[:, 1],
        palette=sns.color_palette("bright", np.unique(rotulos_da_pior_solucao).size),
        hue=rotulos_da_pior_solucao,
        ax=eixos[1],
        legend=False
    )

    eixos[1].scatter(x=pior_solucao[:, 1], y=pior_solucao[:, 0], c="k", marker="s")
    eixos[1].set_title("Pior solução")

    return figura, eixos


if __name__ == "__main__":
    # noinspection SpellCheckingInspection
    cliente = distributed.Client(threads_per_worker=1)

    # noinspection SpellCheckingInspection
    aeroporto = pd.read_csv("../dados/coordenadas_aeroporto.csv", index_col=0)
    # noinspection SpellCheckingInspection
    bairros = pd.read_csv("../dados/coordenadas_bairros_final.csv", index_col=0)

    for data_frame in [aeroporto, bairros]:
        # noinspection SpellCheckingInspection
        data_frame["Longitude"] = data_frame["Coordenadas"].apply(lambda x: x.split(", ")[1])
        # noinspection SpellCheckingInspection
        data_frame["Latitude"] = data_frame["Coordenadas"].apply(lambda x: x.split(", ")[0])

        # noinspection SpellCheckingInspection
        data_frame.drop("Coordenadas", axis=1, inplace=True)

        data_frame["Longitude"] = pd.to_numeric(data_frame["Longitude"])
        data_frame["Latitude"] = pd.to_numeric(data_frame["Latitude"])

    # noinspection SpellCheckingInspection
    aeroporto, bairros = aeroporto.to_numpy(), bairros.to_numpy()

    # noinspection SpellCheckingInspection
    figura, eixo = gera_grafico_de_dispersao(dados=bairros, centroides=aeroporto)

    figura.show()

    # noinspection SpellCheckingInspection
    solucoes = clusteriza_em_paralelo(
        dados=bairros, numero_de_centroides=4, centroides_fixos=aeroporto, cliente=cliente, numero_de_execucoes=1000
    )

    # noinspection SpellCheckingInspection
    melhor_solucao, pior_solucao = solucoes.iloc[0, :], solucoes.iloc[-1, :]

    # noinspection SpellCheckingInspection
    figura, eixos = gera_graficos_de_dispersao(
        dados=bairros, melhor_solucao=melhor_solucao[0], pior_solucao=pior_solucao[0]
    )

    figura.show()

