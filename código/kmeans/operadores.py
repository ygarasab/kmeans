import numba as nb
import numpy as np

from . import caixinha


# noinspection SpellCheckingInspection
@nb.jit(nopython=True)
def roda_k_means(dados, centroides, centroides_fixos, casas_decimais=4, numero_maximo_de_iteracoes=200):
    iteracao = 0

    while iteracao < numero_maximo_de_iteracoes:
        rotulos = rotula_dados(dados, centroides)
        centroides_centralizados = caixinha.centraliza_centroides(dados, centroides, centroides_fixos, rotulos)
        ha_igualdade = caixinha.verifica_igualdade_aproximada_entre_grupos(
            centroides, centroides_centralizados, casas_decimais
        )
        centroides = centroides_centralizados

        if ha_igualdade is True:
            break
        else:
            iteracao += 1

    return centroides


# noinspection SpellCheckingInspection
@nb.jit(nopython=True)
def gera_centroides(dados, numero_de_centroides):
    comprimento, dimensionalidade = dados.shape
    indices = np.arange(comprimento, dtype=np.int_)
    indices_aleatorios = np.random.choice(indices, size=numero_de_centroides, replace=False)
    centroides = dados[indices_aleatorios].copy()

    return centroides


# noinspection SpellCheckingInspection
@nb.jit(nopython=True)
def rotula_dados(dados, centroides):
    distancias = caixinha.calcula_distancia_entre_grupos(dados, centroides)
    numero_de_observacoes = dados.shape[0]
    rotulos = np.empty(numero_de_observacoes, dtype=np.int_)

    for d in nb.prange(numero_de_observacoes):
        rotulos[d] = distancias[d].argmin()

    return rotulos
