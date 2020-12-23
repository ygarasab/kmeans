import numba as nb
import numpy as np

from . import caixinha


# noinspection SpellCheckingInspection
@nb.jit(nopython=True)
def roda_k_means(dados, centroides, centroides_fixos, casas_decimais=4, numero_maximo_de_iteracoes=1000):
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


# noinspection SpellCheckingInspection
@nb.jit(nopython=True, parallel=True)
def calcula_erro_da_solucao(dados, centroides):
    rotulos, numero_de_centroides = rotula_dados(dados, centroides), centroides.shape[0]
    erro_dos_agrupamentos = np.empty(numero_de_centroides)

    for rotulo in nb.prange(numero_de_centroides):
        membros_do_agrupamento = np.where(rotulos == rotulo)[0]

        if membros_do_agrupamento.shape[0] == 0:
            return np.inf
        else:
            centroide = centroides[rotulo, :].reshape(1, -1)
            membros = dados[membros_do_agrupamento, :]
            erro_dos_agrupamentos[rotulo] = caixinha.calcula_distancia_entre_grupos(
                centroide, membros, tira_raiz=False
            ).sum()

    erro_total = erro_dos_agrupamentos.sum()

    return erro_total
