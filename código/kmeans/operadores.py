import typing as t

import numba as nb
import numpy as np

from . import caixinha
from . import checagens


# noinspection SpellCheckingInspection
def roda_k_means(*, dados, centroides, centroides_fixos, casas_decimais=4, numero_maximo_de_iteracoes=200):
    dados = checagens.verifica_tipo(dados=(dados, "parâmetro", np.ndarray))
    centroides = checagens.verifica_tipo(centroides=(centroides, "parâmetro", np.ndarray))
    centroides_fixos = checagens.verifica_tipo(centroides_fixos=(centroides_fixos, "parâmetro", np.ndarray))
    centroides_fixos = checagens.verifica_dtype(centroides_fixos=(centroides_fixos, "parametro", np.bool_))
    casas_decimais = checagens.verifica_tipo(casas_decimais=(casas_decimais, "parâmetro", t.SupportsInt))
    numero_maximo_de_iteracoes = checagens.verifica_tipo(
        numero_maximo_de_iteracoes=(numero_maximo_de_iteracoes, "parâmetro", t.SupportsInt)
    )

    checagens.verifica_ndim(
        dados=(dados, "parâmetro", 2),
        centroides=(centroides, "parâmetro", 2),
        centroides_fixos=(centroides_fixos, "parâmetro", 1),
    )
    checagens.verifica_nao_negatividade(
        casas_decimais=(casas_decimais, "parâmetro"),
        numero_maximo_de_iteracoes=(numero_maximo_de_iteracoes, "parâmetro"),
    )
    checagens.verifica_comprimento_igual_a(
        centroides=(centroides, "parâmetro"), centroides_fixos=(centroides_fixos, "parâmetro")
    )

    centroides_centralizados = _roda_k_means(
        dados, centroides, centroides_fixos, casas_decimais, numero_maximo_de_iteracoes
    )

    return centroides_centralizados


# noinspection SpellCheckingInspection
@nb.jit(nopython=True)
def _roda_k_means(dados, centroides, centroides_fixos, casas_decimais=4, numero_maximo_de_iteracoes=200):
    iteracao = 0

    while iteracao < numero_maximo_de_iteracoes:
        rotulos = _rotula_dados(dados, centroides)
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
def gera_centroides(*, dados, numero_de_centroides):
    dados = checagens.verifica_tipo(dados=(dados, "parâmetro", np.ndarray))
    numero_de_centroides = checagens.verifica_tipo(k=(numero_de_centroides, "parâmetro", t.SupportsInt))

    checagens.verifica_ndim(dados=(dados, "parâmetro", 2))
    checagens.verifica_nao_negatividade(numero_de_centroides=(numero_de_centroides, "parâmetro"))

    centroides = _gera_centroides(dados, numero_de_centroides)

    return centroides


# noinspection SpellCheckingInspection
@nb.jit(nopython=True)
def _gera_centroides(dados, numero_de_centroides):
    comprimento, dimensionalidade = dados.shape
    indices = np.arange(comprimento, dtype=np.int_)
    indices_aleatorios = np.random.choice(indices, size=numero_de_centroides, replace=False)
    centroides = dados[indices_aleatorios].copy()

    return centroides


# noinspection SpellCheckingInspection
def rotula_dados(*, dados, centroides):
    dados = checagens.verifica_tipo(dados=(dados, "parâmetro", np.ndarray))
    centroides = checagens.verifica_tipo(centroides=(centroides, "parâmetro", np.ndarray))

    checagens.verifica_ndim(dados=(dados, "parâmetro", 2))
    checagens.verifica_ndim(centroides=(centroides, "parâmetro", 2))

    rotulos = _rotula_dados(dados, centroides)

    return rotulos


# noinspection SpellCheckingInspection
@nb.jit(nopython=True)
def _rotula_dados(dados, centroides):
    distancias = caixinha.calcula_distancia_entre_grupos(dados, centroides)
    numero_de_observacoes = dados.shape[0]
    rotulos = np.empty(numero_de_observacoes, dtype=np.int_)

    for d in nb.prange(numero_de_observacoes):
        rotulos[d] = distancias[d].argmin()

    return rotulos
