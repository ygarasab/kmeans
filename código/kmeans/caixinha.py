import numba as nb
import numpy as np
import typing as t

from . import checagens


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


# noinspection SpellCheckingInspection,DuplicatedCode
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
    distancias = _calcula_distancia_entre_grupos(dados, centroides)
    numero_de_observacoes = dados.shape[0]
    rotulos = np.empty(numero_de_observacoes, dtype=np.int_)

    for d in nb.prange(numero_de_observacoes):
        rotulos[d] = distancias[d].argmin()

    return rotulos


# noinspection SpellCheckingInspection,DuplicatedCode
def calcula_distancia_entre_grupos(*, grupo_um, grupo_dois):
    grupo_um = checagens.verifica_tipo(grupo_um=(grupo_um, "parâmetro", np.ndarray))
    grupo_dois = checagens.verifica_tipo(grupo_dois=(grupo_dois, "parâmetro", np.ndarray))

    checagens.verifica_ndim(grupo_um=(grupo_um, "parâmetro", 2))
    checagens.verifica_ndim(grupo_dois=(grupo_dois, "parâmetro", 2))

    distancias = _calcula_distancia_entre_grupos(grupo_um, grupo_dois)

    return distancias


# noinspection SpellCheckingInspection
@nb.jit(nopython=True, parallel=True)
def _calcula_distancia_entre_grupos(grupo_um, grupo_dois):
    comprimento_um, comprimento_dois = grupo_um.shape[0], grupo_dois.shape[0]
    distancias = np.empty(shape=(comprimento_um, comprimento_dois))

    for i_um in nb.prange(comprimento_um):
        for i_dois in nb.prange(comprimento_dois):
            distancias[i_um, i_dois] = _calcula_distancia_entre_observacoes(grupo_um[i_um], grupo_dois[i_dois])

    return distancias


# noinspection SpellCheckingInspection
def calcula_distancia_entre_observacoes(*, vetor_um, vetor_dois):
    vetor_um = checagens.verifica_tipo(vetor_um=(vetor_um, "parâmetro", np.ndarray))
    vetor_dois = checagens.verifica_tipo(vetor_dois=(vetor_dois, "parâmetro", np.ndarray))

    checagens.verifica_ndim(vetor_um=(vetor_um, "parâmetro", 1))
    checagens.verifica_ndim(vetor_dois=(vetor_dois, "parâmetro", 1))
    checagens.verifica_comprimento_igual_a(vetor_um=(vetor_um, "parâmetro"), vetor_dois=(vetor_dois, "parâmetro"))

    distancia = _calcula_distancia_entre_observacoes(vetor_um, vetor_dois)

    return distancia


# noinspection SpellCheckingInspection
@nb.jit(nopython=True, parallel=True)
def _calcula_distancia_entre_observacoes(vetor_um, vetor_dois):
    comprimento = vetor_um.shape[0]
    diferencas = np.empty(shape=comprimento)

    for i in nb.prange(comprimento):
        diferencas[i] = np.square(vetor_um[i] - vetor_dois[i])

    distancia = np.sqrt(np.sum(diferencas))

    return distancia


# noinspection SpellCheckingInspection,DuplicatedCode
def verifica_igualdade_aproximada_entre_grupos(*, grupo_um, grupo_dois, casas_decimais=4):
    grupo_um = checagens.verifica_tipo(grupo_um=(grupo_um, "parâmetro", np.ndarray))
    grupo_dois = checagens.verifica_tipo(grupo_dois=(grupo_dois, "parâmetro", np.ndarray))
    casas_decimais = checagens.verifica_tipo(casas_decimais=(casas_decimais, "parâmetro", t.SupportsInt))

    checagens.verifica_ndim(grupo_um=(grupo_um, "parâmetro", 2))
    checagens.verifica_ndim(grupo_dois=(grupo_dois, "parâmetro", 2))
    checagens.verifica_nao_negatividade(casas_decimais=(casas_decimais, "parâmetro"))

    ha_igualdade = _verifica_igualdade_aproximada_entre_grupos(grupo_um, grupo_dois, casas_decimais)

    return ha_igualdade


# noinspection SpellCheckingInspection,PyUnresolvedReferences
@nb.jit(nopython=True)
def _verifica_igualdade_aproximada_entre_grupos(grupo_um, grupo_dois, casas_decimais):
    grupo_um, grupo_dois = _arredonda_matriz(grupo_um, casas_decimais), _arredonda_matriz(grupo_dois, casas_decimais)
    ha_igualdade = (grupo_um == grupo_dois).all()

    return ha_igualdade


# noinspection SpellCheckingInspection
def arredonda_matriz(*, matriz, casas_decimais):
    matriz = checagens.verifica_tipo(matriz=(matriz, "parâmetro", np.ndarray))
    casas_decimais = checagens.verifica_tipo(casas_decimais=(casas_decimais, "parâmetro", t.SupportsInt))

    checagens.verifica_nao_negatividade(casas_decimais=(casas_decimais, "parâmetro"))

    matriz_arredondada = _arredonda_matriz(matriz, casas_decimais)

    return matriz_arredondada


# noinspection SpellCheckingInspection
@nb.jit(nopython=True, parallel=True)
def _arredonda_matriz(matriz, casas_decimais):
    vetor = matriz.ravel()
    comprimento = vetor.shape[0]

    for i in nb.prange(comprimento):
        vetor[i] = np.round_(vetor[i], casas_decimais)

    matriz_arredondada = vetor.reshape(matriz.shape)

    return matriz_arredondada


# noinspection SpellCheckingInspection
def centraliza_centroides(*, dados, centroides, centroides_fixos, rotulos):
    dados = checagens.verifica_tipo(dados=(dados, "parâmetro", np.ndarray))
    rotulos = checagens.verifica_tipo(rotulos=(rotulos, "parâmetro", np.ndarray))
    centroides = checagens.verifica_tipo(centroides=(centroides, "parâmetro", np.ndarray))
    centroides_fixos = checagens.verifica_tipo(centroides_fixos=(centroides_fixos, "parâmetro", np.ndarray))

    rotulos = checagens.verifica_dtype(rotulos=(rotulos, "parâmetro", np.int_))
    centroides_fixos = checagens.verifica_dtype(centroides_fixos=(centroides_fixos, "parâmetro", np.bool_))

    checagens.verifica_ndim(
        dados=(dados, "parâmetro", 2),
        rotulos=(rotulos, "parâmetro", 1),
        centroides=(centroides, "parâmetro", 2),
        centroides_fixos=(centroides_fixos, "parâmetro", 1)
    )
    checagens.verifica_comprimento_igual_a(dados=(dados, "parâmetro"), rotulos=(rotulos, "parâmetro"))
    checagens.verifica_comprimento_igual_a(
        centroides=(centroides, "parâmetro"), centroides_fixos=(centroides_fixos, "parâmetro")
    )

    centroides_centralizados = _centraliza_centroides(dados, centroides, centroides_fixos, rotulos)

    return centroides_centralizados


# noinspection SpellCheckingInspection
@nb.jit(nopython=True, parallel=True)
def _centraliza_centroides(dados, centroides, centroides_fixos, rotulos):
    rotulos_unicos = np.unique(rotulos)[np.logical_not(centroides_fixos)]
    numero_de_centroides, dimensionalidade = rotulos_unicos.shape[0], dados.shape[1]
    centroides_centralizados = np.empty((numero_de_centroides, dimensionalidade))
    centroides_centralizados[centroides_fixos, :] = centroides[centroides_fixos, :]

    for r in nb.prange(numero_de_centroides):
        rotulo = rotulos_unicos[r]
        membros_do_cluster = np.where(rotulos == rotulo)[0]
        cluster = dados[membros_do_cluster, :]
        centroides_centralizados[rotulo, :] = _tira_media_das_colunas(cluster)

    return centroides_centralizados


# noinspection SpellCheckingInspection
def tira_media_das_colunas(*, dados):
    dados = checagens.verifica_tipo(dados=(dados, "parâmetro", np.ndarray))

    checagens.verifica_ndim(dados=(dados, "parâmetro", 2))

    media_das_colunas_dos_dados = _tira_media_das_colunas(dados)

    return media_das_colunas_dos_dados


# noinspection SpellCheckingInspection
@nb.jit(nopython=True, parallel=True)
def _tira_media_das_colunas(dados):
    dimensionalidade = dados.shape[1]
    media_das_colunas_dos_dados = np.empty(shape=dimensionalidade)

    for f in nb.prange(dimensionalidade):
        media_das_colunas_dos_dados[f] = dados[:, f].mean()

    return media_das_colunas_dos_dados
