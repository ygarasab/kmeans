import numba as nb
import numpy as np


# noinspection SpellCheckingInspection
@nb.jit(nopython=True, parallel=True)
def arredonda_matriz(matriz, casas_decimais):
    vetor = matriz.copy().ravel()
    comprimento = vetor.shape[0]

    for i in nb.prange(comprimento):
        vetor[i] = np.round_(vetor[i], casas_decimais)

    matriz_arredondada = vetor.reshape(matriz.shape)

    return matriz_arredondada


# noinspection SpellCheckingInspection
@nb.jit(nopython=True, parallel=True)
def calcula_distancia_entre_grupos(grupo_um, grupo_dois, tira_raiz=True):
    comprimento_um, comprimento_dois = grupo_um.shape[0], grupo_dois.shape[0]
    distancias = np.empty(shape=(comprimento_um, comprimento_dois))

    for i_um in nb.prange(comprimento_um):
        for i_dois in nb.prange(comprimento_dois):
            distancias[i_um, i_dois] = calcula_distancia_entre_observacoes(
                grupo_um[i_um, :], grupo_dois[i_dois, :], tira_raiz
            )

    return distancias


# noinspection SpellCheckingInspection
@nb.jit(nopython=True, parallel=True)
def calcula_distancia_entre_observacoes(vetor_um, vetor_dois, tira_raiz):
    comprimento = vetor_um.shape[0]
    diferencas = np.empty(shape=comprimento)

    for i in nb.prange(comprimento):
        diferencas[i] = np.square(vetor_um[i] - vetor_dois[i])

    distancia = np.sum(diferencas)

    if tira_raiz is True:
        distancia = np.sqrt(distancia)

    return distancia


# noinspection SpellCheckingInspection,PyUnresolvedReferences
@nb.jit(nopython=True)
def verifica_igualdade_aproximada_entre_grupos(grupo_um, grupo_dois, casas_decimais):
    grupo_um, grupo_dois = arredonda_matriz(grupo_um, casas_decimais), arredonda_matriz(grupo_dois, casas_decimais)
    ha_igualdade = bool((grupo_um == grupo_dois).all())

    return ha_igualdade


# noinspection SpellCheckingInspection
@nb.jit(nopython=True, parallel=True)
def centraliza_centroides(dados, centroides, centroides_fixos, rotulos):
    numero_de_centroides, dimensionalidade = centroides.shape
    rotulos_unicos = np.arange(numero_de_centroides)[np.logical_not(centroides_fixos)]
    numero_de_rotulos = rotulos_unicos.shape[0]
    centroides_centralizados = np.empty((numero_de_centroides, dimensionalidade))
    centroides_centralizados[centroides_fixos, :] = centroides[centroides_fixos, :]

    for r in nb.prange(numero_de_rotulos):
        rotulo = rotulos_unicos[r]
        membros_do_agrupamento = np.where(rotulos == rotulo)[0]
        agrupamento = dados[membros_do_agrupamento, :]
        centroides_centralizados[rotulo, :] = tira_media_das_colunas(agrupamento)

    return centroides_centralizados


# noinspection SpellCheckingInspection
@nb.jit(nopython=True, parallel=True)
def tira_media_das_colunas(dados):
    dimensionalidade = dados.shape[1]
    media_das_colunas_dos_dados = np.empty(shape=dimensionalidade)

    for f in nb.prange(dimensionalidade):
        media_das_colunas_dos_dados[f] = dados[:, f].mean()

    return media_das_colunas_dos_dados
