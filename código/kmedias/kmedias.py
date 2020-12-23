import typing as t

import numpy as np
import pandas as pd

from . import checagens
from . import operadores


# noinspection SpellCheckingInspection
class KMedias:
    def __init__(self, *, numero_de_centroides):
        self.numero_de_centroides = numero_de_centroides
        self.__centroides, self.__centroides_fixos, self.__dados = None, None, None

    @property
    def numero_de_centroides(self):
        return self.__numero_de_centroides

    @numero_de_centroides.setter
    def numero_de_centroides(self, novo_numero_de_centroides):
        novo_numero_de_centroides = checagens.verifica_tipo(
            numero_de_centroides=(novo_numero_de_centroides, "atributo", t.SupportsInt)
        )

        checagens.verifica_nao_negatividade(numero_de_centroides=(novo_numero_de_centroides, "atributo"))

        self.__numero_de_centroides = novo_numero_de_centroides

    @property
    def centroides(self):
        return self.__centroides

    @centroides.setter
    def centroides(self, novos_centroides):
        if novos_centroides is None:
            novos_centroides = operadores.gera_centroides(
                dados=self.dados, numero_de_centroides=self.numero_de_centroides
            )
            self.centroides_fixos = novos_centroides.shape[0] * [False]
        elif isinstance(novos_centroides, np.ndarray):
            novos_centroides = checagens.verifica_tipo(centroides=(novos_centroides, "atributo", np.ndarray))

            checagens.verifica_ndim(centroides=(novos_centroides, "atributo", 2))
            checagens.verifica_comprimento_menor_ou_igual_a(
                centroides=(novos_centroides, "atributo"), dados=(self.dados, "atributo")
            )

            centroides_faltantes = self.numero_de_centroides - novos_centroides.shape[0]

            if centroides_faltantes > 0:
                centroides_complementares = operadores.gera_centroides(
                    dados=self.dados, numero_de_centroides=centroides_faltantes
                )
                self.centroides_fixos = novos_centroides.shape[0] * [True] + centroides_faltantes * [False]
                novos_centroides = np.concatenate((novos_centroides, centroides_complementares), axis=0)
            elif centroides_faltantes < 0:
                novos_centroides = novos_centroides[:centroides_faltantes]
                self.centroides_fixos = novos_centroides.shape[0] * [False]

        self.__centroides = novos_centroides

    @property
    def centroides_fixos(self):
        return self.__centroides_fixos

    @centroides_fixos.setter
    def centroides_fixos(self, novos_centroides_fixos):
        novos_centroides_fixos = checagens.verifica_tipo(
            centroides_fixos=(novos_centroides_fixos, "atributo", np.ndarray)
        )

        checagens.verifica_ndim(centroides_fixos=(novos_centroides_fixos, "atributo", 1))
        checagens.verifica_dtype(centroides_fixos=(novos_centroides_fixos, "atributo", np.bool_))
        checagens.verifica_comprimento_igual_a(
            centroides_fixos=(novos_centroides_fixos, "atributo"), centroides=(self.centroides, "atributo")
        )

        self.__centroides_fixos = novos_centroides_fixos

    @property
    def dados(self):
        return self.__dados

    @dados.setter
    def dados(self, novos_dados):
        novos_dados = checagens.verifica_tipo(dados=(novos_dados, "atributo", np.ndarray))

        checagens.verifica_ndim(dados=(novos_dados, "atributo", 2))

        self.__dados = novos_dados

    @property
    def rotulos(self):
        return operadores.rotula_dados(dados=self.dados, centroides=self.centroides)

    def clusteriza_dados(self, *, dados, centroides_fixos, numero_de_execucoes=1000):
        melhor_solucao, erro_da_melhor_solucao = None, None
        data_frame = pd.DataFrame(index=range(numero_de_execucoes), columns=["Solução", "Erro"])

        for iteracao in range(numero_de_execucoes):
            print(f"Execução {iteracao}...", end="\r")

            solucao = self._clusteriza_dados(dados=dados, centroides_fixos=centroides_fixos)
            erro_da_solucao = operadores.calcula_erro_da_solucao(dados, solucao)

            data_frame.iloc[iteracao, :] = np.array([solucao, erro_da_solucao], dtype=object)

            if melhor_solucao is None or erro_da_solucao < erro_da_melhor_solucao:
                melhor_solucao, erro_da_melhor_solucao = solucao, erro_da_solucao

        return data_frame.infer_objects()

    def _clusteriza_dados(self, *, dados, centroides_fixos):
        self.dados = dados
        self.centroides = centroides_fixos

        self.centroides = operadores.roda_k_means(
            dados=dados, centroides=self.centroides, centroides_fixos=self.centroides_fixos
        )

        return self.centroides
