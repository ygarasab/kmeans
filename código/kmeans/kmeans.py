import numpy as np
import typing as t

from . import caixinha, checagens


# noinspection SpellCheckingInspection
class KMeans:
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
            novos_centroides = caixinha.gera_centroides(
                dados=self.dados, numero_de_centroides=self.numero_de_centroides
            )
            self.centroides_fixos = novos_centroides.shape[0] * [False]
        elif isinstance(novos_centroides, np.ndarray):
            novos_centroides = checagens.verifica_tipo(centroides=(novos_centroides, "atributo", np.ndarray))

            checagens.verifica_ndim(centroides=(novos_centroides, "atributo", 2))
            checagens.verifica_comprimento_menor_ou_igual_a(centroides=(novos_centroides, "atributo"),
                                                            dados=(self.dados, "atributo"))

            centroides_faltantes = self.numero_de_centroides - novos_centroides.shape[0]

            if centroides_faltantes > 0:
                centroides_complementares = caixinha.gera_centroides(
                    dados=self.dados, numero_de_centroides=centroides_faltantes
                )
                novos_centroides = np.concatenate((novos_centroides, centroides_complementares))
                self.centroides_fixos = novos_centroides.shape[0] * [True] + centroides_faltantes * [False]
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
        checagens.verifica_comprimento_igual_a(centroides_fixos=(novos_centroides_fixos, "atributo"),
                                               numero_de_centroides=(self.centroides, "atributo"))

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
        return caixinha.rotula_dados(dados=self.dados, centroides=self.centroides)

    def clusteriza(self, *, dados, centroides_fixos):
        self.dados = dados
        self.centroides = centroides_fixos

        iteracao = 0
        while True:
            print(f"Iteração {iteracao}")

            centroides_centralizados = caixinha.centraliza_centroides(
                dados=self.dados,
                centroides=self.centroides,
                centroides_fixos=self.centroides_fixos,
                rotulos=self.rotulos
            )

            if self.deve_continuar(centroides_centralizados) is False:
                self.centroides = centroides_centralizados
                break
            else:
                self.centroides = centroides_centralizados
                iteracao += 1

        return self.centroides

    def deve_continuar(self, novos_centroides):
        novos_centroides = checagens.verifica_tipo(novos_centroides=(novos_centroides, "parâmetro", np.ndarray))

        checagens.verifica_ndim(novos_centroides=(novos_centroides, "parâmetro", 2))
        checagens.verifica_comprimento_igual_a(novos_centroides=(novos_centroides, "parâmetro"),
                                               centroides=(self.centroides, "atributo"))

        ha_igualdade = caixinha.verifica_igualdade_aproximada_entre_grupos(grupo_um=self.centroides,
                                                                           grupo_dois=novos_centroides)
        deve_continuar = not ha_igualdade

        return deve_continuar
