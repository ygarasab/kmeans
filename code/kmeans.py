from . import caixinha as cx

class Kmeans:

    def __init__(self, dados):

        self.dados = dados
        self.dados_ativos = []
        self.centroids = []

    def clusteriza(self, k):

        self.centroids = cx.busca_k_pontos_aleatorios(self.dados, k)
        self.dados_ativos = self.dados.copy()

        while not self.criterio_de_parada_foi_atingido():

            self.agrupa_amostras()
            self.atualiza_centroides()

        return self.dados_ativos


    def agrupa_amostras(self):

        pass

    def atualiza_centroides(self):

        pass

    def criterio_de_parada_foi_atingido(self):

        pass