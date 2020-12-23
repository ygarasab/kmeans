import numpy as np
import typing as t


# noinspection SpellCheckingInspection
def verifica_comprimento_igual_a(**parametros):
    numero_de_parametros = len(parametros.keys())

    if numero_de_parametros != 2:
        raise ValueError(f"Apenas um parâmetro pode ser passado para esta função. Foram recebidos "
                         f"{numero_de_parametros}.")

    parametro, outro_parametro = parametros.keys()

    valor, descricao = parametros[parametro]
    outro_valor, outra_descricao = parametros[outro_parametro]

    if outro_valor is not None and len(valor) != len(outro_valor):
        raise ValueError(f"O {descricao} {parametro} precisa ter um comprimento igual ao {outra_descricao} "
                         f"{outro_parametro}.")


# noinspection SpellCheckingInspection
def verifica_comprimento_maior_ou_igual_a(**parametros):
    numero_de_parametros = len(parametros.keys())

    if numero_de_parametros != 2:
        raise ValueError(f"Apenas um parâmetro pode ser passado para esta função. Foram recebidos "
                         f"{numero_de_parametros}.")

    parametro, outro_parametro = parametros.keys()

    valor, descricao = parametros[parametro]
    outro_valor, outra_descricao = parametros[outro_parametro]

    if outro_valor is not None and len(valor) < len(outro_valor):
        raise ValueError(f"O {descricao} {parametro} precisa ter um comprimento, no mínimo, igual ao {outra_descricao} "
                         f"{outro_parametro}.")


# noinspection SpellCheckingInspection
def verifica_comprimento_menor_ou_igual_a(**parametros):
    numero_de_parametros = len(parametros.keys())

    if numero_de_parametros != 2:
        raise ValueError(f"Apenas um parâmetro pode ser passado para esta função. Foram recebidos "
                         f"{numero_de_parametros}.")

    parametro, outro_parametro = parametros.keys()

    valor, descricao = parametros[parametro]
    outro_valor, outra_descricao = parametros[outro_parametro]

    if outro_valor is not None and len(valor) > len(outro_valor):
        raise ValueError(f"O {descricao} {parametro} precisa ter um comprimento, no máximo, igual ao {outra_descricao} "
                         f"{outro_parametro}.")


# noinspection SpellCheckingInspection
def verifica_dtype(**parametro_dict):
    numero_de_parametros = len(parametro_dict.keys())

    if numero_de_parametros != 1:
        raise ValueError(f"Apenas um parâmetro pode ser passado para esta função. Foram recebidos "
                         f"{numero_de_parametros}.")

    parametro = list(parametro_dict.keys())[0]
    valor, descricao, dtype = parametro_dict[parametro]

    if dtype == np.int_ and valor.dtype != dtype:
        if valor.dtype == np.float_:
            return valor.astype(np.int_)
        else:
            raise TypeError(f"O {descricao} {parametro} precisa ser um numpy array com atributo dtype igual a "
                            f"{dtype}. O dtype do numpy array recebido é {valor.dtype}.")

    if valor.dtype != dtype:
        raise TypeError(f"O {descricao} {parametro} precisa ser um numpy array com atributo dtype igual a {dtype}. "
                        f"O dtype do numpy array recebido é {valor.dtype}.")
    else:
        return valor


# noinspection SpellCheckingInspection
def verifica_maior_ou_igual_a(**parametros):
    numero_de_parametros = len(parametros.keys())

    if numero_de_parametros != 2:
        raise ValueError(f"Apenas um parâmetro pode ser passado para esta função. Foram recebidos "
                         f"{numero_de_parametros}.")

    parametro, outro_parametro = parametros.keys()

    valor, descricao = parametros[parametro]
    outro_valor, outra_descricao = parametros[outro_parametro]

    if outro_valor is not None and valor < outro_valor:
        raise ValueError(f"O {descricao} {parametro} precisa receber um valor, no mínimo, igual ao {outra_descricao} "
                         f"{outro_parametro}.")


# noinspection SpellCheckingInspection
def verifica_menor_ou_igual_a(**parametros):
    numero_de_parametros = len(parametros.keys())

    if numero_de_parametros != 2:
        raise ValueError(f"Apenas um parâmetro pode ser passado para esta função. Foram recebidos "
                         f"{numero_de_parametros}.")

    parametro, outro_parametro = parametros.keys()

    valor, descricao = parametros[parametro]
    outro_valor, outra_descricao = parametros[outro_parametro]

    if outro_valor is not None and valor > outro_valor:
        raise ValueError(f"O {descricao} {parametro} precisa receber um valor, no máximo, igual ao {outra_descricao} "
                         f"{outro_parametro}.")


# noinspection SpellCheckingInspection
def verifica_nao_negatividade(**parametros):
    for parametro in parametros.keys():
        valor, descricao = parametros[parametro]

        if valor < 0:
            raise ValueError(f"O {descricao} {parametro} precisa receber um número não-negativo.")


# noinspection SpellCheckingInspection
def verifica_ndim(**parametros):
    for parametro in parametros.keys():
        valor, descricao, ndim = parametros[parametro]

        if valor.ndim != ndim:
            raise ValueError(f"O o atributo ndim do {descricao} {parametro} precisa ser igual a {ndim}.")


# noinspection SpellCheckingInspection
def verifica_tipo(**parametro_dict):
    numero_de_parametros = len(parametro_dict.keys())

    if numero_de_parametros != 1:
        raise ValueError(f"Apenas um parâmetro pode ser passado para esta função. Foram recebidos "
                         f"{numero_de_parametros}.")

    parametro = list(parametro_dict.keys())[0]
    valor, descricao, tipos = parametro_dict[parametro]

    if tipos == t.SupportsFloat:
        if not isinstance(valor, tipos):
            raise TypeError(f"O {descricao} {parametro} precisa receber um número de ponto flutuante ou um objeto que "
                            f"possa ser convertido para tal.")
        else:
            return float(valor)

    if tipos == t.SupportsInt:
        if not isinstance(valor, tipos):
            raise TypeError(f"O {descricao} {parametro} precisa receber um número inteiro ou um objeto que possa ser "
                            f"convertido para tal.")
        else:
            return int(valor)

    if tipos == np.ndarray:
        if not isinstance(valor, np.ndarray):
            if not isinstance(valor, (list, tuple)):
                raise TypeError(f"O {descricao} {parametro} precisa receber um array numpy ou um objeto que possa ser "
                                f"convertido para tal.")
            else:
                return np.array(valor)

    if tipos == bool:
        if not isinstance(valor, bool):
            if isinstance(valor, np.bool_):
                return bool(valor)
            else:
                raise TypeError(f"O {descricao} {parametro} precisa receber um objeto booleano ou um objeto que possa "
                                f"ser convertido para tal.")
        else:
            return valor

    if not isinstance(valor, tipos):
        raise TypeError(f"O {descricao} {parametro} precisa receber um objeto de classe {tipos} ou que herde dela.")
    else:
        return valor


# noinspection SpellCheckingInspection
def verifica_tipo_operador(operador, valor, tipo):
    if not isinstance(valor, tipo):
        raise TypeError(f"O operador '{operador}' precisa ser do tipo {tipo}.")
