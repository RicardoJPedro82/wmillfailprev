"Implementação de uma classe que disponibilizará funções para podermos usar de modo iterativo ao longo do projecto"

import os
import pandas as pd


class windmills():
    'Obter informação da pasta RawData'

    def get_data():
        data = {}
        'obter um dicinário com todos as tabelas dos ficheiros em dataframes'
        #Obter o caminho do ficheiros
        root_dir = os.path.abspath("..//..")
        #Fazer uma lista dos ficheiros na pasta
        csv_path = os.path.join(root_dir, 'wmillfailprev', 'rawdata')
        #Lista dos ficheiros a importar
        lista = os.listdir(csv_path)
        lista = lista[1:]
        #Obter os nomes dos dataframes e criar o dicionário
        for i in range(len(lista)):
            if len(lista[i]) < 22:
                df_name = lista[i][:(len(lista[i]) - 4)]
            else:
                df_name = lista[i][12:len(lista[i]) - 4]
            data[df_name] = pd.read_csv(os.path.join(csv_path, lista[i]), sep=';')
        return  data

