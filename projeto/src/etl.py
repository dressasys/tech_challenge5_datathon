"""
ETL do dataset PEDE 2022/2023/2024.
Lê as três abas do Excel e gera um dataset longitudinal (uma linha por aluno-ano)
com colunas padronizadas para alimentar EDA, modelo e app.
"""

import os
import pandas as pd
import numpy as np

EXCEL_PATH_INTERNO = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data', 'raw', 'BASE DE DADOS PEDE 2024 - DATATHON.xlsx'
)
EXCEL_PATH_EXTERNO = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    '..',
    'BASE DE DADOS PEDE 2024 - DATATHON.xlsx'
)
EXCEL_PATH = EXCEL_PATH_INTERNO if os.path.exists(EXCEL_PATH_INTERNO) else EXCEL_PATH_EXTERNO
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

INDICADORES = ['INDE', 'IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN']


def _to_float(serie):
    return pd.to_numeric(serie, errors='coerce')


def carregar_2022(xl):
    df = xl.parse('PEDE2022')
    out = pd.DataFrame({
        'RA': df['RA'],
        'Ano': 2022,
        'Nome': df['Nome'],
        'Fase': df['Fase'].astype(str),
        'Turma': df['Turma'],
        'Genero': df['Gênero'],
        'Ano_ingresso': df['Ano ingresso'],
        'Instituicao': df['Instituição de ensino'],
        'Idade': df['Idade 22'],
        'Pedra': df['Pedra 22'],
        'INDE': _to_float(df['INDE 22']),
        'IAA': _to_float(df['IAA']),
        'IEG': _to_float(df['IEG']),
        'IPS': _to_float(df['IPS']),
        'IPP': np.nan,  # PEDE2022 não tinha IPP
        'IDA': _to_float(df['IDA']),
        'IPV': _to_float(df['IPV']),
        'IAN': _to_float(df['IAN']),
        'Defasagem': _to_float(df['Defas']),
        'Mat': _to_float(df['Matem']),
        'Por': _to_float(df['Portug']),
        'Ing': _to_float(df['Inglês']),
        'Atingiu_PV': df['Atingiu PV'],
        'Indicado_bolsa': df['Indicado'],
        'Fase_ideal': df['Fase ideal'].astype(str),
    })
    return out


def carregar_2023(xl):
    df = xl.parse('PEDE2023')
    # idade pode vir como string com lixo
    out = pd.DataFrame({
        'RA': df['RA'],
        'Ano': 2023,
        'Nome': df['Nome Anonimizado'],
        'Fase': df['Fase'].astype(str),
        'Turma': df['Turma'],
        'Genero': df['Gênero'],
        'Ano_ingresso': df['Ano ingresso'],
        'Instituicao': df['Instituição de ensino'],
        'Idade': _to_float(df['Idade']),
        'Pedra': df['Pedra 2023'],
        'INDE': _to_float(df['INDE 2023']),
        'IAA': _to_float(df['IAA']),
        'IEG': _to_float(df['IEG']),
        'IPS': _to_float(df['IPS']),
        'IPP': _to_float(df['IPP']),
        'IDA': _to_float(df['IDA']),
        'IPV': _to_float(df['IPV']),
        'IAN': _to_float(df['IAN']),
        'Defasagem': _to_float(df['Defasagem']),
        'Mat': _to_float(df['Mat']),
        'Por': _to_float(df['Por']),
        'Ing': _to_float(df['Ing']),
        'Atingiu_PV': df['Atingiu PV'],
        'Indicado_bolsa': df['Indicado'],
        'Fase_ideal': df['Fase Ideal'].astype(str),
    })
    return out


def carregar_2024(xl):
    df = xl.parse('PEDE2024')
    out = pd.DataFrame({
        'RA': df['RA'],
        'Ano': 2024,
        'Nome': df['Nome Anonimizado'],
        'Fase': df['Fase'].astype(str),
        'Turma': df['Turma'],
        'Genero': df['Gênero'],
        'Ano_ingresso': df['Ano ingresso'],
        'Instituicao': df['Instituição de ensino'],
        'Idade': _to_float(df['Idade']),
        'Pedra': df['Pedra 2024'],
        'INDE': _to_float(df['INDE 2024']),
        'IAA': _to_float(df['IAA']),
        'IEG': _to_float(df['IEG']),
        'IPS': _to_float(df['IPS']),
        'IPP': _to_float(df['IPP']),
        'IDA': _to_float(df['IDA']),
        'IPV': _to_float(df['IPV']),
        'IAN': _to_float(df['IAN']),
        'Defasagem': _to_float(df['Defasagem']),
        'Mat': _to_float(df['Mat']),
        'Por': _to_float(df['Por']),
        'Ing': _to_float(df['Ing']),
        'Atingiu_PV': df['Atingiu PV'],
        'Indicado_bolsa': df['Indicado'],
        'Fase_ideal': df['Fase Ideal'].astype(str),
    })
    return out


def construir_long(excel_path=None):
    excel_path = excel_path or EXCEL_PATH
    xl = pd.ExcelFile(excel_path)
    df = pd.concat(
        [carregar_2022(xl), carregar_2023(xl), carregar_2024(xl)],
        ignore_index=True
    )
    # higiene básica
    df['Pedra'] = df['Pedra'].replace({'#NULO!': np.nan, 'D9891/2A': np.nan})
    df['Pedra'] = df['Pedra'].replace({'Agata': 'Ágata'})
    df = df.dropna(subset=['INDE'])
    # tempo na PM no ano corrente
    df['Tempo_PM'] = df['Ano'] - pd.to_numeric(df['Ano_ingresso'], errors='coerce')
    df.loc[df['Tempo_PM'] < 0, 'Tempo_PM'] = np.nan
    # flag binaria de defasagem (alvo do modelo)
    df['Em_defasagem'] = (df['Defasagem'] < 0).astype(int)
    return df


def salvar(df, output_dir=None):
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    caminho = os.path.join(output_dir, 'pede_long.csv')
    df.to_csv(caminho, index=False)
    return caminho


if __name__ == '__main__':
    df = construir_long()
    caminho = salvar(df)
    print('linhas:', len(df))
    print('alunos unicos:', df['RA'].nunique())
    print('por ano:')
    print(df['Ano'].value_counts().sort_index())
    print('arquivo:', caminho)
