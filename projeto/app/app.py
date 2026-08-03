"""
Aplicação Streamlit do Datathon Passos Mágicos.

Três páginas:
  1. Visão geral - números do programa e evolução dos indicadores.
  2. Explorador - filtros por ano, fase, pedra e gênero.
  3. Preditor - estima o risco de defasagem no ano seguinte para um aluno.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# permite importar de src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, 'src'))

from etl import construir_long  # noqa: E402

DATA_PATH = os.path.join(ROOT, 'data', 'pede_long.csv')
MODEL_PATH = os.path.join(ROOT, 'models', 'modelo_risco.joblib')

st.set_page_config(
    page_title='Passos Mágicos - Datathon',
    page_icon='?',
    layout='wide',
)


@st.cache_data
def carregar_dados():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = construir_long()
    return df


@st.cache_resource
def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


df = carregar_dados()
modelo = carregar_modelo()

st.sidebar.title('Passos Mágicos')
st.sidebar.caption('Datathon FIAP - PosTech Fase 5')
pagina = st.sidebar.radio('Navegação', ['Visão geral', 'Explorador', 'Preditor de risco'])


def card(label, valor, unidade=''):
    st.metric(label, f'{valor}{unidade}')


if pagina == 'Visão geral':
    st.title('Visão geral do programa')
    st.markdown(
        'Análise consolidada do PEDE 2022, 2023 e 2024 da Associação Passos Mágicos. '
        'Os números abaixo respondem às perguntas de adequação ao nível, desempenho '
        'e efetividade do programa.'
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        card('Anos cobertos', df['Ano'].nunique())
    with col2:
        card('Alunos únicos', df['RA'].nunique())
    with col3:
        card('INDE médio', round(df['INDE'].mean(), 2))
    with col4:
        pct_def = (df['Em_defasagem'].mean() * 100)
        card('Em defasagem', round(pct_def, 1), '%')

    st.subheader('Evolução dos indicadores por ano')
    indicadores = ['INDE', 'IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN']
    serie = df.groupby('Ano')[indicadores].mean().reset_index()
    serie_long = serie.melt('Ano', var_name='Indicador', value_name='Valor médio')

    chart = (
        alt.Chart(serie_long)
        .mark_line(point=True)
        .encode(
            x=alt.X('Ano:O'),
            y=alt.Y('Valor médio:Q', scale=alt.Scale(zero=False)),
            color='Indicador:N',
            tooltip=['Ano', 'Indicador', alt.Tooltip('Valor médio', format='.3f')],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader('Distribuição da Pedra por ano')
    ord_pedra = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
    df_pedra = df[df['Pedra'].isin(ord_pedra)]
    contagem = (
        df_pedra.groupby(['Ano', 'Pedra']).size().reset_index(name='alunos')
    )
    chart_pedra = (
        alt.Chart(contagem)
        .mark_bar()
        .encode(
            x=alt.X('Ano:O'),
            y=alt.Y('alunos:Q', stack='normalize', title='% de alunos'),
            color=alt.Color('Pedra:N', sort=ord_pedra),
            tooltip=['Ano', 'Pedra', 'alunos'],
        )
        .properties(height=380)
    )
    st.altair_chart(chart_pedra, use_container_width=True)

    st.subheader('Perfil de adequação ao nível')

    def faixa(d):
        if pd.isna(d):
            return 'Sem dado'
        if d >= 0:
            return 'Adequado/Adiantado'
        if d == -1:
            return 'Defasagem leve'
        if d == -2:
            return 'Defasagem moderada'
        return 'Defasagem severa'

    df_f = df.copy()
    df_f['Faixa'] = df_f['Defasagem'].apply(faixa)
    perfil = (
        df_f.groupby(['Ano', 'Faixa']).size().reset_index(name='qtd')
    )
    chart_perfil = (
        alt.Chart(perfil)
        .mark_bar()
        .encode(
            x=alt.X('Ano:O'),
            y=alt.Y('qtd:Q', stack='normalize', title='% de alunos'),
            color=alt.Color('Faixa:N'),
            tooltip=['Ano', 'Faixa', 'qtd'],
        )
        .properties(height=380)
    )
    st.altair_chart(chart_perfil, use_container_width=True)

elif pagina == 'Explorador':
    st.title('Explorador de indicadores')

    with st.sidebar:
        st.subheader('Filtros')
        anos = sorted(df['Ano'].dropna().unique().tolist())
        ano_sel = st.multiselect('Ano', anos, default=anos)

        pedras = sorted([p for p in df['Pedra'].dropna().unique().tolist()])
        pedra_sel = st.multiselect('Pedra', pedras, default=pedras)

        generos = sorted([g for g in df['Genero'].dropna().unique().tolist()])
        genero_sel = st.multiselect('Gênero', generos, default=generos)

    df_f = df[
        df['Ano'].isin(ano_sel)
        & df['Pedra'].isin(pedra_sel)
        & df['Genero'].isin(genero_sel)
    ]

    st.metric('Linhas filtradas', len(df_f))
    if len(df_f) == 0:
        st.warning('Nenhum registro com esse filtro.')
    else:
        st.subheader('Estatísticas dos indicadores')
        st.dataframe(
            df_f[['INDE', 'IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN']]
            .describe().round(3)
        )

        st.subheader('Distribuição do INDE')
        chart = (
            alt.Chart(df_f)
            .mark_bar()
            .encode(
                x=alt.X('INDE:Q', bin=alt.Bin(maxbins=30)),
                y='count()',
                color='Pedra:N',
                tooltip=['count()'],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader('Correlação entre indicadores')
        corr = df_f[['INDE', 'IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN']].corr()
        corr_long = corr.reset_index().melt('index')
        corr_long.columns = ['x', 'y', 'corr']
        chart_corr = (
            alt.Chart(corr_long)
            .mark_rect()
            .encode(
                x='x:N',
                y='y:N',
                color=alt.Color('corr:Q', scale=alt.Scale(scheme='redblue', domain=(-1, 1))),
                tooltip=['x', 'y', alt.Tooltip('corr', format='.2f')],
            )
            .properties(height=350)
        )
        st.altair_chart(chart_corr, use_container_width=True)

        st.subheader('Amostra dos dados')
        st.dataframe(df_f.head(50))

elif pagina == 'Preditor de risco':
    st.title('Preditor de risco de defasagem')
    st.caption(
        'Estimativa de probabilidade de o aluno entrar em defasagem no próximo ano '
        '(Defasagem < 0 no ano t+1) com base nos indicadores do ano corrente.'
    )

    if modelo is None:
        st.error(
            'Modelo não encontrado. Rode `python src/modelo.py` para treinar e salvar '
            'o pipeline em `models/modelo_risco.joblib`.'
        )
    else:
        modo = st.radio('Modo', ['Aluno existente na base', 'Entrada manual'])

        if modo == 'Aluno existente na base':
            ano_pred = st.selectbox(
                'Ano de referência',
                sorted(df['Ano'].unique().tolist()),
                index=len(df['Ano'].unique()) - 1,
            )
            opcoes_ra = sorted(df.loc[df['Ano'] == ano_pred, 'RA'].unique().tolist())
            ra_sel = st.selectbox('RA do aluno', opcoes_ra)
            registro = df[(df['RA'] == ra_sel) & (df['Ano'] == ano_pred)].iloc[0]
            st.write('**Indicadores do aluno:**')
            mostrar = ['INDE', 'IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'IPV', 'IAN',
                       'Idade', 'Tempo_PM', 'Genero', 'Pedra']
            st.json(
                {k: (None if pd.isna(registro.get(k)) else registro.get(k))
                 for k in mostrar}
            )

            X = pd.DataFrame([{
                'INDE': registro['INDE'], 'IAA': registro['IAA'], 'IEG': registro['IEG'],
                'IPS': registro['IPS'], 'IPP': registro['IPP'], 'IDA': registro['IDA'],
                'IPV': registro['IPV'], 'IAN': registro['IAN'],
                'Idade': registro['Idade'], 'Tempo_PM': registro['Tempo_PM'],
                'Genero': registro['Genero'], 'Pedra': registro['Pedra'],
            }])

        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                inde = st.number_input('INDE', 0.0, 10.0, 7.0, 0.1)
                iaa = st.number_input('IAA', 0.0, 10.0, 7.5, 0.1)
                ieg = st.number_input('IEG', 0.0, 10.0, 7.0, 0.1)
                ips = st.number_input('IPS', 0.0, 10.0, 6.5, 0.1)
            with c2:
                ipp = st.number_input('IPP', 0.0, 10.0, 6.5, 0.1)
                ida = st.number_input('IDA', 0.0, 10.0, 7.0, 0.1)
                ipv = st.number_input('IPV', 0.0, 10.0, 7.0, 0.1)
                ian = st.number_input('IAN', 0.0, 10.0, 7.0, 0.1)
            with c3:
                idade = st.number_input('Idade', 5, 30, 12, 1)
                tempo = st.number_input('Anos no programa', 0, 15, 1, 1)
                genero = st.selectbox('Gênero', ['Feminino', 'Masculino', 'Menina', 'Menino', 'Outro'])
                pedra = st.selectbox('Pedra', ['Quartzo', 'Ágata', 'Ametista', 'Topázio'])

            X = pd.DataFrame([{
                'INDE': inde, 'IAA': iaa, 'IEG': ieg, 'IPS': ips, 'IPP': ipp,
                'IDA': ida, 'IPV': ipv, 'IAN': ian,
                'Idade': idade, 'Tempo_PM': tempo,
                'Genero': genero, 'Pedra': pedra,
            }])

        if st.button('Calcular risco'):
            try:
                proba = float(modelo.predict_proba(X)[0, 1])
            except Exception as e:
                st.error(f'Falha ao prever: {e}')
            else:
                st.metric('Probabilidade de defasagem no próximo ano', f'{proba * 100:.1f}%')
                if proba >= 0.7:
                    st.error('Risco alto. Recomenda-se acompanhamento individual.')
                elif proba >= 0.4:
                    st.warning('Risco moderado. Vale revisar plano pedagógico.')
                else:
                    st.success('Risco baixo. Manter monitoramento padrão.')

st.sidebar.markdown('---')
st.sidebar.caption('Construído por Gustavo Camargo - guscamar')
