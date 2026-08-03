# Datathon Passos Mágicos - PosTech FIAP Fase 5

Projeto entregue como parte do Datathon da PosTech (Fase 5). Análise dos dados do PEDE (Pesquisa Extensiva do Desenvolvimento Educacional) da Associação Passos Mágicos para os anos de 2022, 2023 e 2024, com modelo preditivo de risco de defasagem e aplicação Streamlit.

## Estrutura

```
projeto/
├── app/
│   └── app.py                     # aplicação Streamlit (deploy no Community Cloud)
├── data/
│   ├── raw/
│   │   └── BASE DE DADOS PEDE 2024 - DATATHON.xlsx  # planilha original
│   └── pede_long.csv              # dataset longitudinal (gerado pelo ETL)
├── models/
│   └── modelo_risco.joblib        # pipeline final treinado
├── notebooks/
│   ├── 01_eda.ipynb               # análise exploratória cobrindo as 11 perguntas
│   └── 02_modelo_risco.ipynb      # feature engineering + modelagem + avaliação
├── src/
│   ├── etl.py                     # leitura do Excel e construção do dataset longitudinal
│   └── modelo.py                  # pipeline de modelagem e persistência
├── requirements.txt
└── README.md
```

## Como rodar localmente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Gerar o dataset longitudinal
python src/etl.py

# 2. Treinar e salvar o modelo
python src/modelo.py

# 3. Rodar os notebooks (opcional)
jupyter lab notebooks/

# 4. Subir o app Streamlit
streamlit run app/app.py
```

## Dados de entrada

O arquivo de origem (`BASE DE DADOS PEDE 2024 - DATATHON.xlsx`) já está incluído em `data/raw/`. O ETL prioriza esse caminho interno e cai para `../` se não existir, então o projeto roda autocontido.

## Modelo

O modelo prevê a probabilidade de o aluno estar em defasagem no ano seguinte (`Defasagem < 0`) usando como features os indicadores do ano corrente (`INDE, IAA, IEG, IPS, IPP, IDA, IPV, IAN`), idade, tempo no programa e variáveis categóricas (gênero, pedra).

Foram comparados três modelos: regressão logística, random forest e gradient boosting. O melhor (gradient boosting) atinge AUC em torno de 0,84 no conjunto de teste e é o pipeline persistido em `models/modelo_risco.joblib`.

## Deploy no Streamlit Community Cloud

1. Subir o conteúdo de `projeto/` para um repositório público no GitHub.
2. Em [share.streamlit.io](https://share.streamlit.io), apontar para o repositório e definir `app/app.py` como entrypoint.
3. Garantir que `requirements.txt` está na raiz do repositório.
4. O modelo pré-treinado (`models/modelo_risco.joblib`) deve ser commitado para o app não precisar treinar online.

## Perguntas respondidas

O notebook `01_eda.ipynb` cobre as 11 perguntas do enunciado:

1. Adequação do nível (IAN) - perfil de defasagem e evolução por ano.
2. Desempenho acadêmico (IDA) - tendência por ano e fase.
3. Engajamento (IEG) - relação com IDA e IPV.
4. Autoavaliação (IAA) - coerência com IDA e IEG.
5. Aspectos psicossociais (IPS) - sinal antecedente de queda de IDA.
6. Aspectos psicopedagógicos (IPP) - relação com a defasagem (IAN).
7. Ponto de virada (IPV) - drivers do indicador.
8. Multidimensionalidade - peso relativo de IDA, IEG, IPS, IPP no INDE.
9. Modelo preditivo de risco de defasagem (notebook `02`).
10. Efetividade do programa - distribuição de Pedras por ano.
11. Insights extras - tempo no programa, gênero, retenção.

## Autor

Gustavo Camargo (guscamar) - PosTech FIAP Data Analytics.
