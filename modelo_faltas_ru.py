"""Modelo simples de Regressão Linear para estimar as faltas diárias no RU da UFSM.

FALTAS = TOTAL_AGENDAMENTOS - TOTAL_REFEICAO (coluna já calculada na view do CSV)

Variáveis independentes:
  - médias móveis de faltas de 7, 30 e 90 dias;
  - TOTAL_AGENDAMENTOS do dia (precisa ser conhecido no momento da previsão);
  - DIA_SEMANA, transformado em variáveis binárias (one-hot).
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Variáveis numéricas usadas pelo modelo
MEDIAS_MOVEIS = ["MEDIA_MOVEL_7_DIAS", "MEDIA_MOVEL_30_DIAS", "MEDIA_MOVEL_90_DIAS"]
FEATURES_NUMERICAS = MEDIAS_MOVEIS + ["TOTAL_AGENDAMENTOS"]
COLUNAS_NECESSARIAS = ["DATA", "TOTAL_REFEICAO", "FALTAS", "DIA_SEMANA"] + FEATURES_NUMERICAS
DIAS_DA_SEMANA = range(7)  # 0 a 6, como na view (0 = domingo)
PROPORCAO_TREINO = 0.8


def ler_argumentos():
    """Lê o caminho do CSV informado no terminal."""
    parser = argparse.ArgumentParser(description="Previsão de faltas diárias no RU (Regressão Linear).")
    parser.add_argument("caminho_csv", help="Caminho do arquivo CSV exportado da view")
    return parser.parse_args()


def carregar_dados(caminho):
    """Lê o CSV, padroniza os nomes das colunas e trata o texto 'NULL' como valor ausente."""
    df = pd.read_csv(caminho, na_values=["NULL", "null"])
    df.columns = df.columns.str.strip().str.upper()  # ex.: "data" -> "DATA"

    faltando = [c for c in COLUNAS_NECESSARIAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {faltando}")

    df["DATA"] = pd.to_datetime(df["DATA"])
    return df


def preparar_dados(df):
    """Remove registros com valores ausentes e ordena por data."""
    df = df[COLUNAS_NECESSARIAS].dropna()
    return df.sort_values("DATA").reset_index(drop=True)


def validar_faltas(df):
    """Avisa se a coluna FALTAS diverge de TOTAL_AGENDAMENTOS - TOTAL_REFEICAO."""
    esperado = df["TOTAL_AGENDAMENTOS"] - df["TOTAL_REFEICAO"]
    divergentes = int((df["FALTAS"] != esperado).sum())
    if divergentes:
        print(f"AVISO: {divergentes} registro(s) com FALTAS diferente de "
              "TOTAL_AGENDAMENTOS - TOTAL_REFEICAO.", file=sys.stderr)


def dividir_treino_teste(df):
    """Separa cronologicamente: os primeiros 80% para treino e os 20% finais para teste."""
    corte = int(len(df) * PROPORCAO_TREINO)
    return df.iloc[:corte], df.iloc[corte:]


def montar_variaveis(df):
    """Monta a matriz de variáveis independentes.

    DIA_SEMANA vira colunas binárias (one-hot), pois o número bruto (0 a 6) imporia
    uma ordem que não existe. As categorias são fixadas em 0..6 para que treino e
    teste tenham sempre as mesmas colunas; o domingo (0) é a categoria de referência.
    """
    dias = pd.Categorical(df["DIA_SEMANA"].astype(int), categories=DIAS_DA_SEMANA)
    dummies = pd.get_dummies(dias, prefix="DIA", drop_first=True, dtype=float)
    dummies.index = df.index
    return pd.concat([df[FEATURES_NUMERICAS], dummies], axis=1)


def treinar_modelo(treino):
    """Ajusta a Regressão Linear com os dados de treino."""
    modelo = LinearRegression()
    modelo.fit(montar_variaveis(treino), treino["FALTAS"])
    return modelo


def tratar_previsoes(previsoes):
    """Impede valores negativos e arredonda para números inteiros."""
    return np.clip(previsoes, 0, None).round().astype(int)


def prever(modelo, teste):
    """Gera as previsões da Regressão Linear."""
    return tratar_previsoes(modelo.predict(montar_variaveis(teste)))


def prever_baseline(treino, teste):
    """Baseline: média histórica de faltas do mesmo dia da semana (calculada no treino)."""
    media_por_dia = treino.groupby("DIA_SEMANA")["FALTAS"].mean()
    previsoes = teste["DIA_SEMANA"].map(media_por_dia).fillna(treino["FALTAS"].mean())
    return tratar_previsoes(previsoes.values)


def avaliar(reais, estimadas):
    """Calcula MAE, RMSE, R² e WAPE (erro absoluto total / total real, em %)."""
    return {
        "MAE": mean_absolute_error(reais, estimadas),
        "RMSE": np.sqrt(mean_squared_error(reais, estimadas)),
        "R²": r2_score(reais, estimadas),
        "WAPE (%)": 100 * np.abs(reais - estimadas).sum() / reais.sum(),
    }


def montar_tabela(teste, estimadas):
    """Monta a tabela final com DATA, FALTAS_REAIS e FALTAS_ESTIMADAS."""
    return pd.DataFrame({
        "DATA": teste["DATA"].dt.strftime("%d/%m/%Y").values,
        "FALTAS_REAIS": teste["FALTAS"].values,
        "FALTAS_ESTIMADAS": estimadas,
    })


def main():
    args = ler_argumentos()

    df = preparar_dados(carregar_dados(args.caminho_csv))
    validar_faltas(df)
    treino, teste = dividir_treino_teste(df)
    print(f"Registros válidos: {len(df)} | treino: {len(treino)} | teste: {len(teste)}")

    modelo = treinar_modelo(treino)
    estimadas = prever(modelo, teste)
    baseline = prever_baseline(treino, teste)

    # Compara o modelo com o baseline para saber se a regressão realmente agrega valor
    metricas = pd.DataFrame({
        "Regressão Linear": avaliar(teste["FALTAS"], estimadas),
        "Baseline (média por dia da semana)": avaliar(teste["FALTAS"], baseline),
    }).T
    print("\nMétricas no conjunto de teste:")
    print(metricas.round(4).to_string())

    print("\nFaltas reais x estimadas:")
    print(montar_tabela(teste, estimadas).to_string(index=False))


if __name__ == "__main__":
    main()