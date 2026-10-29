"""Pipeline completo: leitura da view -> limpeza/diagnóstico -> EDA -> modelos -> validação -> exemplo.

Uso:
    export DATABASE_URL="postgresql+psycopg2://usuario:senha@localhost:5432/banco"
    python run_pipeline.py                      # lê a view vw_agendamentos_faltas_chuva
    python run_pipeline.py --csv view.csv       # alternativa: CSV exportado da view
    python run_pipeline.py --data 2026-05-15 --refeicao almoco --horizonte 1
"""
import argparse
import os

import pandas as pd

import ru_forecast as rf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default=rf.VIEW_NAME)
    ap.add_argument("--csv", help="CSV exportado da view (em vez de ler do PostgreSQL)")
    ap.add_argument("--saida", default="saida")
    ap.add_argument("--data", default="2026-05-15", help="data do exemplo de previsão")
    ap.add_argument("--refeicao", default="almoco")
    ap.add_argument("--horizonte", type=int, default=1, help="dias entre a última informação usada e a data prevista")
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)

    # 1) dados
    if a.csv:
        raw = pd.read_csv(a.csv)
    else:
        from sqlalchemy import create_engine
        raw = rf.load_view(create_engine(os.environ["DATABASE_URL"]), a.view)
    clean, diag = rf.clean_view(raw)
    clean.to_csv(f"{a.saida}/base_limpa.csv", index=False)
    print("=== DIAGNÓSTICO DA VIEW ===")
    for k, v in diag.items():
        print(f"{k}: {v}")

    # 2) análise exploratória
    print("\n=== EDA ===")
    for nome, tab in rf.eda(clean).items():
        print(f"\n[{nome}]\n{tab}")
        tab.to_csv(f"{a.saida}/eda_{nome}.csv")
    rf.salvar_graficos(clean, f"{a.saida}/graficos")

    # 3) validação temporal de todos os modelos, para os dois alvos
    feat = rf.build_features(clean, a.horizonte)
    pred = pd.concat(
        [rf.walk_forward(feat, alvo, horizon=a.horizonte) for alvo in rf.TARGETS], ignore_index=True
    )
    pred.to_csv(f"{a.saida}/previsoes_vs_real.csv", index=False)
    metricas = rf.avaliar(pred)
    metricas.to_csv(f"{a.saida}/metricas.csv", index=False)
    print("\n=== MÉTRICAS (validação temporal, mesmas linhas para todos os modelos) ===")
    print(metricas.round(2).to_string(index=False))
    print("\n=== MÉTRICAS POR REFEIÇÃO ===")
    print(rf.avaliar(pred, ("alvo", "refeicao", "modelo")).round(2).to_string(index=False))

    # 4) exemplo de uso (o valor real só é consultado depois, para medir o erro)
    melhor = {alvo: metricas[metricas["alvo"] == alvo].iloc[0]["modelo"] for alvo in rf.TARGETS}
    data, ref = pd.Timestamp(a.data), rf._norm(a.refeicao)
    linha = clean[(clean["data"] == data) & (clean["refeicao"] == ref)]
    chuva = None if linha.empty or pd.isna(linha["chuva_mm"].iloc[0]) else float(linha["chuva_mm"].iloc[0])
    prev = rf.prever(data, ref, chuva, clean, a.horizonte, modelo=melhor)
    print(f"\n=== EXEMPLO: {ref} de {data.date()} (modelos: {melhor}; chuva usada: {chuva}) ===")
    linhas = []
    for alvo, nome in (("agendamentos", "Agendamentos"), ("faltas", "Faltas")):
        real = float(linha[alvo].iloc[0]) if not linha.empty else float("nan")
        linhas.append({"Indicador": nome, "Previsão": round(prev[alvo], 1), "Valor real": real,
                       "Erro absoluto": round(abs(prev[alvo] - real), 1)})
    exemplo = pd.DataFrame(linhas)
    print(exemplo.to_string(index=False))
    exemplo.to_csv(f"{a.saida}/exemplo_previsao.csv", index=False)
    print("\nObs.: o 'melhor' modelo foi escolhido pelo WAPE no mesmo período de teste; "
          "a estimativa de erro é otimista. A chuva usada é a OBSERVADA (em produção seria previsão).")


if __name__ == "__main__":
    main()
