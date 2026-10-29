"""Testes automatizados (dados sintéticos; não dependem do banco)."""
import numpy as np
import pandas as pd
import pytest

import ru_forecast as rf

DIAS = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]


def make_raw(seed=0, fan_out=6):
    """Simula a view: 1 valor por (data, refeição) repetido `fan_out` vezes, chuva em 1 linha."""
    rng = np.random.default_rng(seed)
    linhas = []
    for d in pd.date_range("2026-03-02", "2026-05-27"):
        if d.dayofweek >= 5:
            continue
        chuva = float(rng.choice([0.0, 0.0, 0.0, 3.0, 12.0]))
        for ref, base in (("Almoço", 4500), ("Jantar", 1800)):
            ag = int(base * (1 - 0.01 * d.dayofweek) * (1 - 0.01 * chuva) + rng.normal(0, 40))
            fa = int(ag * (0.15 + 0.004 * chuva) + rng.normal(0, 10))
            for k in range(fan_out):
                linhas.append({
                    "Data": d.date(), "Dia da semana": DIAS[d.dayofweek], "Refeição": ref,
                    "Chuva (mm)": f"{chuva:.1f}".replace(".", ",") if k == 2 else np.nan,
                    "Agendamentos": ag, "Faltas": fa,
                })
    return pd.DataFrame(linhas)


@pytest.fixture(scope="module")
def clean():
    return rf.clean_view(make_raw())[0]


def test_clean_deduplica_e_converte_chuva():
    raw = make_raw()
    c, diag = rf.clean_view(raw)
    assert not c.duplicated(["data", "refeicao"]).any()
    assert len(c) == len(raw) // 6
    assert c["chuva_mm"].notna().all() and c["chuva_mm"].max() == 12.0  # "12,0" -> 12.0
    assert diag["linhas_brutas"] == len(raw)
    assert diag["dia_da_semana_texto_inconsistente"] == 0
    assert set(c["refeicao"]) == {"almoco", "jantar"}


def test_clean_aborta_se_linhas_repetidas_discordam():
    raw = make_raw()
    raw.loc[0, "Agendamentos"] += 1
    with pytest.raises(ValueError, match="DIFERENTES"):
        rf.clean_view(raw)


def test_faltas_invalidas_viram_nan():
    raw = make_raw()
    raw.loc[raw["Data"] == raw["Data"].iloc[0], "Faltas"] = -5
    c, diag = rf.clean_view(raw)
    assert diag["faltas_negativas_ou_maiores_que_agendamentos"] > 0
    assert c.loc[c["data"] == c["data"].min(), "faltas"].isna().all()


def test_features_nao_usam_o_futuro(clean):
    D = pd.Timestamp("2026-04-15")
    alterado = clean.copy()
    alterado.loc[alterado["data"] == D, ["agendamentos", "faltas"]] *= 10
    a = rf.build_features(clean, 1)
    b = rf.build_features(alterado, 1)
    antes = a["data"] <= D  # features do próprio dia D só enxergam até D-1
    pd.testing.assert_frame_equal(a.loc[antes, rf.LAGS], b.loc[antes, rf.LAGS])
    assert not a.loc[~antes, rf.LAGS].equals(b.loc[~antes, rf.LAGS])  # e o futuro muda


def test_walk_forward_nao_vaza(clean):
    D = pd.Timestamp("2026-05-06")
    alterado = clean.copy()
    alterado.loc[alterado["data"] == D, ["agendamentos", "faltas"]] *= 10
    specs = [s for s in rf.default_specs() if s.name in ("baseline_media", "poisson_chuva")]
    p1 = rf.walk_forward(rf.build_features(clean), "agendamentos", specs)
    p2 = rf.walk_forward(rf.build_features(alterado), "agendamentos", specs)
    k = ["data", "refeicao", "modelo"]
    m = p1.merge(p2, on=k, suffixes=("_1", "_2"))
    ate_D = m["data"] <= D
    assert ate_D.sum() > 0
    np.testing.assert_allclose(m.loc[ate_D, "previsto_1"], m.loc[ate_D, "previsto_2"])


def test_modelos_retornam_previsoes_validas(clean):
    feat = rf.build_features(clean)
    pred = rf.walk_forward(feat, "faltas")
    assert set(pred["modelo"]) == {s.name for s in rf.default_specs()}
    assert (pred["previsto"] >= 0).all() and pred["previsto"].notna().all()
    res = rf.avaliar(pred)
    assert {"MAE", "RMSE", "WAPE_%"} <= set(res.columns)
    # os modelos são comparados nas mesmas linhas
    assert pred.groupby("modelo").size().nunique() == 1


def test_wape():
    assert rf.wape([100, 100], [90, 120]) == pytest.approx(30 / 200)


def test_prever_ignora_valor_real_da_propria_data(clean):
    D = pd.Timestamp("2026-05-15")
    r1 = rf.prever(D, "Almoço", 0.0, clean)
    alterado = clean.copy()
    alterado.loc[alterado["data"] == D, ["agendamentos", "faltas"]] *= 10
    r2 = rf.prever(D, "almoco", 0.0, alterado)
    assert r1 == pytest.approx(r2)
    assert set(r1) == {"agendamentos", "faltas"} and min(r1.values()) >= 0


def test_prever_aceita_modelo_por_alvo_e_chuva_ausente(clean):
    r = rf.prever("2026-05-15", "Jantar", None, clean,
                  modelo={"agendamentos": "baseline_media", "faltas": "binomial_negativa"})
    assert r["agendamentos"] > 0


def test_binomial_negativa_estima_alpha_positivo():
    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame({"dow": rng.integers(0, 5, n), "refeicao": "almoco", "x": rng.normal(size=n)})
    mu = np.exp(5 + 0.2 * X["x"])
    y = rng.negative_binomial(5, 5 / (5 + mu))
    m = rf.GLMCount("nb").fit(X, y)
    assert m.alpha_ > 0 and (m.predict(X) > 0).all()
