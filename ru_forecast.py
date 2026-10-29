"""Previsão de agendamentos e faltas (no-shows) do RU — Projeto Integrador I (UFSM).

Fonte: view `vw_agendamentos_faltas_chuva` no PostgreSQL.

Princípios:
  * granularidade final = 1 observação por (data, refeição); a view repete cada
    combinação muitas vezes (junções com tabelas de granularidade mais fina), então
    o código DEDUPLICA e ABORTA se as linhas repetidas discordarem entre si;
  * nenhuma feature da data D usa informação posterior a D - horizonte;
  * a validação é walk-forward (janela expansível), reajustando só com dados passados.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor

VIEW_NAME = "vw_agendamentos_faltas_chuva"
RENAME = {
    "Data": "data",
    "Dia da semana": "dia_semana_txt",
    "Refeição": "refeicao",
    "Chuva (mm)": "chuva_mm",
    "Agendamentos": "agendamentos",
    "Faltas": "faltas",
}
TARGETS = ("agendamentos", "faltas")
CAL = ["dow", "refeicao"]
LAGS = ["ag_media_mesmo_dia", "ag_ult7", "fa_media_mesmo_dia", "fa_ult7", "taxa_falta_ult7"]
RAIN = ["chuva_mm", "choveu", "chuva_ausente"]
_DIAS = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]


def _norm(s) -> str:
    """minúsculas, sem acentos e sem espaços nas pontas ('Almoço' -> 'almoco')."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().strip().lower()


# --------------------------------------------------------------------------- dados
def load_view(engine, view: str = VIEW_NAME) -> pd.DataFrame:
    """Lê a view inteira. `engine` = sqlalchemy.create_engine(...)."""
    from sqlalchemy import text

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", view):
        raise ValueError(f"Nome de view inválido: {view!r}")
    return pd.read_sql(text(f'SELECT * FROM "{view}"'), engine)


def clean_view(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Converte tipos, deduplica para 1 linha por (data, refeição) e devolve diagnóstico."""
    df = raw.rename(columns=RENAME).copy()
    faltando = set(RENAME.values()) - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas ausentes na view: {sorted(faltando)}")

    df["data"] = pd.to_datetime(df["data"]).dt.normalize()
    df["refeicao"] = df["refeicao"].map(_norm)
    # 'Chuva (mm)' vem como texto com vírgula decimal ("0,0") e muitos nulos
    df["chuva_mm"] = pd.to_numeric(
        df["chuva_mm"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    for c in TARGETS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby(["data", "refeicao"])
    conflito = g[list(TARGETS)].nunique().gt(1).any(axis=1)
    if conflito.any():
        ex = conflito[conflito].index[:5].tolist()
        raise ValueError(
            f"{int(conflito.sum())} combinações (data, refeição) têm linhas repetidas com "
            f"agendamentos/faltas DIFERENTES (ex.: {ex}). A view não está em granularidade "
            "única por data/refeição (ex.: mais de um RU ou tabela de origem mais fina). "
            "Verifique a definição da view antes de modelar."
        )
    chuva_nuniq = g["chuva_mm"].nunique()
    agg = g.agg(
        dia_semana_txt=("dia_semana_txt", "first"),
        agendamentos=("agendamentos", "first"),
        faltas=("faltas", "first"),
        chuva_mm=("chuva_mm", "max"),
        linhas_brutas=("agendamentos", "size"),
    ).reset_index()

    agg["dow"] = agg["data"].dt.dayofweek
    dow_txt_ok = [_norm(t).startswith(_DIAS[d]) for t, d in zip(agg["dia_semana_txt"], agg["dow"])]
    valida = (agg["faltas"] >= 0) & (agg["faltas"] <= agg["agendamentos"])
    diag = {
        "linhas_brutas": len(raw),
        "combinacoes_data_refeicao": len(agg),
        "linhas_por_combinacao(min/mediana/max)": (
            int(agg["linhas_brutas"].min()),
            float(agg["linhas_brutas"].median()),
            int(agg["linhas_brutas"].max()),
        ),
        "periodo": (str(agg["data"].min().date()), str(agg["data"].max().date())),
        "datas_distintas": int(agg["data"].nunique()),
        "dias_sem_registro_no_calendario": int(
            (agg["data"].max() - agg["data"].min()).days + 1 - agg["data"].nunique()
        ),
        "observacoes_por_refeicao": agg["refeicao"].value_counts().to_dict(),
        "observacoes_por_dia_da_semana": agg["dow"].value_counts().sort_index().to_dict(),
        "agendamentos_nulos": int(agg["agendamentos"].isna().sum()),
        "faltas_nulas": int(agg["faltas"].isna().sum()),
        "faltas_negativas_ou_maiores_que_agendamentos": int((~valida & agg["faltas"].notna()).sum()),
        "chuva_ausente_%": round(100 * float(agg["chuva_mm"].isna().mean()), 1),
        "combinacoes_com_mais_de_um_valor_de_chuva": int((chuva_nuniq > 1).sum()),
        "dia_da_semana_texto_inconsistente": int(sum(not ok for ok in dow_txt_ok)),
    }
    # faltas fora do intervalo plausível não entram como alvo
    agg.loc[~valida, "faltas"] = np.nan
    return agg.sort_values(["data", "refeicao"]).reset_index(drop=True), diag


# --------------------------------------------------------------------------- features
def build_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Features calendário + defasagens + chuva.

    Toda defasagem usa apenas dias <= D - horizon (horizon em 1..7), por refeição.
    A linha de D pode ter alvo NaN (caso de previsão) sem afetar nada.
    """
    if not 1 <= horizon <= 7:
        raise ValueError("horizon deve estar entre 1 e 7 dias")
    partes = []
    for ref, g in df.groupby("refeicao"):
        g = g.sort_values("data").set_index("data")
        if g.index.has_duplicates:
            raise ValueError("Há datas duplicadas para a refeição " + str(ref))
        cal = g[list(TARGETS)].reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        f = pd.DataFrame(index=cal.index)
        for col, pre in (("agendamentos", "ag"), ("faltas", "fa")):
            s = cal[col]
            f[f"{pre}_media_mesmo_dia"] = pd.concat(
                [s.shift(k) for k in (7, 14, 21, 28)], axis=1
            ).mean(axis=1)
            f[f"{pre}_ult7"] = s.shift(horizon).rolling(7, min_periods=3).mean()
        f["taxa_falta_ult7"] = f["fa_ult7"] / f["ag_ult7"]
        f = f.loc[g.index].copy()
        f["refeicao"] = ref
        f["chuva_mm"] = g["chuva_mm"]
        f["chuva_ausente"] = f["chuva_mm"].isna().astype(float)
        f["chuva_mm"] = f["chuva_mm"].fillna(0.0)
        f["choveu"] = (f["chuva_mm"] >= 1.0).astype(float)
        f["dow"] = f.index.dayofweek
        for c in TARGETS:
            f[c] = g[c]
        partes.append(f.rename_axis("data").reset_index())
    return pd.concat(partes, ignore_index=True).sort_values(["data", "refeicao"]).reset_index(drop=True)


# --------------------------------------------------------------------------- modelos
class BaselineMedia:
    """Média histórica por (dia da semana, refeição) — modelo de referência."""

    def fit(self, X, y):
        d = X[CAL].assign(y=np.asarray(y, float))
        self.m_ = d.groupby(CAL)["y"].mean()
        self.por_ref_ = d.groupby("refeicao")["y"].mean()
        self.geral_ = float(d["y"].mean())
        return self

    def predict(self, X):
        saida = []
        for dow, ref in zip(X["dow"], X["refeicao"]):
            v = self.m_.get((dow, ref), np.nan)
            saida.append(self.por_ref_.get(ref, self.geral_) if np.isnan(v) else v)
        return np.asarray(saida, float)


class GLMCount:
    """GLM de contagem: Poisson ou binomial negativa (alpha estimado por Cameron-Trivedi)."""

    def __init__(self, familia: str = "poisson"):
        if familia not in ("poisson", "nb"):
            raise ValueError(familia)
        self.familia = familia

    def _design(self, X, ajustar=False):
        num = [c for c in X.columns if c not in CAL]
        if ajustar:
            self.levels_ = {c: sorted(X[c].astype(str).unique()) for c in CAL}
            sd = X[num].astype(float).std(ddof=0)
            self.num_ = [c for c in num if sd[c] > 0]  # colunas constantes saem (evita matriz singular)
            self.mu_ = X[self.num_].astype(float).mean()
            self.sd_ = sd[self.num_]
        D = pd.DataFrame({"const": 1.0}, index=X.index)
        for c in CAL:
            for lv in self.levels_[c][1:]:  # 1º nível = referência
                D[f"{c}={lv}"] = (X[c].astype(str) == lv).astype(float)
        Z = (X[self.num_].astype(float) - self.mu_) / self.sd_
        return pd.concat([D, Z], axis=1)

    def fit(self, X, y):
        y = np.asarray(y, float)
        D = self._design(X, ajustar=True)
        pois = sm.GLM(y, D, family=sm.families.Poisson()).fit()
        self.alpha_ = 0.0
        self.res_ = pois
        if self.familia == "nb":
            mu = np.asarray(pois.mu)
            z = ((y - mu) ** 2 - y) / mu
            self.alpha_ = max(float((mu * z).sum() / (mu**2).sum()), 1e-4)
            self.res_ = sm.GLM(y, D, family=sm.families.NegativeBinomial(alpha=self.alpha_)).fit()
        return self

    def predict(self, X):
        return np.asarray(self.res_.predict(self._design(X)), float)


class TreeCount:
    """Gradient boosting (histograma) com perda de Poisson."""

    def fit(self, X, y):
        self.cols_ = list(X.columns)
        self.levels_ = sorted(X["refeicao"].astype(str).unique())
        self.m_ = HistGradientBoostingRegressor(
            loss="poisson", max_depth=3, learning_rate=0.05, max_iter=200,
            min_samples_leaf=10, random_state=0,
        ).fit(self._prep(X), np.asarray(y, float))
        return self

    def _prep(self, X):
        Z = X[self.cols_].copy()
        mapa = {lv: i for i, lv in enumerate(self.levels_)}
        Z["refeicao"] = Z["refeicao"].astype(str).map(mapa).fillna(-1)
        return Z.astype(float)

    def predict(self, X):
        return np.asarray(self.m_.predict(self._prep(X)), float)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], object]
    features: tuple


def default_specs() -> list[ModelSpec]:
    base = tuple(CAL + LAGS)
    com_chuva = tuple(CAL + LAGS + RAIN)
    return [
        ModelSpec("baseline_media", BaselineMedia, tuple(CAL)),
        ModelSpec("poisson", lambda: GLMCount("poisson"), base),
        ModelSpec("poisson_chuva", lambda: GLMCount("poisson"), com_chuva),
        ModelSpec("binomial_negativa", lambda: GLMCount("nb"), base),
        ModelSpec("binomial_negativa_chuva", lambda: GLMCount("nb"), com_chuva),
        ModelSpec("arvores_chuva", TreeCount, com_chuva),
    ]


# --------------------------------------------------------------------------- validação
def walk_forward(
    feat: pd.DataFrame,
    target: str,
    specs: list[ModelSpec] | None = None,
    horizon: int = 1,
    min_train_days: int = 42,
    refit_every: int = 7,
) -> pd.DataFrame:
    """Validação temporal (janela expansível).

    Para cada bloco de `refit_every` dias de teste, o modelo é treinado só com
    observações cujo alvo já era conhecido (data <= primeiro dia do bloco - horizon).
    Todos os modelos são avaliados exatamente nas mesmas linhas.
    """
    specs = specs or default_specs()
    feat = feat.sort_values(["data", "refeicao"]).reset_index(drop=True)
    avaliavel = feat[LAGS].notna().all(axis=1) & feat[target].notna()
    datas = np.sort(feat["data"].unique())
    primeiro = datas[0] + np.timedelta64(min_train_days, "D")
    datas_teste = [d for d in datas if d >= primeiro]
    saida = []
    for i in range(0, len(datas_teste), refit_every):
        bloco = datas_teste[i : i + refit_every]
        corte = pd.Timestamp(bloco[0]) - pd.Timedelta(days=horizon)
        treino_all = feat[(feat["data"] <= corte) & feat[target].notna()]
        teste = feat[feat["data"].isin(bloco) & avaliavel]
        if teste.empty:
            continue
        for sp in specs:
            cols = list(sp.features)
            tr = treino_all.dropna(subset=cols)
            if len(tr) < 20:
                continue
            m = sp.factory().fit(tr[cols], tr[target].to_numpy(float))
            pred = np.clip(m.predict(teste[cols]), 0, None)
            saida.append(pd.DataFrame({
                "data": teste["data"].values, "refeicao": teste["refeicao"].values,
                "alvo": target, "modelo": sp.name,
                "real": teste[target].values, "previsto": pred,
            }))
    if not saida:
        raise ValueError("Dados insuficientes para a validação temporal.")
    return pd.concat(saida, ignore_index=True)


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def avaliar(pred: pd.DataFrame, por: tuple = ("alvo", "modelo")) -> pd.DataFrame:
    """MAE, RMSE e WAPE (em %) por alvo/modelo (ou outro agrupamento)."""
    linhas = []
    for chave, d in pred.groupby(list(por)):
        e = d["real"].to_numpy(float) - d["previsto"].to_numpy(float)
        linhas.append({**dict(zip(por, np.atleast_1d(chave))), "n": len(d),
                       "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e**2).mean()),
                       "WAPE_%": 100 * wape(d["real"], d["previsto"])})
    return pd.DataFrame(linhas).sort_values(list(por[:-1]) + ["WAPE_%"]).reset_index(drop=True)


# --------------------------------------------------------------------------- previsão
def prever(
    data,
    refeicao: str,
    chuva_mm: float | None = None,
    historico: pd.DataFrame | None = None,
    horizonte_dias: int = 1,
    modelo: str | dict = "poisson_chuva",
) -> dict:
    """Prevê agendamentos e faltas de uma refeição.

    historico: saída de `clean_view`. Só linhas com data <= data - horizonte_dias são
    usadas (se o valor real da própria data estiver no histórico, é ignorado).
    chuva_mm: precipitação diária (mm) observada ou PREVISTA para a data; None = ausente.
    modelo: nome em `default_specs()` ou dict {"agendamentos": ..., "faltas": ...}.
    """
    if historico is None:
        raise ValueError("Informe o histórico limpo (saída de clean_view).")
    data = pd.Timestamp(data).normalize()
    ref = _norm(refeicao)
    corte = data - pd.Timedelta(days=horizonte_dias)
    hist = historico.loc[historico["data"] <= corte, ["data", "refeicao", "agendamentos", "faltas", "chuva_mm"]]
    if hist.empty:
        raise ValueError("Histórico vazio antes da data informada.")
    novo = pd.DataFrame({
        "data": [data], "refeicao": [ref], "agendamentos": [np.nan], "faltas": [np.nan],
        "chuva_mm": [np.nan if chuva_mm is None else float(chuva_mm)],
    })
    feat = build_features(pd.concat([hist, novo], ignore_index=True), horizonte_dias)
    x = feat[(feat["data"] == data) & (feat["refeicao"] == ref)]
    specs = {s.name: s for s in default_specs()}
    saida = {}
    for alvo in TARGETS:
        nome = modelo[alvo] if isinstance(modelo, dict) else modelo
        sp = specs[nome]
        cols = list(sp.features)
        if x[cols].isna().any(axis=None):
            raise ValueError("Histórico insuficiente para montar as features desta data/refeição.")
        tr = feat[feat["data"] <= corte].dropna(subset=cols + [alvo])
        m = sp.factory().fit(tr[cols], tr[alvo].to_numpy(float))
        saida[alvo] = float(np.clip(m.predict(x[cols])[0], 0, None))
    return saida


# --------------------------------------------------------------------------- EDA
def eda(clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Efeito do dia da semana e da chuva (descritivo; não é causal)."""
    d = clean.copy()
    d["taxa_falta_%"] = 100 * d["faltas"] / d["agendamentos"]
    por_dia = d.groupby(["refeicao", "dow"]).agg(
        n=("data", "size"), ag_media=("agendamentos", "mean"), ag_dp=("agendamentos", "std"),
        fa_media=("faltas", "mean"), taxa_falta_media_pct=("taxa_falta_%", "mean"),
    ).round(1)
    for c in TARGETS:  # desvio relativo à média do mesmo (dia da semana, refeição)
        d[f"{c}_rel_%"] = 100 * (d[c] / d.groupby(["dow", "refeicao"])[c].transform("mean") - 1)
    d["faixa_chuva"] = pd.cut(d["chuva_mm"], [-0.01, 0.99, 10, 25, np.inf],
                              labels=["<1 mm", "1-10 mm", "10-25 mm", ">25 mm"])
    por_chuva = d.dropna(subset=["chuva_mm"]).groupby("faixa_chuva", observed=True).agg(
        n=("data", "size"), ag_rel_media_pct=("agendamentos_rel_%", "mean"),
        fa_rel_media_pct=("faltas_rel_%", "mean"),
    ).round(2)
    com = d.dropna(subset=["chuva_mm"])
    spearman = pd.DataFrame({
        "alvo": ["agendamentos", "faltas"],
        "spearman_com_chuva_mm": [
            com["chuva_mm"].corr(com["agendamentos_rel_%"], method="spearman"),
            com["chuva_mm"].corr(com["faltas_rel_%"], method="spearman"),
        ],
        "n": len(com),
    }).round(3)
    return {"por_dia_semana": por_dia, "por_faixa_chuva": por_chuva, "spearman_chuva": spearman}


def salvar_graficos(clean: pd.DataFrame, pasta: str) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(pasta, exist_ok=True)
    dias = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
    for alvo in TARGETS:
        fig, ax = plt.subplots(figsize=(7, 4))
        for ref, g in clean.groupby("refeicao"):
            m = g.groupby("dow")[alvo].mean()
            ax.plot([dias[i] for i in m.index], m.values, marker="o", label=ref)
        ax.set(title=f"Média de {alvo} por dia da semana", ylabel=alvo)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{pasta}/{alvo}_por_dia_semana.png", dpi=150)
        plt.close(fig)
    d = clean.dropna(subset=["chuva_mm"]).copy()
    if not d.empty:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, alvo in zip(axs, TARGETS):
            rel = 100 * (d[alvo] / d.groupby(["dow", "refeicao"])[alvo].transform("mean") - 1)
            ax.scatter(d["chuva_mm"], rel, s=14, alpha=0.7)
            ax.axhline(0, color="gray", lw=0.8)
            ax.set(xlabel="chuva (mm)", ylabel=f"{alvo}: desvio vs média do dia/refeição (%)")
        fig.tight_layout()
        fig.savefig(f"{pasta}/efeito_chuva.png", dpi=150)
        plt.close(fig)
