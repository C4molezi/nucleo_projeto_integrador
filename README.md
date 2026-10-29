# Previsão de agendamentos e faltas do RU (UFSM, Campus I)

Projeto Integrador I — modelo preditivo que estima, para uma **data** e uma **refeição** do Restaurante Universitário:

1. **Agendamentos** — quantas pessoas agendarão a refeição;
2. **Faltas** — quantas pessoas que agendaram não comparecerão (*no-shows*).

> **Status:** o código foi testado com dados sintéticos. Nenhum resultado em dados reais está registrado aqui. As métricas e a meta de WAPE ≤ 10% só podem ser afirmadas depois de rodar `run_pipeline.py` na sua base e conferir a saída.

---

## 1. Estrutura

| Arquivo | Função |
|---|---|
| `ru_forecast.py` | Biblioteca: leitura da view, limpeza, features, modelos, validação temporal, métricas, função `prever` e análise exploratória |
| `run_pipeline.py` | Executa tudo de ponta a ponta e grava os resultados na pasta `saida/` |
| `test_ru_forecast.py` | Testes automatizados (pytest), com dados sintéticos |
| `requirements.txt` | Dependências Python |

## 2. Dados de entrada

Fonte única: a view `vw_agendamentos_faltas_chuva` no PostgreSQL, com as colunas:

| Coluna | Tipo | Observação |
|---|---|---|
| `Data` | date | dia da refeição |
| `Dia da semana` | text | conferido contra a data (o código usa o dia calculado a partir da data) |
| `Refeição` | varchar | almoço, jantar |
| `Chuva (mm)` | varchar | texto com vírgula decimal (`"0,0"`), muitos nulos |
| `Agendamentos` | integer | total de agendamentos |
| `Faltas` | integer | agendamentos − refeições realizadas |

### O que a limpeza (`clean_view`) faz

- **Deduplica.** A view repete cada combinação data × refeição muitas vezes (vários milhares de linhas para algumas centenas de combinações). O código reduz para **uma linha por data e refeição**. Se as linhas repetidas tiverem agendamentos ou faltas **diferentes**, ele **interrompe com erro**, porque nesse caso a granularidade da view não é única (por exemplo, mais de um RU).
- **Converte a chuva** de texto com vírgula para número e usa o valor não nulo do dia (se houver mais de um valor, usa o maior e registra no diagnóstico).
- **Invalida faltas implausíveis:** faltas negativas ou maiores que os agendamentos viram `NaN` e não entram como alvo.
- **Normaliza a refeição** (`Almoço` → `almoco`).
- **Gera um diagnóstico** (linhas brutas, período, dias sem registro, observações por refeição e dia da semana, nulos, chuva ausente, inconsistências).

> A subtração `agendamentos − refeições` só pode ser validada na tabela de origem (cancelamentos, ajustes e correspondência entre tabelas). A consulta SQL para isso está na seção 8.

## 3. O modelo

### Features (sem vazamento)

Para prever o dia *D* com horizonte *h* (padrão 1 dia), só entram informações de dias ≤ *D − h*, separadas por refeição:

| Feature | Descrição |
|---|---|
| `dow` | dia da semana (0 = segunda) |
| `refeicao` | almoço ou jantar |
| `ag_media_mesmo_dia`, `fa_media_mesmo_dia` | média de agendamentos e de faltas no mesmo dia da semana, 1 a 4 semanas antes |
| `ag_ult7`, `fa_ult7` | média móvel dos últimos 7 dias disponíveis |
| `taxa_falta_ult7` | `fa_ult7 / ag_ult7` |
| `chuva_mm`, `choveu` (≥ 1 mm), `chuva_ausente` | chuva do dia (apenas nas variantes `_chuva`) |

O horizonte `h` pode ir de 1 a 7 dias.

### Modelos comparados

| Nome | Descrição |
|---|---|
| `baseline_media` | média histórica por dia da semana e refeição (referência) |
| `poisson` / `poisson_chuva` | regressão de Poisson (GLM), sem e com chuva |
| `binomial_negativa` / `binomial_negativa_chuva` | binomial negativa, com `alpha` estimado pelo método de Cameron-Trivedi |
| `arvores_chuva` | gradient boosting com perda de Poisson (profundidade 3) |

Agendamentos e faltas têm **modelos separados**, ajustados com as mesmas features.

### Validação temporal (walk-forward)

- Janela **expansível**: os primeiros 42 dias só servem de treino.
- A cada 7 dias de teste, os modelos são reajustados usando **apenas observações cujo alvo já era conhecido** (`data ≤ primeiro dia do bloco − h`).
- Todos os modelos são avaliados **exatamente nas mesmas linhas**.
- Métricas por alvo e modelo (e também por refeição): **MAE**, **RMSE** e **WAPE** = Σ|erro| / Σ real, em %.

## 4. Instalação

Python 3.10 ou superior.

```bash
cd ru_forecast
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Como executar

### 5.1. Testes automatizados

```bash
pytest -q
```

Cobrem: deduplicação e conversão da chuva, erro quando as linhas repetidas discordam, faltas inválidas, **ausência de vazamento** nas features e no walk-forward, métricas, e a função `prever` ignorando o valor real da própria data. Não precisam de banco.

### 5.2. Pipeline completo a partir do PostgreSQL

```bash
export DATABASE_URL="postgresql+psycopg2://usuario:senha@localhost:5432/banco"
python run_pipeline.py --data 2026-05-15 --refeicao almoco
```

(No Windows PowerShell: `$env:DATABASE_URL="postgresql+psycopg2://..."`.)
Não coloque a senha dentro do código nem em repositórios.

### 5.3. Pipeline a partir de um CSV exportado da view

```bash
python run_pipeline.py --csv view.csv
```

O CSV deve ter as mesmas colunas da view.

### 5.4. Opções do `run_pipeline.py`

| Opção | Padrão | Significado |
|---|---|---|
| `--view` | `vw_agendamentos_faltas_chuva` | nome da view |
| `--csv` | — | lê de um CSV em vez do banco |
| `--saida` | `saida` | pasta de resultados |
| `--data` | `2026-05-15` | data do exemplo de previsão |
| `--refeicao` | `almoco` | refeição do exemplo |
| `--horizonte` | `1` | dias entre a última informação usada e a data prevista (1 a 7) |

### 5.5. O que é gerado em `saida/`

| Arquivo | Conteúdo |
|---|---|
| `base_limpa.csv` | base com uma linha por data e refeição |
| `eda_por_dia_semana.csv`, `eda_por_faixa_chuva.csv`, `eda_spearman_chuva.csv` | análise exploratória do dia da semana e da chuva |
| `graficos/` | média por dia da semana e dispersão do efeito da chuva |
| `previsoes_vs_real.csv` | todas as previsões do walk-forward ao lado do valor real |
| `metricas.csv` | MAE, RMSE e WAPE por alvo e modelo |
| `exemplo_previsao.csv` | tabela *Indicador / Previsão / Valor real / Erro absoluto* da data pedida |

O diagnóstico, as métricas e a tabela do exemplo também são impressos no terminal.

## 6. Usando a função de previsão

```python
import os
import pandas as pd
from sqlalchemy import create_engine
import ru_forecast as rf

raw = rf.load_view(create_engine(os.environ["DATABASE_URL"]))
clean, diag = rf.clean_view(raw)

previsao = rf.prever(
    data="2026-05-15",
    refeicao="almoço",
    chuva_mm=0.0,            # chuva observada ou PREVISTA; None se não houver
    historico=clean,
    horizonte_dias=1,
    modelo="poisson_chuva",  # ou {"agendamentos": "...", "faltas": "..."}
)
print(previsao)   # {"agendamentos": ..., "faltas": ...}
```

Só dados com data ≤ `data − horizonte_dias` são usados; se o valor real da própria data estiver no histórico, ele é ignorado.

## 7. Limitações e cuidados

- **Chuva observada × prevista.** Na avaliação histórica, a chuva usada é a observada no dia. Em uso real ela viria de uma previsão do tempo, com erro. Por isso, as métricas dos modelos com chuva tendem a ser otimistas. As variantes sem chuva permitem medir o ganho real da variável.
- **Seleção do melhor modelo.** O `run_pipeline.py` escolhe o melhor modelo pelo WAPE no mesmo período de teste, o que torna a estimativa de erro um pouco otimista.
- **Amostra pequena.** Com poucas centenas de observações (dias úteis × 2 refeições), as estimativas do efeito da chuva e os modelos de árvores são instáveis.
- **Efeito da chuva é descritivo.** A análise exploratória compara cada dia com a média do mesmo dia da semana e refeição, mas não prova causalidade (feriados, provas e greves não estão nos dados).
- **WAPE ≤ 10%.** A meta é interpretável quando os valores observados são grandes; para as faltas, verifique o valor médio diário antes de comparar.
- **Fim do período.** Se a view terminar antes de 30/06/2026, confira a causa (tabela de chuva ou junção) antes de usar o período inteiro.

## 8. Validação da fórmula de faltas na tabela de origem

Para conferir se `total_agendamentos − total_refeicao` é consistente (totais iguais dentro da mesma data e refeição, sem faltas negativas):

```sql
SELECT dh_refeicao::date AS data, tipo_refeicao, COUNT(*) AS linhas,
       COUNT(DISTINCT total_agendamentos) AS n_agend, COUNT(DISTINCT total_refeicao) AS n_ref,
       MIN(total_agendamentos - total_refeicao) AS min_falta,
       MAX(total_agendamentos - total_refeicao) AS max_falta
FROM avaliacoes_ru
GROUP BY 1, 2
HAVING COUNT(DISTINCT total_agendamentos) > 1 OR COUNT(DISTINCT total_refeicao) > 1
    OR MIN(total_agendamentos - total_refeicao) < 0;
```

Se a consulta retornar linhas, a granularidade ou o cálculo de faltas precisa ser revisto antes da modelagem.
