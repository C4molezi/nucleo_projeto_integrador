import pandas as pd

# 1. Carregar a base de dados original
arquivo_entrada = 'vw_agendamentos_faltas_chuva.csv'
df = pd.read_csv(arquivo_entrada)

print(f"Total de registros originais: {len(df):,}")

# 2. Remover linhas duplicadas mantendo apenas a primeira ocorrência
df_limpo = df.drop_duplicates(keep='first')

print(f"Total de registros após remoção de duplicatas: {len(df_limpo):,}")
print(f"Duplicatas removidas: {len(df) - len(df_limpo):,}")

# 3. Salvar a base de dados limpa
arquivo_saida = 'vw_agendamentos_faltas_chuva_limpo.csv'
df_limpo.to_csv(arquivo_saida, index=False)

print(f"Arquivo limpo salvo com sucesso em: {arquivo_saida}")