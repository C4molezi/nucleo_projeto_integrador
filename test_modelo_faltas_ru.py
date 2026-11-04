import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
import sys
import numpy as np
import pandas as pd

# Importando o módulo a ser testado
# (Certifique-se de que o arquivo modelo_faltas_ru.py esteja no mesmo diretório ao rodar)
import modelo_faltas_ru

class TestModeloFaltasRU(unittest.TestCase):

    def setUp(self):
        # Configuração de um DataFrame fictício válido para uso nos testes
        data = {
            "DATA": pd.date_range(start="2024-01-01", periods=10),
            "TOTAL_REFEICAO": [100, 150, 120, 130, 140, 100, 150, 120, 130, 140],
            "TOTAL_AGENDAMENTOS": [120, 160, 130, 150, 160, 120, 160, 130, 150, 160],
            "FALTAS": [20, 10, 10, 20, 20, 20, 10, 10, 20, 20],
            "DIA_SEMANA": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "MEDIA_MOVEL_7_DIAS": [15] * 10,
            "MEDIA_MOVEL_30_DIAS": [15] * 10,
            "MEDIA_MOVEL_90_DIAS": [15] * 10
        }
        self.df_valido = pd.DataFrame(data)

    @patch("pandas.read_csv")
    def test_carregar_dados_sucesso(self, mock_read_csv):
        """Testa se o carregamento padroniza os nomes das colunas e formata a data."""
        df_csv = self.df_valido.copy()
        # Simulando colunas em minúsculo e com espaços
        df_csv.columns = [f" {c.lower()} " for c in df_csv.columns]
        mock_read_csv.return_value = df_csv

        resultado = modelo_faltas_ru.carregar_dados("caminho_falso.csv")
        
        # Verifica se todas as colunas necessárias estão presentes
        for col in modelo_faltas_ru.COLUNAS_NECESSARIAS:
            self.assertIn(col, resultado.columns)
        
        # Verifica se a coluna DATA foi convertida para datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(resultado["DATA"]))

    @patch("pandas.read_csv")
    def test_carregar_dados_colunas_ausentes(self, mock_read_csv):
        """Testa se a função levanta ValueError quando faltam colunas."""
        df_incompleto = self.df_valido.drop(columns=["FALTAS", "TOTAL_REFEICAO"])
        mock_read_csv.return_value = df_incompleto

        with self.assertRaises(ValueError) as context:
            modelo_faltas_ru.carregar_dados("caminho_falso.csv")
        
        self.assertIn("Colunas ausentes no CSV", str(context.exception))
        self.assertIn("FALTAS", str(context.exception))

    def test_preparar_dados(self):
        """Testa a remoção de NAs e a ordenação cronológica."""
        df_com_na = self.df_valido.copy()
        df_com_na.loc[0, "FALTAS"] = np.nan
        # Embaralhar datas para testar ordenação
        df_com_na = df_com_na.sample(frac=1, random_state=42)

        resultado = modelo_faltas_ru.preparar_dados(df_com_na)

        self.assertEqual(len(resultado), 9)  # Removeu 1 linha com NA
        self.assertTrue(resultado["DATA"].is_monotonic_increasing)

    @patch("sys.stderr", new_callable=StringIO)
    def test_validar_faltas(self, mock_stderr):
        """Testa se o aviso é disparado ao haver divergência no cálculo das faltas."""
        df_erro = self.df_valido.copy()
        df_erro.loc[0, "FALTAS"] = 999  # Criando uma divergência proposital
        
        modelo_faltas_ru.validar_faltas(df_erro)
        
        # Verifica se a mensagem de aviso foi enviada para o stderr
        self.assertIn("AVISO: 1 registro(s) com FALTAS diferente", mock_stderr.getvalue())

    def test_dividir_treino_teste(self):
        """Testa o particionamento temporal de 80/20."""
        treino, teste = modelo_faltas_ru.dividir_treino_teste(self.df_valido)
        
        self.assertEqual(len(treino), 8)  # 80% de 10
        self.assertEqual(len(teste), 2)   # 20% de 10
        # O último elemento do treino deve ser anterior ao primeiro do teste
        self.assertTrue(treino.iloc[-1]["DATA"] < teste.iloc[0]["DATA"])

    def test_montar_variaveis(self):
        """Testa o one-hot encoding para dias da semana."""
        variaveis = modelo_faltas_ru.montar_variaveis(self.df_valido)
        
        # O DataFrame tem 4 features numéricas. O dia da semana (0-6) gerará 6 dummies (drop_first=True)
        # Total esperado de colunas = 4 + 6 = 10
        self.assertEqual(variaveis.shape[1], 10)
        
        colunas = variaveis.columns.tolist()
        self.assertTrue(any(c.startswith("DIA_1") for c in colunas))
        self.assertFalse(any(c.startswith("DIA_0") for c in colunas)) # Domingo foi dropado

    def test_tratar_previsoes(self):
        """Testa o clipping para zeros e arredondamento das previsões."""
        previsoes = np.array([-2.5, 3.4, 7.8, -0.1])
        resultado = modelo_faltas_ru.tratar_previsoes(previsoes)
        
        np.testing.assert_array_equal(resultado, np.array([0, 3, 8, 0]))
        self.assertTrue(np.issubdtype(resultado.dtype, np.integer))

    def test_avaliar(self):
        """Testa as métricas de regressão calculadas."""
        reais = np.array([10, 20, 30])
        estimadas = np.array([12, 18, 30]) # Erros: 2, 2, 0 -> Soma=4. MAE = 4/3 = 1.333
        
        resultado = modelo_faltas_ru.avaliar(reais, estimadas)
        
        self.assertIn("MAE", resultado)
        self.assertIn("RMSE", resultado)
        self.assertIn("R²", resultado)
        self.assertIn("WAPE (%)", resultado)
        
        self.assertAlmostEqual(resultado["MAE"], 4/3)
        self.assertAlmostEqual(resultado["WAPE (%)"], 100 * 4 / 60) # Erro absoluto total(4) / Reais(60)

    def test_montar_tabela(self):
        """Testa a geração do DataFrame final de comparação."""
        estimadas = np.array([20, 10])
        df_teste = self.df_valido.head(2).copy()
        
        resultado = modelo_faltas_ru.montar_tabela(df_teste, estimadas)
        
        self.assertListEqual(list(resultado.columns), ["DATA", "FALTAS_REAIS", "FALTAS_ESTIMADAS"])
        # Verifica a conversão da data para string dd/mm/yyyy
        self.assertEqual(resultado.iloc[0]["DATA"], df_teste.iloc[0]["DATA"].strftime("%d/%m/%Y"))
        self.assertEqual(resultado.iloc[0]["FALTAS_ESTIMADAS"], 20)

if __name__ == "__main__":
    unittest.main()
