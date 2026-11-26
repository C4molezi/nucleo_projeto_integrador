# Roteiro de fala: Previsão de faltas no RU da UFSM

**Gabriel Augusto Camolezi** — Projeto Integrador I · Curso de Ciência de Dados e Inteligência Artificial  
**Tempo total estimado:** cerca de 8,5 minutos, em ritmo de fala calmo.

---

### Slide 1: Título (cerca de 20 s)
Olá, pessoal! Meu nome é Gabriel, e hoje vou contar sobre um trabalho de «Previsão de Absenteísmo no Restaurante Universitário da UFSM». A ideia é simples: ensinar um computador a prever quantas pessoas vão faltar às refeições.

### Slide 2: Contexto — o restaurante (cerca de 40 s)
Na UFSM, existe o Restaurante Universitário, chamado de RU. Antes da refeição, os alunos marcam no sistema que vão comer. Com base nessas marcações, a cozinha decide quanta comida preparar. Ou seja, a comida é feita pensando nos alunos que agendaram.

### Slide 3 (Partes 1 e 2): O problema do absenteísmo (cerca de 55 s)
*(Começa exibindo apenas a analogia da festa)*  
Só que nem todo mundo que marcou aparece. Pensem numa festa em que vocês convidam 100 amigos, mas só 60 vão. Se vocês fizeram comida para 100, sobram 40 pratos.

*(Avança para revelar os dados do RU da UFSM)*  
No RU do Campus I, acontece algo parecido: mais de um em cada três agendamentos termina em ausência — uma taxa de 35,8% de não comparecimento. Em uma média de cerca de 6.835 agendamentos por dia, isso representa aproximadamente 2.447 faltas diárias. É muita comida preparada e não consumida.

### Slide 4: O impacto e a pergunta norteadora (cerca de 40 s)
Essa comida que sobra vira desperdício. Ela custa dinheiro público, usa água, energia e o trabalho de muita gente, e esse recurso poderia ajudar outros estudantes. Por isso, a pergunta do trabalho é: é possível prever a quantidade de usuários que agendarão refeições e a quantidade de usuários que não comparecerão aos Restaurantes Universitários da UFSM? Conhecendo essa informação, a cozinha planeja melhor a produção.

### Slide 5: Abordagens anteriores (cerca de 65 s)
Outros pesquisadores já enfrentaram problemas parecidos. Em 2011, Rocha e colegas usaram uma rede neural para estimar quantas refeições um restaurante universitário deveria preparar por dia. Uma rede neural é um modelo com muitos parâmetros ajustados a partir do histórico.

O resultado pode ser muito bom, mas a rede funciona como uma caixa-preta: se ela sugerir 3.000 refeições, fica difícil explicar para a equipe da cozinha como chegou a esse número. Além disso, aquela abordagem previa o total de refeições, e não diretamente as faltas. Ainda assim, pretendemos implementar também redes neurais como modelos concorrentes, para comparar o desempenho preditivo das diferentes abordagens.

### Slide 6: Modelo proposto (cerca de 85 s)
Como abordagem inicial, investigamos a regressão linear múltipla. Nossa variável resposta é o número diário de faltas — agendamentos menos refeições servidas. As covariáveis são cinco: as médias móveis de faltas dos últimos 7, 30 e 90 dias, o total de agendamentos do dia e o dia da semana.

Cada variável recebe um peso interpretável. Por exemplo, o peso dos agendamentos é 0,22: cada 100 marcações a mais associam-se a cerca de 22 faltas a mais. É uma receita aberta, fácil de auditar. Além disso, ao longo do projeto, decidimos investigar também a regressão de Poisson com múltiplas covariáveis, já que o número de faltas corresponde a dados de contagem. Para avaliar os ganhos, adotamos um modelo de referência — a baseline — baseado na média histórica de faltas por dia da semana.

### Slide 7: Metodologia (cerca de 65 s)
Como avaliamos o modelo? Usamos validação temporal, preservando a ordem cronológica, sem embaralhar os dados. Dos 1.034 dias válidos disponíveis, os primeiros 827 dias — 80% da base — foram usados no treinamento para estimação dos parâmetros do modelo, isto é, a amostra *in-sample*. Os 207 dias seguintes — os 20% finais — foram usados no teste para avaliação do desempenho preditivo fora da amostra, ou *out-of-sample*. O modelo só acessa informações anteriores ao dia previsto, como as médias móveis.

### Slide 8: Resultados preliminares (cerca de 95 s)
Agora, os resultados preliminares, obtidos com a regressão linear múltipla. Retomando, comparamos esse modelo com a baseline — a média histórica de faltas do mesmo dia da semana.

Pelo WAPE, que mede o erro percentual total em relação às faltas observadas, a regressão linear alcançou 12,87%, contra 29,92% da baseline — menos da metade do erro. Pelo R², que vai de zero a um, nosso modelo atingiu 0,865, frente a 0,368 da baseline. Em um exemplo real do conjunto de teste, no dia 31 de agosto de 2025, ocorreram 1.163 faltas e o modelo previu 1.013. Esses resultados da regressão linear múltipla também servirão de referência para a comparação com os próximos modelos.

### Slide 9: Próximos passos (cerca de 55 s)
O modelo atual tem limitações: depende dos agendamentos do próprio dia, tem desempenho menor nos fins de semana e ainda não inclui fatores externos. Nos próximos passos, pretendemos incorporar novas covariáveis — dados de chuva do INMET, calendário acadêmico e cardápio — e comparar três abordagens: regressão linear múltipla, regressão de Poisson com múltiplas covariáveis e redes neurais artificiais, avaliando tanto o desempenho preditivo quanto a interpretabilidade.

### Slide 10: Agradecimento (cerca de 15 s)
Muito obrigado! Espero que tenham percebido como a matemática e a ciência de dados podem apoiar a rotina do RU. Fico à disposição para as perguntas de vocês.

---

### Notas de conferência
- **Contagem de palavras:** ~990 palavras no texto falado e títulos (aumento de aproximadamente **+4%**, dentro do limite máximo de **+10%** em relação às ~950 palavras do roteiro original).
- **Referência de agendamentos (Slide 3):** o valor de *~6.835 agendamentos/dia* foi derivado diretamente dos dados do artigo (`2.447 faltas / 0,358`). Caso deseje citar o total acumulado exato do CSV nos 1.034 dias, basta substituí-lo nesse trecho.
- **Exemplo real (Slide 8):** preenchido com os valores do dia `31/08/2025` (`1.163` faltas reais e `1.013` previstas), já constantes no slide original.