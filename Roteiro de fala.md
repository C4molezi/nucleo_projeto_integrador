# Roteiro de fala: Modelo preditivo de faltas no RU da UFSM (até 7 min)

## Introdução (cerca de 1 min 30 s)

Bom dia a todos. Meu nome é Gabriel e hoje vou apresentar o modelo preditivo de faltas diárias no Restaurante Universitário da UFSM, Campus I, desenvolvido no Projeto Integrador I.

Para começar, o problema. O RU prepara as refeições com base nos agendamentos feitos pelos estudantes. Mas nem todo agendamento vira refeição: quando a pessoa agenda e não comparece, surge uma falta. Quando essas faltas são muitas e não são previstas, sobra comida, aumenta o desperdício e o planejamento de suprimentos perde precisão. Por isso, estimar quantas faltas vão ocorrer em cada dia ajuda a planejar melhor a produção.

Nosso objetivo foi prever a quantidade diária de faltas a partir do histórico do próprio restaurante.

## Solução (cerca de 1 min 30 s)

A solução é um modelo de regressão linear múltipla, implementado em Python com a biblioteca scikit-learn. Escolhemos a regressão linear por ser simples, fácil de explicar e de baixo custo computacional, o que facilita a leitura por quem gerencia o restaurante.

A variável-alvo é definida pela relação entre agendamentos e refeições: faltas são iguais ao total de agendamentos menos o total de refeições realizadas.

## Como o modelo funciona (cerca de 2 min)

O modelo usa três tipos de informação. Primeiro, as médias móveis de faltas dos últimos 7, 30 e 90 dias, que capturam tendências recentes e a inércia do comportamento dos usuários. Segundo, o total de agendamentos do dia, que é o principal indicativo de quantas pessoas devem aparecer. Terceiro, o dia da semana, transformado em variáveis binárias (one-hot), com o domingo como referência, já que o número de 0 a 6 não representa uma ordem real.

O fluxo tem algumas etapas. Os dados são lidos de um CSV exportado de uma view do PostgreSQL, os registros com valores ausentes são removidos e tudo é ordenado por data. Em seguida, fazemos uma divisão cronológica: 80% para treino e 20% para teste. Optamos por dividir no tempo, e não de forma aleatória, para que o modelo não tenha acesso a informações do futuro.

O modelo aprende os coeficientes pelo método dos mínimos quadrados. Depois, as previsões passam por um pós-processamento: valores negativos são zerados e os resultados são arredondados para inteiros, já que faltas são contagens de pessoas.

Para verificar se o modelo realmente agrega valor, comparamos com um baseline simples, que estima as faltas pela média histórica do mesmo dia da semana, calculada apenas com os dados de treino.

## Resultados obtidos (cerca de 2 min)

Os resultados foram medidos no conjunto de teste, com 207 registros, começando em 30 de agosto de 2025.

O MAE, o erro absoluto médio, da regressão linear foi de cerca de 336 faltas por dia, contra 781 do baseline. O RMSE, que penaliza mais os erros grandes, ficou em 433, frente a 937. Ou seja, o modelo erra menos em média e evita desvios grandes.

O R² foi de 0,865, o que indica que o modelo explica cerca de 86,5% da variação das faltas. O baseline explica 36,8%.

Por fim, o WAPE, o erro percentual absoluto ponderado, foi de 12,87%, contra 29,92% do baseline. Esse valor é o erro total dividido pelo total real de faltas. Ele não significa que o modelo acerta 87% dos casos.&#32;

## Melhorias futuras e conclusão (cerca de 1 min)

Concluindo, o modelo se mostrou uma estimativa útil e superou o baseline de forma clara. Reconhecemos, porém, algumas limitações. Chuva e avaliações de cardápio foram consideradas no planejamento, mas não integram o modelo atual. Feriados, recessos e o calendário acadêmico também não são tratados.

Como próximos passos, pretendemos incluir dados meteorológicos do INMET e variáveis de calendário, ampliar a validação temporal com mais de uma janela de teste e criar testes automatizados. Também será preciso rever a dependência do total de agendamentos: se a previsão precisar ser feita antes do fechamento dos agendamentos, esse valor terá de ser estimado separadamente.

Agradeço a atenção e fico à disposição para perguntas.
