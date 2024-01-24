# Computacao-Paralela

Este repositório possui o conteúdo produzido para realização da componente prática da Unidade Curricular de Computação Paralela.
O objetivo deste trabalho era a otimização dividida em 3 diferentes fases de um programa inicialmente fornecido pelos docentes.

## Fase 1 -Otimização de Código Sequencial
Na primeira fase o objetivo seria implementar paralelismo ao nível das instruçóes com o objetivo de reduzir o tempo de execução de um programa de cálculos físicos, que calculam a força resultante, bem como a energia potencial interação entre duas partículas.
Optamos por fazer diversas otimizaçóes e abaixo estão descritas as mesmas .

![Fase1](https://github.com/jbtescudeiro16/Computacao-Paralela/raw/main/PICS/cp_fase1.png)

Para executar o código da primeira fase basta fazer make e make run.

## Fase 2 -Otimização usando OPENMP (paralelização)

Na segunda fase e como pedido era necessário implementar paralelismo ao nível de dados. Assim seriam criadas várias threads que seriam executadas no processador e que paralelizavam uma parte do código.
Após analisarmos qual seria o pedaço de código que seria mais benéfico de ser paralelizado, implementamos então as alterações.
Abaixo está presente um gráfico que demonstra as melhorias obtidas variando o número de threads no Cluster dos professores

![Fase2](https://github.com/jbtescudeiro16/Computacao-Paralela/blob/main/PICS/desempenho.png)
 

## Fase 3 -Otimização usando Aceleradores (GPU)
Na terceira fase o objetivo seria implementar paralelismo utilizando CUDA.
Assim e após realizar diversas alterações os resultados obtidos estão presentes nas figuras abaixo.

![Fase3](https://github.com/jbtescudeiro16/Computacao-Paralela/blob/main/PICS/fase3.png)

![Fase3.Graph](https://github.com/jbtescudeiro16/Computacao-Paralela/blob/main/PICS/fase33.png)


![Resultados Variando o nr de Particulas para as 3 fases](https://github.com/jbtescudeiro16/Computacao-Paralela/blob/main/PICS/cp_fase3_2.png)

![Comparacao entre as 3 versoes](https://github.com/jbtescudeiro16/Computacao-Paralela/blob/main/PICS/fase3-2.png)
