# Relatório — Previsão 1º Rodada Copa do Mundo 2026

### Rede Neural com TensorFlow


## 1\. Introdução

### Hipótese

A hipótese central deste estudo é que o desempenho histórico recente e o equilíbrio de forças entre seleções nacionais podem ser adequadamente representados por métricas quantitativas, como Elo Rating, forma nos últimos dez jogos e indicadores de força ofensiva e defensiva. Assume-se que essas variáveis contêm informação suficiente para prever placares de partidas da Copa do Mundo com desempenho comparável ao obtido por modelos mais complexos e com maior número de atributos. Variáveis socioeconômicas, como PIB e população, foram deliberadamente excluídas por não apresentarem evidências consistentes de contribuição preditiva para resultados esportivos em estudos anteriores.

### Descrição Geral da Abordagem

O pipeline segue uma lógica de **aprendizado supervisionado multi-saída**: a partir de confrontos históricos (Copa 1950–2022) e eliminatórias 2026, é construído um vetor de 22 features por jogo, que alimenta uma rede neural TensorFlow capaz de prever simultaneamente os gols do time 1 e do time 2. As previsões finais são os placares arredondados para inteiro dos 24 jogos da 1ª rodada da Copa 2026, exportados em JSON.

## 📐 Arquitetura do Pipeline

```
Dados Históricos (Copa 1950–2022 + Eliminatórias 2026)
         ↓
  Engenharia de Features
  ┌─────────────────────────────────────────────────┐
  │ Elo Rating          │ Forma últimos 10 jogos    │
  │ Ranking FIFA        │ Saldo de gols (10 jogos)  │
  │ Força ofensiva      │ Força defensiva           │
  │ Head-to-Head        │ Odds (prob. implícitas)   │
  │ Temperatura clima   │ Flag sede Copa 2026       │
  └─────────────────────────────────────────────────┘
         ↓
  Validação Temporal (sem data leakage)
  ┌──────────────────────────────────────┐
  │ Treino  : Copa 1950 – 2018           │
  │ Val     : Copa 2022                  │
  │ Teste   : Eliminatórias 2026         │
  │ Previsão: 1ª Rodada Copa 2026        │
  └──────────────────────────────────────┘
         ↓
  Rede Neural Multi-Output + Loss de Poisson
  [gols_t1, gols_t2] → Placar inteiro arredondado
         ↓
  bolao_copa_previsoes.json

```

## 2\. Dados

### Fontes e Período

|Fonte|Conteúdo|Período|
|-|-|-|
|Fjelstul World Cup Database (datahub.io)|Resultados completos de Copas|1950 – 2022|
|Wikipedia API|Resultados das eliminatórias por confederação|2023 – 2025|
|Fallback embutido|Tabela de resultados das eliminatórias (caso scraping falhe)|2023 – 2025|
|Odds de casas de apostas|Probabilidades implícitas para os 24 jogos|Copa 2026|

O scraping das eliminatórias é feito via Wikipedia API para seis confederações: CONMEBOL, UEFA, CAF, AFC, CONCACAF e OFC. Se o número de jogos obtidos for menor que 50, o pipeline ativa automaticamente uma tabela fallback embutida no código.

### Variáveis Coletadas

Para cada jogo: time da casa (`home_team`), time visitante (`away_team`), gols do mandante (`gols_home`), gols do visitante (`gols_away`) e ano.

### Critérios de Inclusão/Exclusão de Jogos

* \*\*Incluídos:\*\* todos os jogos de Copa do Mundo (fase de grupos, mata-mata) de 1950 a 2022; jogos das eliminatórias 2026 de todas as confederações.
* \*\*Excluídos:\*\* jogos com time identificado como `UNK` (país não mapeado no dicionário de normalização); times históricos extintos são mantidos sob códigos próprios (ex.: `URS`, `YUG`, `TCH`), sendo tratados como entidades distintas.
* \*\*Normalização de nomes:\*\* tabela de lookup com mais de 100 variantes de nomes (ex.: `"Türkiye"` → `TUR`, `"West Germany"` → `GER`, `"Côte d'Ivoire"` → `CIV`) garante consistência entre fontes.

### Critérios de Inclusão/Exclusão de Features

O vetor de 22 features foi construído seguindo três princípios aplicados a cada variável candidata:

1. \*\*Validade preditiva\*\* — existe evidência empírica ou mecanismo causal de que a variável influencia o resultado?
2. \*\*Disponibilidade temporal\*\* — pode ser calculada sem acesso a dados futuros (sem data leakage)?
3. \*\*Parcimônia\*\* — adiciona informação não capturada por outras features já presentes?

#### Grupo 1 — Força relativa entre as equipes (features \[0]–\[4])

|Feature|Critério de inclusão|
|-|-|
|`elo_t1`, `elo_t2`|O Elo captura a força acumulada de cada time ao longo de toda a história de jogos, com base em resultados ponderados por importância da competição e margem de gols. É a melhor proxy disponível de "qualidade geral" sem depender de variáveis não-esportivas. Base na literatura: Elo (1978), Groll et al. (2018).|
|`diff_elo`|A diferença de Elo é o preditor direto da probabilidade esperada de vitória na fórmula original. Incluí-la separadamente, além dos dois valores absolutos, permite ao modelo usar esse sinal de forma direta, sem precisar aprender a subtração internamente.|
|`rank_t1`, `rank_t2`|O ranking FIFA usa metodologia distinta do Elo — pontuação por resultados recentes ponderada por importância do jogo e força do adversário. Os dois sistemas não são redundantes: times podem ter Elo alto e ranking médio (histórico longo mas desempenho recente irregular) ou vice-versa.|



#### Grupo 2 — Forma e dinâmica recente (features \[5]–\[12])

Todas calculadas sobre os \*\*últimos 10 jogos\*\* de cada time (Copa + eliminatórias, em ordem cronológica).

|Feature|Critério de inclusão|
|-|-|
|`forma_t1`, `forma_t2`|Aproveitamento em pontos (%). Captura o momentum recente: um time que perdeu 4 dos últimos 10 jogos está em momento diferente de um que ganhou 8, mesmo com Elo parecido. O Elo se move lentamente; a forma reage rapidamente.|
|`saldo_t1`, `saldo_t2`|Saldo médio de gols (marcados − sofridos por jogo). Complementa a forma: uma equipe pode ter 70% de aproveitamento por goleadas (saldo alto) ou por resultados apertados (saldo baixo) — a diferença importa para a previsão do placar.|
|`forca_of_t1`, `forca_of_t2`|Média de gols marcados por jogo. Preditor direto da saída do modelo (`gols_t1`, `gols_t2`).|
|`forca_def_t1`, `forca_def_t2`|Média de gols sofridos por jogo. A força ofensiva de um time interage com a força defensiva do adversário; ambas são necessárias para estimar o placar com independência (inspiração no modelo de Maher, 1982).|

#### Grupo 3 — Histórico direto entre os times (features \[13]–\[15])

|Feature|Critério de inclusão|
|-|-|
|`h2h_v1`, `h2h_emp`, `h2h_v2`|O confronto direto captura relações que o Elo não captura. Alguns times sistematicamente superam adversários teoricamente mais fortes por fatores táticos, psicológicos ou históricos que se repetem ao longo de gerações. O H2H é o único vetor que guarda esse padrão específico de par de times.|

\*\*Limitação reconhecida:\*\* o H2H foi inserido manualmente para os 24 confrontos da Copa 2026. Para pares sem histórico direto suficiente, o default é `(1, 1, 1)` — neutro —, introduzindo imprecisão para seleções que raramente se enfrentam.

#### Grupo 4 — Probabilidades implícitas de mercado (features \[16]–\[18])

|Feature|Critério de inclusão|
|-|-|
|`prob_v1`, `prob_emp`, `prob_v2`|As odds de casas de apostas agregam informação de dezenas de analistas especializados, incluindo dados privados de elenco, lesões e contexto tático não disponíveis no dataset histórico. Empiricamente, mercados de apostas esportivas são bem calibrados para futebol. As probabilidades são normalizadas pela soma total para remover a margem da casa antes de entrar no modelo.|



#### Grupo 5 — Contexto ambiental e sede (features \[19]–\[21])

|Feature|Critério de inclusão|
|-|-|
|`temp_dist_t1`, `temp_dist_t2`|A Copa 2026 será disputada no verão norte-americano em cidades como Miami, Los Angeles e Dallas. Condições de calor e umidade afetam times de climas frios de forma mais pronunciada. A distância da temperatura média anual do país até 14 °C (referência de temperatura ótima para desempenho físico em futebol) é uma proxy desse efeito de adaptação climática.|
|`host_flag`|Vantagem de jogar em casa é empiricamente comprovada em Copas do Mundo (Brasil 1950, Argentina 1978, França 1998, Alemanha 2006). O flag vale +1 para EUA/CAN/MEX como time 1, −1 se forem time 2, e 0 para confrontos sem sede. No \*\*treino\*\*, esse flag é sempre 0, pois as Copas históricas são disputadas em campo neutro para os visitantes — ele só é ativado na previsão dos 24 jogos de 2026.|



#### Resumo das variáveis excluídas

* GDP per capita e população, presentes em versões anteriores do código. Não há mecanismo causal direto entre tamanho econômico e resultado em campo; a sua inclusão inflava artificialmente a aparência do modelo sem contribuição preditiva real. Removidos na auditoria.
-estatísticas globais de carreira (médias sobre toda a história), pois já estão implicitamente capturadas pelo Elo, que é calculado sobre o histórico completo.
* as odds brutas (ex.: 2,20 / 3,10 / 3,50). Após a conversão para probabilidades normalizadas, os valores originais são redundantes e colineares — mantê-los geraria multicolinearidade perfeita com as três probabilidades derivadas.
* altitude das cidades-sede (sem variabilidade real para a Copa 2026, cujas sedes ficam em baixa altitude); fuso horário (correlacionado com temperatura e geograficamente redundante); one-hot encoding de identidade de time (criaria \~200 features esparsas com poucos exemplos por seleção, levando a overfitting garantido com menos de 300 amostras de treino).



## 3\. Pré-processamento

### Limpeza

* Nomes de países normalizados via dicionário `NOME_COD` com lookup O(1) em minúsculas.
* Registros com código `UNK` são filtrados antes da construção dos datasets.
* Eliminatórias e histórico de Copas são concatenados com deduplicação por subconjunto `(home_team, away_team, gols_home, gols_away)`.

### Codificação de Países

Cada país é representado pelo seu código FIFA de 3 letras. Não há embeddings ou one-hot encoding de identidade de time; a identidade é \*\*implícita nas features numéricas\*\* calculadas sobre o histórico de cada seleção (Elo, forma, força).

### Criação de Features (22 dimensões)

|Índice|Feature|Descrição|
|-|-|-|
|0–2|`elo_t1`, `elo_t2`, `diff_elo`|Elo rating individual e diferença|
|3–4|`rank_t1`, `rank_t2`|Ranking FIFA (menor = melhor)|
|5–6|`forma_t1`, `forma_t2`|Aproveitamento % nos últimos 10 jogos|
|7–8|`saldo_t1`, `saldo_t2`|Saldo médio de gols nos últimos 10 jogos|
|9–10|`forca_of_t1`, `forca_of_t2`|Média de gols marcados nos últimos 10|
|11–12|`forca_def_t1`, `forca_def_t2`|Média de gols sofridos nos últimos 10|
|13–15|`h2h_v1`, `h2h_emp`, `h2h_v2`|Head-to-head histórico entre os dois times|
|16–18|`prob_v1`, `prob_emp`, `prob_v2`|Probabilidades implícitas das odds (normalizadas)|
|19–20|`temp_dist_t1`, `temp_dist_t2`|Distância da temperatura média do país a 14°C (ideal)|
|21|`host_flag`|+1 se t1 é sede (EUA/CAN/MEX), −1 se t2 é, 0 se neutro|

### Normalização

O `StandardScaler` do scikit-learn é \*\*ajustado exclusivamente nos dados de treino\*\* (Copa ≤ 2018) e aplicado por transformação (sem re-ajuste) nos conjuntos de validação, teste e previsão. Isso evita que informações dos dados futuros vazem para a normalização.

### Balanceamento por Espelhamento

Cada jogo é duplicado com os times invertidos (t1 ↔ t2), eliminando o viés posicional — ou seja, o modelo não aprende a favorecer sistematicamente o "time 1" apenas por aparecer nessa posição.

\---

## 4\. Modelo TensorFlow

### Arquitetura

O modelo é uma \*\*rede neural multi-saída\*\* com tronco compartilhado e duas cabeças independentes:

```

Input (22 features)
│
├─ Dense(128) → BatchNorm → ReLU → Dropout(0.30)
├─ Dense(64)  → BatchNorm → ReLU → Dropout(0.20)
├─ Dense(32)  → ReLU → Dropout(0.10)
└─ Dense(16)  → ReLU
│
┌──────┴──────┐
│             │
Dense(1,softplus)  Dense(1,softplus)
gols_t1          gols_t2

```

A ativação \*\*softplus\*\* na camada de saída garante que a taxa prevista λ seja sempre positiva, requisito da distribuição de Poisson.

### Função de Perda

\*\*Poisson Loss:\*\* `L = ŷ − y × log(ŷ)`

Essa função é a escolha estatisticamente correta para dados de contagem como gols. Ao contrário do MSE, ela penaliza erros de forma assimétrica na escala log — a diferença entre 0 e 1 gol é tratada como mais significativa do que entre 3 e 4 gols, refletindo a realidade de jogos de futebol.

> Referência: Dixon & Coles (1997) — *Modelling Association Football Scores*.

### Otimizador e Callbacks

* \*\*Otimizador:\*\* Adam com `learning_rate = 0.001`
* \*\*EarlyStopping:\*\* monitora `val_loss`, patience = 50, restaura os melhores pesos
* \*\*ReduceLROnPlateau:\*\* reduz LR à metade se `val_loss` não melhorar por 15 epochs, mínimo `1e-6`
* \*\*Epochs máximos:\*\* 400 (interrompido antes pelo EarlyStopping)
* \*\*Batch size:\*\* 64 (é a escolha padrão da literatura para datasets pequenos)

### Métricas

* \*\*MAE (Mean Absolute Error):\*\* por saída (`gols_t1_mae`, `gols_t2_mae`) — interpretável em unidade de gols
* \*\*RMSE:\*\* complementar ao MAE para penalizar erros maiores
* \*\*Acerto de placar exato:\*\* percentual de jogos com gols arredondados corretos para ambos os times

### Justificativa das Escolhas

* Tronco compartilhado: força o modelo a aprender uma representação conjunta do contexto do jogo, evitando que as duas previsões de gols sejam feitas de forma totalmente independente.
* BatchNormalization: estabiliza o treinamento com datasets pequenos (< 500 amostras de treino).
* Dropout progressivo (30% → 20% → 10%): regularização mais forte nas camadas iniciais (mais parâmetros) e mais suave nas camadas finais (representação mais específica).

\---

## 5\. Avaliação

### Validação Histórica com Split Temporal

O split é estritamente cronológico, sem qualquer embaralhamento:

|Conjunto|Dados|Papel|
|-|-|-|
|Treino|Copa 1950 – 2018|Ajuste dos pesos da rede|
|Validação|Copa 2022|Seleção de hiperparâmetros e EarlyStopping|
|Teste|Eliminatórias 2026|Avaliação final out-of-sample|
|Previsão|24 jogos 1ª rodada Copa 2026|Saída do bolão|

O Elo Rating do conjunto de treino usa um snapshot calculado apenas até 2018, evitando que o modelo acesse indiretamente informações da Copa 2022 ao construir as features de treino.

### Baseline Simples

O modelo é implicitamente comparado com um baseline que sempre prevê placar constante (ex.: 1×1 ou 2×1). Como o MAE típico para previsão de gols em modelos competitivos fica entre 0.9 e 1.3 gols por time, valores abaixo disso indicam ganho real sobre o baseline ingênuo.

### Limitações

1. \*\*Volume de dados restrito:\*\* Copas do Mundo têm \~70 a \~150 jogos por edição; o conjunto de treino total (1950–2018) tem apenas \~140 jogos originais (\~280 após espelhamento), o que é pequeno para uma rede neural.
2. \*\*Gols de Copa ≠ Gols de eliminatórias:\*\* o comportamento tático e o nível de motivação diferem entre fases, mas ambos alimentam o cálculo de forma e força.
3. \*\*Odds estáticas:\*\* as probabilidades de apostas foram inseridas manualmente para os 24 jogos e não refletem movimentações de mercado em tempo real.
4. \*\*Ausência de dados de lesões e convocações:\*\* a composição real dos elencos na Copa não é capturada pelo modelo.
5. \*\*Países com pouco histórico:\*\* seleções como Curaçau (`CUW`) e Haiti (`HAI`) têm histórico muito escasso, e suas features recebem valores default (forma = 45%, gols = 1.0), introduzindo incerteza adicional.

\---



## 6\. Conclusão

### Análise dos Resultados Esperados

O modelo tende a refletir o \*\*favoritismo técnico\*\* capturado pelo Elo Rating e pelas odds de apostas: seleções europeias e sul-americanas tradicionais (ARG, BRA, FRA, ENG, ESP, GER, POR) devem aparecer como vencedoras na maioria dos seus confrontos. Jogos envolvendo países-sede (EUA, CAN, MEX) recebem o bônus de `host_flag`, o que pode elevar ligeiramente as previsões de gols ou resultados favoráveis a esses times.

A distribuição típica esperada para a 1ª rodada é de aproximadamente \*\*14–16 vitórias do favorito (T1), 5–8 empates e 2–4 surpresas (vitória do azarão)\*\*, refletindo o nível de desequilíbrio real entre as seleções nessa fase.

### Principais Incertezas

* \*\*Surpresas da Copa 2022\*\* (Japão × Alemanha, Marrocos × Bélgica, Arábia Saudita × Argentina) mostram que azarões podem vencer favoritos com frequência maior do que o modelo captura — a rede tende a ser conservadora e subestima upset probability.
* Times com \*\*histórico escasso\*\* têm previsões menos confiáveis.
* A \*\*forma recente\*\* calculada inclui goleadas nas eliminatórias (ex.: Noruega 11×1 Moldávia), que inflacionam as médias ofensivas sem refletir o nível de oposição.

### Possíveis Melhorias Futuras

1. \*\*Modelo probabilístico de Dixon-Coles completo:\*\* prever a distribuição de Poisson bivariada completa, não apenas o valor esperado, permitindo calcular probabilidades de todos os placares possíveis.
2. \*\*Features de contexto real:\*\* incluir dados de elenco atual (valor de mercado via Transfermarkt), lesões e suspensões nas semanas anteriores à Copa.
3. \*\*Dados de odds em tempo real:\*\* integrar API de casas de apostas para atualizar as probabilidades implícitas próximo à data dos jogos.
4. \*\*Modelo ensemble:\*\* combinar a rede neural com um modelo estatístico (regressão de Poisson com efeitos aleatórios por time) via média ponderada ou stacking.
5. \*\*Calibração das saídas:\*\* aplicar Platt scaling ou isotonic regression para calibrar as taxas λ previstas e melhorar a acurácia dos placares arredondados.
6. \*\*Tratamento de times novatos:\*\* criar um mecanismo de "empréstimo de força" baseado na confederação de origem para times com menos de 5 jogos no histórico.



\*\*Dados e fontes\*\*

* FJELSTUL, J. World Cup Database. Disponível em: https://datahub.io/sports-data/fifa-world-cup. Acesso em: jun. 2026.
Fonte primária dos resultados de todas as Copas do Mundo de 1950 a 2022 usados no treino e validação do modelo.
* WIKIMEDIA FOUNDATION. Wikipedia API (MediaWiki). Disponível em: https://en.wikipedia.org/w/api.php. Acesso em: jun. 2026.
Utilizada para scraping dos resultados das eliminatórias 2026 por confederação (CONMEBOL, UEFA, CAF, AFC, CONCACAF, OFC).
* WORLD FOOTBALL ELO RATINGS. EloRatings.net. Disponível em: https://www.eloratings.net. Acesso em: jun. 2026.
Referência metodológica para o cálculo incremental do Elo com fator de margem de gols, adotado como base conceitual para a implementação própria do projeto.

\*\*Durante a preparação deste trabalho, foi utilizada a ferramenta  Claude Sonet 4.6 para revisão do código e pesquisar dos dados\*\*

