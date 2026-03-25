# Notebooks educativos que desmistificam LLMs e modelos de visão. Do bigrama ao Transformer e Difusão,  tudo do zero, com Python puro e PyTorch. Sem hype, só código.

> *"Um LLM não está a caminho da consciência. Ele está a caminho da próxima palavra."*

---

## Por que este repositório existe

O mercado está intoxicado de hype.

Nos últimos anos acumulamos narrativas sobre dot-com, Web 2.0, IoT, cloud, blockchain, big data, metaverso e NFTs. Em cada onda, a proporção entre promessa de vendor e resultado real para o cliente foi desfavorável ao cliente. Em cada onda, quem perdeu mais foi quem acreditou mais cedo e sem critério.

A onda atual dos LLMs tem elementos de fato diferentes das anteriores, e eu argumento isso com seriedade no paper [*Abundance of Tools and the Limits of Human Efficiency*](https://zenodo.org/records/18301091). Modelos de linguagem representam, pela primeira vez na história, a capacidade artificial de raciocínio retórico entre domínios e linguagens. Isso tem implicações reais para como organizações se estruturam, como humanos se coordenam, e como sistemas produtivos são arquitetados.

Mas uma coisa específica me irrita profundamente: a narrativa de que LLMs nos aproximam da AGI, de que terão autoconsciência, de que "pensam" ou "entendem" no sentido que atribuímos a humanos.

Isso não é apenas errado. É prejudicial. Porque quem acredita nisso toma decisões de adoção baseadas em fantasia, e quem não acredita descarta a tecnologia por reação ao exagero, e perde o que ela oferece.

**Este repositório é uma resposta prática a esse problema.**

Construí seis notebooks que percorrem, do zero, a lógica interna de modelos de linguagem e visão. Qualquer pessoa com noções básicas de Python consegue acompanhar. O objetivo não é ensinar machine learning formal. O objetivo é tirar o LLM do campo da magia e colocá-lo no campo da engenharia, onde ele pertence.

---

## O que um LLM realmente faz

Antes de abrir qualquer notebook, uma única frase para guardar:

> **Um modelo de linguagem prevê o próximo token. Sempre. Só isso.**

O que muda entre um modelo de bigramas de 1950 e o GPT-4 de 2023 não é o objetivo. O objetivo é idêntico: dado o que veio antes, qual é a sequência mais provável a seguir?

O que muda é:
- **Como o contexto é representado**: de uma tabela de contagem para vetores densos em espaços de altíssima dimensão
- **Quanto contexto é considerado**: de 1 palavra para dezenas de milhares de tokens
- **A arquitetura que processa esse contexto**: de n-gramas para transformers com bilhões de parâmetros
- **A escala de dados e computação**: de dezenas de frases para trilhões de tokens e meses de GPU

Mas o mecanismo fundamental não mudou. E ao entender esse mecanismo, os limites tornam-se visíveis.

Um modelo de linguagem não tem intenção. Não tem memória persistente entre conversas. Não raciocina causalmente. Não verifica se o que diz é verdade. Não tem modelo de mundo. Não tem consciência de si mesmo.

Ele é, como argumento no paper, **infraestrutura de alta largura de banda e baixa latência para transporte e transformação de informação**. Uma infraestrutura extraordinariamente útil,  mas infraestrutura. Não agente. Não mente.

---

## Os seis notebooks

### `aula1_modelo_linguagem_ingenuo.ipynb`

**O que é:** Um modelo de linguagem construído do zero com Python puro,  sem PyTorch, sem numpy, sem nenhuma dependência além da biblioteca padrão.

**O que você verá:** Um corpus de frases do agro brasileiro é tokenizado, seus bigramas são contados, e o modelo gera frases sorteando a próxima palavra com probabilidade proporcional à frequência observada. Isso já é um modelo de linguagem. Um modelo raso, limitado, mas funcionalmente equivalente ao GPT em seu objetivo central.

**Por que importa:** Quando você vê que dá para escrever um modelo de linguagem em 30 linhas de Python usando `defaultdict` e `random.choice`, a aura mística desaparece. O que resta é a pergunta certa: o que a escala e a arquitetura adicionam a isso?

**Conceito central:** Tokenização, bigramas, amostragem proporcional à frequência, o limite de vocabulário.

---

### `aula2_modelo_linguagem_limite_do_contexto.ipynb`

**O que é:** Uma demonstração cirúrgica de onde o modelo ingênuo falha,  e por que a falha é estrutural, não cosmética.

**O que você verá:** A frase *"A fazenda teve estiagem severa, então a produtividade do milho caiu"* processada por modelos com janelas de 1 a 6 palavras. Uma visualização mostra, token a token, quando a palavra *"estiagem"* sai do campo de visão de cada modelo,  e, com ela, a informação que determina o desfecho correto. Uma simulação de esvanecimento de gradiente mostra por que mesmo RNNs sofrem com isso.

**Por que importa:** O problema do contexto não é um bug de implementação. É uma consequência direta de processar linguagem como um fluxo com memória de tamanho fixo. Nenhum aumento de janela resolve isso definitivamente,  porque sempre haverá dependências que excedem a janela.

**Conceito central:** Dependências longas, esparsidade de n-gramas, esvanecimento de gradiente simulado.

---

### `aula2_complexa_modelo_linguagem_RNN_e_LSTM.ipynb`

**O que é:** RNN e LSTM reais,  implementadas com as equações verdadeiras, treinadas de verdade, com backpropagation real via PyTorch.

**O que você verá:** A equação `h_t = tanh(W_xh·x + W_hh·h + b)` implementada linha a linha em NumPy. Os quatro portões da LSTM,  esquecimento, entrada, célula, saída,  implementados com as fórmulas de Hochreiter & Schmidhuber (1997). Ambos os modelos treinados na tarefa de prever próxima palavra. Uma análise de divergência mostra quanto os estados internos de *"contexto de estiagem"* e *"contexto de chuva"* se distinguem ao longo da frase,  e onde essa distinção começa a colapsar.

**Por que importa:** A LSTM foi um avanço real. Os portões permitem que informação importante persista por mais passos. Mas o gargalo do estado oculto e a natureza sequencial do processamento impõem limites duros. Este notebook torna esses limites quantitativos e visíveis.

**Conceito central:** Recorrência, portões, cell state, vanishing gradient, divergência de contextos.

> **Nota:** Este notebook é um complemento técnico ao `aula2_modelo_linguagem_limite_do_contexto.ipynb`, que usa simulações simples. Se o objetivo é uma aula introdutória, o aula2 original é suficiente. Este é para audiências que querem ver o mecanismo real.

---

### `aula3_modelo_linguagem_atencao.ipynb`

**O que é:** O mecanismo de atenção implementado do zero em NumPy, com três formas de visualização.

**O que você verá:** Embeddings construídos a partir de co-ocorrências no corpus. A fórmula `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V` implementada e comentada linha a linha. Para a frase *"A produtividade caiu porque a estiagem afetou a lavoura"*, três visualizações da mesma matriz de atenção:

1. **Tabela de texto**,  linhas são tokens que perguntam, colunas são tokens que respondem
2. **Heatmap**,  a mesma matriz como imagem, com intensidade de cor proporcional ao peso
3. **Texto com pesos**,  para cada token, um ranking visual dos tokens que ele mais atende, com barras ASCII e blocos unicode inline na frase

Em seguida, side-by-side: a frase com estiagem ao lado da frase com chuva. A atenção do token *"caiu"* vai para *"estiagem"*; a atenção do token *"subiu"* vai para *"chuva"*. Sem sequencialidade. Sem esvanecimento. Conexão direta.

**Por que importa:** Isto é exatamente o que a RNN não conseguia fazer. A atenção permite que qualquer token consulte diretamente qualquer outro token da sequência, independente da distância. Esse é o mecanismo que viabilizou a escala dos LLMs modernos.

**Conceito central:** Q/K/V, produto escalar normalizado, softmax, atenção multi-cabeça conceitual.

---

### `aula3_complexa_modelo_linguagem_transformer.ipynb`

**O que é:** Um transformer real treinado em PyTorch, com extração e visualização de pesos de atenção genuínos.

**O que você verá:** Positional encoding sinusoidal implementado com as fórmulas do paper *"Attention Is All You Need"* (Vaswani et al., 2017). Multi-Head Attention implementada manualmente,  não com `nn.MultiheadAttention`, mas com as projeções W_Q, W_K, W_V, W_O explícitas. Encoder blocks com Add&Norm e feed-forward. O modelo treinado por 500 épocas. Os pesos de atenção extraídos por camada e por cabeça. As mesmas três visualizações do demo3 didático,  agora com pesos reais, aprendidos por backpropagation. Uma análise de qual cabeça especializou-se em capturar a relação estiagem→caiu versus chuva→subiu. Uma comparação de atenção versus distância mostrando que o transformer não tem decaimento,  ao contrário da RNN.

**Por que importa:** Quando os pesos são reais, as visualizações revelam o que o modelo genuinamente aprendeu. Cabeças diferentes capturam relações diferentes. Isso não é interpretabilidade perfeita,  atenção não é explicação,  mas é uma janela real para o mecanismo.

**Conceito central:** Positional encoding, multi-head attention real, extração de atenção, especialização de cabeças.

---

### `aula4_complexa_vision_model.ipynb`

**O que é:** Modelos de visão desmistificados,  da convolução manual ao processo de difusão completo (DDPM), tudo implementado do zero em PyTorch.

**O que você verá:** Imagens tratadas como matrizes de inteiros, não conceitos visuais. Filtros de Sobel implementados manualmente e comparados com os filtros aprendidos por uma CNN,  a matemática é idêntica. Uma CNN simples treinada para classificar círculos, quadrados e triângulos com ~99% de acurácia. O processo forward de difusão (adição progressiva de ruído) e o processo reverse (geração via denoising iterativo), seguindo o paper DDPM (Ho et al., 2020). Dois denoisers treinados com os mesmos hiperparâmetros,  um aprende a distribuição de círculos, outro a de triângulos,  gerando imagens a partir de ruído puro.

**Por que importa:** O mesmo argumento dos notebooks de linguagem se aplica aqui: DALL-E, Stable Diffusion e Veo não "imaginam" imagens. Eles aprendem distribuições condicionais de pixels e amostram dessas distribuições. A "magia" é escala,  o mecanismo é o mesmo do demo1.

**Conceito central:** Convolução, CNN, classificação, noise schedule DDPM, denoising MLP, processo forward e reverse de difusão.

---

## O que estes notebooks não mostram

Eles não mostram treinamento em escala. Não mostram RLHF. Não mostram fine-tuning. Não mostram como o ChatGPT funciona end-to-end.

Isso é intencional.

O objetivo é mostrar o mecanismo mínimo,  o suficiente para que a pessoa entenda por que um LLM é o que é, e por que não é o que o mercado diz que é.

**O que falta para ir de um transformer de brinquedo para o GPT-4:**

- Corpus de trilhões de tokens (toda a internet + livros + código)
- Bilhões de parâmetros em vez de milhares
- Meses de treino em milhares de GPUs
- Embeddings ricos o suficiente para capturar estrutura semântica em escala
- RLHF para alinhar o comportamento ao que humanos consideram útil
- Infraestrutura de inferência para servir milhões de usuários simultâneos

Tudo isso é engenharia extraordinária. É também o motivo pelo qual poucos atores no mundo conseguem construir modelos de fronteira.

Mas o objetivo,  prever o próximo token,  é o mesmo que o do bigrama do Demo 1.

---

## A posição que defendo

LLMs são a mais importante inovação em infraestrutura de informação das últimas décadas. Pela primeira vez, é possível ter um interlocutor que opera em linguagem natural, em escala, com latência próxima de zero, sobre qualquer domínio de conhecimento.

Isso não é pouco. É transformador para como organizações se coordenam, como humanos interagem com sistemas complexos, como produtividade pode ser amplificada.

Mas o limite estrutural da tecnologia é preciso: **um modelo de linguagem não tem intenção, não tem consciência, não tem modelo causal do mundo**. Ele generaliza padrões de linguagem com uma eficácia sem precedentes. Generalização não é compreensão.

As narrativas de AGI e autoconsciência não são apenas prematuras,  são categorialmente equivocadas. Elas confundem fluência linguística com raciocínio, e raciocínio com consciência. São confusões que servem a quem vende e prejudicam quem compra.

A forma mais eficaz de combater esse equívoco não é um argumento filosófico. É mostrar o mecanismo.

Quando você vê que um modelo de linguagem é, na essência, uma tabela de probabilidades condicionais sofisticada,  e que todo o progresso dos últimos décadas foi sobre como tornar essa tabela mais rica, mais densa, e mais capaz de capturar contexto longo,  você para de atribuir mistério onde há matemática.

E quando você para de atribuir mistério, você começa a usar a ferramenta certa, no lugar certo, com as expectativas certas.

---

## Como usar este repositório

**Se você tem 30 minutos:** Abra o `aula1_modelo_linguagem_ingenuo.ipynb` e execute célula por célula. Só isso já muda a percepção.

**Se você tem 2 horas:** Percorra os notebooks em ordem. Cada um termina com uma ponte para o próximo.

**Se você quer usar em aula:** Os notebooks são autocontidos e progressivos. O `aula1` e o `aula2` funcionam com Python puro. O `aula2_complexa`, os `aula3` e o `aula4_complexa` precisam de `torch` instalado.

```
pip install numpy matplotlib torch
```

**Se você quer aprofundar:** O paper [*Abundance of Tools and the Limits of Human Efficiency*](https://zenodo.org/records/18301091) desenvolve a tese de que LLMs são infraestrutura de coordenação,  não inteligência autônoma,  e as implicações arquiteturais disso para organizações produtivas.

---

## Estrutura dos notebooks

```
aula1_modelo_linguagem_ingenuo.ipynb                | Python puro. Sem dependências externas.
aula2_modelo_linguagem_limite_do_contexto.ipynb     | Python puro. Sem dependências externas.
aula2_complexa_modelo_linguagem_RNN_e_LSTM.ipynb    | numpy + matplotlib + torch
aula3_modelo_linguagem_atencao.ipynb                | numpy + matplotlib
aula3_complexa_modelo_linguagem_transformer.ipynb   | numpy + matplotlib + torch
aula4_complexa_vision_model.ipynb                   | numpy + matplotlib + torch
```

---

## Autor

José Ricardo de Oliveira Damico
[jose.damico@scicrop.com](mailto:jose.damico@scicrop.com)
