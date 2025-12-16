# BERT Masked Language & Attention Visualization
## Descrição do Projeto

Este projeto demonstra o funcionamento de um modelo BERT pré-treinado em Masked Language Modeling (MLM) e fornece uma visualização gráfica do mecanismo de atenção (attention) presente nas camadas do modelo.

O programa realiza duas funções principais.

1. Previsão de palavras mascaradas:

O usuário insere uma frase contendo o token [MASK].

O modelo BERT tenta prever a palavra mais provável para esse local com base no contexto completo da frase.

São apresentadas as top K previsões para o token mascarado.

2. Visualização de atenção:

Para cada camada e cada cabeça do BERT, o programa gera uma matriz de atenção mostrando como cada token “olha” para os outros tokens na frase.

Essa matriz é convertida em uma imagem em tons de cinza, onde tons mais claros indicam maior atenção e tons mais escuros menor atenção.

As imagens ajudam a compreender como o modelo distribui a atenção entre palavras para formar suas previsões.

## Conceitos-Chave

Masked Language Modeling (MLM):
Treinamento de modelos de linguagem em que palavras são mascaradas e o modelo aprende a prever a palavra correta usando o contexto.

Attention Mechanism:
Mecanismo que permite ao modelo ponderar a importância relativa de cada token em relação aos outros tokens de uma frase.

Multi-Head Attention:
Cada camada do BERT contém várias “cabeças” de atenção, permitindo ao modelo capturar diferentes tipos de relações entre palavras.
