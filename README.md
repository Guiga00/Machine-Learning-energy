# Previsão de Conta de Energia Elétrica com Redes Neurais

Este projeto utiliza **Redes Neural** implementado com **TensorFlow/Keras** para prever o valor da conta de energia elétrica (`Valor_R$`) com base em um conjunto de dados históricos. A aplicação é disponibilizada através de uma interface interativa construída com **Streamlit**.

-----

## Objetivo

O objetivo principal é construir e avaliar um modelo de Machine Learning capaz de estimar o custo da fatura de energia elétrica.

### Características Utilizadas

O modelo utiliza as seguintes variáveis como *features* de entrada para a previsão do `Valor_R$`:

  * `Consumo_kWh` (Consumo total em kWh)
  * `TempMedia` (Temperatura Média em °C)
  * `ArCond_dias` (Número de dias em que o ar-condicionado foi usado)
  * `ArCond_kWh` (Consumo em kWh atribuído ao ar-condicionado)

### Performance do Modelo

O modelo é uma rede neural sequencial. As métricas de desempenho no conjunto de **teste** são:

  * **MAE (Erro Médio Absoluto):** R$ 13.91
  * **RMSE (Raiz do Erro Quadrático Médio):** R$ 19.34
  * **R² (Coeficiente de Determinação):** 0.8105

-----

## Tutorial de Execução

Siga os passos abaixo para instalar as dependências, treinar o modelo (opcional) e rodar a aplicação Streamlit localmente. 

O repositório já possui o modelo treinado com os arquivos necessários para serem testados, mas se quiser treinar do zedo, delete os arquivos: 

```
correlacao_variaveis.png
previsoes_vs_reais.png
temperatura_vs_gasto.png
treinamento_metricas.png
scaler_X.pkl
scaler_Y.pkl
metricas.pkl
modelo_energia.h5
```

### Requerimentos Locais

Você precisará ter o **Python** (versão 3.x) e o **`pip`** (gerenciador de pacotes do Python) instalados em seu ambiente.

### Instalação das Dependências

As bibliotecas necessárias para o projeto são:

```text
pandas
numpy
tensorflow
scikit-learn
matplotlib
seaborn
streamlit
plotly
Pillow
```

1.  **Instale as bibliotecas:**
    ```bash
    pip install -r requirements.txt
    ```

### Treinamento do Modelo (Opcional)

Se você deseja treinar o modelo do zero, execute o script `trainer.py`. O repositório já possui os modelos treinados.

1.  **Rode o script de treinamento:**
    ```bash
    python trainer.py
    ```

### Abrindo a Aplicação Interativa

O modelo treinado pode ser explorado e testado através da aplicação web interativa Streamlit:

1.  **Rode a aplicação Streamlit:**
    ```bash
    python -m streamlit run app.py
    ```
    *A aplicação será aberta no seu navegador, exibindo tabelas, gráficos e permitindo testar o modelo*.

-----

## Análises e Visualizações

O projeto inclui as seguintes visualizações geradas durante o treinamento:

  * **Correlação entre Variáveis** [correlacao\_variaveis.png]: Matriz de correlação entre as features e o valor da conta.
  * **Temperatura vs Gasto de Energia** [temperatura\_vs\_gasto.png]: Dispersão entre temperatura e valor, colorida pelo uso do ar-condicionado.
  * **Evolução do Treinamento** [treinamento\_metricas.png]: Gráficos de Loss e MAE ao longo das épocas de treinamento.
  * **Previsões vs Valores Reais** [previsoes\_vs\_reais.png]: Comparação gráfica entre os valores previstos e os valores reais do conjunto de teste.
