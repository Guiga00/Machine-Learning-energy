import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Configuração da página
st.set_page_config(
    page_title="Previsão de Conta de Energia",
    page_icon="",
    layout="wide"
)

# Título principal
st.title("Sistema de Previsão de Conta de Energia Elétrica")
st.markdown("### Análise baseada em Inteligência Artificial")
st.markdown("---")

# Carregar modelo e scalers
@st.cache_resource
def carregar_modelo():
    model = keras.models.load_model('modelo_energia.h5', compile=False)
    with open('scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    with open('metricas.pkl', 'rb') as f:
        metrics = pickle.load(f)
    return model, scaler_X, scaler_y, metrics

try:
    model, scaler_X, scaler_y, metrics = carregar_modelo()
    df = pd.read_csv('dataset.csv')
    
    # Sidebar
    st.sidebar.header("Navegação")
    pagina = st.sidebar.radio(
        "Escolha uma seção:",
        ["Visão Geral", "Fazer Previsão", "Análises", "Performance do Modelo"]
    )
    
    # ==================== PÁGINA: VISÃO GERAL ====================
    if pagina == "Visão Geral":
        st.header("Visão Geral do Projeto")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Registros", f"{len(df)}")
        with col2:
            st.metric("Valor Médio", f"R$ {df['Valor_R$'].mean():.2f}")
        with col3:
            st.metric("Temperatura Média", f"{df['TempMedia'].mean():.1f}°C")
        with col4:
            st.metric("Precisão (R²)", f"{metrics['r2_test']:.2%}")
        
        st.markdown("---")
        
        # Gráfico de evolução temporal
        st.subheader("Evolução dos Valores ao Longo do Tempo")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Valor_R$'],
            mode='lines+markers',
            name='Valor da Conta',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=4)
        ))
        fig.update_layout(
            xaxis_title="Mês",
            yaxis_title="Valor (R$)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar imagens geradas
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(" Temperatura × Gasto")
            st.image('temperatura_vs_gasto.png')
        with col2:
            st.subheader(" Correlação entre Variáveis")
            st.image('correlacao_variaveis.png')
    
    # ==================== PÁGINA: FAZER PREVISÃO ====================
    elif pagina == "Fazer Previsão":
        st.header("Fazer Nova Previsão")
        
        col1, col2 = st.columns(2)
        
        with col1:
            consumo = st.number_input("Consumo (kWh)", min_value=0.0, value=300.0, step=10.0)
            temperatura = st.number_input("Temperatura Média (°C)", min_value=0.0, value=28.0, step=0.5)
        
        with col2:
            ar_dias = st.number_input("Dias com Ar-Condicionado", min_value=0, value=15, step=1)
            ar_kwh = st.number_input("Consumo do Ar (kWh)", min_value=0.0, value=80.0, step=5.0)
        
        if st.button("PREVER VALOR", type="primary"):
            # Fazer previsão
            entrada = np.array([[consumo, temperatura, ar_dias, ar_kwh]])
            entrada_scaled = scaler_X.transform(entrada)
            previsao_scaled = model.predict(entrada_scaled, verbose=0)
            previsao = scaler_y.inverse_transform(previsao_scaled)[0][0]
            
            st.markdown("---")
            st.success(f"### Valor Previsto: R$ {previsao:.2f}")
            
            # Comparação com média
            diferenca = previsao - df['Valor_R$'].mean()
            if diferenca > 0:
                st.warning(f"R$ {abs(diferenca):.2f} acima da média histórica")
            else:
                st.info(f"R$ {abs(diferenca):.2f} abaixo da média histórica")
    
    # ==================== PÁGINA: ANÁLISES ====================
    elif pagina == "Análises":
        st.header("Análises Detalhadas")
        
        st.subheader("Consumo por Temperatura")
        fig = px.scatter(
            df, x='TempMedia', y='Valor_R$', size='ArCond_kWh',
            color='ArCond_dias', color_continuous_scale='YlOrRd',
            labels={'TempMedia': 'Temperatura (°C)', 'Valor_R$': 'Valor (R$)'},
            hover_data=['Consumo_kWh', 'Bandeira']
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Impacto do Ar-Condicionado")
            fig = px.box(df, x='Bandeira', y='Valor_R$', color='Bandeira',
                         labels={'Valor_R$': 'Valor (R$)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Distribuição de Valores")
            fig = px.histogram(df, x='Valor_R$', nbins=30,
                               labels={'Valor_R$': 'Valor (R$)'})
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== PÁGINA: PERFORMANCE ====================
    elif pagina == "Performance do Modelo":
        st.header("Performance do Modelo")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE (Treino)", f"R$ {metrics['mae_train']:.2f}")
        with col2:
            st.metric("MAE (Teste)", f"R$ {metrics['mae_test']:.2f}")
        with col3:
            st.metric("R² (Teste)", f"{metrics['r2_test']:.2%}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Evolução do Treinamento")
            st.image('treinamento_metricas.png')
        
        with col2:
            st.subheader("Previsões vs Valores Reais")
            st.image('previsoes_vs_reais.png')
        
        st.markdown("---")
        st.info("**Dica:** MAE baixo significa que o modelo erra pouco. R² próximo de 1 indica ótima precisão!")

except FileNotFoundError:
    st.error("Modelo não encontrado! Execute primeiro: `python trainer_melhorado.py`")
    st.stop()