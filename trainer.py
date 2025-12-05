import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# ==================== CARREGAR E EXPLORAR DADOS ====================
df = pd.read_csv('dataset.csv')
print(f" Dataset carregado: {len(df)} registros")

# Usar múltiplas features para melhorar previsão
features = ['Consumo_kWh', 'TempMedia', 'ArCond_dias', 'ArCond_kWh']
target = 'Valor_R$'

# Preparar dados
X = df[features].values
y = df[target].values.reshape(-1, 1)

print(f"\n Features selecionadas: {features}")
print(f" Target: {target}")
print(f" Shape X: {X.shape}")
print(f" Shape y: {y.shape}")

# ==================== ANÁLISE DE CORRELAÇÃO ====================
correlation_data = df[features + [target]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_data, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlação entre Variáveis', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlacao_variaveis.png', dpi=300, bbox_inches='tight')
print(" Salvo: correlacao_variaveis.png")
plt.close()

# ==================== NORMALIZAÇÃO ====================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ==================== DIVISÃO TREINO/TESTE ====================
split_idx = int(0.8 * len(X_scaled))
X_train = X_scaled[:split_idx]
y_train = y_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_test = y_scaled[split_idx:]

print(f"\n Treino: {len(X_train)} amostras")
print(f" Teste: {len(X_test)} amostras")

# ==================== CONSTRUIR MODELO ====================
model = Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(128, activation='relu', name='camada_1'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', name='camada_2'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu', name='camada_3'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu', name='camada_4'),
    layers.Dense(1, name='saida')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n Arquitetura do modelo:")
model.summary()

# ==================== TREINAR ====================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=16,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)
    ]
)

# ==================== AVALIAR ====================
y_train_pred = model.predict(X_train, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)

# Desescalar
y_train_real = scaler_y.inverse_transform(y_train)
y_train_pred_real = scaler_y.inverse_transform(y_train_pred)
y_test_real = scaler_y.inverse_transform(y_test)
y_test_pred_real = scaler_y.inverse_transform(y_test_pred)

mae_train = mean_absolute_error(y_train_real, y_train_pred_real)
rmse_train = np.sqrt(mean_squared_error(y_train_real, y_train_pred_real))
r2_train = r2_score(y_train_real, y_train_pred_real)

mae_test = mean_absolute_error(y_test_real, y_test_pred_real)
rmse_test = np.sqrt(mean_squared_error(y_test_real, y_test_pred_real))
r2_test = r2_score(y_test_real, y_test_pred_real)

print("\n" + "="*60)
print(" MÉTRICAS DO MODELO")
print("="*60)
print(f"\n TREINO:")
print(f"   MAE:  R$ {mae_train:.2f}")
print(f"   RMSE: R$ {rmse_train:.2f}")
print(f"   R²:   {r2_train:.4f}")
print(f"\n TESTE:")
print(f"   MAE:  R$ {mae_test:.2f}")
print(f"   RMSE: R$ {rmse_test:.2f}")
print(f"   R²:   {r2_test:.4f}")

# Salvar métricas
metrics = {
    'mae_train': mae_train, 'rmse_train': rmse_train, 'r2_train': r2_train,
    'mae_test': mae_test, 'rmse_test': rmse_test, 'r2_test': r2_test
}
with open('metricas.pkl', 'wb') as f:
    pickle.dump(metrics, f)

# ==================== GRÁFICO TEMPERATURA VS GASTO ====================
plt.figure(figsize=(12, 6))
plt.scatter(df['TempMedia'], df['Valor_R$'], alpha=0.6, s=50, c=df['ArCond_dias'], 
            cmap='YlOrRd', edgecolors='black', linewidth=0.5)
plt.colorbar(label='Dias com Ar-Condicionado')
plt.xlabel('Temperatura Média (°C)', fontsize=12)
plt.ylabel('Valor da Conta (R$)', fontsize=12)
plt.title('Temperatura × Gasto de Energia', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('temperatura_vs_gasto.png', dpi=300, bbox_inches='tight')
print(" Salvo: temperatura_vs_gasto.png")
plt.close()

# ==================== GRÁFICOS ====================
print("\n Gerando visualizações...")

# Gráfico 1: Loss
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(history.history['loss'], label='Treino', linewidth=2, color='#2E86AB')
axes[0].plot(history.history['val_loss'], label='Validação', linewidth=2, color='#A23B72')
axes[0].set_title('Evolução do Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico 2: MAE
axes[1].plot(history.history['mae'], label='Treino', linewidth=2, color='#2E86AB')
axes[1].plot(history.history['val_mae'], label='Validação', linewidth=2, color='#A23B72')
axes[1].set_title('Evolução do MAE', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('treinamento_metricas.png', dpi=300, bbox_inches='tight')
print(" Salvo: treinamento_metricas.png")
plt.close()

# Gráfico 3: Previsões vs Reais
plt.figure(figsize=(12, 6))
indices = range(len(y_test_real))
plt.plot(indices, y_test_real, 'o-', label='Valores Reais', linewidth=2, 
         markersize=8, color='#2E86AB', alpha=0.7)
plt.plot(indices, y_test_pred_real, 's--', label='Previsões', linewidth=2, 
         markersize=8, color='#E63946', alpha=0.7)
plt.xlabel('Índice de Teste', fontsize=12)
plt.ylabel('Valor (R$)', fontsize=12)
plt.title('Previsões vs Valores Reais', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('previsoes_vs_reais.png', dpi=300, bbox_inches='tight')
print(" Salvo: previsoes_vs_reais.png")
plt.close()

# ==================== SALVAR MODELO ====================
model.save('modelo_energia.h5')
with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print("\n" + "="*60)
print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("="*60)
print("\n Arquivos gerados:")
print("   • modelo_energia.h5")
print("   • scaler_X.pkl, scaler_y.pkl")
print("   • metricas.pkl")
print("   • correlacao_variaveis.png")
print("   • temperatura_vs_gasto.png")
print("   • treinamento_metricas.png")
print("   • previsoes_vs_reais.png")
print("\n Agora rode: python -m streamlit run app.py")
