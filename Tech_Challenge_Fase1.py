import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy import stats


# Carregando o arquivo CSV e exibindo as primeiras linhas do dataset
# Referência base: https://www.kaggle.com/code/mragpavank/medical-cost-personal-datasets/input

try:
    dados = pd.read_csv("C:/Users/DELL/Desktop/FIAP/Pós-Tech/insurance_PT_BR.csv")
    print("Arquivo carregado com sucesso!")
except FileNotFoundError:
    print("Erro: O arquivo não foi encontrado. Verifique o caminho e o nome do arquivo.")
except Exception as e:
    print(f"Erro ao carregar o arquivo: {e}")

print("Exibindo as primeiras linhas do dataset:")
print(dados.head(20))

# Verificando quantidade de linhas e colunas

print("\nNúmero de Linhas e Colunas: ",dados.shape)

# Verificando dados nulos
celulas_nulas = dados.isnull()
total_nulos_por_coluna = celulas_nulas.sum(axis=0)
print("\nVerificando dados nulos")
print(total_nulos_por_coluna)

# Análise Descritiva dos Dados Originais

print("\nEstatísticas Descritivas dos Dados Originais:")
print(dados.describe())

# Avaliando distribuição por genero

print("\nAvaliando distribuição por genero\n")
print(dados["genero"].value_counts())

# Agrupando dados para entendimento de causualidade por regiao

print("\nAgrupando dados para entendimento de causualidade por regiao\n")
print(dados.groupby('regiao').count())


# Retirando FutureWarning dos gráficos
warnings.filterwarnings("ignore", category=FutureWarning)

plt.figure(figsize=(12, 8))
plt.subplot(3, 3, 1)
sb.histplot(dados['idade'], kde=True)
plt.title('Distribuição de Idade')

plt.subplot(3, 3, 2)
sb.histplot(dados['imc'], kde=True)
plt.title('Distribuição de IMC')

plt.subplot(3, 3, 3)
sb.histplot(dados['filhos'], kde=True)
plt.title('Distribuição de Número de Filhos')

plt.subplot(3, 3, 4)
sb.histplot(dados['encargos'], kde=True)
plt.title('Distribuição de Encargos')

plt.subplot(3, 3, 5)
sb.countplot(x='genero', data=dados)
plt.title('Distribuição de Gênero')

plt.subplot(3, 3, 6)
sb.countplot(x='fumante', data=dados)
plt.title('Distribuição de Fumantes')

plt.subplot(3, 3, 7)
sb.countplot(x='regiao', data=dados)
plt.title('Distribuição de regiões')

plt.tight_layout()
plt.show()

# Análise Comparativa Pré e Pós-Tratamento

print("\nAnálise Comparativa Pré e Pós-Tratamento\n")
contagem_fumantes = dados.groupby('idade')['fumante'].value_counts().unstack().fillna(0)
print("Contagem Fumantes por Idade\n")
print(contagem_fumantes,"\n")

contagem_fumantes = dados.groupby('filhos')['fumante'].value_counts().unstack().fillna(0)
print("Contagem Fumantes por Números de Filhos\n")
print(contagem_fumantes,"\n")

contagem_fumantes = dados.groupby('regiao')['fumante'].value_counts().unstack().fillna(0)
print("Contagem Fumantes por Região\n")
print(contagem_fumantes,"\n")


# Analisando Outliers
# Abaixo seguem os códigos para geração dos boxplot que encontram-se no relatório entregue
sb.boxplot(x=dados["encargos"])
sb.boxplot(x=dados["imc"])
sb.boxplot(x="fumante",y="encargos", data=dados, palette="hls")
sb.boxplot(x="genero",y="encargos", data=dados, palette="hls")

df_homens = dados[(dados['genero'] == 'homem') & (dados['encargos'] > 40000)]
df_homens['encargos'].count()

df_mulheres = dados[(dados['genero'] == 'mulher') & (dados['encargos'] > 30000)]
df_mulheres['encargos'].count()

sb.boxplot(x="filhos",y="encargos", data=dados, palette="hls")

plt.figure(figsize=(15, 6))
sb.boxplot(x="idade",y="encargos", data=dados, palette="hls")
plt.show()

# Abaixo seguem os códigos para geração dos diagramas de dispersão que encontram-se no relatório entregue

dados.plot.scatter(x = 'fumante', y = 'encargos')

plt.figure(figsize=(8, 6))
plt.scatter(dados['idade'],dados['encargos'], alpha=0.5)
plt.xlabel('Idade')
plt.ylabel('Encargos')
plt.title('Diagrama de dispersão')
plt.show()

# Remoção dos Outliers
print("\n Remoção dos Outliers\n")

linhas_com_outliers = dados.genero.count()
index = dados[(dados['genero'] == 'mulher') & (dados['encargos'] > 15000)].index
dados.drop(index, inplace=True)

index = dados[(dados['genero'] == 'homem') & (dados['encargos'] > 15000)].index
dados.drop(index, inplace=True)

plt.figure(figsize=(8, 6))
plt.scatter(dados['idade'],dados['encargos'], alpha=0.5)
plt.xlabel('Idade')
plt.ylabel('Encargos')
plt.title('Diagrama de dispersão')
plt.show()

sb.boxplot(x="genero",y="encargos", data=dados, palette="hls")

plt.figure(figsize=(12, 8))
plt.subplot(3, 3, 5)
sb.countplot(x='genero', data=dados)
plt.title('Distribuição de Gênero')
plt.show()

plt.subplot(3, 3, 6)
sb.countplot(x='fumante', data=dados)
plt.title('Distribuição de Fumantes')
plt.show()

contagem_fumantes = dados.groupby('filhos')['fumante'].value_counts().unstack().fillna(0)
print("Contagem Fumantes por Números de Filhos\n")
print(contagem_fumantes,"\n")

contagem_fumantes = dados.groupby('regiao')['fumante'].value_counts().unstack().fillna(0)
print("Contagem Fumantes por Região\n")
print(contagem_fumantes,"\n")

print("Estatísticas Descritivas dos Dados sem outliers:\n")
print(dados.describe())

print("\nPercentual de Outliers Removidos da Base")
linhas_sem_outliers = dados.genero.count()
percentual_outliers_removidos = (linhas_com_outliers - linhas_sem_outliers) / linhas_com_outliers
print(f"\nTotal de outliers removidos: {percentual_outliers_removidos:.2%}")

# Padronização dos Dados 
# Foi escolhida a padronização LabelEncoder 

label_enconder = LabelEncoder()
dados_antigos = dados.copy(deep=True)

dados["fumante"] = label_enconder.fit_transform(dados["fumante"])
dados["genero"] = label_enconder.fit_transform(dados["genero"])

df = pd.get_dummies(dados["regiao"], prefix = "dummy")
dados = pd.concat([dados, df], axis=1)
dados.drop(["regiao"], axis=1, inplace = True)

# Apresentando os dados categóricos padronizados
print(dados.head())

# Reapresentando os dados categóricos antigos para comparação
print(dados_antigos.head())

# Analisando as Correlações
matrix_correlacao = dados.corr()
plt.figure(figsize=(10, 8))
sb.heatmap(matrix_correlacao, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Matriz de Correlação')
plt.show()

# Teste de normalidade
_, p_value_idade = stats.shapiro(dados['idade'])
_, p_value_encargos = stats.shapiro(dados['encargos'])
print(f"\nTeste de Shapiro-Wilk para normalidade:")
print(f"Idade: p-value = {p_value_idade:.5f}")
print(f"Encargos: p-value = {p_value_encargos:.5f}")

# Teste T para comparação de média de fumantes e não- fumantes

fumantes = dados[dados['fumante'] == 1]['encargos']
nao_fumantes = dados[dados['fumante'] == 0]['encargos']
t_stat, p_value_t = stats.ttest_ind(fumantes, nao_fumantes)
print(f"\nTeste t para comparação de médias entre fumantes e não fumantes:")
print(f"t-statistic = {t_stat:.5f}, p-value = {p_value_t:.5f}")


# Iniciando a Validação Cruzada

print("\nIniciando a Validação Cruzada\n")
print("Definindo a target\n")
x = dados.drop(columns=['encargos'])
y = dados["encargos"]
print(x.head())
print("\n")
print(y.head())
print("\n")

# Aplicando a Validação Cruzada K-Fold

forest_model = RandomForestRegressor(n_estimators=10, random_state=42)
kfold = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(forest_model, x, y, cv=5)
mean_mse = scores.mean()
print(f"\nK-Fold (R^2) Scores: {scores}")
print(f"Média do Erro Médio Quadrático (MSE) utilizando Cross-Validation: {mean_mse:.5f}")
print(f"Raiz do Erro Médio Quadrático (RMSE) utilizando Cross-Validation: {np.sqrt(mean_mse):.5f}")

# Divisão entre Treino e Teste

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Padronizando Dados de Treino e Teste

scaler_std = StandardScaler()
scaler_std.fit(x_train)
x_dados_std_train = scaler_std.fit_transform(x_train)
x_dados_std_test = scaler_std.transform(x_test)
print("\n",x_dados_std_train,"\n")
print(x_dados_std_test,"\n")

# Verificando frequência após padronização

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sb.histplot(x_train['imc'], kde=True, color='blue', label='Antes')
plt.title('Distribuição de IMC - Antes')
plt.legend()


plt.subplot(2, 2, 2)
sb.histplot(x_dados_std_train[:, 2], kde=True, color='orange', label='Depois')
plt.title('Distribuição de IMC - Depois')
plt.legend()
plt.show()

# Iniciando o Modelo de Random Forest

print("\nIniciando o modelo de Random Forest")
forest_model.fit(x_dados_std_train, y_train) # Treinando o modelo
y_pred_forest = forest_model.predict(x_dados_std_test) #Fazendo previsões nos dados de teste

# Avaliando o desempenho do modelo
mse_forest = mean_squared_error(y_test, y_pred_forest)
rmse_forest = np.sqrt(mse_forest)
r2_forest = r2_score(y_test, y_pred_forest)
print(f"\nErro médio quadrático (MSE) - Floresta Aleatória: {mse_forest:.5f}")
print(f"Raiz do erro médio quadrático (RMSE) - Floresta Aleatória: {rmse_forest:.5f}")
print(f"Coeficiente de determinação (R^2) - Floresta Aleatória: {r2_forest:.5f}\n")

def calc_mape(targets, predictions):
  errors = np.abs(targets - predictions)
  relative_errors = errors / np.abs(targets)
  mape = np.mean(relative_errors) * 100
  return mape
print(f"MAPE: {calc_mape(y_test, y_pred_forest): .2f}%")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_forest, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Linha diagonal para referência
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title('Previsões vs. Valores Reais - Floresta Aleatória')
plt.grid(True)
plt.show()

# Iniciando o Modelo de Regressão Linear

print("\nIniciando Modelo de Regressão Linear")
model = LinearRegression()
model.fit(x_dados_std_train, y_train)
y_pred_linear = model.predict(x_dados_std_test)
mse = mean_squared_error(y_test,y_pred_linear)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred_linear)
print(f"\nErro médio quadrático (MSE): {mse:.5f}")
print(f"Erro médio quadrático (RMSE): {rmse:.5f}")
print(f"Coeficiente de determinação (R^2): {r2:.5f}\n")

print(f"MAPE: {calc_mape(y_test, y_pred_linear): .2f}%")

plt.figure(figsize=(8, 6))
plt.scatter(y_test,y_pred_linear, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Linha diagonal para referência
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title('Previsões vs. Valores Reais - Regressão Linear')
plt.grid(True)
plt.show()