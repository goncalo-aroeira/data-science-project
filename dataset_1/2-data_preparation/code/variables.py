import pandas as pd
from dslabs_functions import plot_bar_chart
import matplotlib.pyplot as plt

# Carregar o DataFrame a partir do arquivo CSV
file_path = "../../../class_pos_covid.csv"
data = pd.read_csv(file_path)


column_to_plot = 'State'

# Contar o número de registros por região
contagem_por_regiao = data[column_to_plot].value_counts()

file_tag = "CovidPos_State_per_region"

mapeamento_regioes = {
    'Alabama': 'Sul',
    'Alaska': 'Oeste',
    'Arizona': 'Oeste',
    'Arkansas': 'Sul',
    'Califórnia': 'Oeste',
    'Colorado': 'Oeste',
    'Connecticut': 'Nordeste',
    'Delaware': 'Nordeste',
    'Flórida': 'Sul',
    'Geórgia': 'Sul',
    'Havaí': 'Oeste',
    'Idaho': 'Oeste',
    'Illinois': 'Meio-Oeste',
    'Indiana': 'Meio-Oeste',
    'Iowa': 'Meio-Oeste',
    'Kansas': 'Meio-Oeste',
    'Kentucky': 'Sul',
    'Louisiana': 'Sul',
    'Maine': 'Nordeste',
    'Maryland': 'Nordeste',
    'Massachusetts': 'Nordeste',
    'Michigan': 'Meio-Oeste',
    'Minnesota': 'Meio-Oeste',
    'Mississippi': 'Sul',
    'Missouri': 'Meio-Oeste',
    'Montana': 'Oeste',
    'Nebraska': 'Meio-Oeste',
    'Nevada': 'Oeste',
    'New Hampshire': 'Nordeste',
    'New Jersey': 'Nordeste',
    'Novo México': 'Oeste',
    'Nova York': 'Nordeste',
    'Carolina do Norte': 'Sul',
    'Dakota do Norte': 'Meio-Oeste',
    'Ohio': 'Meio-Oeste',
    'Oklahoma': 'Sul',
    'Oregon': 'Oeste',
    'Pensilvânia': 'Nordeste',
    'Rhode Island': 'Nordeste',
    'Carolina do Sul': 'Sul',
    'Dakota do Sul': 'Meio-Oeste',
    'Tennessee': 'Sul',
    'Texas': 'Sul',
    'Utah': 'Oeste',
    'Vermont': 'Nordeste',
    'Virgínia': 'Sul',
    'Washington': 'Oeste',
    'Virgínia Ocidental': 'Sul',
    'Wisconsin': 'Meio-Oeste',
    'Wyoming': 'Oeste'
}

# Adicionar uma nova coluna 'regiao' ao DataFrame com base no mapeamento
data['regiao'] = data[column_to_plot].map(mapeamento_regioes)

# Contar o número de registros por região
contagem_por_regiao = data['regiao'].value_counts()

# Criar o histograma
contagem_por_regiao.plot(kind='bar', color='green')
plt.xlabel('Regiões Geográficas')
plt.ylabel('Número de Registros')
plt.title('Número de Registros por Região')
plt.savefig(f"{file_tag}_study_for_granularity.png", bbox_inches='tight')
plt.show()

file_tag = "CovidPos_State"
# Calcular contagem de valores
value_counts = data[column_to_plot].value_counts()

# Chamar a função de plotagem de gráfico de barras
plt.figure(figsize=(20, 6))
plot_bar_chart(value_counts.index, value_counts.values, title='Granularity Study for variable LastCheckupTime', xlabel='Value', ylabel='Number of records')
plt.xticks(rotation=45, ha='right')  # Ajuste de rotação
plt.savefig(f"{file_tag}_study_for_granularity.png", bbox_inches='tight')

# Mostrar o gráfico
plt.show()

# Extrair os valores únicos da coluna 'TetanusLast10Tdap'
#informacoes_restantes = data['Sex'].unique()

# Exibir as informações únicas restantes
#print(informacoes_restantes)