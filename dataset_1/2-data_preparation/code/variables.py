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
    'Alabama': 'South',
    'Alaska': 'West',
    'Arizona': 'West',
    'Arkansas': 'South',
    'Califórnia': 'West',
    'Colorado': 'West',
    'Connecticut': 'NorthEast',
    'Delaware': 'NorthEast',
    'Flórida': 'South',
    'Geórgia': 'South',
    'Havaí': 'West',
    'Idaho': 'West',
    'Illinois': 'MidWest',
    'Indiana': 'MidWest',
    'Iowa': 'MidWest',
    'Kansas': 'MidWest',
    'Kentucky': 'South',
    'Louisiana': 'South',
    'Maine': 'NorthEast',
    'Maryland': 'NorthEast',
    'Massachusetts': 'NorthEast',
    'Michigan': 'MidWest',
    'Minnesota': 'MidWest',
    'Mississippi': 'South',
    'Missouri': 'MidWest',
    'Montana': 'West',
    'Nebraska': 'MidWest',
    'Nevada': 'West',
    'New Hampshire': 'NorthEast',
    'New Jersey': 'NorthEast',
    'Novo México': 'West',
    'Nova York': 'NorthEast',
    'Carolina do Norte': 'South',
    'Dakota do Norte': 'MidWest',
    'Ohio': 'MidWest',
    'Oklahoma': 'South',
    'Oregon': 'West',
    'Pensilvânia': 'NorthEast',
    'Rhode Island': 'NorthEast',
    'Carolina do South': 'South',
    'Dakota do South': 'MidWest',
    'Tennessee': 'South',
    'Texas': 'South',
    'Utah': 'West',
    'Vermont': 'NorthEast',
    'Virgínia': 'South',
    'Washington': 'West',
    'Virgínia Ocidental': 'South',
    'Wisconsin': 'MidWest',
    'Wyoming': 'West'
}

# Adicionar uma nova coluna 'regiao' ao DataFrame com base no mapeamento
data['regiao'] = data[column_to_plot].map(mapeamento_regioes)

# Contar o número de registros por região
contagem_por_regiao = data['regiao'].value_counts()

# Criar o histograma
contagem_por_regiao.plot(kind='bar', color='green')
plt.xlabel('Geographical Regions')
plt.ylabel('Number of Records')
plt.title('Number of Records per Region')
plt.savefig(f"{file_tag}_study_for_granularity.png", bbox_inches='tight')
plt.show()

file_tag = "CovidPos_State"
# Calcular contagem de valores
value_counts = data[column_to_plot].value_counts()

# Chamar a função de plotagem de gráfico de barras
plt.figure(figsize=(20, 6))
plot_bar_chart(value_counts.index, value_counts.values, title='Granularity Study for variable State', xlabel='State', ylabel='Number of records')
plt.xticks(rotation=45, ha='right')  # Ajuste de rotação
plt.savefig(f"{file_tag}_study_for_granularity.png", bbox_inches='tight')

# Mostrar o gráfico
plt.show()

# Extrair os valores únicos da coluna 'TetanusLast10Tdap'
#informacoes_restantes = data['Sex'].unique()

# Exibir as informações únicas restantes
#print(informacoes_restantes)