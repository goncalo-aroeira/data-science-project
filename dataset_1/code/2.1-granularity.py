import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from dslabs_functions import plot_bar_chart

# Supondo que 'data' seja o DataFrame carregado do seu arquivo CSV
# Certifique-se de substituir o caminho do arquivo e o nome do arquivo pelo seu caso
data = pd.read_csv("../../class_pos_covid.csv")

##################################################################################################################
################################################ LastCheckupTime ######################################################
####################################################################################################################

file_tag = "CovidPos_LastCheckupTime"

column_to_plot = 'LastCheckupTime'

# Calcular contagem de valores
value_counts = data[column_to_plot].value_counts()

# Chamar a função de plotagem de gráfico de barras
plt.figure(figsize=(10, 6))
plot_bar_chart(value_counts.index, value_counts.values, title='Granularity Study for variable LastCheckupTime', xlabel='Value', ylabel='Number of records')
plt.xticks(rotation=45, ha='right')  # Ajuste de rotação
plt.savefig(f"../images/{file_tag}_study_for_granularity.png", bbox_inches='tight')

# Mostrar o gráfico
plt.show()
'''
'''
##################################################################################################################
################################################ SmokerStatus ######################################################
####################################################################################################################

file_tag = "CovidPos_SmokerStatus"

column_to_plot = 'SmokerStatus'

# Calcular contagem de valores
value_counts = data[column_to_plot].value_counts()

# Chamar a função de plotagem de gráfico de barras
plt.figure(figsize=(10, 6))
plot_bar_chart(value_counts.index, value_counts.values, title='Granularity Study for variable SmokerStatus', xlabel='Value', ylabel='Number of records')

plt.savefig(f"../images/{file_tag}_study_for_granularity.png")

# Mostrar o gráfico
plt.show()


##################################################################################################################
################################################ ECigaretteUsage ######################################################
####################################################################################################################

file_tag = "CovidPos_ECigaretteUsage"

column_to_plot = 'ECigaretteUsage'

# Calcular contagem de valores
value_counts = data[column_to_plot].value_counts()

# Chamar a função de plotagem de gráfico de barras
plt.figure(figsize=(10, 6))
plot_bar_chart(value_counts.index, value_counts.values, title='Granularity Study for variable ECigaretteUsage', xlabel='Value', ylabel='Number of records')

plt.savefig(f"../images/{file_tag}_study_for_granularity.png")

# Mostrar o gráfico
plt.show()


##################################################################################################################
################################################ RaceEthnicityCategory ######################################################
####################################################################################################################

file_tag = "CovidPos_RaceEthnicityCategory"

column_to_plot = 'RaceEthnicityCategory'

# Calcular contagem de valores
value_counts = data[column_to_plot].value_counts()

# Chamar a função de plotagem de gráfico de barras
plt.figure(figsize=(10, 6))
plot_bar_chart(value_counts.index, value_counts.values, title='Granularity Study for variable RaceEthnicityCategory', xlabel='Value', ylabel='Number of records')

plt.savefig(f"../images/{file_tag}_study_for_granularity.png")

# Mostrar o gráfico
plt.show()



##################################################################################################################
################################################ AgeCategory ######################################################
####################################################################################################################

file_tag = "CovidPos_AgeCategory"

categories = ['Age 80 or older', 'Age 40 to 44', 'Age 75 to 79', 'Age 70 to 74', 'Age 55 to 59',
              'Age 65 to 69', 'Age 60 to 64', 'Age 50 to 54', 'Age 45 to 49', 'Age 35 to 39',
              'Age 30 to 34', 'Age 25 to 29', 'Age 18 to 24']

# Filtrar e ordenar as categorias específicas
ordered_categories = sorted(data['AgeCategory'].dropna().loc[data['AgeCategory'].isin(categories)].unique(), key=lambda x: int(x.split()[1]))

# Criar a primeira figura (primeiro gráfico)
plt.figure(figsize=(15, 6))

# Adicionar o primeiro subplot (o histograma para categorias específicas de idade)
plt.subplot(1, 2, 1)
data['AgeCategory'].dropna().loc[data['AgeCategory'].isin(ordered_categories)].value_counts().loc[ordered_categories].plot(kind='bar', color='skyblue')
plt.title('5 years interval')
plt.xlabel('Age Group')
plt.ylabel('Number of records')
plt.xticks(rotation=45, ha='right')  # Ajuste de rotação

# Adicionar o segundo subplot (o histograma para intervalos de idade)
plt.subplot(1, 2, 2)
interval_mapping = {
    'Age 18 to 24': 'Age 18 to 29',
    'Age 25 to 29': 'Age 18 to 29',
    'Age 30 to 34': 'Age 30 to 39',
    'Age 35 to 39': 'Age 30 to 39',
    'Age 40 to 44': 'Age 40 to 49',
    'Age 45 to 49': 'Age 40 to 49',
    'Age 50 to 54': 'Age 50 to 59',
    'Age 55 to 59': 'Age 50 to 59',
    'Age 60 to 64': 'Age 60 to 69',
    'Age 65 to 69': 'Age 60 to 69',
    'Age 70 to 74': 'Age 70 to 79',
    'Age 75 to 79': 'Age 70 to 79',
    'Age 80 or older': 'Age 80 or older'
}

# Criar a coluna 'AgeInterval' usando o mapeamento
data['AgeInterval'] = data['AgeCategory'].map(interval_mapping)

# Filtrar e ordenar as categorias específicas
ordered_intervals = ['Age 18 to 29', 'Age 30 to 39', 'Age 40 to 49', 'Age 50 to 59', 'Age 60 to 69', 'Age 70 to 79', 'Age 80 or older']
data['AgeInterval'].dropna().loc[data['AgeInterval'].isin(ordered_intervals)].value_counts().loc[ordered_intervals].plot(kind='bar', color='skyblue')
plt.title('10 years interval')
plt.xlabel('Age Group')
plt.ylabel('Number of records')
plt.xticks(rotation=45, ha='right')  # Ajuste de rotação

# Ajustar o layout para evitar sobreposição
plt.tight_layout()

# Salvar a figura combinada
plt.savefig(f"../images/{file_tag}_study_for_granularity.png")

# Mostrar a figura
plt.show()




##################################################################################################################
################################################ TETANUSLAST10TDAP ################################################
####################################################################################################################

file_tag = "CovidPos_TetanusLast10Tdap"

# Criar uma figura com dois subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot para os quatro valores distintos
counts_all_values = data['TetanusLast10Tdap'].value_counts()
bars = axes[1].bar(counts_all_values.index, counts_all_values, color='skyblue')
axes[1].set_title('Histograma para os 4 Valores')
axes[1].set_xlabel('Tetanus Last 10 Tdap')
axes[1].set_ylabel('Number of records')
axes[1].tick_params(axis='x', rotation=90)  # Ajuste de rotação

# Dividir o texto em várias linhas usando textwrap
labels = [textwrap.fill(label.get_text(), width=10) for label in axes[1].get_xticklabels()]

# Configurar explicitamente as posições das marcas no eixo x
axes[1].set_xticks(range(len(labels)))
axes[1].set_xticklabels(labels)

# Mapear os valores 'No, did not receive any tetanus shot in the past 10 years' para 'No' e os outros para 'Yes'
data['TetanusLast10Tdap'] = data['TetanusLast10Tdap'].map({'No, did not receive any tetanus shot in the past 10 years': 'No', 'Yes, received Tdap': 'Yes', 'Yes, received tetanus shot but not sure what type': 'Yes', 'Yes, received tetanus shot, but not Tdap': 'Yes'})


# Subplot para 'Yes' e 'No'
counts_yes_no = data['TetanusLast10Tdap'].value_counts()
axes[0].bar(counts_yes_no.index, counts_yes_no, color='skyblue')
axes[0].set_title('Histograma para "Yes" e "No"')
axes[0].set_xlabel('Tetanus Last 10 Tdap')
axes[0].set_ylabel('Number of records')
axes[0].tick_params(axis='x', rotation=0)  # Ajuste de rotação


# Ajustar layout para evitar sobreposição
plt.tight_layout()

# Salvar a figura
plt.savefig(f"../images/{file_tag}_study_for_granularity.png")

# Exibir a figura
plt.show()


'''
import pandas as pd

# Carregar o DataFrame a partir do arquivo CSV
file_path = "../../class_pos_covid.csv"
data = pd.read_csv(file_path)

# Extrair os valores únicos da coluna 'TetanusLast10Tdap'
informacoes_restantes = data['TetanusLast10Tdap'].unique()

# Exibir as informações únicas restantes
print(informacoes_restantes)
'''