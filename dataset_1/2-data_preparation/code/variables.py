import pandas as pd
from dslabs_functions import plot_bar_chart
import matplotlib.pyplot as plt

# Carregar o DataFrame a partir do arquivo CSV
file_path = "../../../class_pos_covid.csv"
data = pd.read_csv(file_path)

file_tag = "CovidPos_HadDiabetes"

column_to_plot = 'HadDiabetes'

# Calcular contagem de valores
value_counts = data[column_to_plot].value_counts()

# Chamar a função de plotagem de gráfico de barras
plt.figure(figsize=(10, 6))
plot_bar_chart(value_counts.index, value_counts.values, title='Granularity Study for variable LastCheckupTime', xlabel='Value', ylabel='Number of records')
plt.xticks(rotation=45, ha='right')  # Ajuste de rotação
plt.savefig(f"{file_tag}_study_for_granularity.png", bbox_inches='tight')

# Mostrar o gráfico
plt.show()
# Extrair os valores únicos da coluna 'TetanusLast10Tdap'
#informacoes_restantes = data['Sex'].unique()

# Exibir as informações únicas restantes
#print(informacoes_restantes)