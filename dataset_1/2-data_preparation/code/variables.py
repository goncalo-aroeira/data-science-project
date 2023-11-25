import pandas as pd

# Carregar o DataFrame a partir do arquivo CSV
file_path = "../../../class_pos_covid.csv"
data = pd.read_csv(file_path)

# Extrair os valores únicos da coluna 'TetanusLast10Tdap'
informacoes_restantes = data['Sex'].unique()

# Exibir as informações únicas restantes
print(informacoes_restantes)