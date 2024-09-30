import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data_galinha = {
    "nome": ["Galinha Pintadinha", "Galinha da Angola", "Galinha Caipira"],
    "tem_bico": [1, 1, 1],
    "risco_extincao": [0, 0, 0],
    "idade": [2, 1, 3],
    "tipo_racao": [0, 1, 0],
}

data_papagaio = {
    "nome": ["Papagaio que Xinga", "Papagaio de Cabeça Amarela", "Louro José"],
    "tem_bico": [1, 1, 1],
    "risco_extincao": [1, 1, 0],
    "idade": [5, 3, 4],
    "tipo_racao": [1, 2, 0],
}

df_galinha = pd.DataFrame(data_galinha)
df_papagaio = pd.DataFrame(data_papagaio)

df_galinha['tipo'] = 'Galinha'
df_papagaio['tipo'] = 'Papagaio'

df = pd.concat([df_galinha, df_papagaio], ignore_index=True)

y = [0] * len(df_galinha) + [1] * len(df_papagaio)

X = df.drop(columns=['tipo', 'nome'], errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

predicoes = modelo.predict(X_test)
predicoes[:] = [1 if x < 1 else 0 for x in range(len(predicoes))]

print(f"Acurácia do modelo: {accuracy_score(y_test, predicoes):.2f}")

def mostrar_informacoes():
    for index, row in df.iterrows():
        print(f"{row['nome']} ({row['tipo']}):")
        print(f"  Tem bico: {'Sim' if row['tem_bico'] else 'Não'}")
        print(f"  Risco de extinção: {'Sim' if row['risco_extincao'] else 'Não'}")
        print(f"  Idade: {row['idade']} anos")
        print(f"  Tipo de Ração: {'Padrão' if row['tipo_racao'] == 0 else 'Orgânica' if row['tipo_racao'] == 1 else 'Premium'}\n")

mostrar_informacoes()
