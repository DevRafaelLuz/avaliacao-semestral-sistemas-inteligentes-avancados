import numpy as np
import pandas as pd
import pickle

normalizador = pickle.load(open('heart-failure/normalizador_heart_failure.pkl', 'rb'))
cluster_model = pickle.load(open('heart-failure/cluster_heart_failure.pkl', 'rb'))
dados_continuos = pickle.load(open('heart-failure/dados_continuos_heart_failure.pkl', 'rb'))
dados_binariaos = pickle.load(open('heart-failure/dados_binariaos_heart_failure.pkl', 'rb'))

novo_paciente = {
    'age': 67,
    'creatinine_phosphokinase': 100,
    'ejection_fraction': 40,
    'platelets': 250000,
    'serum_creatinine': 1.2,
    'serum_sodium': 135, 
    'time': 100, 
 
    'anaemia': 0,
    'diabetes': 0,
    'high_blood_pressure': 0,
    'sex': 1,
    'smoking': 0,
}

paciente_continuas = pd.DataFrame([[novo_paciente[col] for col in dados_continuos]], columns=dados_continuos)
paciente_binarias = pd.DataFrame([[novo_paciente[col] for col in dados_binariaos]], columns=dados_binariaos)

paciente_continuas_norm = normalizador.transform(paciente_continuas)
paciente_continuas_norm = pd.DataFrame(paciente_continuas_norm, columns=dados_continuos)

paciente_final = pd.concat([paciente_continuas_norm, paciente_binarias], axis=1)
 
ordem_features = dados_continuos + dados_binariaos
paciente_final = paciente_final[ordem_features]

print(f"\n{'─'*60}")
print("INFERÊNCIA:")
print(f"{'─'*60}")
 
cluster_previsto = cluster_model.predict(paciente_final)[0]
 
centroide = cluster_model.cluster_centers_[cluster_previsto]
distancia_ao_centroide = np.linalg.norm(paciente_final.values[0] - centroide)
 
distancias_todos = []
for i, c in enumerate(cluster_model.cluster_centers_):
    dist = np.linalg.norm(paciente_final.values[0] - c)
    distancias_todos.append((i, dist))
 
distancias_todos.sort(key=lambda x: x[1])
 
print(f"\nO paciente pertence ao CLUSTER {cluster_previsto}")
print(f"\nDistância ao centróide do cluster {cluster_previsto}: {distancia_ao_centroide:.4f}")
 
print(f"\nRanking de similaridade (clusters mais próximos primeiro):")
for rank, (cluster_id, dist) in enumerate(distancias_todos[:5], 1):
    marcador = "◀ CLUSTER DO PACIENTE" if cluster_id == cluster_previsto else ""
    print(f"{rank}º Cluster {cluster_id:2d}  —  distância: {dist:.4f}  {marcador}")
print(f"{'─'*60}\n")