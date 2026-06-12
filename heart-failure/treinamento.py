"""
METAESTIMADOR ESCOLHIDO: K-Means
-------------------------------
Justificativa:
  - K-Means é um algoritmo de clustering não supervisionado que agrupa pacientes
    em K grupos com base na similaridade de seus atributos clínicos.
  - É adequado para este problema porque:
      1. Queremos descobrir grupos NATURAIS de pacientes (perfis clínicos),
         sem rótulos pré-definidos de treino.
      2. O dataset possui variáveis numéricas contínuas (age, creatinine_phosphokinase,
         ejection_fraction, platelets, serum_creatinine, serum_sodium) que se
         beneficiam da métrica euclidiana usada pelo K-Means.
      3. Interpretação clínica: cada centroide representa um "perfil típico" de
         paciente, facilitando a comunicação com profissionais de saúde.
      4. Escalável e eficiente para o tamanho do dataset (299 pacientes).
"""

import math
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

dados = pd.read_csv('heart-failure\heart_failure_clinical_records_dataset.csv', sep=',')

dados_binariaos = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
dados_continuos = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
coluna_excluida = ['DEATH_EVENT']

print(f"\n[INFO] Variáveis BINÁRIAS (mantidas como 0/1, sem escalonamento):")
print(f"       {dados_binariaos}")
print(f"\n[INFO] Variáveis CONTÍNUAS (serão normalizadas com MinMaxScaler):")
print(f"       {dados_continuos}")
print(f"\n[INFO] Variável EXCLUÍDA (desfecho clínico, não é feature):")
print(f"       {coluna_excluida}")

X = dados[dados_continuos + dados_binariaos].copy()
print(f"\n[INFO] Shape do conjunto de features para clustering: {X.shape}")

for col in dados_binariaos:
    valores_unicos = sorted(dados[col].unique())
    assert set(valores_unicos).issubset({0, 1}), f"Coluna {col} possui valores não binários: {valores_unicos}"
    print(f"  ✓ {col}: valores únicos = {valores_unicos} — OK")

print(f"\n[INFO] Estatísticas das variáveis contínuas (antes da normalização):")
print(dados[dados_continuos].describe().round(2))

scaler = MinMaxScaler()
X_continuas_norm = scaler.fit_transform(X[dados_continuos])
X_continuas_norm = pd.DataFrame(X_continuas_norm, columns=dados_continuos)
 
X_binarias = X[dados_binariaos].reset_index(drop=True)
X_normalizado = pd.concat([X_continuas_norm, X_binarias], axis=1)

print(f"\n[INFO] Estatísticas após normalização (contínuas em [0,1], binárias em {{0,1}}):")
print(X_normalizado.describe().round(3))

pickle.dump(scaler, open('heart-failure/normalizador_heart_failure.pkl', 'wb'))
pickle.dump(dados_continuos, open('heart-failure/dados_continuos_heart_failure.pkl', 'wb'))
pickle.dump(dados_binariaos, open('heart-failure/dados_binariaos_heart_failure.pkl', 'wb'))

K_MAX = 50
K = range(1, K_MAX + 1)
distorcoes = []

for i in K:
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(X_normalizado)
    distorcoes.append(
        sum(
            np.min(
                cdist(X_normalizado.values, km.cluster_centers_, 'euclidean'),
                axis=1
            ) / X_normalizado.shape[0]
        )
    )
 
x0, y0 = K[0], distorcoes[0]
xn, yn = K[-1], distorcoes[-1]
distancias = []
 
for i in range(len(distorcoes)):
    x, y = K[i], distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
    distancias.append(numerador / denominador)
 
numero_clusters_otimo = K[distancias.index(max(distancias))]

cluster_heart_failure = KMeans(n_clusters=numero_clusters_otimo, random_state=42, n_init=10)
cluster_heart_failure.fit(X_normalizado)

dados['cluster'] = cluster_heart_failure.labels_
 
pickle.dump(cluster_heart_failure, open('heart-failure/cluster_heart_failure.pkl', 'wb'))