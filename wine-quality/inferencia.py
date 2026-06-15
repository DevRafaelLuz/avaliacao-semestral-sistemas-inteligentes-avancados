import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def categorizar(quality):
    if quality <= 5:
        return 'baixa'
    elif quality == 6:
        return 'media'
    else:
        return 'alta'

def treinar_modelo_final(caminho_csv: str, modelo, caminho_saida: str = 'modelo_final.pkl'):
    dados = pd.read_csv(caminho_csv, sep=';')
    dados['quality'] = dados['quality'].apply(categorizar)

    X = dados.drop(columns=['quality'])
    y = dados['quality']

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', modelo)
    ])
    pipe.fit(X, y)

    joblib.dump(pipe, caminho_saida)
    print(f"\n[Inferência] Modelo salvo em: {caminho_saida}")
    return pipe

def carregar_modelo(caminho_modelo: str = 'modelo_final.pkl'):
    return joblib.load(caminho_modelo)

def prever_amostra(pipe, amostra: dict):
    df = pd.DataFrame([amostra])
    pred = pipe.predict(df)[0]

    proba = None
    if hasattr(pipe, 'predict_proba'):
        probas = pipe.predict_proba(df)[0]
        classes = pipe.classes_
        proba = dict(zip(classes, probas))

    return pred, proba

def demonstrar_inferencia(caminho_csv: str, pipe, n_amostras: int = 3):
    dados = pd.read_csv(caminho_csv, sep=';')
    dados['quality_real'] = dados['quality'].apply(categorizar)

    amostras = dados.sample(n=n_amostras, random_state=123)

    print("\n" + "═" * 65)
    print("  DEMONSTRAÇÃO DE INFERÊNCIA — amostras reais do dataset")
    print("═" * 65)

    for i, (_, linha) in enumerate(amostras.iterrows(), 1):
        entrada = linha.drop(['quality', 'quality_real']).to_dict()
        real = linha['quality_real']

        pred, proba = prever_amostra(pipe, entrada)

        print(f"\n  Amostra {i}")
        print("  " + "-" * 40)
        for k, v in entrada.items():
            print(f"    {k:<22}: {v}")
        print(f"\n    Classe real      : {real}")
        print(f"    Classe prevista  : {pred}")
        if proba:
            proba_fmt = ", ".join(f"{c}={p:.2%}" for c, p in proba.items())
            print(f"    Probabilidades   : {proba_fmt}")
        status = "✓ CORRETO" if pred == real else "✗ ERRADO"
        print(f"    Resultado        : {status}")

    print("\n" + "═" * 65)