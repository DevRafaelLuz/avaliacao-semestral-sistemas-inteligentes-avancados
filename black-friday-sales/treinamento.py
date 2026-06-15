import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

dados = pd.read_csv('black-friday-sales/retail_black_friday_sales_100k.csv')

features = [
    'gender', 'city', 'customer_segment', 'discount_pct', 
    'quantity', 'is_weekend', 'is_black_friday'
]

targets = ['product_category', 'payment_method', 'age_group']

X = dados[features]

categorical_features = ['gender', 'city', 'customer_segment']
numerical_features = ['discount_pct', 'quantity', 'is_weekend', 'is_black_friday']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

modelos_treinados = {}

print("=== INICIANDO O TREINAMENTO E AVALIAÇÃO ===\n")

for target in targets:
    print(f"{"="*10} TARGET: {target.upper()} {"="*10}")
    
    y = dados[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline_modelo = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    ])
    
    pipeline_modelo.fit(X_train, y_train)
    modelos_treinados[target] = pipeline_modelo
    
    y_pred = pipeline_modelo.predict(X_test)

    acc_global = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"Acurácia Global: {acc_global:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}\n")
    
    cm = confusion_matrix(y_test, y_pred)
    classes = pipeline_modelo.classes_
    
    print("Métricas por Classe:")
    print(f"{'Classe':<20} | {'Sensibilidade (Recall)':<22} | {'Especificidade':<15} | {'F1-Score':<10}")
    print("-" * 75)
    
    for i, classe_nome in enumerate(classes):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(sum(cm)) - tp - fn - fp
        
        sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
        especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        f1_classe = f1_score(y_test, y_pred, labels=[classe_nome], average='macro')
        
        print(f"{str(classe_nome):<20} | {sensibilidade:<22.4f} | {especificidade:<15.4f} | {f1_classe:.4f}")
    
    print("\n")

def modulo_inferencia(circunstancia_venda):
    df_input = pd.DataFrame([circunstancia_venda])
    
    print("=" * 50)
    print("          MÓDULO DE INFERÊNCIA EM AÇÃO          ")
    print("=" * 50)
    print("Circunstâncias do Evento:")
    for k, v in circunstancia_venda.items():
        print(f"  - {k}: {v}")
    print("-" * 50)
    
    for target in targets:
        modelo = modelos_treinados[target]
        
        predicao = modelo.predict(df_input)[0]
        
        probabilidades = modelo.predict_proba(df_input)[0]
        idx_classe = list(modelo.classes_).index(predicao)
        grau_certeza = probabilidades[idx_classe] * 100
        
        print(f"► Predição para [{target.upper()}]: {predicao}")
        print(f"  Grau de Certeza (Score): {grau_certeza:.2f}%")
        print("-" * 50)

venda_exemplo_1 = {
    'gender': 'Female',
    'city': 'Miami',
    'customer_segment': 'New',
    'discount_pct': 10,
    'quantity': 1,
    'is_weekend': 0,
    'is_black_friday': 0
}

venda_exemplo_2 = {
    'gender': 'Male',
    'city': 'Dallas',
    'customer_segment': 'VIP',
    'discount_pct': 45,
    'quantity': 3,
    'is_weekend': 1,
    'is_black_friday': 1
}

modulo_inferencia(venda_exemplo_1)
modulo_inferencia(venda_exemplo_2)