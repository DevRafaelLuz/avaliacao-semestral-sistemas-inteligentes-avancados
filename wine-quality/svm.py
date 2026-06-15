import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def categorizar(quality):
    if quality <= 5:
        return 'baixa'
    elif quality == 6:
        return 'media'
    else:
        return 'alta'

def avaliar_modelo(caminho_csv: str = 'wine-quality/winequality.csv') -> dict:
    dados = pd.read_csv(caminho_csv, sep=';')

    dados['quality'] = dados['quality'].apply(categorizar)

    X = dados.drop('quality', axis=1)
    y = dados['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo_svm = SVC(kernel='rbf', C=1.0)
    modelo_svm.fit(X_train_scaled, y_train)
    y_pred = modelo_svm.predict(X_test_scaled)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, random_state=42))
    ])
    scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    score_cross = cross_validate(pipe, X, y, scoring=scoring, cv=10, n_jobs=-1)

    return {
        'modelo': 'SVM',
        'acuracia': accuracy_score(y_test, y_pred),
        'precisao': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro'),
        'acuracia_cv_media': score_cross['test_accuracy'].mean(),
        'acuracia_cv_std': score_cross['test_accuracy'].std(),
        'f1_cv_media': score_cross['test_f1_macro'].mean(),
        'f1_cv_std': score_cross['test_f1_macro'].std(),
        'matriz_confusao': confusion_matrix(y_test, y_pred).tolist(),
        'relatorio': classification_report(y_test, y_pred),
        'melhores_params': {'kernel': 'rbf', 'C': 1.0},
        'labels': sorted(y.unique()),
    }