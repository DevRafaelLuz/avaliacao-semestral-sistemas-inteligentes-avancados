from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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

    X = dados.drop(columns=['quality'])
    y = dados['quality']

    resampler = SMOTE(random_state=42, k_neighbors=3)
    X_bal, y_bal = resampler.fit_resample(X, y)
    print(f'[Random Forest] Frequência das classes após SMOTE: {Counter(y_bal)}')

    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.3, random_state=42)

    rf_grid = {
        'n_estimators': [int(x) for x in np.linspace(10, 100, 10)],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 10],
        'max_depth': [int(x) for x in np.linspace(10, 100, 20)],
        'max_features': ['sqrt', 'log2'],
    }
    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=rf_grid,
        n_iter=10, cv=3, verbose=0, n_jobs=-1, random_state=42,
    )
    rf_search.fit(X_train, y_train)

    rf = RandomForestClassifier(**rf_search.best_params_, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    score_cross = cross_validate(rf, X_bal, y_bal, scoring=scoring, cv=10, n_jobs=-1)

    return {
        'modelo': 'Random Forest',
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
        'melhores_params': rf_search.best_params_,
        'labels': sorted(y_bal.unique()),
    }