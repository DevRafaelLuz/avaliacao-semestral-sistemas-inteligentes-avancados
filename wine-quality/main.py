import inferencia
import random_forest
import regressao_logistica
import svm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

CAMINHO_CSV = 'wine-quality/winequality.csv'

METRICAS_EXIBIR = [
    ('acuracia',          'Acurácia (hold-out)'),
    ('precisao',          'Precisão (macro)'),
    ('recall',            'Recall (macro)'),
    ('f1',                'F1-Score (macro)'),
    ('acuracia_cv_media', 'Acurácia CV média (10-fold)'),
    ('acuracia_cv_std',   'Acurácia CV desvio padrão'),
    ('f1_cv_media',       'F1 CV média (10-fold)'),
    ('f1_cv_std',         'F1 CV desvio padrão'),
]

def separador(char='─', largura=65):
    print(char * largura)

def cabecalho(texto: str):
    separador('═')
    print(f'  {texto}')
    separador('═')

def imprimir_tabela(resultados: list):
    nomes = [r['modelo'] for r in resultados]
    col_w = 22

    separador()
    print(f"{'Métrica':<28}" + ''.join(f"{n:>{col_w}}" for n in nomes))
    separador()

    for chave, label in METRICAS_EXIBIR:
        linha = f"{label:<28}"
        for r in resultados:
            val = r.get(chave, 0)
            if 'std' in chave:
                linha += f"{'±' + f'{val:.4f}':>{col_w}}"
            else:
                linha += f"{f'{val:.4f}':>{col_w}}"
        print(linha)

    separador()

def imprimir_matrizes(resultados: list):
    for r in resultados:
        print(f"\n  Matriz de Confusão — {r['modelo']}")
        separador('-', 40)
        mc = r['matriz_confusao']
        labels = r['labels']
        print("           " + "".join(f"{l:>10}" for l in labels))
        for i, linha in enumerate(mc):
            print(f"  {labels[i]:<8} " + "".join(f"{v:>10}" for v in linha))

def imprimir_relatorios(resultados: list):
    for r in resultados:
        cabecalho(f"Relatório de Classificação — {r['modelo']}")
        print(r['relatorio'])

def recomendar(resultados: list) -> dict:
    """
    Pontuação composta: prioriza F1 CV (robustez), penaliza alta variância.
    score = f1_cv_media - 0.5 * f1_cv_std + 0.3 * acuracia_cv_media
    """
    for r in resultados:
        r['_score'] = (
            r['f1_cv_media']
            - 0.5 * r['f1_cv_std']
            + 0.3 * r['acuracia_cv_media']
        )
    return max(resultados, key=lambda r: r['_score'])

def main():
    cabecalho('AVALIANDO MODELOS — por favor aguarde...')

    print('\n[1/3] Treinando Regressão Logística...')
    res_lr = regressao_logistica.avaliar_modelo(CAMINHO_CSV)

    print('\n[2/3] Treinando SVM...')
    res_svm = svm.avaliar_modelo(CAMINHO_CSV)

    print('\n[3/3] Treinando Random Forest (inclui busca de hiperparâmetros)...')
    res_rf = random_forest.avaliar_modelo(CAMINHO_CSV)

    resultados = [res_lr, res_svm, res_rf]

    cabecalho('TABELA COMPARATIVA DE MÉTRICAS')
    imprimir_tabela(resultados)

    cabecalho('MATRIZES DE CONFUSÃO')
    imprimir_matrizes(resultados)

    imprimir_relatorios(resultados)

    melhor = recomendar(resultados)
    cabecalho('RECOMENDAÇÃO PARA PRODUÇÃO')

    ranking = sorted(resultados, key=lambda r: r['_score'], reverse=True)
    for i, r in enumerate(ranking, 1):
        marker = '★ RECOMENDADO' if i == 1 else f'  #{i}'
        print(f"  {marker}  {r['modelo']:<26} score={r['_score']:.4f}")

    separador()
    print(f"""
        Modelo recomendado : {melhor['modelo']}
        F1 CV médio        : {melhor['f1_cv_media']:.4f} ± {melhor['f1_cv_std']:.4f}
        Acurácia CV média  : {melhor['acuracia_cv_media']:.4f}
        Melhores parâmetros: {melhor['melhores_params']}

        Critério de seleção:
            score = F1_cv_média − 0.5×F1_cv_std + 0.3×Acurácia_cv_média
            (prioriza generalização e penaliza instabilidade)
        """)
    separador('═')

    # ── MÓDULO DE INFERÊNCIA ──
    cabecalho('TREINANDO MODELO FINAL PARA PRODUÇÃO')

    construtores = {
        'Regressão Logística': lambda: LogisticRegression(max_iter=1000, random_state=42),
        'SVM': lambda: SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'Random Forest': lambda: RandomForestClassifier(**melhor['melhores_params'], random_state=42),
    }

    estimador_final = construtores[melhor['modelo']]()
    pipe_final = inferencia.treinar_modelo_final(
        CAMINHO_CSV, estimador_final, caminho_saida='wine-quality/modelo_final.pkl'
    )

    inferencia.demonstrar_inferencia(CAMINHO_CSV, pipe_final, n_amostras=3)

if __name__ == '__main__':
    main()