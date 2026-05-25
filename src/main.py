# =====================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# =====================================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================================
# 2. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
# =====================================================

df = pd.read_csv("creditcard.csv")

print(df.head())
print(df.info())
print("\nDistribuição das classes:")
print(df["Class"].value_counts())

# =====================================================
# 3. SEPARAÇÃO DAS FEATURES E RÓTULO
# =====================================================

X = df.drop("Class", axis=1)
y = df["Class"]

# Guardar Amount separadamente (para custo financeiro)
amount = df["Amount"]

# =====================================================
# 4. NORMALIZAÇÃO DOS DADOS
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Converter de volta para DataFrame (necessário para SHAP)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# =====================================================
# 5. DIVISÃO TREINO / TESTE
# =====================================================

X_train, X_test, y_train, y_test, amount_train, amount_test = train_test_split(
    X_scaled,
    y,
    amount,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 6. FUNÇÃO DE AVALIAÇÃO CLÁSSICA
# =====================================================

def avaliar_modelo(modelo, X_train, X_test, y_train, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nResultados para {modelo.__class__.__name__}")
    print(f"Acurácia : {acuracia:.4f}")
    print(f"Precisão : {precisao:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred

# =====================================================
# 7. FUNÇÃO DE CUSTO FINANCEIRO
# =====================================================

CUSTO_FP = 50.0  # custo operacional estimado (R$)

def custo_financeiro(y_test, y_pred, amount_test):
    fn_mask = (y_test == 1) & (y_pred == 0)
    fp_mask = (y_test == 0) & (y_pred == 1)

    custo_fn = amount_test[fn_mask].sum()
    custo_fp = fp_mask.sum() * CUSTO_FP

    custo_total = custo_fn + custo_fp
    return custo_fn, custo_fp, custo_total

# =====================================================
# 8. TREINAMENTO E AVALIAÇÃO DOS MODELOS
# =====================================================

modelos = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=1000),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42,
    )
}

resultados_custo = []

for nome, modelo in modelos.items():
    y_pred = avaliar_modelo(modelo, X_train, X_test, y_train, y_test)

    custo_fn, custo_fp, custo_total = custo_financeiro(
        y_test, y_pred, amount_test
    )

    resultados_custo.append([
        nome,
        custo_fn,
        custo_fp,
        custo_total
    ])

# =====================================================
# 9. TABELA DE CUSTO FINANCEIRO
# =====================================================

df_custo = pd.DataFrame(
    resultados_custo,
    columns=["Modelo", "Custo FN (R$)", "Custo FP (R$)", "Custo Total (R$)"]
)

print("\n=== ANÁLISE DE CUSTO FINANCEIRO ===")
print(df_custo.sort_values("Custo Total (R$)"))

# =====================================================
# 10. XAI – SHAP APLICADO AO XGBOOST
# =====================================================

print("\nGerando explicações SHAP para o XGBoost...")

model_xgb = modelos["XGBoost"]

explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)

# =====================================================
# 10.1 VISUALIZAÇÃO GLOBAL (IMPORTÂNCIA DAS VARIÁVEIS)
# =====================================================

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.show()

# =====================================================
# 10.2 VISUALIZAÇÃO LOCAL (UMA TRANSAÇÃO FRAUDULENTA)
# =====================================================

y_pred_xgb = model_xgb.predict(X_test)
fraud_indices = np.where((y_test == 1) & (y_pred_xgb == 1))[0]

if len(fraud_indices) > 0:
    idx = fraud_indices[0]

    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X_test.iloc[idx],
        matplotlib=True
    )
else:
    print("Nenhuma fraude corretamente detectada para visualização local.")
