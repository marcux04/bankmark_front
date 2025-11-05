# dashboard/app_streamlit.py
import streamlit as st
import pandas as pd
import requests
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

# =========================
# CONFIGURACIÓN Y CONEXIONES
# =========================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")  # Asegúrate de que la API esté activa

client = MongoClient(MONGO_URI)
db = client["bank_marketing_db"]

st.set_page_config(page_title="🏦 Bank Marketing Dashboard", layout="wide")
st.title("🏦 Bank Marketing - Random Forest Dashboard")

st.sidebar.header("Opciones")
tab = st.sidebar.radio("Navegación", ["🔮 Predicción manual", "📥 Subir dataset", "📈 Métricas y gráficos"])

# =========================
# 1️⃣ PREDICCIÓN MANUAL
# =========================
if tab == "🔮 Predicción manual":
    st.header("🔮 Predicción manual mediante API")

    with st.form("predict_form"):
        age = st.number_input("Edad", value=35, min_value=18, max_value=120)
        job = st.selectbox("Trabajo", options=["admin.","technician","services","management","blue-collar",
                                               "retired","student","unemployed","entrepreneur","housemaid","unknown"])
        marital = st.selectbox("Estado civil", options=["married","single","divorced"])
        education = st.selectbox("Educación", options=["primary","secondary","tertiary","unknown"])
        default = st.selectbox("¿Crédito en default?", options=["no","yes"])
        balance = st.number_input("Balance promedio (€)", value=1000)
        housing = st.selectbox("¿Crédito de vivienda?", options=["no","yes"])
        loan = st.selectbox("¿Préstamo personal?", options=["no","yes"])
        contact = st.selectbox("Tipo de contacto", options=["cellular","telephone"])
        day = st.number_input("Día de contacto", min_value=1, max_value=31, value=15)
        month = st.selectbox("Mes", options=["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
        duration = st.number_input("Duración de llamada (seg)", value=300)
        campaign = st.number_input("Número de contactos durante campaña", value=1)
        pdays = st.number_input("Días desde última campaña", value=-1)
        previous = st.number_input("Número de contactos previos", value=0)
        poutcome = st.selectbox("Resultado campaña anterior", options=["unknown","failure","other","success"])
        submitted = st.form_submit_button("🔍 Predecir")

    if submitted:
        payload = {
            "age": int(age), "job": job, "marital": marital, "education": education,
            "default": default, "balance": float(balance), "housing": housing,
            "loan": loan, "contact": contact, "day": int(day), "month": month,
            "duration": int(duration), "campaign": int(campaign), "pdays": int(pdays),
            "previous": int(previous), "poutcome": poutcome
        }

        try:
            resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                st.success(result.get("message", "Predicción realizada"))
                st.metric("Probabilidad de contratación (yes)", f"{result.get('probability_yes', 0)*100:.2f}%")
                
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"❌ Error conectando a la API: {e}")

# =========================
# 2️⃣ SUBIR DATASET
# =========================
elif tab == "📥 Subir dataset":
    st.header("📥 Subir nuevo dataset (CSV)")
    uploaded_file = st.file_uploader("Selecciona tu archivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Detectar separador automáticamente
            sample = pd.read_csv(uploaded_file, sep=None, engine="python", nrows=5)
            sep = sample.attrs.get("delimiter", ",")
            uploaded_file.seek(0)
            df_new = pd.read_csv(uploaded_file, sep=sep, engine="python")

            st.write("Vista previa del dataset cargado:")
            st.dataframe(df_new.head())

            # Botón para actualizar MongoDB
            if st.button("Actualizar colección 'bank_clients'"):
                db["bank_clients"].delete_many({})
                db["bank_clients"].insert_many(df_new.to_dict(orient="records"))
                st.success("✅ Datos cargados correctamente ")

            # Checkbox independiente para reentrenar modelo
            if st.checkbox("🔄 Reentrenar modelo "):
                try:
                    with st.spinner("Entrenando modelo, espera unos segundos..."):
                        r = requests.post(f"{API_BASE}/retrain", timeout=120)
                    if r.status_code == 200:
                        st.success("✅ Reentrenamiento completado correctamente.")
                        st.write(r.json().get("output", ""))
                    else:
                        st.error(f"⚠️ Error durante reentrenamiento: {r.status_code} {r.text}")
                except Exception as e:
                    st.error(f"❌ Error conectando a la API: {e}")

        except Exception as e:
            st.error(f"❌ Error leyendo el archivo CSV: {e}")
# =========================
# 3️⃣ MÉTRICAS Y GRÁFICOS
# =========================
else:
    st.header("📊 Métricas y Visualizaciones")

    # Buscar métricas más recientes
    metrics_doc = db["results"].find_one({}, {"_id": 0}, sort=[("timestamp", -1)])

    st.subheader("📈 Últimas métricas registradas")

    if metrics_doc:
        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)

        with col1:
            st.metric("🎯 Accuracy", f"{metrics_doc.get('accuracy', 0)*100:.2f} %")
        with col2:
            st.metric("⚖️ Precision", f"{metrics_doc.get('precision', 0)*100:.2f} %")
        with col3:
            st.metric("📈 Recall", f"{metrics_doc.get('recall', 0)*100:.2f} %")
        with col4:
            st.metric("🧮 F1 Score", f"{metrics_doc.get('f1_score', 0)*100:.2f} %")
        with col5:
            st.metric("💹 AUC ROC", f"{metrics_doc.get('roc_auc', 0)*100:.2f} %")

        st.markdown("---")
        st.subheader("📊 Matriz de Confusión")

        cm = metrics_doc.get("confusion_matrix")
        if cm:
            import seaborn as sns
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Real")
            st.pyplot(fig)
    else:
        st.warning("⚠️ No se encontraron métricas registradas en MongoDB.")

    # Cargar dataset desde MongoDB o CSV local
    try:
        data = pd.DataFrame(list(db["bank_clients"].find({}, {"_id": 0}).limit(10000)))
        if data.empty:
            csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "bank.csv")
            data = pd.read_csv(csv_path, sep=";")
    except Exception:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "bank.csv")
        data = pd.read_csv(csv_path, sep=";")

    st.subheader("📋 Vista previa del dataset")
    st.dataframe(data.head())

    # =========================
    # Distribuciones
    # =========================
    st.subheader("📊 Distribuciones principales")
    col1, col2 = st.columns(2)

    with col1:
        if "job" in data.columns:
            st.bar_chart(data["job"].value_counts())
    with col2:
        if "marital" in data.columns:
            st.bar_chart(data["marital"].value_counts())

    # =========================
    # Métricas del modelo (ROC, PR, Confusion Matrix)
    # =========================
    try:
        enc_path = os.path.join(os.path.dirname(__file__), "..", "model", "label_encoders.pkl")
        model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

        model_local = joblib.load(model_path)
        encoders = joblib.load(enc_path) if os.path.exists(enc_path) else None

        df_proc = data.copy()
        if "y" in df_proc.columns:
            df_proc["y"] = df_proc["y"].map({"yes": 1, "no": 0})

        if encoders:
            for col, le in encoders.items():
                if col in df_proc.columns:
                    df_proc[col] = df_proc[col].fillna("unknown")
                    df_proc[col] = df_proc[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        if "y" in df_proc.columns:
            X = df_proc.drop(columns=["y"])
            y = df_proc["y"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            y_proba = model_local.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig1, ax1 = plt.subplots()
            ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax1.plot([0, 1], [0, 1], "k--")
            ax1.set_title("Curva ROC")
            ax1.legend()
            st.pyplot(fig1)

            # Precision-Recall
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            fig2, ax2 = plt.subplots()
            ax2.plot(recall, precision)
            ax2.set_title("Curva Precision-Recall")
            st.pyplot(fig2)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig3, ax3 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
            ax3.set_xlabel("Predicción")
            ax3.set_ylabel("Real")
            ax3.set_title("Matriz de Confusión (threshold=0.5)")
            st.pyplot(fig3)
        else:
            st.info("ℹ️ El dataset no contiene la columna 'y', no se pueden generar métricas.")
    except Exception as e:
        st.warning(f"⚠️ No se pudieron generar las gráficas: {e}")
