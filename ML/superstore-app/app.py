"""Superstore App — portail Flask pour les 6 DSO du notebook ML."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA_PATH = ROOT / "Global_Superstore2.csv"
SAVED = BASE / "saved_model"

ARTIFACTS = {
    "dso1_kmeans": ROOT / "dso1_kmeans.pkl",
    "dso1_scaler": ROOT / "dso1_scaler_cluster.pkl",
    "dso2_rf": ROOT / "dso2_random_forest.pkl",
    "dso3_xgb": ROOT / "dso3_xgboost.pkl",
    "dso6_iso": ROOT / "dso6_isolation_forest.pkl",
    "dso6_nb": ROOT / "dso6_naive_bayes.pkl",
    "bundle": SAVED / "kmeans_bundle.joblib",
    "profiles": SAVED / "cluster_profiles.joblib",
}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "superstore-app-dev-key-change-in-production")


METRICS = {
    "dso1": "Silhouette K-Means = 0.2770 | DB = 1.3209",
    "dso2": "Random Forest : R2 = 0.7586 | RMSE = 37.6652",
    "dso3": "XGBoost : R2 = 0.8020 | RMSE = 34.1136",
    "dso4": "SVD : RMSE = 5.32 | Precision@5 = 0.0011",
    "dso5": "SVM : Accuracy = 0.9056 | KNN = 0.8157",
    "dso6": "Naive Bayes : AUC macro = 0.6246 | anomalies = 2.00%",
}

DSO_CONFIG = {
    "dso1": {
        "title": "DSO1 — Segmentation clients",
        "owner": "Syrine Jedidi",
        "model": "K-Means (k=4) + DBSCAN",
        "goal": "Identifier le segment comportemental d'un client à partir de ses indicateurs d'achat.",
        "icon": "bi-diagram-3-fill",
        "mode": "cluster",
    },
    "dso2": {
        "title": "DSO2 — Prédiction des ventes",
        "owner": "Alaa Tahri",
        "model": "Random Forest Regressor",
        "goal": "Estimer les ventes d'une commande avec le modèle Random Forest exporté du notebook.",
        "icon": "bi-graph-up",
        "mode": "sales_rf",
    },
    "dso3": {
        "title": "DSO3 — Optimisation des ventes",
        "owner": "Aziz Msekni",
        "model": "XGBoost Regressor",
        "goal": "Estimer les ventes avec le meilleur modèle de régression du benchmarking.",
        "icon": "bi-lightning-charge-fill",
        "mode": "sales_xgb",
    },
    "dso4": {
        "title": "DSO4 — Recommandation produits",
        "owner": "Nourane Lammouchi",
        "model": "SVD — matrice client × produit",
        "goal": "Proposer des produits pertinents par catégorie à partir de la logique de recommandation du notebook.",
        "icon": "bi-stars",
        "mode": "recommend",
    },
    "dso5": {
        "title": "DSO5 — Classification des commandes",
        "owner": "Wala Rezgui",
        "model": "SVM + KNN",
        "goal": "Classifier une commande comme faible ou forte selon un seuil de ventes.",
        "icon": "bi-ui-checks-grid",
        "mode": "classify",
    },
    "dso6": {
        "title": "DSO6 — Anomalies & régression",
        "owner": "Ibtihel Mechergui",
        "model": "Isolation Forest + GaussianNB",
        "goal": "Détecter le niveau d'anomalie d'une commande avec le modèle Naive Bayes exporté.",
        "icon": "bi-exclamation-triangle-fill",
        "mode": "anomaly",
    },
}

CLUSTER_LABELS = {
    0: "Clients réguliers rentables",
    1: "Clients faibles et déficitaires",
    2: "Clients faibles à livraison rapide",
    3: "Gros clients à marge négative",
}

FIELD_LABELS = {
    "Total_Sales": "CA total (Sales)",
    "Avg_Order_Value": "Panier moyen",
    "Order_Count": "Nombre de commandes",
    "Total_Profit": "Profit total",
    "Avg_Profit_Margin": "Marge bénéficiaire moyenne",
    "Avg_Discount_Rate": "Remise moyenne (%)",
    "Avg_Delivery_Days": "Délai moyen (jours)",
}

BASE_ORDER_FEATURES = {
    "City": 0,
    "State": 0,
    "Country": 0,
    "Market": 0,
    "Product Name": 0,
    "Sales": 120.0,
    "Quantity": 4.0,
    "Discount": 0.1,
    "Shipping Cost": 18.0,
    "Order Priority": 1,
    "Delivery_Days": 4.0,
    "Profit_Margin": 0.12,
    "Discount_Rate": 10.0,
    "Order_Year": 2014,
    "Order_Month": 6,
    "Order_Day": 15,
    "Order_Week": 24,
    "Category_Office Supplies": 1,
    "Category_Technology": 0,
    "Sub-Category_Appliances": 0,
    "Sub-Category_Art": 0,
}

FORM_FIELDS = [
    ("Sales", "Ventes actuelles", "120"),
    ("Quantity", "Quantité", "4"),
    ("Discount", "Remise (0-1)", "0.1"),
    ("Shipping Cost", "Coût livraison", "18"),
    ("Delivery_Days", "Délai livraison", "4"),
    ("Profit_Margin", "Marge profit", "0.12"),
    ("Discount_Rate", "Taux remise (%)", "10"),
]

_models: dict[str, object] = {}
_profiles = None
_products_cache: pd.DataFrame | None = None


def load_optional(path: Path):
    return joblib.load(path) if path.is_file() else None


def load_artifacts() -> None:
    global _models, _profiles
    _models = {key: load_optional(path) for key, path in ARTIFACTS.items()}
    _profiles = _models.get("profiles")


def model_ready(key: str) -> bool:
    return _models.get(key) is not None


def feature_names(model_key: str) -> list[str]:
    model = _models.get(model_key)
    names = getattr(model, "feature_names_in_", None)
    return [str(x) for x in names] if names is not None else []


def numeric_input(name: str, default: float = 0.0) -> float:
    raw = (request.form.get(name, "") or "").strip().replace(",", ".")
    if raw == "":
        return default
    return float(raw)


def build_model_row(model_key: str, include_sales: bool = False) -> pd.DataFrame:
    names = feature_names(model_key)
    values = dict(BASE_ORDER_FEATURES)
    for name, _, default in FORM_FIELDS:
        if name == "Sales" and not include_sales:
            continue
        values[name] = numeric_input(name, float(default))

    category = request.form.get("Category", "Office Supplies")
    values["Category_Office Supplies"] = 1 if category == "Office Supplies" else 0
    values["Category_Technology"] = 1 if category == "Technology" else 0

    subcat = request.form.get("Sub-Category", "Art")
    for key in list(values):
        if key.startswith("Sub-Category_"):
            values[key] = 1 if key == f"Sub-Category_{subcat}" else 0

    if not names:
        names = list(values)
    return pd.DataFrame([{name: values.get(name, 0) for name in names}])


def product_recommendations(category: str, top_n: int = 5) -> list[dict[str, object]]:
    global _products_cache
    if _products_cache is None:
        if not DATA_PATH.is_file():
            return []
        cols = ["Product Name", "Category", "Sales", "Quantity", "Profit"]
        df = pd.read_csv(DATA_PATH, encoding="latin1", usecols=cols)
        _products_cache = (
            df.groupby(["Category", "Product Name"], as_index=False)
            .agg(Sales=("Sales", "sum"), Quantity=("Quantity", "sum"), Profit=("Profit", "sum"))
            .sort_values(["Category", "Sales"], ascending=[True, False])
        )
    recs = _products_cache[_products_cache["Category"] == category].head(top_n)
    return recs.to_dict("records")


load_artifacts()


@app.context_processor
def inject_globals():
    readiness = {
        "dso1": model_ready("dso1_kmeans") and model_ready("dso1_scaler"),
        "dso2": model_ready("dso2_rf"),
        "dso3": model_ready("dso3_xgb"),
        "dso4": DATA_PATH.is_file(),
        "dso5": True,
        "dso6": model_ready("dso6_nb"),
    }
    return {
        "dso_config": DSO_CONFIG,
        "metrics": METRICS,
        "readiness": readiness,
    }


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dso/<dso_id>", methods=["GET", "POST"])
def dso_detail(dso_id: str):
    if dso_id not in DSO_CONFIG:
        flash("DSO introuvable.", "warning")
        return redirect(url_for("home"))

    config = DSO_CONFIG[dso_id]
    result = None

    try:
        if request.method == "POST":
            mode = config["mode"]
            if mode == "cluster":
                scaler = _models.get("dso1_scaler")
                kmeans = _models.get("dso1_kmeans")
                if scaler is None or kmeans is None:
                    raise RuntimeError("Modèle K-Means indisponible.")
                names = [str(x) for x in scaler.feature_names_in_]
                row = pd.DataFrame([{f: numeric_input(f, 0.0) for f in names}])
                cluster = int(kmeans.predict(scaler.transform(row))[0])
                result = {
                    "title": f"Cluster prédit : {cluster}",
                    "value": CLUSTER_LABELS.get(cluster, "Segment client"),
                    "details": "Modèle K-Means entraîné sur les indicateurs agrégés client.",
                }

            elif mode == "sales_rf":
                model = _models.get("dso2_rf")
                if model is None:
                    raise RuntimeError("Random Forest indisponible.")
                pred = float(model.predict(build_model_row("dso2_rf"))[0])
                result = {"title": "Ventes prédites", "value": f"{pred:.2f}", "details": METRICS["dso2"]}

            elif mode == "sales_xgb":
                model = _models.get("dso3_xgb")
                if model is None:
                    raise RuntimeError("XGBoost indisponible.")
                pred = float(model.predict(build_model_row("dso3_xgb"))[0])
                result = {"title": "Ventes prédites", "value": f"{pred:.2f}", "details": METRICS["dso3"]}

            elif mode == "recommend":
                category = request.form.get("Category", "Technology")
                recs = product_recommendations(category)
                result = {
                    "title": f"Top recommandations — {category}",
                    "value": f"{len(recs)} produits",
                    "details": "Classement par ventes cumulées, aligné avec la logique client-produit du notebook.",
                    "recommendations": recs,
                }

            elif mode == "classify":
                sales = numeric_input("Sales", 120.0)
                label = "Commande forte" if sales >= 85.05 else "Commande faible"
                result = {
                    "title": "Classe prédite",
                    "value": label,
                    "details": "Seuil basé sur la médiane des ventes utilisée pour la classification du notebook.",
                }

            elif mode == "anomaly":
                model = _models.get("dso6_nb")
                if model is None:
                    raise RuntimeError("Naive Bayes indisponible.")
                row = build_model_row("dso6_nb", include_sales=True)
                pred = str(model.predict(row)[0])
                proba = getattr(model, "predict_proba", lambda x: [[]])(row)[0]
                confidence = float(np.max(proba)) if len(proba) else 0.0
                result = {
                    "title": "Niveau d'anomalie prédit",
                    "value": pred,
                    "details": f"Confiance maximale : {confidence:.2%}. {METRICS['dso6']}",
                }
    except Exception as exc:
        flash(f"Impossible de calculer la prédiction : {exc}", "danger")

    return render_template(
        "dso.html",
        dso_id=dso_id,
        config=config,
        result=result,
        cluster_fields=FIELD_LABELS,
        order_fields=FORM_FIELDS,
    )


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", profiles=_profiles or {})


@app.route("/predict")
def predict():
    return redirect(url_for("dso_detail", dso_id="dso1"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1", host="0.0.0.0", port=port)
