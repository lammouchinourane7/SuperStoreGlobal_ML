"""
Entraîne K-Means (k=4) sur les agrégats client — aligné avec le notebook projet.
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA_PATH = ROOT / "Global_Superstore2.csv"
OUT_DIR = BASE / "saved_model"
PLOT_DIR = BASE / "static" / "plots"


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="latin-1",
        parse_dates=["Order Date", "Ship Date"],
        dayfirst=True,
    )
    df["Delivery_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Profit_Margin"] = df["Profit"] / df["Sales"].replace(0, np.nan)
    df["Discount_Rate"] = df["Discount"] * 100
    return df


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Customer ID", as_index=False).agg(
        Total_Sales=("Sales", "sum"),
        Avg_Order_Value=("Sales", "mean"),
        Order_Count=("Sales", "count"),
        Total_Profit=("Profit", "sum"),
        Avg_Profit=("Profit", "mean"),
        Avg_Profit_Margin=("Profit_Margin", "mean"),
        Avg_Discount_Rate=("Discount_Rate", "mean"),
        Avg_Delivery_Days=("Delivery_Days", "mean"),
    )
    g = g.rename(columns={"Customer ID": "Customer_ID"})
    return g


CLUSTERING_FEATURES = [
    "Total_Sales",
    "Avg_Order_Value",
    "Order_Count",
    "Total_Profit",
    "Avg_Profit_Margin",
    "Avg_Discount_Rate",
    "Avg_Delivery_Days",
]


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_PATH}\n"
            "Placez Global_Superstore2.csv à la racine du dossier ML (parent de superstore-app/)."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(DATA_PATH)
    customer_features = build_customer_features(df)

    available = [f for f in CLUSTERING_FEATURES if f in customer_features.columns]
    X = customer_features[available].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    customer_features = customer_features.copy()
    customer_features["Cluster"] = labels

    bundle = {
        "kmeans": kmeans,
        "scaler": scaler,
        "feature_names": available,
        "silhouette": float(sil),
        "inertia": float(kmeans.inertia_),
    }
    joblib.dump(bundle, OUT_DIR / "kmeans_bundle.joblib")
    customer_features.to_csv(OUT_DIR / "customers_with_clusters.csv", index=False)

    # Profils par cluster (moyennes sur features brutes)
    profiles = {}
    for c in range(4):
        sub = customer_features[customer_features["Cluster"] == c]
        profiles[str(c)] = {
            "count": int(len(sub)),
            "pct": float(len(sub) / len(customer_features) * 100),
            "means": {f: float(sub[f].mean()) for f in available},
        }
    joblib.dump(profiles, OUT_DIR / "cluster_profiles.joblib")

    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="viridis", alpha=0.65, s=18)
    cxy = pca.transform(kmeans.cluster_centers_)
    plt.scatter(cxy[:, 0], cxy[:, 1], c="red", marker="X", s=200, label="Centroïdes")
    plt.colorbar(sc, label="Cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("K-Means (k=4) — projection ACP (2D)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "kmeans_pca.png", dpi=120)
    plt.close()

    print(f"Modèle entraîné — Silhouette: {sil:.4f}, inertie: {kmeans.inertia_:.2f}")
    print(f"Sauvegardé : {OUT_DIR / 'kmeans_bundle.joblib'}")
    print(f"CSV clients : {OUT_DIR / 'customers_with_clusters.csv'}")
    print(f"Graphique : {PLOT_DIR / 'kmeans_pca.png'}")


if __name__ == "__main__":
    main()
