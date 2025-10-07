import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import pickle
import os


# =========================
# PATH CONFIGURATION
# =========================

# ✅ Windows host path
WINDOWS_DATA = r"D:\vsCOde\yashwanth\Airflow\data\Iris - all-numbers.csv"
WINDOWS_WORKDIR = r"D:\vsCOde\yashwanth\Airflow\working_data"

# ✅ Docker container path
DOCKER_DATA = "/opt/airflow/data/Iris - all-numbers.csv"
DOCKER_WORKDIR = "/opt/airflow/working_data"

# Detect environment dynamically
if os.path.exists(DOCKER_DATA):
    DATA_FILE = DOCKER_DATA
    WORK_DIR = DOCKER_WORKDIR
else:
    DATA_FILE = WINDOWS_DATA
    WORK_DIR = WINDOWS_WORKDIR

MODEL_FILE = os.path.join(WORK_DIR, "iris_kmeans_model.pkl")
PLOT_FILE = os.path.join(WORK_DIR, "elbow_plot.png")


# =========================
# TASK FUNCTIONS
# =========================

def load_data():
    """Load Iris dataset and serialize it."""
    print(f"[INFO] Loading dataset from {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found at: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"[INFO] Dataset shape: {df.shape}")
    return pickle.dumps(df)


def data_preprocessing(data):
    """Normalize numeric features using MinMaxScaler."""
    # 🔧 Robust type handling for Airflow XComs
    if isinstance(data, str):
        try:
            # Handle strings like "b'...'"
            data = eval(data) if data.startswith("b'") or data.startswith('b"') else data.encode()
        except Exception:
            data = data.encode()

    df = pickle.loads(data)
    print(f"[INFO] Preprocessing... Shape: {df.shape}")

    df = df.dropna()
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    print(f"[INFO] Scaled data shape: {df_scaled.shape}")
    return pickle.dumps(df_scaled)


def build_save_model(data):
    """Train KMeans model and save it."""
    if isinstance(data, str):
        try:
            data = eval(data) if data.startswith("b'") or data.startswith('b"') else data.encode()
        except Exception:
            data = data.encode()

    df = pickle.loads(data)
    print("[INFO] Training KMeans model...")

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(df)

    os.makedirs(WORK_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"[INFO] Model saved to {MODEL_FILE}")
    return pickle.dumps(kmeans)


def load_model_elbow(data):
    """Generate Elbow curve and determine optimal k."""
    if isinstance(data, str):
        try:
            data = eval(data) if data.startswith("b'") or data.startswith('b"') else data.encode()
        except Exception:
            data = data.encode()

    df = pickle.loads(data)
    sse = []
    k_values = range(1, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(k_values, sse, curve="convex", direction="decreasing")
    optimal_k = kl.elbow
    print(f"[INFO] Optimal number of clusters: {optimal_k}")

    os.makedirs(WORK_DIR, exist_ok=True)
    plt.plot(k_values, sse, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("SSE")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    print(f"[INFO] Elbow plot saved to {PLOT_FILE}")
    plt.close()

    return pickle.dumps({"optimal_k": optimal_k, "sse": sse})


# =========================
# LOCAL TEST ENTRYPOINT
# =========================

if __name__ == "__main__":
    raw = load_data()
    scaled = data_preprocessing(raw)
    build_save_model(scaled)
    load_model_elbow(scaled)
    print("[✅] Local test completed successfully.")
