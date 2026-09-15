from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import add_features, add_weekly_target


def load_data(data_dir="data/raw"):
    p = Path(data_dir)
    ventas = pd.read_csv(p / "ventas_historicas.csv", parse_dates=["fecha"])
    inventario = pd.read_csv(p / "inventario_actual.csv")
    catalogo = pd.read_csv(p / "catalogo_productos.csv")
    tiendas = pd.read_csv(p / "maestro_tiendas.csv")
    trends = pd.read_csv(p / "ground_truth_trends.csv")
    return ventas, inventario, catalogo, tiendas, trends


def validate_inputs(ventas, inventario, catalogo, tiendas):
    required = {
        "ventas": {"fecha", "id_tienda", "id_producto", "unidades_vendidas"},
        "inventario": {"id_tienda", "id_producto", "stock_actual"},
        "catalogo": {"id_producto", "costo_unitario", "precio_venta", "costo_almacenamiento_semanal"},
        "tiendas": {"id_tienda", "ciudad", "tamaño_m2"},
    }
    for name, df in [("ventas", ventas), ("inventario", inventario), ("catalogo", catalogo), ("tiendas", tiendas)]:
        missing = required[name] - set(df.columns)
        if missing:
            raise ValueError(f"{name}: faltan columnas {sorted(missing)}")
    if ventas.duplicated(["fecha", "id_tienda", "id_producto"]).any():
        raise ValueError("Ventas contiene duplicados por fecha-tienda-producto")
    return True


def enrich_sales(ventas, catalogo, tiendas):
    out = ventas.merge(tiendas, on="id_tienda", how="left", validate="many_to_one")
    out = out.merge(catalogo, on="id_producto", how="left", validate="many_to_one")
    out["margen_unitario"] = out["precio_venta"] - out["costo_unitario"]
    out["margen_pct"] = out["margen_unitario"] / out["precio_venta"]
    return out


def build_processed_datasets(raw_dir="data/raw", processed_dir="data/processed"):
    """Construye y persiste datasets derivados reproducibles a partir de data/raw."""
    raw = Path(raw_dir)
    processed = Path(processed_dir)
    weekly_dir = processed / "weekly"
    daily_dir = processed / "daily"
    weekly_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)

    ventas, inventario, catalogo, tiendas, trends = load_data(raw)
    validate_inputs(ventas, inventario, catalogo, tiendas)
    enriched = enrich_sales(ventas, catalogo, tiendas)

    daily = add_weekly_target(add_features(enriched))
    weekly = daily.copy()
    weekly = weekly.dropna(subset=["target_week", "lag_28", "rolling_mean_28"]).reset_index(drop=True)

    daily_path = daily_dir / "dataset_daily_features.csv"
    weekly_path = weekly_dir / "dataset_weekly_features.csv"
    daily.to_csv(daily_path, index=False)
    weekly.to_csv(weekly_path, index=False)

    return daily, weekly, daily_path, weekly_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    daily, weekly, daily_path, weekly_path = build_processed_datasets(
        root / "data" / "raw", root / "data" / "processed"
    )
    print(f"Dataset diario generado: {daily.shape[0]:,} filas x {daily.shape[1]} columnas")
    print(f"Dataset semanal generado: {weekly.shape[0]:,} filas x {weekly.shape[1]} columnas")
    print(f"Diario: {daily_path}")
    print(f"Semanal: {weekly_path}")
