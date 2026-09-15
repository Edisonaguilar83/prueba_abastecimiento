"""Ejecución completa y reproducible del Caso A."""
from pathlib import Path
from src.data_preparation import build_processed_datasets
from src.pipeline import run_end_to_end

ROOT = Path(__file__).resolve().parent


def main():
    build_processed_datasets(ROOT / "data" / "raw", ROOT / "data" / "processed")
    metrics, rec, bt = run_end_to_end(ROOT)
    print("\nMétricas de validación temporal:")
    print(metrics.to_string(index=False))
    print(f"\nForecast final: {len(rec):,} recomendaciones SKU-tienda")
    print(f"Backtest de política: {len(bt):,} observaciones OOF")
    print("Outputs actualizados en outputs/")


if __name__ == "__main__":
    main()
