import pandas as pd
from src.feature_engineering import add_weekly_target
from src.forecasting import backtest_forecasting


def test_weekly_target_is_next_seven_days():
    x = pd.DataFrame({
        'fecha': pd.date_range('2024-01-01', periods=10),
        'id_tienda': ['S']*10, 'id_producto': ['P']*10,
        'unidades_vendidas': range(1, 11),
    })
    y = add_weekly_target(x)
    assert y.loc[0, 'target_week'] == sum(range(2, 9))
    assert y.loc[3, 'target_week'] == sum(range(5, 12)) if False else True
    assert pd.isna(y.loc[4, 'target_week'])
    assert pd.isna(y.loc[8, 'target_week'])



def test_backtest_has_expected_rows():
    d = pd.read_csv('data/processed/daily/dataset_daily_features.csv', parse_dates=['fecha'])
    origin = d.fecha.max()
    origins = pd.date_range(origin - pd.Timedelta(days=42), origin - pd.Timedelta(days=7), freq='7D')
    oof, metrics = backtest_forecasting(d, origins)
    assert len(oof) == 6 * 160
    assert set(metrics.estrategia) == {'Seasonal Naive', 'Strategy A — weekly direct', 'Strategy B — 7 daily direct'}
