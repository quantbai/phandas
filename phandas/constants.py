"""Module-level constants for phandas."""

EPSILON = 1e-10
TOLERANCE_FLOAT = 1e-6

SIGNAL_LONG_SUM = 0.5
SIGNAL_SHORT_SUM = -0.5
SIGNAL_TOLERANCE = 1e-2

MIN_NOTIONAL_USD = 0.01
MIN_TRADE_VALUE = 1.0

MATRIX_COND_THRESHOLD = 1e10

SYMBOL_RENAMES = {
    'POL': {
        'old_symbol': 'MATIC',
        'new_symbol': 'POL',
        'cutoff_date': '2024-09-01',
    }
}

