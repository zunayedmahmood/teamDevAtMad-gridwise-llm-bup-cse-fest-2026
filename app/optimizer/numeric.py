import math

EPS = 1e-9
VALIDATION_TOLERANCE = 1e-7
REPLAY_TOLERANCE = VALIDATION_TOLERANCE


def clean_number(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        return number
    if abs(number) < EPS:
        return 0.0
    return number
