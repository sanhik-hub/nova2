import sys
from pathlib import Path

# Allow importing from backend/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.wolfram_engine import WolframEngine


def main():
    engine = WolframEngine()

    print("=" * 60)
    print("WOLFRAM ENGINE TEST")
    print("=" * 60)

    print("\nExecutable:")
    print(engine.executable)

    print("\n[1] Basic calculation")
    result = engine.calculate("2 + 2")
    print(result)

    print("\n[2] Solve equation")
    result = engine.solve("x^2 - 5*x + 6 == 0")
    print(result)

    print("\n[3] Integration")
    result = engine.integrate("x^2")
    print(result)

    print("\n[4] Differentiation")
    result = engine.differentiate("x^3 + 2*x^2 + x")
    print(result)

    print("\n[5] Simplification")
    result = engine.simplify(
        "(x^2 - 1)/(x - 1)"
    )
    print(result)

    print("\n[6] Factorization")
    result = engine.factor(
        "x^4 - 1"
    )
    print(result)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()