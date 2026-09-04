import requests


BASE_URL = "http://127.0.0.1:8000"


def compute(query):
    response = requests.post(
        f"{BASE_URL}/api/compute",
        json={"query": query},
        timeout=60,
    )

    print("HTTP:", response.status_code)
    print(response.json())


compute("2 + 2")

compute("Solve[x^2 - 5*x + 6 == 0, x]")

compute("Integrate[x^2, x]")

compute("D[x^3 + 2*x^2 + x, x]")