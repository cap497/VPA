# safety_policies.py
def derive_risk_level(speed_kmh: float, classes: list[float]) -> str:
    """
    classes: [central, fedavg, fedprox] -> p.ex. 0=calmo, 1=normal, 2=agressivo
    """
    agg = max(classes) if classes else 0.0
    if speed_kmh < 10:
        return "stopped"
    if speed_kmh < 60 and agg < 1.5:
        return "low"
    if speed_kmh < 100:
        return "medium"
    return "high"

def apply_safety_policies(answer: str, style: str, sensors_state: dict) -> str:
    # extrai velocidade aproximada
    graph = sensors_state.get("graph") or []
    speed = 0.0
    if graph:
        last = graph[-1]["data"]
        speed = float(last.get("speed", 0.0))

    risk = derive_risk_level(speed, sensors_state.get("classes") or [])
    if risk in ("high", "medium"):
        # política de resposta curta
        return (
            "Por segurança, estou fornecendo uma resposta resumida, "
            "pois o veículo está em movimento.\n\n"
            + answer[:400]
        )
    return answer
