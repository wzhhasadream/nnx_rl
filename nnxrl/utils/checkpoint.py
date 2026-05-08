from collections.abc import Mapping

from flax import nnx, serialization
from .normalization import RMS


def _serialize_value(value):
    if value is None:
        return {"kind": "none", "data": None}
    if isinstance(value, int):
        return {"kind": "int", "data": value}
    if isinstance(value, RMS):
        state = nnx.state(value)
        return {
            "kind": "rms_state",
            "data": nnx.to_pure_dict(state),
            "epsilon": value.epsilon,
        }
    try:
        return {
            "kind": "nnx_state",
            "data": nnx.to_pure_dict(nnx.state(value)),
        }
    except Exception:
        return {"kind": "raw", "data": value}


def _restore_value(template, payload):
    kind = payload["kind"]
    if kind in ("none", "int", "raw"):
        return payload["data"]

    if kind == "rms_state":
        state = nnx.State(payload["data"])
        if template is None:
            return RMS(
                mean=state["mean"],
                var=state["var"],
                count=state["count"],
                epsilon=payload["epsilon"],
            )
        return template.replace(**dict(state))

    if kind == "nnx_state":
        state = nnx.State(payload["data"])
        if template is None:
            return state
        nnx.update(template, state)
        return template

    raise ValueError(f"Unsupported checkpoint payload kind: {kind}")


def save_states(path: str, state_map: Mapping[str, object]) -> None:
    payload = {key: _serialize_value(value) for key, value in state_map.items()}

    with open(path, "wb") as f:
        f.write(serialization.to_bytes(payload))


def load_states(
    path: str,
    template_map: Mapping[str, object]
) -> dict[str, object]:
    template = {key: _serialize_value(value) for key, value in template_map.items()}

    with open(path, "rb") as f:
        state_map = serialization.from_bytes(template, f.read())


    restored = {
        key: _restore_value(template_map[key], state_map.get(key))
        for key in template_map
    }

    return restored

