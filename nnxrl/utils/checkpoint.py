from collections.abc import Mapping

from flax import nnx, serialization


def _state_or_none(value):
    if value is None:
        return None
    return nnx.state(value)



def restore_module(module, state) -> None:
    if module is None or state is None:
        return
    nnx.update(module, state)


def save_states(path: str, state_map: Mapping[str, object]) -> None:
    payload = {}
    for key, value in state_map.items():
        if isinstance(value, int):
            payload[key] = value
        else:
            payload[key] = _state_or_none(value)

    with open(path, "wb") as f:
        f.write(serialization.to_bytes(payload))


def load_states(
    path: str,
    template_map: Mapping[str, object]
) -> dict[str, object]:
    template = {}
    for key, value in template_map.items():
        if isinstance(value, int):
            template[key] = value
        else:
            template[key] = _state_or_none(value)

    with open(path, "rb") as f:
        state_map = serialization.from_bytes(template, f.read())


    for key, value in template_map.items():
        if isinstance(value, int):
            continue
        restore_module(value, state_map.get(key))

    return state_map



