import inspect
from functools import wraps
from typing import Callable


def project_operator(f: Callable, lift: Callable, project: Callable, args: str = "x") -> Callable:
    signature = inspect.signature(f)
    fn_params = signature.parameters
    if args not in fn_params:
        raise ValueError(f"{getattr(f, '__name__', f)} has no argument {args!r}")
    forwards_extras = any(p.kind is p.VAR_KEYWORD for p in fn_params.values())

    @wraps(f)
    def fn(*call_args, **call_kwargs):
        if not forwards_extras:
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in fn_params}
        bound = signature.bind(*call_args, **call_kwargs)
        if args not in bound.arguments:
            bound.apply_defaults()
        x = bound.arguments[args]
        bound.arguments[args] = lift(x)
        return project(f(*bound.args, **bound.kwargs), x)

    return fn
