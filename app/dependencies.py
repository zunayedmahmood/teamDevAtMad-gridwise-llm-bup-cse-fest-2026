from functools import lru_cache

from app.interpreter.client import DirectiveInterpreter


@lru_cache(maxsize=1)
def get_interpreter() -> DirectiveInterpreter:
    return DirectiveInterpreter()
