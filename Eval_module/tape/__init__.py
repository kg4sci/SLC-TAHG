import sys

# When the project is executed via the Eval_module package (e.g. python -m Eval_module.tape.models...)
# make sure "tape" is also registered as a top-level package alias so legacy absolute imports
# such as "from tape.models.core import ..." continue to work.
_this_module = sys.modules[__name__]
if __name__ != "tape":
    sys.modules.setdefault("tape", _this_module)

__all__ = []
