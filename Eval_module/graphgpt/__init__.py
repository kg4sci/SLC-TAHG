"""GraphGPT package initializer.

When accessed via ``Eval_module.graphgpt`` (e.g. ``python -m Eval_module.graphgpt...``),
register ``graphgpt`` as a top-level alias so absolute imports like
``from graphgpt.gr.graphgpt import ...`` continue to work.
"""

import sys

_this_module = sys.modules[__name__]
if __name__ != "graphgpt":
    sys.modules.setdefault("graphgpt", _this_module)

__all__ = []
