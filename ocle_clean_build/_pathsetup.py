"""Internal helper for ocle_clean_build's re-export shims.

ocle_clean_build's shim files (engine.py, tokenizer.py, memory.py, ...) are
deliberately named the same as the top-level modules they wrap, e.g.::

    # ocle_clean_build/engine.py
    from engine import *

This is what lets ``ocle_clean_build.engine.OpenClosedLoopEngine`` and the
top-level ``engine.OpenClosedLoopEngine`` be *the same* class object. But it
means the shim relies on the project root being found on ``sys.path``
*before* this package's own directory -- and if it isn't (for example, if a
caller's working directory or sys.path happens to put this directory first),
``from engine import *`` can silently import this very file again instead of
the real top-level engine.py, producing an empty module with no error. See
CHANGELOG.

Each shim calls :func:`ensure_project_root_on_path` (via a *relative*
import, so that if this file is ever accidentally executed as a bare
top-level module instead of as ``ocle_clean_build.X``, Python itself raises
an immediate, clear ``ImportError`` rather than silently mis-resolving)
before doing its wildcard import, which makes the project root win the
resolution race regardless of ambient sys.path order.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def ensure_project_root_on_path() -> None:
    """Insert the project root at the front of sys.path (deduplicated)."""
    if _PROJECT_ROOT in sys.path:
        sys.path.remove(_PROJECT_ROOT)
    sys.path.insert(0, _PROJECT_ROOT)


def check_reexport(module_globals: dict, *expected_names: str) -> None:
    """Fail loudly if a wildcard re-export didn't bring in what it should.

    Belt-and-suspenders alongside ensure_project_root_on_path(): if this
    module's namespace is still missing the names it's supposed to
    re-export after the wildcard import, something resolved wrong (most
    likely this file importing itself). Raise a clear, actionable error
    instead of letting callers hit a confusing AttributeError far away from
    the real cause.
    """
    missing = [name for name in expected_names if name not in module_globals]
    if missing:
        raise ImportError(
            f"ocle_clean_build.{module_globals.get('__name__', '?')} failed to "
            f"re-export {missing!r} from the top-level module of the same "
            "name -- 'from X import *' likely resolved to this shim file "
            "itself instead of the real project-root module. This usually "
            "means the project root wasn't found on sys.path from wherever "
            "this was imported. See CHANGELOG."
        )
