from ._pathsetup import check_reexport, ensure_project_root_on_path

ensure_project_root_on_path()
from decomposer import *  # noqa: F401,F403

check_reexport(globals(), 'TaskDecomposer', 'DecompositionResult')
