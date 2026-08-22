# Repository Working Instructions

These instructions apply to the entire `MoSE_spec` repository.

## Environment

- Use the `SPE` virtual environment by default. Prefer the explicit interpreter
  `/home/Xiaohan/anaconda3/envs/SPE/bin/python` for Python commands, tests, and
  experiment launchers unless the user explicitly requests another environment.

## Implementation style

- Avoid excessive defensive programming. Add checks, fallbacks, wrappers, and
  abstractions only when they address a concrete requirement or realistic
  failure mode. Prefer direct control flow, clear naming, and readable code.

## Change disclosure

- If any change outside the scope discussed with the user appears during
  programming, explicitly disclose it in the next user-facing response. State
  the affected path, what changed, and why; never silently include unrelated or
  incidental changes.
