# Contributor guidance

## Agent coordination

- Default to one agent. Delegate only an explicitly requested or clearly useful,
  bounded independent subtask while the parent advances other work. Use the
  smallest useful team and a self-contained brief; avoid unnecessary full-history
  forks, recursive delegation, duplicate work, and overlapping edits.
- Prefer completion notifications. When blocked on a result, call the native wait
  tool directly with an explicit timeout suited to the expected duration and the
  active runtime and communication limits. Avoid repeated short waits, wrapping
  native agent waits in another yielding tool, and checking status after every
  unchanged timeout.
- Send follow-up messages only for new information, changed scope, or a concrete
  blocker. If a final result conflicts with a running status, inspect once and
  reconcile it instead of polling indefinitely. Respect required progress updates.
- Use bounded waits and incremental output for CI and long commands too. A timeout
  is neither completion nor approval; required checks must still pass before merge.
