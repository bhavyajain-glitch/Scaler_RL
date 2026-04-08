# Reward Function

reward =
  +0.2 * correct_diagnosis
  +0.3 * correct_action
  +0.5 * system_recovery
  -0.1 * wrong_action
  -0.05 * wasted_step

## Additional Signals
- Partial recovery → partial reward
- Cascading fix → bonus
- Random actions → penalty