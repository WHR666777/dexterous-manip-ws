# Releases

## Unreleased

### Added
- Added adaptive contact loss for pinch retargeting: `L_contact = w_contact * alpha * ||thumb_tip - partner_tip||²`.
- Enabled `w_contact: 6.0` across all adaptive configuration files.
- Documented contact loss behavior and tuning notes in README.md and README.zh.md.
- Added Linker L20 real-output integration files and teleoperation wiring.

### Notes
- Contact loss only affects `AdaptiveOptimizerAnalytical` when pinch alpha is non-zero.
- The `model/` experimental training files are intentionally not included in this release update.
