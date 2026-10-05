# V7 asymmetric carry-only freeze

Changes are deliberately narrow and frozen before the V7 robustness run.

1. Keep V6 joint PUT/CALL/CASH HJB and once-per-session management.
2. Standalone short calls are not executable. A call is executable only when the same ticker/expiry PUT is independently HJB-selected; both are then paired as SHORT_STRANGLE_APPROVED.
3. Standalone short puts remain executable with their V6 full package-style risk basis.
4. Legacy local-SABR/index-fly branches are still constructed for diagnostics, but are research-only and cannot receive orders.
5. No IV, spread, MC path-count, HJB objective, stop, trail, or position-size thresholds were relaxed for V7.

V7_RECHECK_DEV (Jan-2023) and V7_RECHECK_CHECK (Jun-2025) are already-seen engineering rechecks. Do not call them OOS.
V7_ROBUSTNESS (Jul-2026) was registered at this freeze as a fast historical robustness check. It is not a pristine prospective OOS claim.
