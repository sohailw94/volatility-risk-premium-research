# Research chronology

This document records the logic of the main ablations so the final strategy is not presented as if it appeared fully formed.

1. **Original VRP / surface thesis:** negative walk-forward performance.
2. **Executable-price audit:** midpoint assumptions were replaced with executable quote logic.
3. **Sign-inversion diagnostic:** realistic inverse fills also lost, rejecting the hypothesis that the signal was simply backwards.
4. **Exit ablation:** removing individual-leg exits exposed a strong interaction between holding horizon and realized carry, while also revealing assignment / package-accounting contamination.
5. **Clean package replay:** once mechanical bugs were removed, full short strangles still underperformed.
6. **Leg decomposition:** put-side and call-side economics were materially asymmetric.
7. **V5:** independent leg research and asymmetric sizing; still underperformed because standalone calls were over-selected and exit/model cadence differed.
8. **V6:** joint PUT/CALL/CASH competition and once-per-session management aligned the execution policy more closely with the MC cadence.
9. **V7:** standalone calls became research-only unless the corresponding put was also independently selected; legacy fly execution was disabled. No threshold loosening was used for the V7 rechecks.

The purpose of documenting failed versions is to make the research falsifiable and auditable.
