# Nova2 Pre-Change Baseline Audit

Source: `scripts/results/baseline_20261008_103618.json`

## Overall

- 12 questions.
- 36 standard runs: Off, Concise, Detailed.
- 12 guided runs.
- All 12 guided runs failed validation with **"Guided generator returned fewer than two valid step records."**
- No guided MCQs were validly produced, so no guided accuracy claim is made.
- Standard timing is highly variable; several LLM-backed runs take tens of seconds to minutes.
- Raw API responses remain preserved. Presentation fields have been normalized to rendered answer text.

## Confirmed standard-output issues

| Test | Mode | Finding |
|---|---|---|
| Math derivative | Off | CAS result is mathematically equivalent but returned as `0 == 7 - 10*x + 12*x^3` instead of a normalized derivative. |
| Definite integral | Off | **Incorrect:** returned `-1/4`; correct value is `-1/9`. |
| Vertical projectile | Off | **Incomplete:** only total time was returned; maximum height was omitted. |
| Vertical projectile | Concise | **Incorrect:** total time stated as 2 s; correct value is 4 s. |
| Circular orbit | Off | **Incomplete:** only escape/orbital-speed relation was returned; requested speed, kinetic energy and total energy were omitted. |
| NH3 geometry | Off | **Incorrect:** electron-pair geometry stated as trigonal pyramidal; correct is tetrahedral. |
| NH3 geometry | Concise | **Incorrect:** same electron-pair geometry error. |
| Complexity | Off | Gives O(n), O(1) rather than the tighter requested Θ(n), Θ(1). |
| Complexity | Concise | Same asymptotic bounds are valid but not expressed as tight Θ bounds. |
| Binary search | Off | **Incomplete:** only the words "Binary Search" were returned. |
| Dijkstra | Off | **Incomplete:** only O(log V) was returned. |
| Dijkstra | Concise/Detailed | Negative-edge discussion incorrectly implies a negative edge itself can produce infinitely decreasing path cost; that requires a negative cycle. |

## Outputs that are substantively correct

- Math infinite integral: `π^4/15`.
- Rod motional emf: `(1/2)BωL²`.
- Chemistry buffer: pH ≈ 4.26, consistent with the ≈4.27 reference after rounding.
- N2O4 equilibrium: α ≈ 0.173.
- Binary-search concise/detailed: algorithm and Θ/O(log n)-type complexity reasoning are substantively correct.
- Most detailed math/physics/chemistry responses are substantially correct where they reached a complete answer.

## Important baseline-level issues

1. **Guided mode is currently the largest reliability failure:** 12/12 guided runs failed validation.
2. **Explanation-off mode is not reliably result-only:** several questions returned incomplete answers or malformed/overly narrow outputs.
3. **The local LLM can produce numerically wrong answers even when the prose is fluent:** the definite-integral `-1/4` result is a clear example.
4. **Chemistry geometry needs deterministic verification:** NH3 electron-pair geometry was wrong in Off and Concise despite correct molecular geometry.
5. **The baseline report previously exposed raw metadata/JSON-like text:** display fields are now normalized to answer text while raw responses remain available.
6. **The DOCX table width was too large for the previous default page geometry:** the renderer is now explicitly fixed to Letter landscape with reduced table widths.

## Not changed

The baseline's measured outputs are not silently corrected. Incorrect/incomplete outputs are flagged for evaluation; the original `response` and `raw` fields remain intact.
