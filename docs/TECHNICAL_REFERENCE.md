# PALEO Technical Reference

How every component of both pipelines works — the GP prototype and the neural
pipeline, Stage 1 and Stage 2 — mechanism only, no results. Code lives in
`paleo/` (neural pipeline) and in notebooks 00–08 (GP prototype).

---

## Part A — Shared foundations

### A.1 Light-curve ingestion

Every light curve, for every purpose, passes the same ingestion rules:

1. **Source pipelines**: only SPOC, TESS-SPOC, or QLP products are accepted, in
   that priority order. Community pipelines (eleanor, CDIPS) lack the
   systematic-trend removal the rest of the pipeline assumes.
2. **Flux column**: SPOC/TESS-SPOC → `pdcsap_flux` (systematics-corrected).
   QLP renamed its corrected column at Sector 56, so the column is resolved by
   priority search: `kspsap_flux` → `sys_rm_flux` → `det_flux`. The library
   default (`sap_flux`, raw photometry) is never used: uncorrected instrumental
   ramps dominate its periodogram.
3. **Quality masking**: only cadences with `quality == 0` survive. Nonzero
   flags mark scattered light, momentum dumps, and thermal events; leaving them
   in plants instrumental artefacts that downstream statistics will find.
4. **Validity + positivity**: finite time, finite flux, flux > 0.
5. **Normalisation**: `f = flux / median(flux) − 1`, so flux is a fractional
   deviation centred on zero. A transit of depth d appears as a dip of −d.

### A.2 Rotation-period estimator (Lomb–Scargle + FAP gate)

Used by the GP prototype to set its kernel prior (the neural detector does not
need a rotation period). Construction:

- **Search band**: `min_period = max(2 × median cadence, 0.05 d)` — a
  Nyquist-like floor — up to `max_period = baseline / 2`, requiring at least
  two complete cycles. Wider bounds admit long-timescale instrumental trends
  as spurious "periods."
- **Frequency grid**: linear in frequency from `1/max_period` to
  `1/min_period` with `oversample × baseline / min_period` points
  (oversample = 10), the standard density for resolving LS peaks.
- **Peak selection**: astropy `LombScargle` power on the true, irregular time
  stamps (never a uniform-sampling periodogram); the period is the reciprocal
  of the arg-max frequency.
- **Reliability gate**: the Baluev analytic false-alarm probability of the peak
  is computed; if FAP ≥ 10⁻³ the star is declared to have no reliable rotation
  period rather than being assigned a guess. This prevents a garbage prior from
  propagating into the GP kernel constraint.

---

## Part B — GP prototype, Stage 1

The GP Stage 1 answers: *given one light curve, which short windows are
statistically inconsistent with the star's own variability?* Its chain is
bin → prewhiten → GP model → interval test → event extraction.

### B.1 Binning to fixed cadence

Exact GP inference is O(n³) in the number of points, so the light curve is
binned to a fixed **36-minute cadence** (`n_bins = baseline_days × 40`): bin
edges are uniform in time, each bin takes the *median* flux of its points
(median for outlier robustness), and empty bins are dropped. Fixed *cadence*
rather than fixed *count* keeps bin width — and therefore the number of bins
across a transit — identical for every baseline length.

### B.2 Balmung prewhitening

Balmung removes coherent periodic stellar signal by **iterative sinusoid
subtraction**:

1. Compute a Lomb–Scargle amplitude spectrum of the current residual.
2. Find the highest-amplitude peak; least-squares fit a single sinusoid
   (frequency, amplitude, phase) at that frequency; subtract it.
3. Repeat until a stopping rule fires or `maxiter` is reached.

The stopping rule compares the current component's amplitude `a₀` against a
noise floor `minimum_snr × median(amplitude spectrum)`. Two parameters matter:

- `minimum_snr` (pipeline value **4**): the floor multiplier. At the library
  default of 0 the floor is identically zero, the rule never fires, and
  iteration runs to `maxiter` subtracting real signal, noise, and transits
  alike. At ≥ 3 the criterion halts within ~20 iterations on real stars.
- `maxiter` (**50**): a backstop only; not the operative parameter once
  `minimum_snr ≥ 3`.

Why a nonzero floor protects transits: stellar rotation concentrates its power
in a few strong sinusoids (the rotation frequency and harmonics) that tower
over the median spectrum level and are removed; a transit is a brief localised
dip whose power is spread thinly across many frequencies, so no *single*
sinusoid component of it ever clears a 4× floor.

### B.3 The Gaussian-process model

The star's remaining variability is modelled as a zero-mean GP over time with
a **quasi-periodic kernel**: a periodic kernel (period parameter constrained
in a window around the Lomb–Scargle rotation estimate — tightly for periods
above ~4 d) multiplied by an RBF kernel that lets the periodic pattern evolve
slowly, plus a learned white-noise term in the Gaussian likelihood.
Implementation is gpytorch `ExactGP`; hyperparameters are fitted by maximising
the exact marginal log-likelihood with Adam, with early stopping when training
loss begins increasing (minimum 50 iterations). The flux is standardised
(zero mean, unit variance) before fitting.

The constrained period is why the rotation estimator upstream must be
trustworthy: for slow rotators the optimiser cannot escape a bad prior.

### B.4 ExhaustiveSearch — the interval test

Candidate anomalies are **contiguous intervals** of the binned light curve.
The interval grid is built from four parameters, all derived from the target
transit duration `dur` and the bin cadence `cad` (with
`n_tr = round(dur/cad)`):

- `min_anomaly_len = max(2, n_tr // 2)` — shortest window considered,
- `max_anomaly_len = 3 × n_tr` — longest,
- `window_slide_step = window_size_step = max(2, n_tr)` — stride between
  candidate start positions and between window sizes.

For each candidate interval the test is a **held-out predictive check**:

1. Fit/condition the GP on the *complement* of the interval (all data outside
   it). Dynamic programming re-uses shared computations across the grid so the
   full sweep does not refit from scratch per interval.
2. Predict the held-out interval: a multivariate normal with mean vector μ and
   covariance Σ from the GP posterior.
3. Compute the Mahalanobis distance of the observed flux y in the interval:
   D² = (y−μ)ᵀ Σ⁻¹ (y−μ).
4. Under the null (the GP describes this data), D² ~ χ² with dof = interval
   length; the upper-tail probability is the interval's **p-value**.

Training on the complement is the structural point: the posterior at held-out
points is non-degenerate. (The rejected GreedySearch evaluated the GP at its
*own training points*, where the posterior collapses and the Cholesky fails;
its window-growing loop also lacked a terminating metric. ExhaustiveSearch has
neither pathology.)

The parametric p-values are **used only as rankings**, never as absolute
probabilities — the χ² assumption inherits every imperfection of the GP model.
All absolute significance in the pipeline comes from Stage 2's permutation
test.

### B.5 Event extraction (shared logic with the NN, different scores)

1. Sort *all* scored intervals by p-value, best first.
2. Walk down the ranking, keeping an interval only if it does not overlap any
   already-kept interval (overlap checked in bin indices with a 1-bin margin).
   Duplicated coverage of one feature is thus skipped, not merged, so the
   selection keeps descending to genuinely distinct events. Stop at k = 8.
3. **Knee cut**: convert the kept p-values to −log₁₀ p; find the largest drop
   between consecutive ranked values (considering cuts that leave at least 3
   events); if that drop exceeds one decade, truncate the list there,
   otherwise keep all 8. Clean stars self-truncate at their few strong events
   (junk is orders of magnitude weaker); weak-signal stars keep a full list.
4. Each kept interval becomes one **event time**: the midpoint of its time
   span. The ranked event times are Stage 2's input.

---

## Part C — Neural pipeline, Stage 1

The neural Stage 1 replaces B.1–B.4 with a learned per-timestep detector; the
extraction step (C.6) mirrors B.5 with scores in place of p-values. It needs no
rotation period and no prewhitening: the network learns to ignore stellar
variability directly.

### C.1 Resampling to a uniform 10-minute grid

`resample_uniform(time, flux)`:

1. Grid step = 10 min; cell index of each point =
   `floor((t − t₀)/step)`; each cell's value = *mean* of its points (computed
   via two `bincount` calls: sum and count).
2. **Hole interpolation**: a 30-min-cadence star leaves 2 of 3 cells empty by
   construction — sampling artefacts, not gaps. Runs of empty cells no longer
   than 45 minutes, with data on both sides, are filled by linear
   interpolation and marked live. Longer runs (the mid-sector downlink gap,
   edges) stay flagged.
3. **Gap channel**: a parallel array, 1.0 where a cell is genuinely
   unobserved, 0.0 where live. Gap cells have flux set to exactly 0.
4. **Noise scale σ**: `1.4826 × median(|Δf|)/√2` computed on consecutive
   *gridded* live cells — the robust point-to-point scatter, insensitive to
   slow variability, and measured after gridding so all cadences are on the
   same footing.

Output per star: `{t0, step, flux[N], gap[N], sigma}`.

### C.2 Input representation

The network sees a `(2, N)` tensor: channel 0 = `flux / σ` (the star's flux in
units of its own noise), channel 1 = the gap mask. The σ-division means the
network never learns "detect dips of X ppm" — it learns "detect dips of X
sigma," making its sensitivity **star-relative** and transferring across quiet
and noisy stars automatically. The gap channel lets it condition on
missingness rather than mistake gap edges for signals.

### C.3 The training-data generator

There is no fixed dataset. `make_training_sample(grid, rng)` synthesises a new
labelled example on demand:

- With probability 0.8, inject a periodic transit; otherwise return the star
  untouched (a pure negative).
- Injection parameters per sample: depth log-uniform in [0.2%, 3%]; duration
  uniform in [1, 6] h; period uniform in [2, 10] d; epoch fraction uniform in
  [0.05, 0.95]; ingress fraction uniform in [0.1, 0.5].
- **Shape**: a trapezoid — flat bottom of half-width `(dur/2)(1−f_ingress)`,
  linear ramps to zero at ±dur/2 — deliberately *not* the box used in
  evaluation, so the network cannot overfit the evaluation artefact. (A `box`
  mode exists for protocol-matched evaluation injections.)
- The deficit is subtracted from live cells only.
- **Label** `y[N]`: 1.0 for live cells within ±dur/2 of any transit centre,
  else 0. Gap cells are never labelled.
- Stars are drawn with replacement each step, so a star is revisited many
  times but never with the same transits: examples are unlimited; the finite
  resource is the diversity of stellar backgrounds.

### C.4 UNet1D architecture

A 1-D U-Net, ~1.04 M parameters, mapping `(B, 2, N) → (B, N)` logits:

- **ConvBlock(c_in, c_out)**: Conv1d(kernel 9, same padding) → BatchNorm →
  ReLU, twice. Kernel 9 at 10-min cells spans 1.5 h — transit-edge scale.
- **Encoder**: four ConvBlocks with channels 24 → 48 → 96 → 192, each after a
  MaxPool(2) except the first. Successive halvings give deep layers a
  receptive field of many hours: enough context to model local variability
  around a candidate dip.
- **Decoder**: three levels of ConvTranspose1d(stride 2) upsampling, each
  concatenated with the same-scale encoder output (skip connections preserve
  cell-level timing) followed by a ConvBlock.
- **Head**: a 1×1 convolution to a single channel; `.squeeze(1)` yields
  per-cell logits.
- **`pad_to(x, 8)`** right-pads inputs to a multiple of 8 so three pooling
  halvings divide evenly; padded cells are treated as gap everywhere.

### C.5 Loss and training loop

**Loss** — `masked_bce(logits, target, gap, pos_weight=25)`: element-wise
binary cross-entropy with logits, where each element's weight is
`(1 + 24·y) × 1[gap<0.5]` — i.e. gap and padded cells contribute nothing, and
in-transit cells are up-weighted 25× to counter the ~1–3% duty cycle that
would otherwise let "predict zero everywhere" be a near-optimal solution. The
sum is normalised by the count of live cells.

**Batching**: 16 samples per step; each batch pads to its own longest star
(rounded to a multiple of 8); padding is marked gap in the loss mask.

**Optimisation**: Adam, lr 3×10⁻⁴, fixed seeds. Every 200 steps the model is
scored on a **fixed dev set** (12 pre-generated batches = 192 samples from the
30 held-out dev stars, same seed always, so evaluations are comparable across
runs). Improvements of the dev loss checkpoint the model to disk; six
consecutive non-improvements trigger early stopping.

### C.6 Inference and event extraction

- **Scoring**: `sigmoid(model(x))` per cell; scores in gap cells forced to 0.
- **`nn_events`** — ranked, credible, non-overlapping peaks:
  walk cells in descending score order; a cell becomes an event only if
  (a) the window of ±half a transit duration around it is at least 80%
  live — a candidate surrounded by gap is not a credible transit — and
  (b) it lies at least one transit duration from every already-kept event.
  Keep up to k = 8. Events are the cell's timestamp plus its score, in rank
  order.
- **Knee cut on scores**: identical logic to B.5 step 3, but on logits
  `ln(s/(1−s))` with a required drop of 2.0 (≈ one decade in odds), min 3,
  max 8 events.

The extracted, ranked, truncated event list is Stage 2's input — the identical
interface the GP Stage 1 produced, which is what makes the two Stage 1s
drop-in interchangeable above the same Stage 2.

---

## Part D — Stage 2: sparse-event period search (shared by both pipelines)

Input: event times `e₁…e_m` (3 ≤ m ≤ 8), the observing span [t_start, t_end],
and a tolerance `tol` = transit duration. Output: best period, its aligned
events, and a permutation false-alarm probability. Fewer than 3 events →
abstain (return no period, FAP 1).

### D.1 Candidate periods

For every pair (i, j) and every integer k = 1…8, the value `(e_j − e_i)/k` is
a candidate if it lies in [0.5 d, baseline/2]. Rationale: if the events contain
transits of true period P, some pair of them is separated by an integer number
of periods, so P itself appears in this set. The grid is therefore
data-implied and finite (≤ 28 pairs × 8 divisors), not a blind scan — which is
also what makes the permutation null (D.4) affordable.

### D.2 Alignment scoring

For a candidate period P:

1. **Phase clustering** (`_phase_cluster_size`): fold the event times mod P,
   sort the phases, duplicate the array shifted by +P (so windows can wrap the
   phase boundary), and slide a window of width `tol` with two pointers to
   find the largest number of events m_al falling inside one window, plus that
   window's starting phase.
2. **Statistic**: `S(P) = (m_al − 1) × ln(max(P/tol, 1))`. Raw match *count*
   is the wrong statistic because a short trial period makes `tol` a large
   fraction of phase and random events align cheaply; the log factor prices
   each additional aligned event by how improbable alignment is at that
   period (the −1 because a single event trivially "aligns" with itself).
3. **Occupancy gate**: with `n_win = floor(baseline/P)` available transit
   windows, a candidate is vetoed (S set to −1) unless `m_al ≥ 3` **and**
   `m_al / n_win ≥ 0.5`. A genuine periodic transit appears in most of its
   windows; a subharmonic of the true period padded by junk fills only a few
   of many. (Consequence: with ≤ 8 events, periods below ≈ baseline/16 are
   unreachable — a documented domain edge.)

The winning candidate is the arg-max of S; its aligned subset is recovered by
re-folding at the winning period/phase. If no candidate survives the gate,
abstain.

### D.3 Why no analytic p-value

S is a custom statistic with no textbook null distribution, and any analytic
approximation would inherit the same model-trust problem that made the GP's
χ² p-values unusable. Significance is therefore bought empirically.

### D.4 The permutation FAP

Repeat `n_perm = 2000` times: draw m event times uniformly at random over
[t_start, t_end]; run the **entire identical search** on them — their own
pairwise candidate periods, the same scoring, the same occupancy gate — and
record the best surviving statistic (0 if nothing survives). The FAP is the
fraction of these null universes whose best statistic meets or exceeds the
observed one.

Properties this construction guarantees:

- **Look-elsewhere is priced in**: the fake events enjoy every freedom the
  real search had, including data-derived candidate grids.
- **Conditioning on m**: the null uses the same event count as observed, so a
  star that produced many events is judged against an appropriately stronger
  null.
- **Resolution floor**: with 2000 permutations the smallest reportable FAP is
  5×10⁻⁴; `n_perm` scales linearly in cost if smaller values are needed.
- **Calibration is inherited, not assumed**: if the search is changed in any
  way, the null automatically changes with it, because they are the same code
  path.

### D.5 Decision rule and paired controls

FAP < 0.05 = detection; 0.05–0.15 = candidate tier (follow-up, never a
claim). In every benchmark each star is also run as an untouched control; if
a control's best period matches its injected twin's found period (within 5%),
the star carries an intrinsic periodic signal and is excluded from scoring for
all methods — an exclusion rule that uses only control information.

---

## Part E — Orchestration

**GP pipeline (per star)**: ingest (A.1) → bin 36-min (B.1) → rotation period
(A.2) → Balmung `minimum_snr=4` (B.2) → standardise → GP fit (B.3) →
ExhaustiveSearch interval p-values (B.4) → ranked non-overlap + knee (B.5) →
Stage 2 (D) → period + FAP.

**Neural pipeline (per star)**: ingest (A.1) → resample 10-min + gap + σ
(C.1) → UNet1D scores (C.4/C.6) → ranked credible peaks + logit knee (C.6) →
Stage 2 (D) → period + FAP.

Interchangeability is by construction: both Stage 1s emit the same object — a
rank-ordered list of ≤ 8 event times with per-event strengths — and Stage 2
consumes it without knowing which detector produced it. All fixed constants:
36-min bins (GP) / 10-min grid (NN); `minimum_snr=4, maxiter=50`;
`num_perm=2000`; tolerance = transit duration; k = 8; knee minimum 3 events,
drop thresholds 1 decade (p-values) / 2.0 logits (scores); occupancy 0.5;
candidate divisors k ≤ 8; period band [0.5 d, baseline/2]; FAP thresholds
0.05 / 0.15.
