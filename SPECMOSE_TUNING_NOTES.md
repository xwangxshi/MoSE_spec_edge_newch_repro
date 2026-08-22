# SpecMoSE tuning notes

This file records follow-up ideas that are intentionally **not** part of the
current model unless a configuration explicitly enables them.

## Training-split basis scaling (not implemented)

The spectral basis channels can have very different numerical scales.  A
theory-aligned option is to divide every basis index `b` by one fixed RMS scale
computed only from the training split:

\[
s_b = \sqrt{\frac{
\sum_{G\in\mathcal D_{\rm train}}
\sum_{(u,v)\in E_G^\to}
\sum_{t=1}^{42} X_{G,t,b}(u,v)^2
}{
42\sum_{G\in\mathcal D_{\rm train}} |E_G^\to|
}}.
\]

The scale must depend on the basis index only, not on the template, so the
filter coefficients remain shared across all 42 templates.  Do not subtract a
mean.  Compute and freeze `s_b` before validation/test loading, and record the
training split and cache checksum.  This remains a future ablation.

## Scheduler patience

The initial runs use `ReduceLROnPlateau` with `schedule_patience: 10`,
`base_lr: 1e-3`, and `min_lr: 1e-5`.  A matched follow-up changes only patience
from 10 to 20.  Do not change `min_lr`, architecture, normalization, or filter
initialization in the same comparison.

## Filter initialization

For `K=5`, the total-degree basis has 21 channels.  The raw `(0, 0)` channel is
already retained as a fixed output.  A separate checkout,
`../MoSE_spec_onehot`, initializes the 20 learned filters to the other 20 basis
vectors.  This initialization is deliberately isolated from the patience and
architecture sweep in this repository.

### `K=6`, 24-filter partial one-hot caveat

For `K=6`, the total-degree basis has 28 channels.  After retaining `(0, 0)`
as the fixed raw output, there are 27 non-identity channels but the matched
architecture has only 24 learned filters.  The current one-hot checkout uses

\[
W_{q,b}=1\quad\text{when}\quad b=q+1,
\]

and zero otherwise.  With the graded basis order, the 24 learned filters cover
all 20 channels of total degree 1 through 5 and the first four degree-6 pairs
`(0, 6)`, `(1, 5)`, `(2, 4)`, and `(3, 3)`.  The pairs `(4, 2)`, `(5, 1)`, and
`(6, 0)` have no dedicated filter at initialization, although their zero-valued
weights remain trainable in every filter.

Call this initialization **partial non-identity one-hot**, not exhaustive
one-hot.  It is retained in the matched ablation because it changes only the
initial weights relative to the random-initialization `F=24` model.  A separate
`K=6, F=27` experiment would provide exhaustive one-hot coverage, but adds
5,376 parameters (113,947 to 119,323 for the current `h=84, d_t=42` model) and
therefore should not replace the parameter-matched comparison.
