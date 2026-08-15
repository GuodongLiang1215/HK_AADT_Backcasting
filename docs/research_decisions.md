# Phase 1 Research Decisions

## Primary feasibility decision

Proceed with the pilot, subject to three requirements:

1. model labels must be current-year measured AADT rather than values already estimated by an official growth factor;
2. historical station identity must be verified using road descriptions and geometry, not station number alone;
3. 2021 must be interpreted as a shock/sensitivity year rather than an ordinary endpoint of a long-term trend.

## Initial years

- 2011: historical anchor;
- 2016: clean pre-COVID midpoint;
- 2021: census-aligned temporal-transfer stress case;
- 2006: add only after the first three-year extraction and crosswalk are stable;
- 2023: modern comparison with Niu et al., not independent ground truth.

## Required station-year fields

- `year`
- `station_id`
- `station_type`
- `road_type`
- `road_name`
- `road_from`
- `road_to`
- `aadt_previous`
- `aadt_current`
- `previous_aadt_estimated`
- `current_aadt_estimated`
- `source_pdf`
- `source_page`

## First modelling-label rule

The primary training table must satisfy:

```text
current_aadt_estimated == False
```

Rows with estimated current-year AADT may be retained for comparison but not used as independent observed labels in the main model or headline validation.

## Longitudinal matching rule

The same station number is a candidate match, not a confirmed historical segment. The crosswalk must compare:

- station number;
- normalised road name;
- road-from description;
- road-to description;
- current official station geometry;
- manual review reason when definitions differ.

Every longitudinal match should be assigned `high`, `medium`, or `low` confidence.

## Current spatial-anchor rule

The yearly updated official Station Point and Station Line files are treated as
the latest spatial snapshot at download time. They do not contain a census-year
field and therefore cannot establish historical location stability by
themselves.

- permitted: mapping, current-location lookup, manual review prioritisation;
- not permitted: automatically upgrading historical match confidence;
- required for disputed historical cases: archived official geometry, report
  maps, or another year-specific source;
- current road centreline processing is deferred until a defined network
  conflation task justifies its size and computational cost.

## Core-panel freeze rule

The first primary longitudinal panel is frozen before manual recovery:

- 783 stations have measured AADT in 2011, 2016, and 2021 and pass both
  adjacent high-confidence physical-match checks;
- 778 of these have a current official Point anchor and form the initial
  spatial model panel;
- 57 excluded stations could enter a sensitivity panel after historical
  evidence is reviewed;
- 89 additional review stations do not enlarge the clean three-year panel
  because a year or measured label is missing;
- the direct 2011-2021 comparison is diagnostic and must not create a duplicate
  manual-review workload.

Manual recovery is not allowed to delay the first baseline model. Any recovered
station enters a sensitivity extension first, not the frozen primary panel.

## Road-network support rule

The Traffic Flow Census Road Centerline is used as a harmonised current network
support because the official Station Line is defined relative to it. Historical
AADT labels may be attached to that common support for a first reconstruction
baseline, but the current geometry cannot establish that historical topology
was unchanged.

Road attachment must use both spatial distance and road-name consistency.
Nearest-segment assignment alone is not acceptable in grade-separated or
parallel-road settings. Only high-confidence station-to-route links enter the
first network baseline; medium and low links remain sensitivity or review data.

Because an ATC AADT label represents a road cross-section more naturally than
one directed centerline feature, the linkage stores a representative
`selected_route_id` plus a same-road, same-elevation
`label_support_route_ids` bundle. A competing road identity or elevation level
is treated as genuine ambiguity. Shared support routes must not trigger
automatic averaging of station labels.

## First validation-design rule

Only Step 7 high-confidence station-to-centerline links enter the primary
training table. Medium and low links remain review or sensitivity cases and do
not delay the first baseline.

The first comparison uses five regional spatial holdouts rather than a random
row split. Stations sharing any `label_support_route_ids` value are joined into
one support component before regional clustering, and every component remains
entirely within one fold. This prevents the same road cross-section support
from appearing on both sides of a validation split.

The first model comparison is deliberately limited to a training-median
baseline, distance-weighted spatial KNN, and one nonlinear tabular model, fitted
and evaluated separately for 2011, 2016, and 2021. MAE and RMSE on held-out
spatial folds are primary. Added model complexity is justified only if it
produces consistent spatial-holdout improvement rather than a better random
split score.

Direct high-cardinality identifiers such as `route_id`, `street_code`, and
street name remain audit fields, not first-model predictors. The station-report
road type is also excluded because it is unavailable for every prediction
segment on the current centerline.

## First model-comparison rule

The first comparison is fixed before external interpretation: a training-fold
median, a distance-weighted spatial KNN with `k=10`, and one
HistGradientBoosting model with absolute-error loss. Models are fitted
separately for 2011, 2016, and 2021. Outer regional folds are not used for
hyperparameter selection.

The zero-variance `named_street` field is removed from the training feature
set. Overall MAE improvement is necessary but not sufficient. Observed-AADT
quartile calibration must also be reported because regression to the mean can
overstate low-volume roads, understate high-volume roads, and artificially
compress the traffic-burden distribution used in later equity analysis.

## Step 9 evidence and feature-extension decision

The fixed nonlinear baseline lowered pooled regional out-of-fold MAE relative
to both the training median and spatial KNN in 2011, 2016, and 2021. This is
evidence that the available road and network covariates contain genuine spatial
signal. It is not sufficient evidence for equity backcasting because the model
overpredicts the observed low-volume quartile by roughly 104--120% and
underpredicts the high-volume quartile by roughly 32% in all three years.

The first response to this tail compression is a controlled official-feature
ablation. The HistGradientBoosting parameters, years, labels, and regional folds
remain fixed. Current Road Network v2 speed-limit, intersection, roundabout,
bus-lane, traffic-feature, restriction, permit, road-name, street-code, and
strategic-route attributes are added as one predeclared bundle.

The bundle is retained only if no year's pooled MAE degrades by more than 2%
and the absolute high-volume Q4 mean bias improves by at least five percentage
points in at least two years. Reusing the Step 9 folds makes this development
evidence, not a fresh independent final test.

In the current run, the official extension lowers pooled MAE in zero of three
years. It improves Q1 bias by at least five percentage points in two years but
does not meet the Q4 improvement threshold in any year. The official bundle is
therefore not adopted as the new baseline. The next permitted feature experiment
is a small capacity-oriented OSM bundle, principally road class, lanes,
maxspeed, and junction type. It must be tested as another explicit ablation,
not blended into an unconstrained feature search.

The Road Network v2 layers are current monthly snapshots. They are useful for
diagnostic present-day road attributes but do not prove historical road
configuration or capacity. After the capacity feature choice is frozen, a new
validation source or held-out spatial design is required before final claims.

## Full-network preliminary backcasting rule

The exploratory task now prioritises the requested full-network reconstruction
over additional feature searching. The Step 9 HistGradientBoosting model and
its eight predictors are frozen. The failed Step 10 official feature bundle is
reported as a diagnostic negative result and is not adopted. The proposed OSM
capacity ablation is deferred until after the requested backcast, census
harmonisation, and preliminary equity analysis.

Separate year-specific models are fitted to all 679 high-confidence stations
and applied to all 36,107 features of the current official centreline. These
full-fit predictions are not described as out-of-fold estimates; model
validation continues to rely on the Step 9 regional OOF results. Observed AADT
on the 677 directly linked routes is stored in separate fields and does not
silently replace model predictions.

The frozen model produces a small number of nonpositive raw network
predictions. Raw outputs are retained for audit, while the physical output used
for mapping is floored at one vehicle per day. This is a physical constraint,
not a calibration method. No correction is applied to the known Q1
overprediction or Q4 underprediction.

Spatial support is reported descriptively using distance to the nearest
training-route centroid and structural-feature range exceedance. These labels
are not prediction intervals or statistical confidence claims.

Road-level predictions must not yet be summed directly into an equity-exposure
total. The current centreline may represent opposite directions or parallel
carriageways as separate features, whereas an ATC label represents a road
cross-section. Step 12 must define a harmonised road-support and neighbourhood
aggregation rule before socioeconomic comparison.

## Census geography and boundary-audit rule

The first neighbourhood geography is the official Large Tertiary Planning Unit
Group. It retains demographic, household, educational, economic, housing, and
internal-migration variables in all three study years while remaining more
local than District Council districts. The official unit counts are 154 in
2011, 154 in 2016, and 159 in 2021; equal counts in 2011 and 2016 are not treated
as proof of identical geometry.

The 2016 Large TPU Group layer is the common reference because it is the
pre-COVID midpoint and endpoint of the primary 2011--2016 comparison. Boundary
overlap is measured in Hong Kong 1980 Grid. A direct longitudinal match requires
the same code and at least 95% area coverage in both directions. Other units are
retained for repeated cross-sectional analysis but are not silently forced into
the same-neighbourhood panel.

Area-intersection weights are diagnostic only. They are not population weights
and must not be used to transform medians such as household income. The first
equity design therefore combines annual repeated cross-sections with a stable
same-code panel sensitivity analysis.

Median monthly domestic household income is retained primarily as a within-year
rank or quintile. Nominal HKD differences across years are not interpreted as
real-income change without a declared price adjustment. The post-secondary
education share remains deferred until the detailed education-category mapping
is verified across all three official classifications.

The boundary audit does not resolve directed or parallel road-centreline double
counting. A separate road-to-neighbourhood aggregation rule remains required
before any total traffic-burden measure is calculated.

## Interpretation boundary

The first equity outcome is neighbourhood traffic-volume burden. It is not a direct estimate of ambient concentration, personal exposure, health effect, or causal injustice.

## External benchmark rule

Every validation statement up to Step 12 was produced by the same model that
produced the predictions. Section 3.4 of each Annual Traffic Census report
publishes the average daily vehicle-kilometrage on the census road network, split
by region and by major/minor network. This is an official aggregate for exactly
the quantity a full-network reconstruction implies, and it is now the primary
external validation target.

Official daily vehicle-kilometrage on the census network is 33,820,206 in 2011,
37,409,081 in 2016 and 38,747,164 in 2021. The official territory total therefore
rose by 10.61% between 2011 and 2016 and by a further 3.58% between 2016 and
2021. The 2021 increase means that 2021 may not be described as a period of
suppressed travel at network level; it remains a sensitivity year for other
reasons, but not that one.

## Estimand rule

The Step 11 full-network backcast implies 2.12, 2.05 and 1.89 times the official
census aggregate in 2011, 2016 and 2021. The discrepancy is not a uniform scale
error: on the strategic route network the same predictions reproduce the official
major-network total to within 8% in 2011 and 2016, while the unlabelled local
street population is overstated by roughly an order of magnitude.

The reportable estimand is therefore frozen as the strategic route network, the
1,521 current centreline features carrying an official route number. It is the
only nested subset bounded by an official published aggregate. The remaining
34,586 features are retained in the output file with `in_primary_estimand` set to
false and must not enter any reported result, figure or equity measure.

## Official calibration rule

Multiplicative region-by-year factors rake the primary estimand onto the official
regional major-network vehicle-kilometrage. This is a calibration, not a
validation: raking removes the regional level error, it does not measure it.

The factors are 1.388, 1.016 and 0.842 for Hong Kong Island, Kowloon and the New
Territories in 2011; 1.329, 0.969 and 0.864 in 2016; and 1.615, 1.091 and 1.002
in 2021. They agree in direction and rough magnitude with the Step 9 regional
out-of-fold fold biases, which were produced by an independent route. The model's
regional error is systematic and aligned with Hong Kong's income geography.

Calibration is the only point at which a year-varying official quantity enters
the reconstruction. Regional totals therefore now carry official year-to-year
information, while the within-region relative distribution is still produced by
time-invariant predictors and cannot support any segment-level change claim.

## Corrected baseline and calibration rule

A median lookup table over two road-hierarchy proxies -- strategic route number
and corridor-extent quintile, binned inside the training folds -- reaches pooled
out-of-fold MAE of 10,353, 10,118 and 9,980. The frozen nonlinear model reaches
10,044, 9,746 and 9,615. The reportable gain is therefore about 3.7% over an
honest baseline, not 36--39% over a single training median. The larger figure may
not be used as a headline.

The Step 9 quartile calibration bins by the observed value. Conditioning on the
outcome makes an apparent Q1 overprediction and Q4 underprediction arithmetically
unavoidable, so that diagnostic cannot distinguish a defective model from a noisy
one. Binned by the predicted value instead, the Q4 bias is -9.4%, -9.6% and -9.6%
and the bias is close to flat across bins. The regression slope of observed on
predicted is 1.03 to 1.08.

What the diagnostic does reveal is a whole-sample mean bias of -12.2%, -13.1% and
-13.5%. Absolute-error loss fits the conditional median while AADT is right
skewed, so every neighbourhood mean and every burden total inherits this
understatement. It must be reported alongside any aggregated result.

## Step 10 re-scoring rule

The Step 10 bundle was rejected because the Q4 mean bias measured on
observed-value bins did not improve by five percentage points. That criterion is
an artefact and cannot be satisfied by any model. Re-scored on the same folds and
the same frozen model, the official extension changes pooled MAE by +0.21%,
+0.17% and +1.25%, all inside noise, while lowering RMSE and raising R-squared in
all three years.

The recorded verdict is that the rejection used an artefactual criterion and the
bundle should be reopened. Any future feature decision must use forward selection
under a spatial design rather than whole-bundle acceptance or rejection, and must
be judged on RMSE, R-squared, aggregate bias and predicted-bin calibration.

## Year-identification rule

A model trained on one year's labels predicts another year's observations to
within 2% of the own-year model, and the 2016 model predicts 2011 observations
better than the 2011 model does. The year dimension carries no independent
information once the predictors are time invariant.

A 30-draw station bootstrap gives a mean absolute predicted change of 1,378 and
1,538 vehicles per day for 2011--2016 and 2016--2021, against a refit standard
deviation of 1,790 and 1,919. Only 3.7% and 4.8% of segments exceed two bootstrap
standard deviations.

Segment-level change maps are therefore withdrawn. Year-to-year change may be
reported only at station level from observations, and at regional level from the
official vehicle-kilometrage series. Restoring a segment-level change claim
requires predictors that themselves vary by year.

## Neighbourhood-level validation rule

Validation now exists at the unit the conclusions are drawn at. Aggregated to the
2016 Large TPU Group reference geography, 136 of 154 units contain at least one
training station. Neighbourhood-level MAE is 6,497, 6,312 and 6,308 vehicles per
day, about 30% of the observed mean, with a mean bias of -5.4%, -8.3% and -8.6%
and a Spearman rank correlation of about 0.75.

Segment errors do not cancel on aggregation. Rank agreement is adequate, so
direction may be discussed; magnitude may not be reported as a point estimate.

## Equity estimand rule

The income gradient was computed twice on the same neighbourhoods, once from
observed AADT and once from out-of-fold predictions. The lowest-minus-highest
income quintile difference changes sign in all three years.

The more fundamental finding is that the observed relationship is not monotonic.
In 2016 the observed means by income quintile are 23,394, 18,300, 19,426, 20,362
and 24,415: both the poorest and the richest units sit on high-volume corridors.
The unit-level Spearman correlation between household income and mean station
AADT is +0.010, -0.035 and -0.052 across the three years, that is, effectively
zero.

At Large TPU Group scale, with neighbourhood mean AADT as the outcome, there is
no monotonic income-traffic association in the observed data to attenuate. The
quintile difference is noise around zero, which is why its sign is unstable. This
is an estimand problem rather than a model problem: the unit is coarse enough for
modifiable areal unit effects to dominate, and the outcome is not a
population-weighted near-road burden.

No equity magnitude may be reported from the current design. The next equity step
is to change the estimand -- a population-weighted near-road measure on a finer
geography, stratified by density and urban-rural status -- not to improve the
model. The preliminary calibrated description is retained as descriptive material
only, and the interpretation boundary above continues to apply in full.

## Step 13--16 support-correction rule (supersedes parts of the entries above)

Status: current. The earlier External benchmark, Estimand, Official calibration,
Neighbourhood-level validation and Equity estimand entries remain in this file
for audit history, but the conflicting decisions below supersede them.

Official vehicle-kilometrage must be paired with the Appendix H trafficable road
length. The current full centreline is 3,850.95 km, compared with official census
support of 1,813.25, 1,859.51 and 1,922.67 km. Once each VKT total is divided by
its own support length, the model-to-official mean-AADT ratios are approximately
1.00, 0.99 and 0.94. The apparent twofold VKT discrepancy therefore cannot be
used to infer an order-of-magnitude local-road AADT error.

The route-number subset is not the official major road network. In 2016 it is
520.72 km, while Appendix H reports 1,039.12 km of official major roads. The
official definition is the CTS simplified road network, not the presence of a
route number. The previous E1 freeze is revoked: `primary_estimand_frozen = 0`.
No subset may be raked to the full official major total until a matched official
major/minor centreline support has been constructed. The previous regional
calibration factors and calibrated E1 equity description are not reportable.

The official aggregate remains useful as a published consistency constraint,
but it is not an independent label set because Appendix K derives VKT from ATC
AADT and road length. Official VKT change also combines network-length change
with traffic-intensity change; it is not a same-support segment trend target.

The Step 10 observed-bin rejection is superseded, but the official feature bundle
is not automatically retained. Its aggregate changes are small and mixed across
metrics. Individual features may be reopened only under fold-internal or nested
forward selection; the full bundle is not promoted to the frozen baseline.

At Large TPU Group scale, mean station AADT has no stable monotonic income signal.
The Q1-minus-Q5 unit-resampling sensitivity interval includes zero, and the
unit-level Spearman association is near zero. An attenuation factor is therefore
undefined. Overall neighbourhood rank agreement does not validate the direction
of an income-group contrast. Neither equity direction nor magnitude is reportable
under the current estimand.

MAUP is a plausible explanation, not a demonstrated result of the current
single-scale analysis. Station selection and heterogeneous stations per unit are
additional explanations. The next equity gate is a population-weighted near-road
traffic-activity measure on finer geography. The separate temporal gate remains:
segment trends require year-varying predictors or must be labelled as officially
constrained scenarios rather than learned backcasts.

## Step 19 long-horizon temporal-identification rule

The annual Step 17--18 result is now tested at the five-year horizons used by the
pilot. Step 19 uses all recommended high-confidence measured pairs separately:
799 for 2011--2016 and 812 for 2016--2021, rather than requiring membership in
the 679-station three-year intersection. The latest official coordinates assign
the frozen spatial folds and current major/minor stratum; they are not treated as
historical geometry.

Official VKT growth is kept separate from official traffic-intensity growth. The
latter is VKT divided by the corresponding Appendix H road-network length. Across
the primary region-by-network strata, road-length change makes the two growth
signals differ by as much as 5.80 percentage points, so VKT growth may not be
applied directly to an existing segment as an AADT growth factor.

Even when the observed first-year AADT is supplied, no tested official factor,
fold-trained stratum change, or spatial residual model beats no change across
both adjacent five-year transitions. The best 2011--2016 candidate is the
region-by-network intensity factor (MAE 2,371 versus 2,274 for no change); the
best 2016--2021 candidate is the territory intensity factor (2,449 versus 2,441).
No model passes the frozen requirement of positive improvement, positive change
correlation and a loss-difference interval below zero in both transitions.

The held-out-location task also fails. Applying the official region-by-network
intensity factor to a spatially predicted first-year level gives change MAE of
2,393 versus 2,274 in 2011--2016 and 2,569 versus 2,441 in 2016--2021. Independent
year models are worse. The 2011--2021 sensitivity result does not alter the gate:
its small apparent improvement is not separated from zero by the resampling
interval and does not repair the two primary transitions.

The reportable decision is therefore: official aggregates identify temporal
change at their published region-and-network support, but no validated rule
downscales that change to individual segments. A full-network segment backcast
may be shown only as an officially constrained scenario, not as an empirically
identified historical reconstruction. Multi-year equity trends may not be built
from those segment changes until a new temporal signal passes an equivalent
held-out-location gate.

## Step 20 external-data selection and Niu 2023 rule

Step 20 separates data availability from inferential validity. Recent studies
that successfully estimate local-road traffic or multi-year exposure have either
direct local-road labels, repeated dynamic sensors, historical traffic/emission
inventories, or some combination of these. A large static feature set is not a
substitute for a measured time signal. Of the reviewed products, only Niu et al.
provide a directly usable Hong Kong link-level release.

The three public Niu files contain 17,706 directed 2023 OSM-link rows with the
same link keys, eight vehicle classes, 16 hourly fields from 07 through 22, and
matched traffic, NOx and PM2.5 outputs. They are suitable as one 2023 product,
not as three independent sources. Direction must be handled explicitly: 27.5%
of rows belong to endpoint pairs with multiple directed records, their AADT is
usually duplicated across directions, and the released hourly window sums to a
median 87.0% of AADT. It must not be treated as an automatically additive
24-hour directed total.

The Niu links can be connected to the current project: high/moderate matches
cover 71.6% of current TD centreline length and 85.7% of directly measured 2023
stations. This demonstrates geographic usability, not independent predictive
accuracy. Among matched stations, 85.3% of Niu AADT values are exactly equal to
the ATC values. Niu et al. use ATC 2023 and the public release does not expose
out-of-fold predictions or station membership, so the apparent agreement is
principally evidence of shared label lineage.

The approved use is a 2023 cross-sectional benchmark and a population-weighted
near-road equity proof of concept on finer geography, with buffer, weighting and
density sensitivity. It can improve the traffic-activity and exposure estimand
through hourly vehicle classes and link emissions. It cannot validate this
project's AADT model independently, infer 2011--2016--2021 link changes, or make
the failed temporal gate pass. Rebuilding another 2023-only feature stack would
add cross-sectional detail but would not identify historical change. Multi-year
backcasting still requires archived year-varying inputs and held-out temporal
validation.
