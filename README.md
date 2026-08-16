# Hong Kong Historical AADT Backcasting Pilot

## Project status

This repository documents a feasibility study of reconstructing road-segment annual average daily traffic (AADT) in Hong Kong using public data.

The current evidence supports a **bounded cross-sectional reconstruction within the measured and feature-support domain**. OpenStreetMap (OSM) road class adds reproducible predictive skill in 2023. However, the predeclared gates for **full-network reconstruction, segment-level temporal backcasting, and multi-year equity trends have not passed**. Outputs that fail a gate are retained for audit, not presented as validated estimates.

## Questions from the pilot brief

| Question | Current answer |
| --- | --- |
| Can public data reconstruct AADT at measured or well-supported road segments? | **Partly.** OSM road class improves spatial out-of-fold prediction, especially for observed minor-road stations. |
| Can the model be extended credibly to the whole road network? | **Not yet.** Local/service roads lack representative traffic-count labels, and subgroup bias remains too large. |
| Can annual segment-level changes be backcast? | **Not yet.** Static and historical road attributes do not beat a no-change benchmark for measured AADT change. |
| Can current outputs support a multi-year equity trend? | **Not yet.** Large TPU Group mean AADT is too coarse and does not show a stable monotonic income gradient. |

## Public data used

- Hong Kong Transport Department Annual Traffic Census (ATC) station records and road-network files
- Official daily vehicle-kilometre and covered-road-length tables
- Annual measured-station records for 2018–2024
- Historical and current OSM road attributes
- Public strategic-detector and vehicle-class traffic archives
- GTFS public-transport service data
- Hong Kong census geography and socioeconomic variables
- The public 2023 traffic and emissions release associated with Niu et al., used as a descriptive benchmark rather than independent validation

Large raw files and downloaded source archives are not committed. The scripts preserve source and extraction audits so that data provenance can be checked without treating derived products as independent evidence.

## Approach and validation

The workflow builds station-year panels, links stations to the official centreline, constructs deployable road and transport-context features, and evaluates models with frozen spatial folds. Validation is deliberately separated into four questions:

1. **Spatial skill:** out-of-fold prediction at measured stations.
2. **Deployability:** prediction features must exist for unmonitored road segments; ATC-only road-class labels and station-matching diagnostics are excluded.
3. **External consistency:** predicted network totals are checked against official vehicle-kilometres and covered road length.
4. **Temporal identification:** annual changes must improve on a no-change benchmark, not merely reproduce traffic levels.

Simple baselines, subgroup bias, predicted-bin calibration, label support and manually adjudicated road matches are reported alongside model accuracy.

## Main findings

### 1. OSM adds real cross-sectional information

For the 2023 measured-station sample, the deployable OSM feature block reduced MAE relative to the two-variable road-hierarchy median lookup by **6.8%** (four of five spatial folds) and added **6.2%** beyond the deployable road/GTFS context model (all five folds). It reduced minor-road station MAE by **24.4%**.

This is the first external, network-wide feature source in the pilot to produce a material and deployable gain. The gain should not be interpreted as a passed full-network gate.

### 2. The unmonitored local-road domain remains unsupported

The primary absolute-error OSM model still overpredicted observed minor-road AADT by **31.0%**. The OSM `service` group covers a material share of the centreline but has only five measured stations; its class-specific effect is therefore not identifiable under the frozen model settings. Predictions in this group rely on contextual patterns learned mainly from other road classes.

A blind, stratified manual review of 100 station-to-road links found 93 correct, six adjudicable errors and one indeterminate case. No errors were observed in the sampled minor-road or Hong Kong Island strata, although the sample is too small to claim zero population error. This makes obvious geometry mismatch an unlikely explanation for the full subgroup bias; label coverage and transportability remain the main unresolved limitations.

### 3. Mean-targeting losses improve overall totals but not subgroup validity

The Poisson-loss model reduced overall aggregate bias to **-4.3%**, but failed the predeclared subgroup-bias, prediction-bin calibration and RMSE non-inferiority gates. Its maximum region/network bias was **38.0%**. The absolute-error model therefore remains the spatial-skill model, while neither model is authorised for full-network aggregation or equity estimation.

### 4. Official external checks rejected an early full-network estimand

Official territory-wide daily vehicle-kilometres were **33.82**, **37.41** and **38.75 million vehicle-km/day** in 2011, 2016 and 2021. An early full-network reconstruction was roughly twice the official total. Joint auditing of vehicle-kilometres, covered road length and implied length-weighted AADT showed that the discrepancy mainly followed incompatible road support. That early estimand and its major-road comparison were withdrawn rather than calibrated after the fact.

### 5. Historical OSM is not an annual traffic signal

Across 2,784 measured station pairs from 2018–2024, **84.4% of total absolute AADT change occurred where the matched OSM road tag did not change**. The association between an OSM tag-change indicator and absolute AADT change was close to zero (Spearman **0.026**). OSM is useful as road context, but its editing history cannot identify annual traffic change or distinguish mapping edits from physical network changes.

The public Overpass attic cannot supply a valid 2011 state. The practical public-data overlap window is 2018–2024, with 2016 as a supplementary census year; even within that window, measured-station OSM coverage is currently below the predeclared threshold in some years.

### 6. The current equity estimand is not adequate

At Large TPU Group scale, observed mean station AADT has no stable monotonic association with household income. The relationship is non-linear and near-zero in rank correlation, while neighbourhood-level prediction error remains material. This result does not contradict population-weighted near-road exposure studies: mean AADT over a large administrative unit is a different estimand.

The next equity analysis should use a finer population geography and a population-weighted near-road traffic or emissions burden. It should only be interpreted longitudinally after a separate temporal-identification gate passes.

## Decision summary

**Supported now**

- Reproducible 2023 cross-sectional prediction experiments at measured stations
- A deployable OSM road-class feature contribution within the observed support domain
- External consistency auditing with official aggregate traffic statistics
- A clear diagnosis of where public-data support is inadequate

**Not supported now**

- Validated AADT estimates for every Hong Kong road segment
- Segment-level annual backcasting or change maps
- A multi-year socioeconomic/equity trend based on the current exposure definition
- Treating the Niu et al. product as independent validation when it shares ATC inputs

## Key reproducibility entry points

Run from the repository root with the project environment activated:

```powershell
python src/13_extract_official_vkt_benchmark.py
python src/14_validate_network_against_official_vkt.py
python src/15_recheck_model_evidence.py
python src/16_neighbourhood_validation_and_equity_check.py
python src/17_build_annual_panel_and_test_temporal_identification.py
python src/18_validate_full_measured_station_sample.py
python src/19_validate_long_horizon_dynamic_backcast.py
python src/20_audit_external_traffic_data_and_niu_2023.py
python src/21_audit_2023_reconstruction_data_and_leakage.py
python src/22_test_2023_dynamic_reconstruction.py
python src/23a_test_2023_osm_road_class.py
python src/23a1_audit_osm_support_and_mean_models.py
python src/23b_audit_historical_osm_data.py
```

Step 23A.1's blind review atlas is generated separately with:

```powershell
python src/23a1b_build_blind_review_atlas.py
```

The principal machine-readable conclusions are in:

- `outputs/tables/step23a1_decision_audit.csv`
- `outputs/tables/step23a1_subgroup_bias.csv`
- `outputs/tables/step23a1_blind_match_review_results.csv`
- `outputs/tables/step23b_data_decision_audit.csv`
- `outputs/tables/step23b_aadt_change_alignment.csv`
- `docs/research_decisions.md`

## Next evidence needed

1. Representative independent counts for local and service roads, potentially from public-sector video analytics, smart-lamppost sensors, traffic-impact assessments, estate or car-park access counts, or a designed short-duration count sample.
2. A genuine year-varying traffic signal, such as a consistent annual detector archive, screenline flows, vehicle-class counts or public-transport operating intensity.
3. A finer, population-weighted near-road exposure estimand for the socioeconomic analysis.

These are proposed evidence-building directions, not assumptions that adding more data will necessarily make the full-network or temporal gates pass.
