# Quantifying PV Module Mismatch Losses in Real-World Rooftop Systems Using Optimiser Data

## Introduction

* Mismatch losses in rooftop PV systems significantly impact performance due to complex roof geometries, partial shading, and non-uniform orientations
* Previous studies present conflicting conclusions on mismatch loss magnitude, with limited comprehensive real-world field data across diverse conditions
* SolarEdge optimiser technology provides unprecedented access to module-level maximum power point (MPP) data from operational installations
* This study quantifies real-world mismatch losses using optimiser-derived MPP data across 22 diverse rooftop systems spanning multiple continents and climate zones

## Methodology

* **Novel 4-step approach:** (1) I-V curve reconstruction from measured MPP using single-diode model, (2) series connection I-V simulation, (3) combined MPP identification, (4) mismatch loss calculation
* **Single-diode model foundation:** Reconstructed full I-V curves from optimiser MPP measurements using established photovoltaic modelling equations
* **Mismatch quantification:** Mismatch Loss (%) = (Sum of Individual MPP - Series Connection MPP) / Sum of Individual MPP × 100
* **Data quality assurance:** Bypass diode activation events filtered using Voc/Isc outlier detection to isolate true mismatch losses
* **Comprehensive dataset:** 22 installations across Australia, North America, and Europe; 4 seasonal periods; 5-minute measurement intervals
* **Methodology validation:** Monte Carlo simulation using two-diode reference model confirms ±1.2% accuracy of proposed approach
* **Figure 1: Methodology Validation**
  + Representative measured MPP data from optimiser telemetry
  + Reconstructed I-V curves using single-diode model approach
  + Series connection analysis showing mismatch loss quantification

## Results

### Overall Mismatch Loss Quantification

* **Mean mismatch loss:** 14.4 ± 5.0% across 88 site-season combinations after bypass diode filtering
* **Performance range:** 5.3% (best performing single orientation) to 22.0% (worst performing multiple orientation system)
* **Geographic consistency:** Results validated across diverse climate zones and installation conditions

### Critical Design Impact: Orientation Strategy

* **Single orientation installations (16 sites):** 12.6 ± 4.6% average mismatch loss
* Uniform irradiance conditions reduce orientation-related mismatches
* Representative performance: Site 4034376 achieving 7.5% loss
* **Multiple orientation installations (6 sites):** 19.1 ± 2.2% average mismatch loss
* Architectural constraints requiring mixed orientations create additional mismatch sources
* Representative performance: Site 3455043 with 19.0% loss
* **Quantified design penalty:** Multiple orientations add ~6.5% additional mismatch compared to single orientation systems
* **Figure 2: Orientation Impact Analysis**
* Histogram comparing mismatch loss distributions between single orientation (12.6 ± 4.6%) and multiple orientation (19.1 ± 2.2%) installations
* Clear visualisation of 6.5% design penalty for architectural constraints requiring mixed orientations

### Seasonal and Environmental Effects

* **Winter performance degradation:** Higher mismatch losses observed due to lower solar altitude angles increasing shadowing effects
* **Irradiance uniformity impact:** Seasonal variation of 2-4% between winter and summer performance
* **Installation-specific factors:** Complex roof geometries and nearby obstructions significantly influence seasonal mismatch patterns
* **Figure 3: Seasonal Performance Patterns**
* Seasonal mismatch loss variation across geographic locations showing winter degradation effects
* Climate zone impact on seasonal performance consistency

## Conclusions and Design Implications

* **Novel methodology and validation:** First comprehensive field study using optimiser-derived MPP data for direct mismatch quantification with validation across 22 sites on multiple continents and Monte Carlo accuracy confirmation
* **Critical design insight:** Multiple orientation installations incur ~6.5% performance penalty compared to single orientation designs, with single orientation planning recommended where architecturally feasible
* **Technology validation and industry impact:** Quantified benefits of module-level power electronics for complex rooftop installations, providing actionable design guidance for rooftop PV system optimisation, performance modelling, and technology selection decisions