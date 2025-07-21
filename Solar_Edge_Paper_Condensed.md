# Quantifying Photovoltaic Module-to-Module Mismatch Losses with Real-World Rooftop Systems

**Authors:** Sijin Wang, Ziv Hameiri, Baran Yildiz, James Sturch

## Novelty Points

- Novel method to estimate mismatch losses from real-world SolarEdge MPPT data
- Comprehensive benchmark across 22 diverse rooftop systems (multiple countries/climates)
- Advanced bypass diode filtering methodology
- Monte Carlo validation framework (±1.2% error)

## Introduction

### Commercial PV Overview
- PV essential for climate change mitigation with exponential growth
- Rooftop PV suffers more mismatch losses than utility-scale (complex geometries, shading, non-uniform orientations)

### Mismatch Loss Introduction
- **Definition:** Difference between sum of individual module MPPs and actual string power output
- **Sources:** Manufacturing variations, aging, soiling, temperature differences, orientation variations, PID, bypass diode activation
- **Literature gap:** Conflicting conclusions, limited real-world field studies

### SolarEdge Technology Introduction
- DC-DC optimizers enable independent module MPP operation
- Provide real-time MPP data, monitoring, fault detection (5-minute intervals)
- Creates unprecedented opportunity for real-world mismatch assessment

### Study Objectives
- **Primary:** Quantify mismatch losses from real-world SolarEdge MPP data
- **Secondary:** Validate single-diode methodology, categorize patterns, analyze seasonal/geographic variations, develop bypass diode filtering

## Methodology

### Overview of Proposed Method
Four sequential steps:
1. Reconstruct I-V curve from measured MPPT using single-diode model
2. Calculate combined series I-V curve
3. Find MPPT of combined series I-V curve
4. Compare series vs. individual energies to calculate mismatch loss

### I-V Curve Reconstruction
- **Single-Diode Model Foundation:** Standard 5-parameter model
- **Parameter Extraction:** From STC ratings and measured MPPT data
- **Implementation:** Temperature/irradiance corrections applied

### Series Connection I-V Simulation
- Combine individual I-V curves: constant current, additive voltage
- Account for bypass diode effects and voltage limitations

### Power Comparison and Mismatch Calculation
$$\text{Mismatch Loss (\%)} = \frac{\sum P_{individual} - P_{series}}{\sum P_{individual}} \times 100$$

### Bypass Diode Filtering
- **Challenge:** Diode activation creates artificial mismatch signals
- **Detection:** Voc/Isc outlier analysis using IQR methods
- **Classification:** Type 1 (1 diode), Type 2 (2 diodes), Type -1 (variable)
- **Impact:** 10-15% timestamps excluded

#### Bypass Diode Detection Algorithm
- Voc outlier identification (IQR-based)
- Classification thresholds: >Q3+1.5×IQR (fault), <2/3×Q3 (1 diode), <1/3×Q3 (2 diodes)
- Cross-validation with Isc measurements

## Test Site Selection and Aggregation

### Site Characteristics
- **Total:** 22 installations across Australia, North America, Europe
- **Geographic diversity:** Multiple countries and climate zones
- **System specifications:** Detailed in comprehensive database (Table 1)

### Data Collection Protocol
- **Temporal sampling:** 10 days per season (March, June, September, December 2024)
- **Frequency:** 5-minute intervals
- **Parameters:** Module/ambient temperature, MPP current/voltage, power output, inverter data

### Site Classification System
#### Category 1: Single Orientation (16 sites)
- Uniform direction, tilt, azimuth
- Mismatch sources: manufacturing tolerances, temperature gradients, soiling, partial shading

#### Category 2: Multiple Orientation (6 sites)
- Mixed directions due to complex roof geometry
- Additional mismatch: varying irradiance from orientation differences

## Results

### Overall Statistical Analysis
- **Dataset:** 22 sites × 4 seasons × 10 days = 880 site-season combinations
- **Mean mismatch loss:** 14.4 ± 5.0% (after bypass diode filtering)
- **Range:** 5.3% (single orientation) to 22.0% (multiple orientation)

### Orientation-Based Results

#### Case 1: Single Orientation Sites (16 sites)
- **Average mismatch loss:** 12.6 ± 4.6%
- **Range:** 5.3% to 21.5%
- **Sources:** Manufacturing tolerances, temperature gradients, soiling, partial shading, cloud effects

#### Case 2: Multiple Orientation Sites (6 sites)
- **Average mismatch loss:** 19.1 ± 2.2%
- **Range:** 17.0% to 22.0%
- **Additional impact:** ~6.5% penalty from orientation differences

### Seasonal Analysis
- **Winter conditions:** Elevated mismatch losses
- **Geographic patterns:** Climate-dependent variations observed

### Geographic and Climate Analysis
- **Köppen-Geiger classification:** Systematic patterns revealed
- **Latitude dependency:** Some correlation with absolute latitude
- **Global applicability:** Results span diverse climate conditions

## Error Estimation Using Monte Carlo Simulation (MCS)

### Motivation for Validation
- Assess single-diode model accuracy
- Quantify parameter dependencies (temperature/irradiance effects)

### Monte Carlo Methodology
#### Two-Diode Reference Model
- Generate 10,000 dummy modules with realistic parameter distributions
- Create 1000 dummy systems (10 modules each)
- Apply temperature/irradiance variations

#### Validation Process
- **Branch 1:** True results from two-diode model
- **Branch 2:** Reconstructed results from single-diode methodology
- Compare mismatch loss estimates

### Monte Carlo Results
- **Error quantification:** Mean error 1.2% absolute (systematic underestimation)
- **Uncertainty bounds:** ±1.2% confidence interval
- **Validation:** Single-diode approach sufficient for field study accuracy

#### Monte Carlo Simulation Parameters
- **Module distributions:** Normal (Rs), log-normal (Rsh), uniform (n), literature-based (temperature coefficients)
- **Environmental conditions:** 15-85°C, 200-1200 W/m², realistic diurnal/seasonal variations

#### Error Analysis Results
- **Statistical metrics:** 1.2% mean absolute error, 0.8% standard deviation, ±1.6% 95% confidence interval
- **Conclusions:** Single-diode sufficient, minimal temperature dependency impact, proper irradiance effect capture

## Discussion

### Comparison with Previous Studies

#### NREL Distributed Power Electronics Research
**MacAlpine et al. (2009):**
- Building-integrated PV with module-integrated DC-DC converters
- >10% power gains for differing panel orientations
- Simulation model validated with experimental data

**Deline (2010):**
- Multi-string PV systems with/without DC-DC converters
- 5-10% annual power production improvement
- Up to 40% of shading losses from voltage mismatch between parallel strings

**Deline (2011):**
- Comprehensive mismatch loss estimates: 5-15% (residential shade), 5-20% (multi-string), 5-20% (orientation), 1-5% (commercial), 0.2-1% (manufacturing), 1.5-6.2% (soiling)

#### Comparison with Current Study
- **Current findings:** 14.4 ± 5.0% mean loss (22 sites)
- **Convergence:** Results align with NREL's 5-20% estimates
- **Methodology difference:** NREL (controlled experiments/simulations) vs. Current (real-world field data)
- **Validation:** Both studies confirm ~15% mismatch losses in rooftop systems

### Implications for System Designers
- **Orientation planning:** Single (12.6%) vs. multiple (19.1%) orientation difference
- **Installation strategy:** Multiple orientations add ~6.5% mismatch penalty
- **Technology selection:** Quantified benefits of module-level power electronics

## Conclusion

### Key Findings Summary
- **Real-world mismatch losses:** 14.4 ± 5.0% average across 22 diverse installations
- **Orientation impact:** 6.5% additional mismatch for multiple orientations
- **Methodology validation:** ±1.2% error through Monte Carlo simulation
- **NREL alignment:** Results consistent with established research (5-20% range)

### Scientific Contributions
- Novel real-world mismatch quantification methodology
- Comprehensive field study across multiple countries/climates
- Advanced bypass diode filtering algorithm
- Validation of single-diode approach for field studies

### Industry Impact
- Quantified benefits of module-level power electronics
- Design guidelines for orientation planning
- Performance modeling improvements for real-world conditions
- Evidence-based recommendations for complex installations

## Appendices

### Appendix A: Derivation of Dark Saturation Current
#### Known Parameters
- STC ratings, temperature coefficients, ideality factor

#### Mathematical Derivation
- Analytical solution for I₀ from STC conditions
- Temperature scaling relationships

### Appendix B: Bypass Diode Filtering Algorithm
#### Statistical Outlier Detection
- Voc/Isc outlier classification using IQR methods
- Threshold definitions for diode activation states

#### Activation Classification Algorithm
- Decision tree for diode activation types
- Cross-validation with current measurements

### Appendix C: Temperature and Irradiance Dependencies
#### Series Resistance (Rs)
- Temperature dependency: 0.356%/K (PERC modules)
- Physical basis: metal grid resistance increase

#### Shunt Resistance (Rsh)
- Temperature dependency: negligible
- Irradiance dependency: PVsyst exponential model

#### Ideality Factor (n)
- Temperature dependency: minimal (0.006%/K)
- Irradiance dependency: 1.4-1.5 for 0.4-1.0 sun conditions