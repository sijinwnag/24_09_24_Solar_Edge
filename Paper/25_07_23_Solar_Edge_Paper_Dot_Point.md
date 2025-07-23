# Quantifying Photovoltaic Module-to-Module Mismatch Losses with Real-World Rooftop Systems

**Authors:** Sijin Wang, Baran Yildiz, James Sturch, Ziv Hameiri

## Novelty Points

- Development of a novel method to estimate mismatch losses based on the recorded MPPT of each module from real-world SolarEdge systems
- Comprehensive benchmark of module-to-module mismatch losses across 22 diverse rooftop systems spanning multiple countries and climate zones

## Introduction

### Commercial PV Overview

- PV is essential for combating climate change, with exponential growth in deployment
- Recent rapid growth in rooftop PV installations globally
- Rooftop PV systems typically suffer more from mismatch losses compared to utility-scale installations due to:
  - Complex roof geometries
  - Partial shading from nearby objects
  - Non-uniform orientations
  - Installation constraints

### Mismatch Loss Introduction

**Definition:** Mismatch loss in a PV system series connection is the difference between the sum of individual module maximum power points and the actual combined string power output

**Sources of mismatch losses in rooftop PV:**
- Manufacturing inconsistencies and parameter variations
- Light-induced degradation (LID) and ageing effects
- Module ageing at different rates
- Soiling and temperature differences across the array
- Site-specific conditions: partial shading, varying module orientations
- Potential-induced degradation (PID) and damaged module diodes
- Bypass diode activation events

**Literature Review:**
- Historically, studies on mismatch losses have presented conflicting conclusions
- Some research indicates mismatch losses significantly reduce system output [refs]
- Other studies argue that these losses have minimal impact on overall performance [refs]
- A gap exists in comprehensive real-world field studies across diverse conditions

### SolarEdge Technology Introduction

To mitigate mismatch losses, SolarEdge Technologies has developed DC-DC converters known as optimisers

**Key features:**
- Allow modules to operate independently at their maximum power points (MPP)
- Provide real-time MPP data on each module's operating conditions
- Enable module-level monitoring and fault detection
- Continuous 5-minute interval telemetry data collection

**Research Opportunity:**
- The extensive data available from SolarEdge optimisers presents an unprecedented opportunity to assess real-world PV mismatch losses
- Large-scale deployment enables statistical analysis across diverse geographic and climatic conditions

### Study Objectives

Quantify mismatch losses accurately based on MPP data recorded by SolarEdge systems across diverse real-world conditions

## Methodology

### Overview of Proposed Method

The proposed method consists of four sequential steps:

1. Reconstruct the I-V curve of each module based on the measured MPPT using the single-diode model.
2. Calculate the combined I-V curve if the modules were connected in series
3. Find the MPPT of the combined series I-V curve
4. Compare the energy produced by a series connection vs. the sum of individual module energies to calculate mismatch loss

### I-V Curve Reconstruction

#### Single-Diode Model Foundation

The single-diode equation is given by:

```
I = I_L - I_0 * [exp((V + I*R_s)/(n*N*V_th)) - 1] - (V + I*R_s)/R_sh
```

Where:
- I = module output current
- I_L = photogenerated current
- I_0 = dark saturation current
- n = diode ideality factor
- N = number of cells in the module
- V_th = thermal voltage = kT/q
- k = Boltzmann constant
- T = cell temperature in Kelvin
- q = electron charge
- V = module output voltage
- R_s = module series resistance
- R_sh = module shunt resistance

#### Parameter Extraction

Based on the single-diode equation and assuming measured voltage and current represent the maximum power point, the dark saturation current can be derived as:

```
I_0 = [I*(1+R_s/R_sh) - V/R_sh] / [1 - I*R_s/V] * (n*N*V_th/V) * exp(-(V+I*R_s)/(n*N*V_th))
```

The photogenerated current is calculated by rearranging equation (1):

```
I_L = I*(1+R_s/R_sh) + V/R_sh + I_0*[exp((V+I*R_s)/(n*N*V_th)) - 1]
```

#### Implementation Process

For each measurement timestamp:
- Parameters extracted from PV module datasheet (.PAN files)
- V_th calculated based on the measured module temperature
- I, V recorded by the SolarEdge optimiser
- I_0 and I_L calculated using equations (2) and (3)
- Full I-V curve constructed using pvlib.pvsystem.v_from_i function

An example of a reconstructed IV curve from the measured MPP is shown in Figure 1 (a) and (b)

### Series Connection I-V Simulation

- **Methodology:** Simulated I-V curve constructed assuming all modules are connected in series
- **Bypass diode modelling:** Each module is assumed to have at least one bypass diode with a negligible voltage drop
- **Mathematical approach:** Series connection I-V curve calculated as the voltage sum for each current value
- **Current limiting:** When combined current modules into reverse voltage, bypass diodes activate

An example of a series connection IV curve from the individual module IV is shown in Figures 1 (b) and (c)

**Figure 1:** (a) The recorded MPP for each module, (b) The reconstructed IV curve for each module, (c) The reconstructed IV curve if they were connected in series

### Power Comparison and Mismatch Calculation

- MPP of the series connection I-V curve identified through power curve optimisation
- Comparison performed between:
  - Sum of individual module MPP powers (SolarEdge case)
  - Combined series string MPP power (traditional string inverter case)

**Mismatch loss calculation:**
```
Mismatch Loss (%) = (Sum of MPP - Series MPP) / Sum of MPP × 100
```

An example of the power comparison of one test site for 10 days is shown in Figure 2

**Figure 2:** Power comparison between the case of each maximum power and the power if they were connected in series.

### Bypass Diode Filtering

**Motivation:** Internal bypass diode activation creates incorrect MPP signals

**Detection methodology:**
- The rule is defined based on LTSpice simulation
- Voc outlier analysis using IQR methods
- Isc outlier detection for fault identification

**Classification system:**
- Type 1: 1/3 Voc loss (1 diode activated)
- Type 2: 2/3 Voc loss (2 diodes activated)
- Type -1: High Voc + Low Isc (variable activation)

**Data filtering:** Timestamps with diode activation excluded from mismatch analysis

Please see Appendix B for the detailed bypass diode filtering algorithm

## Test Site Selection and Aggregation

### Site Characteristics

- **Total sites:** 22 selected installations with diverse orientations and technologies
- **Geographic distribution:**
  - Australia: Queensland, Victoria, South Australia, New South Wales
  - North America: Texas, Arizona, Nevada, California, Ohio, and Iowa
  - Europe: Netherlands, Germany, France
- **System specifications:** Detailed in the comprehensive site database are shown in Table 1

**Table 1:** Site database details

### Data Collection Protocol

- **Temporal sampling:** 10 days selected per season (March, June, September, December 2024)
- **Measurement frequency:** 5-minute intervals for all parameters
- **Recorded parameters:**
  - Module temperature and ambient temperature
  - Module MPP current and voltage
  - Power output per optimiser
  - Inverter-level data

### Site Classification System

Based on visual analysis using satellite and street view imagery, sites are classified into two categories based on module orientation:

#### Category 1: Single Orientation (16 sites)
- All modules face the same direction
- Uniform tilt and azimuth angles across the installation
- **Mismatch sources:** manufacturing tolerances, temperature gradients, soiling variations, and potential partial shading effects
- Common in standard residential and commercial installations

#### Category 2: Multiple Orientation (6 sites)
- Modules face different directions due to complex roof geometry
- Mixed tilt and/or azimuth angles across the installation
- **Additional mismatch sources:** varying irradiance conditions due to orientation differences
- Typically found in installations with architectural constraints or complex roof designs

## Results

**Table 2:** Summary of mismatch results

### Overall Statistical Analysis

The mismatch results of all test sites are shown in Table 2

- **Comprehensive dataset:** 22 sites × 4 seasons = 88 site-season combinations
- **Mean mismatch loss:** 14.4 ± 5.0% (after bypass diode filtering)
- **Range:** 5.3% (Site 4034376, single orientation) to 22.0% (Site 4111492, multiple orientation)
- **Distribution characteristics:** Site-specific variation attributed to local installation conditions and orientation complexity

### Orientation-Based Results

The comparison of the mismatch losses between single orientation sites and multi orientation sites is shown in Figure 3

**Figure 3:** Mismatch loss comparison between single and multi-orientation sites.

#### Case 1: Single Orientation Sites (16 sites)
- **Representative examples:** Site 4034376 (7.5% loss), Site 4002138 (9.2% loss)
- **Average mismatch loss:** 12.6 ± 4.6%
- **Range:** 5.3% to 21.5%
- **Installation characteristics:** Uniform orientation provides consistent irradiance conditions, but is still subject to environmental factors such as shading

#### Case 2: Multiple Orientation Sites (6 sites)
- **Representative examples:** Site 3455043 (19.0% loss), Site 4111492 (22.0% loss)
- **Average mismatch loss:** 19.1 ± 2.2%
- **Range:** 17.0% to 22.0%
- **Impact quantification:** ~6.5% additional mismatch compared to single orientation sites
- **Installation reality:** Common when architectural constraints require multiple orientations

#### Comparison
- Single orientation sites generally have lower mismatch loss
- However, if there is significant shading nearby, the upper limit of the mismatch of both single orientations can be close to multiple orientation sites

### Seasonal Analysis

**Figure 4:** The mismatch losses comparison between each season.

The mismatch loss comparison between each season is shown in Figure 4

- **Winter pattern:** Generally, higher mismatch losses are observed
- **Physical explanation:** Lower solar altitude increases shadowing effects and reduces overall irradiance uniformity
- **Summer stability:** More consistent performance with reduced mismatch variation
- **Seasonal variation:** 2-4% difference between winter and summer averages

## Error Estimation Using Monte Carlo Simulation (MCS)

### Motivation for Validation

**Model accuracy assessment:** Single-diode model approximation needs quantitative validation

**Parameter dependencies:** Temperature and irradiance effects on model parameters:
- Ideality factor variations
- Series resistance temperature coefficient
- Shunt resistance irradiance dependency

### Monte Carlo Methodology

#### Two-Diode Reference Model
- **Truth generation:** Use a comprehensive two-diode model with temperature/irradiance dependencies
- **Parameter distributions:** Based on literature review and manufacturer data
- **System simulation:**
  - Generate 10,000 dummy modules with realistic parameter distributions
  - Create 1000 dummy systems (10 modules each)
  - Apply temperature and irradiance variations

#### Validation Process

**Branch 1: True Results Generation**
- Calculate actual series connection I-V curves using the two-diode model
- Account for temperature/irradiance parameter dependencies
- Determine true series MPP power - Calculate true mismatch loss

**Branch 2: Reconstructed Results**
- Extract only MPP data (simulating SolarEdge measurements)
- Apply proposed single-diode reconstruction methodology
- Calculate reconstructed mismatch loss
- Compare with true values

### Monte Carlo Results

**Figure 5:** Monte Carlo simulation results

The results of the Monte Carlo simulation are shown in Figure 5

- **Error quantification:** Mean error of 1.2% absolute (methodology underestimates true mismatch)
- **Error distribution:** Systematic bias toward underestimation
- **Uncertainty bounds:** ±1.2% confidence interval for mismatch loss estimates
- **Model validation:** Confirms single-diode approach is sufficiently accurate for field studies

## Discussion

### Comparison with Previous Studies

#### Deline et al. (2010) - University of Colorado/NREL Study:

- **Research question:** how much shading loss is for a typical residential PV system under real-world shading conditions
- **Method:** Applied experimental shade impact factors to a 3kW residential system model using detailed site surveys and NREL's PVWatts with TMY3 weather data
- **System:** 14-module residential installation with two parallel strings
- **Results:** experiencing 10% annual mismatch loss from tree shading, peaking before 10 AM and after 2 PM

#### Current study findings:
- 14.4 ± 5.0% mean mismatch loss in real-world field installations (22 sites)
- Single orientation: 12.6 ± 4.6% (16 sites)
- Multiple orientation: 19.1 ± 2.2% (6 sites)

#### Key differences in methodology:
- **NREL approach:** Controlled experiments with validated simulation models to simulate the electric loss from the optical measurement
- **Current study approach:** Real-world field electrical data from SolarEdge optimiser measurements across 22 diverse installations
- This study also uses a number of sites across the globe, while NREL studies focus on a single test site

#### Validation of mismatch loss estimation:
- Despite different data sources and methodologies
- Both studies agree within the uncertainty range

### Implications for System Designers

- **Orientation planning:** Single orientation installations show lower mismatch losses (12.6%) compared to multiple orientation designs (19.1%)
- **Installation strategy:** Architectural constraints requiring multiple orientations add ~6.5% mismatch penalty
- **Technology selection:** Quantified benefits of module-level power electronics, especially for complex installations

## Conclusion

- **Overall mismatch quantification:** Real-world rooftop systems exhibit 14.4 ± 5.0% average mismatch loss
- **Orientation-based category impacts:**
  - Single orientation installations: 12.6 ± 4.6% (16 sites)
  - Multiple orientation installations: 19.1 ± 2.2% (6 sites)
- **Orientation impact:** Multiple orientations add ~6.5% additional mismatch compared to a single orientation
- **Seasonal patterns:** Winter conditions show elevated mismatch losses due to lower solar altitude
- **Methodology validation:** Monte Carlo simulation confirms ±1.2% accuracy of proposed approach
- **Novel methodology:** First comprehensive field study using optimiser-derived data for mismatch quantification
- **Geographic diversity:** Multi-continental study spanning diverse climate zones
- **Practical relevance:** Direct applicability to rooftop PV system design and modelling

## Appendices

### Appendix A: Derivation of Dark Saturation Current

#### Known Parameters

**From SolarEdge measurements:**
- Maximum power point voltage: V_mp
- Maximum power point current: I_mp

**From PV module datasheet:**
- Module ideality factor: n
- Module cell number: N
- Module series resistance: R_s
- Module shunt resistance: R_sh

#### Mathematical Derivation

**Objective:** Derive an expression for the dark saturation current from the single-diode equation

At maximum power point: `dP/dV = 0`

Since `P = IV`: `dP/dV = I + V(dI/dV) = 0`

This yields: `dI/dV = -I/V` ... (4)

Taking the derivative of equation (1) with respect to voltage:

```
dI/dV = -I_0 * [exp((V+I*R_s)/(n*N*V_th))] * [1/(n*N*V_th) + R_s*dI/dV/(n*N*V_th)] - 1/R_sh - R_s*dI/dV/R_sh
```

Solving for `dI/dV`:

```
dI/dV = [-I_0*exp((V+I*R_s)/(n*N*V_th))/(n*N*V_th) - 1/R_sh] / [1 + I_0*R_s*exp((V+I*R_s)/(n*N*V_th))/(n*N*V_th) + R_s/R_sh]
```

Substituting into equation (4) and solving for I_0 yields equation (2).

### Appendix B: Bypass Diode Filtering Algorithm

The LTSpice simulated IV curves for a PV module with three internal bypass diodes are shown in Figure B-1

**Figure B-1:** LTSpice simulated IV curves with internal bypass diodes activation.

Based on the simulation, the MPP of the case when the internal diode is activated can be divided into the following two cases:

**Case 1:** Voc is a high outlier, and Isc is a lower outlier
**Case 2:** Voc is a lower outlier, and Isc is not an outlier

#### The outliers are defined as follows:

**Voc Outlier Classification:**
- High outliers: `Voc > Q3 + 1.5 × IQR`
- Lower outlier: `Voc < Q1 - 1.5 × IQR`

**Isc Outlier Detection:**
- High outliers: `Isc > Q3 + 3.0 × IQR`
- Low outliers: `Isc < Q1 - 1.5 × IQR`

Figure B-2 shows an example of Case 1 and Case 2 diode activation from the proposed algorithm

**Figure B-2:** Recorded MPP and the reconstructed IV curves with internal bypass diodes activation from measured data.

### Appendix C: Temperature and Irradiance Dependencies

#### Series Resistance (Rs)
- **Temperature dependency:** 0.356%/K (based on PERC module studies) [refs]
- **Physical basis:** Silver grid fingers and copper interconnects resistance increase with temperature
- **Irradiance dependency:** Minimal impact under normal operating conditions

#### Shunt Resistance (Rsh)
- **Temperature dependency:** Negligible effect on overall performance
- **Irradiance dependency:** Decreases with irradiance following the PVsyst exponential model
- **Implementation:** Dynamic adjustment based on measured irradiance levels

#### Ideality Factor (n)
- **Temperature dependency:** Minimal variation (0.006%/K for PERC modules)
- **Irradiance dependency:** Varies between 1.4-1.5 for 0.4-1.0 sun conditions
- **Model treatment:** Constant value approach validated within error bounds