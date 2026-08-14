# U-MAN — TSO–DSO Coordination for Voltage and Reactive Power Management

[![Python syntax](https://github.com/gordav003/U-man/actions/workflows/python-syntax.yml/badge.svg)](https://github.com/gordav003/U-man/actions/workflows/python-syntax.yml)

## Overview

This repository supports research conducted in connection with the [CRESYM U-MAN project](https://cresym.eu/u-man/), which investigates coordination between transmission system operators (TSOs) and distribution system operators (DSOs) for voltage control and reactive power management.

Power systems are undergoing a rapid transformation driven by increasing renewable generation, decentralised electricity production, the retirement of conventional synchronous generators, and the growing use of inverter-based resources. These developments reduce traditionally available reactive power flexibility and make secure voltage control more difficult across transmission, sub-transmission, and distribution networks.

The research represented in this repository focuses on identifying and analysing operating conditions in which reactive power exchange, reverse active-power flows, and low network loading contribute to voltage-control problems.

## Motivation

Lightly loaded high-voltage and sub-transmission networks can produce substantial capacitive reactive power and elevated voltages. At the same time, high generation from distributed energy resources may cause reverse power flows from distribution networks towards the transmission system.

These conditions can expose limitations in existing voltage-control mechanisms, including:

- insufficient coordination between transmission and distribution control actions;
- limited reactive power support from inverter-based resources;
- restricted or non-automatic operation of transformer tap changers;
- reduced reactive power capability following the retirement of synchronous generation;
- insufficient exchange of operational data between TSOs and DSOs;
- voltage-control actions that solve a local issue while worsening conditions elsewhere in the system.

Effective voltage management therefore requires coordinated use of reactive power resources and voltage-control devices across multiple voltage levels.

## Research Objectives

The main objectives of this research are to:

- assess voltage and reactive power behaviour across 400 kV, 220 kV, 110 kV, and medium-voltage networks;
- quantify active and reactive power exchange between transmission and distribution systems;
- investigate correlations between active power, reactive power, voltage, network loading, and operating states;
- identify substations and network elements with persistent capacitive reactive power behaviour;
- detect operating conditions associated with elevated voltages and reverse power flows;
- evaluate the interaction between medium-voltage and high-voltage networks;
- identify limitations of existing voltage and reactive power control mechanisms;
- select representative normal, critical, and extreme operating cases for further studies;
- support the development and validation of coordinated TSO–DSO control strategies.

## Scope of the Study

The work covers the acquisition, preparation, and analysis of operational measurements from several voltage levels. The principal quantities of interest are:

- voltage magnitude;
- active power;
- reactive power;
- transformer operating states and tap positions;
- direction of power flow;
- network loading conditions.

Particular attention is given to the following operating situations:

- high capacitive reactive power contribution from 110 kV and neighbouring 220 kV or 400 kV networks;
- low loading of the 110 kV network;
- overvoltage conditions in medium-voltage networks;
- reverse power flow from distribution towards transmission;
- simultaneous voltage and reactive power issues across several voltage levels;
- insufficient response from existing voltage-regulation equipment.

## Key Research Questions

The study addresses the following questions:

1. Which substations and network areas contribute most strongly to capacitive reactive power exchange?
2. Under which loading and generation conditions do critical voltage states occur?
3. How strongly are voltage variations correlated with active and reactive power flows?
4. How does operation of the medium-voltage network affect voltage and reactive power conditions at 110 kV?
5. Which operating states are suitable as representative cases for detailed simulation and control studies?
6. Which measurements and information exchanges are required for effective TSO–DSO coordination?
7. Which control actions should remain local, and which require system-level coordination?

## Expected Outcomes

The research is expected to provide:

- a structured overview of voltage and reactive power behaviour across the analysed network;
- identification and ranking of critical substations and network elements;
- representative operating cases for simulation and validation;
- improved understanding of interactions between transmission, sub-transmission, and distribution networks;
- evidence for defining TSO–DSO information exchange requirements;
- technical input for coordinated voltage and reactive power control schemes;
- recommendations for further studies, pilot implementations, and operational improvements.

## Relation to the CRESYM U-MAN Project

The wider U-MAN project investigates coordination frameworks that define the participating actors, system interfaces, required information exchanges, and available control actions. Its objectives include developing and testing multiple coordination schemes, validating them on realistic test systems and pilot implementations, and examining synergies with other services enabled by TSO–DSO coordination.

The project partners listed by CRESYM are ELES, RTE, Swissgrid, IPTO, HEDNO, and NTUA.

This repository represents a focused analytical contribution to that broader research context and should not be interpreted as the official repository of the entire U-MAN project.

## Data Confidentiality

The research may use operational power-system measurements and infrastructure information. Such data can be commercially sensitive or security-relevant. Raw measurements, detailed network models, asset identifiers, and results that reveal operational states should only be published when authorised by the relevant data owner.

## Project Reference

Further information about the wider project is available on the official CRESYM project page:

[https://cresym.eu/u-man/](https://cresym.eu/u-man/)

## Installation

Python 3.10 or newer is recommended. Create and activate a virtual environment,
then install the required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Raw SCADA measurements and generated results should remain outside the repository
because they may be large or confidential.

The `powerfactory_to_pandapower_jacobian.py` workflow additionally requires a
licensed DIgSILENT PowerFactory installation and must run with the Python
environment supplied or supported by PowerFactory. The proprietary
`powerfactory` module is therefore not listed in `requirements.txt`.

## Repository tools

| Area | Main scripts | Purpose |
| --- | --- | --- |
| Data preparation | `measurements/prepare_*.py` | Normalize SCADA P/Q/U measurements and transformer tap positions. |
| Voltage analysis | `voltage/*.py` | Analyze voltage levels, high-voltage events, annual duration curves, and shared voltage data. |
| Reactive power | `reactive_power/*.py` | Analyze reactive-power behavior, peaks, annual duration, and compensation scenarios. |
| Correlation | `correlations/*.py`, `continuous_segments.py` | Calculate segmented P/Q/U and cross-voltage-level correlations. |
| Interactive plotting | `parquet_plotter.py` | Build and export plots from prepared Parquet components. |
| Network model | `powerfactory_to_pandapower_jacobian.py` | Convert a PowerFactory model, solve or reconstruct the operating point, and export the sparse classical AC Jacobian. |

Generated measurements, plots, Jacobian matrices, pandapower networks, and
PowerFactory project files are intentionally excluded through `.gitignore`.

## Usage

Prepare the P/Q/U measurements from a directory containing SCADA CSV files:

```powershell
python -m measurements.prepare_measurements "C:\path\to\scada-data" --mode normalize
python -m measurements.prepare_measurements "C:\path\to\scada-data" --mode component_files
```

Prepare transformer tap measurements:

```powershell
python -m measurements.prepare_tap_measurements "C:\path\to\scada-data" --mode all
```

Analyze the generated transformer component files:

```powershell
python -m reactive_power.reactive_power_analysis `
  "C:\path\to\scada-data\urejeno\Uman_parquet\component_files"
```

All scripts accept `--help`. The preprocessing scripts write to a directory under
the input directory by default; use `--output-dir` to select another location.
`reactive_power_analysis.py` saves charts only as editable vector SVG files.
Analytical results remain available in the console; CSV files are not generated.

## Interactive RTP reactive power compensation

To compare the original state with added capacitive or inductive reactive power,
run:

```powershell
python -m reactive_power.reactive_power_compensation_gui
```

In the window, select the `component_files` directory and the RTP, enter the capacitive or
inductive reactive power in MVAr, and click **Calculate and plot**. The sign convention used is
`Q_new = Q_measured + Q_inductive - Q_capacitive`. The plot can be saved
as SVG, PNG, or PDF, and the results for individual measurement points can be exported to CSV.

If the data are stored in another directory, select it in the window or provide it at startup:

```powershell
python -m reactive_power.reactive_power_compensation_gui `
  --data-dir "C:\path\to\Uman_parquet\component_files"
```

## Interactive Parquet plot selector

To create arbitrary plots from the prepared components, run:

```powershell
python parquet_plotter.py
```

In the window, find a component (transformer, transmission line, generator, measurement point, or
TAP), select a measurement, define the period, and add it to the collection. Each collection item
may use a different component, measurement, and period. The collection can be plotted in one window,
saved as PNG, SVG, or PDF, and the selection can be saved as JSON for reuse.

If the Parquet files are stored elsewhere, add the directory in the application or provide it at startup:

```powershell
python parquet_plotter.py --data-dir "C:\path\to\parquet-files"
```

Test data loading without opening the graphical window:

```powershell
python parquet_plotter.py --smoke-test
```

## Q–U correlation by continuous segments

The correlation between reactive power and voltage changes is calculated separately for each continuous
15-minute measurement segment. A time gap always starts a new segment; data before
and after the gap are therefore not combined into the same correlation coefficient or regression.

```powershell
python -m correlations.reactive_power_voltage_delta_15min `
  --input-file "C:\path\to\TR_JESENICE_110_TR1.parquet" `
  --start 2025-10-18 `
  --end 2025-10-18 `
  --min-segment-points 10
```

The main CSV is an index of all detected segments. For each sufficiently long and
variable segment, a separate directory is created containing measurements, changes, statistics,
and plots. Segments that are too short or constant remain documented as
skipped.

## Code organization

- `voltage/voltage_data.py` contains shared element discovery, topology handling, and reading of
  voltage series for voltage analyses.
- `continuous_segments.py` provides the shared segmentation implementation for Q–U and
  400/110-kV correlation analysis.
- `powerfactory_to_pandapower_jacobian.py` converts a PowerFactory model to
  pandapower and exports the sparse classical AC Jacobian matrix and diagnostic
  results.
- `measurements/prepare_measurements.py` prepares P/Q/U measurements, while
  `measurements/prepare_tap_measurements.py` prepares
  discrete tap positions. They are intentionally separate because they use different
  recognition, validation, and export rules.
- All libraries are listed in a single `requirements.txt`.

## Development and contributing

Rules for branches, commits, pull requests, and basic checks are described in
[`CONTRIBUTING.md`](CONTRIBUTING.md). The `main` branch is the only permanent branch;
completed development branches are deleted after merging.
