# U-MAN — TSO–DSO Coordination for Voltage and Reactive Power Management

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

## Usage

Prepare the P/Q/U measurements from a directory containing SCADA CSV files:

```powershell
python ureditev_meritev.py "C:\path\to\scada-data" --mode normalize
python ureditev_meritev.py "C:\path\to\scada-data" --mode component_files
```

Prepare transformer tap measurements:

```powershell
python ureditev_TAP_meritev.py "C:\path\to\scada-data" --mode all
```

Analyze the generated transformer component files:

```powershell
python Reactive_Power_Analysis.py `
  "C:\path\to\scada-data\urejeno\Uman_parquet\component_files"
```

All scripts accept `--help`. The preprocessing scripts write to a directory under
the input directory by default; use `--output-dir` to select another location.
`Reactive_Power_Analysis.py` saves charts only as editable vector SVG files.
Analytical results remain available in the console; CSV files are not generated.

## Interaktivna kompenzacija jalove moči RTP

Za primerjavo izvornega stanja z dodano kapacitivno ali induktivno jalovo močjo
zaženi:

```powershell
python Reactive_Power_Compensation_GUI.py
```

V oknu izberi mapo `component_files` in RTP, vnesi kapacitivno oziroma
induktivno moč v MVAr ter klikni **Izračunaj in izriši**. Uporabljen je predznak
`Q_novi = Q_izmerjeni + Q_induktivni - Q_kapacitivni`. Graf je mogoče shraniti
kot SVG, PNG ali PDF, rezultate posameznih merilnih točk pa izvoziti v CSV.

Če so podatki v drugi mapi, jo izberi v oknu ali podaj ob zagonu:

```powershell
python Reactive_Power_Compensation_GUI.py `
  --data-dir "C:\pot\do\Uman_parquet\component_files"
```

## Interaktivni izbirnik Parquet grafov

Za poljubno sestavljanje grafov iz pripravljenih komponent zaženi:

```powershell
python Parquet_Plotter.py
```

V oknu poišči komponento (transformator, daljnovod, generator, merilno mesto ali
TAP), obkljukaj meritev, določi obdobje in jo dodaj v zbirko. Vsak element zbirke
ima lahko drugo komponento, meritev in obdobje. Zbirko lahko izrišeš v enem oknu,
shraniš kot PNG, SVG ali PDF ter shraniš izbor v JSON za ponovno uporabo.

Če so Parquet datoteke drugje, mapo dodaj v aplikaciji ali jo podaj ob zagonu:

```powershell
python Parquet_Plotter.py --data-dir "C:\pot\do\parquetov"
```

Preizkus branja brez odprtja grafičnega okna:

```powershell
python Parquet_Plotter.py --smoke-test
```

## Korelacija Q–U po zveznih segmentih

Korelacija sprememb jalove moči in napetosti se računa ločeno za vsak zvezni
15-minutni merilni segment. Časovna vrzel vedno začne nov segment; podatki pred
in po vrzeli se zato ne združijo v isti korelacijski koeficient ali regresijo.

```powershell
python Korelacija_dQ_dU_15min.py `
  --input-file "C:\pot\do\TR_JESENICE_110_TR1.parquet" `
  --start 2025-10-18 `
  --end 2025-10-18 `
  --min-segment-points 10
```

Glavni CSV je indeks vseh zaznanih segmentov. Za vsak dovolj dolg in
spremenljiv segment nastane še lastna mapa z meritvami, spremembami, statistiko
ter grafoma. Prekratki in konstantni segmenti ostanejo dokumentirani kot
preskočeni.

## Organizacija kode

- `voltage_data.py` vsebuje skupno odkrivanje elementov, topologijo in branje
  napetostnih vrst za napetostne analize.
- `continuous_segments.py` je skupna implementacija segmentiranja za Q–U in
  400/110-kV korelacijsko analizo.
- `ureditev_meritev.py` pripravlja meritve P/Q/U, `ureditev_TAP_meritev.py` pa
  diskretne položaje regulatorjev. Ločeni sta namenoma, ker imata različna
  pravila prepoznavanja, validacije in izvoza.
- Vse knjižnice so navedene v enem `requirements.txt`.
