"""
PowerFactory -> pandapower sparse classical AC Newton-Raphson Jacobian.

Designed for large PowerFactory transmission-system models.

Current workarounds / features
==============================
1) PowerFactory base load flow.
2) PF -> pandapower conversion.
3) Skip unsupported ElmStactrl objects individually.
4) Work around pandapower 3.5.4 ext_grid Q-capability conversion.
5) Diagnose unsupplied islands.
6) Export unsupplied-island diagnostics.
7) Deactivate unsupplied islands using pandapower's official function.
8) Build pandapower initial voltage from the already-converged
   PowerFactory voltage solution:
       res_bus["pf_vm_pu"]
       res_bus["pf_va_degree"]
9) Robust convergence sequence:
       A) NR from PowerFactory voltage
       B) Iwamoto NR from PowerFactory voltage, if no FACTS
       C) constant-power NR bootstrap, then exact NR
       D) light-injection homotopy continuation
10) If pandapower still does not converge, construct the internal Ybus directly
    and evaluate the classical Jacobian at the converged PowerFactory voltage point.
11) Export PF-point P/Q equation residuals to expose conversion mismatch.
12) Supported controller loop when a pandapower operating point exists.
13) Final standalone NR when a pandapower operating point exists.
14) Classical frozen-control AC bus-voltage Jacobian.
15) Sparse export and PF-vs-pandapower validation.

Jacobian convention
===================

    x = [delta_PV, delta_PQ, U_PQ]

    f = [P_PV, P_PQ, Q_PQ]

    J = df/dx

      = [ dP/d(delta)   dP/dU ]
        [ dQ/d(delta)   dQ/dU ]

Units:
    P, Q  : p.u.
    delta : rad
    U     : p.u.

Important
=========
The exported Jacobian is the classical frozen-control AC bus-voltage
Jacobian. If pandapower converges, it is evaluated at the final pandapower
operating point. If pandapower does not converge, it is evaluated directly
at the already-converged PowerFactory voltage vector using the converted
pandapower Ybus; pf_point_power_mismatch.csv then quantifies the remaining
model-equivalence residual.

Control variables themselves are not appended as Jacobian states.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import re
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy import sparse

import pandapower as pp

from pandapower.auxiliary import (
    LoadflowNotConverged,
)

from pandapower.converter.powerfactory import (
    from_pfd,
)

import pandapower.converter.powerfactory.pp_import_functions as pf_import_functions

from pandapower.pf.create_jacobian import (
    create_jacobian_matrix,
)

from pandapower.topology import (
    create_nxgraph,
    unsupplied_buses,
)

from pandapower.toolbox.grid_modification import (
    set_isolated_areas_out_of_service,
)

from pandapower.pypower.idx_bus import (
    BUS_I,
    BUS_TYPE,
    VM,
    VA,
    PQ,
    PV,
    REF,
    SL_FAC,
)

from pandapower.pypower.bustypes import bustypes
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pd2ppc import _pd2ppc

import powerfactory


# =============================================================================
# USER SETTINGS
# =============================================================================

OUTPUT_ROOT = Path(
    r"C:\Users\David\Desktop\Projekti\CRESYM-Uman\Jacobian_exports"
)


# =============================================================================
# PF -> PANDAPOWER CONVERTER
# =============================================================================

PF_CONVERTER_HANDLE_US = "Nothing"

# Use the tap position resulting from the converged PowerFactory load flow.
# For a frozen operating-point export this is preferable to the pre-control input tap.
PF_TAP_OPTION = "c:nntap"

# Export the actual solved terminal powers from PowerFactory. The converter maps
# m:P:bus1 to m:Q:bus1 internally for the corresponding reactive power.
PF_VARIABLE_P_LOADS = "m:P:bus1"
PF_VARIABLE_P_GEN = "m:P:bus1"

EXPORT_CONTROLLERS = True

SKIP_UNSUPPORTED_STATION_CONTROLLERS = True

PATCH_EXT_GRID_Q_CAPABILITY = True


# =============================================================================
# UNSUPPLIED ISLANDS
# =============================================================================

AUTO_DEACTIVATE_UNSUPPLIED = True

RESPECT_SWITCHES_FOR_SUPPLY = True

STOP_IF_UNSUPPLIED_REMAIN = True


# =============================================================================
# PANDAPOWER POWER FLOW
# =============================================================================

TOLERANCE_MVA = 1e-8

MAX_ITERATION = 100

# Only used as a bootstrap operating point.
BOOTSTRAP_TOLERANCE_MVA = 1e-6

BOOTSTRAP_MAX_ITERATION = 100

ENFORCE_Q_LIMS = False

CHECK_CONNECTIVITY = True

# IMPORTANT: P/Q are exported from PowerFactory result values m:P/m:Q.
# Those are already the powers at the solved PF voltage. Applying the ZIP model
# a second time would move the operating point, so keep the bootstrap/final PF
# constant-power. The exported Jacobian is classical frozen-control anyway.
VOLTAGE_DEPEND_LOADS = False

USE_NUMBA = True

USE_LIGHTSIM2GRID = False


# -----------------------------------------------------------------------------
# PowerFactory warm-start
# -----------------------------------------------------------------------------

USE_POWERFACTORY_VOLTAGE_INITIALIZATION = True

ALIGN_PF_ANGLE_PER_ISLAND = True


# -----------------------------------------------------------------------------
# Robust convergence fallbacks
# -----------------------------------------------------------------------------

USE_IWAMOTO_FALLBACK = True

USE_CONSTANT_POWER_BOOTSTRAP = True

# Continuation / homotopy fallback for difficult converted transmission grids.
USE_HOMOTOPY_FALLBACK = True
HOMOTOPY_START_FACTOR = 0.05
HOMOTOPY_INITIAL_STEP = 0.10
HOMOTOPY_MIN_STEP = 0.0125

# If all pandapower power-flow attempts fail, still export the classical
# Jacobian at the converged PowerFactory voltage vector. This is mathematically
# valid because d(P,Q)/d(delta,U) depends on Ybus, bus types and V, not on
# Newton convergence itself. A residual report is exported so model mismatch
# remains explicit rather than hidden.
ALLOW_PF_POINT_JACOBIAN_FALLBACK = True


# -----------------------------------------------------------------------------
# Controllers
# -----------------------------------------------------------------------------

# Frozen-control Jacobian: do not re-run controls after importing the already
# converged PowerFactory operating point. Set True manually only if you explicitly
# want a new pandapower-controlled operating point instead.
RUN_CONTROLLERS: bool | None = False


# =============================================================================
# OPTIONAL PRIMSKOVO REPAIR
# =============================================================================

# FALSE for the Slovenian full-grid model.
ADD_PRIMSKOVO_ZKS = False

ZKS_FROM_BUS_NAME = "Terminal"

ZKS_TO_BUS_NAME = "110 kV"

ZKS_R_OHM = 0.427

ZKS_L_H = 13.6e-3

ZKS_F_HZ = 50.0

ZKS_PP_NAME = "Z_ks [ElmSfilt RL equivalent]"


# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

SAVE_PANDAPOWER_NET_JSON = True

SAVE_JACOBIAN_COO_CSV = True

SAVE_DEBUG_TABLES = True

SAVE_VOLTAGE_VALIDATION = True

PRINT_TABLE_ROWS = 20


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def pf_print(
    app,
    message="",
) -> None:

    try:

        app.PrintInfo(
            str(message)
        )

    except Exception:

        print(
            message
        )


def pf_warn(
    app,
    message="",
) -> None:

    try:

        app.PrintWarn(
            str(message)
        )

    except Exception:

        try:

            app.PrintInfo(
                "WARNING: "
                + str(message)
            )

        except Exception:

            print(
                "WARNING:",
                message,
            )


def package_version(
    name: str,
) -> str:

    try:

        return importlib.metadata.version(
            name
        )

    except Exception:

        return "unknown"


def safe_project_folder_name(
    name: str,
) -> str:

    text = str(
        name
    ).strip()

    text = re.sub(
        r'[<>:"/\\|?*]+',
        "_",
        text,
    )

    text = text.rstrip(
        ". "
    )

    if not text:

        text = "PowerFactory_project"

    return text


def safe_pf_attribute(
    obj,
    attribute: str,
    default=None,
):

    try:

        return obj.GetAttribute(
            attribute
        )

    except Exception:

        try:

            return getattr(
                obj,
                attribute,
            )

        except Exception:

            return default


def safe_pf_full_name(
    obj,
) -> str:

    try:

        return str(
            obj.GetFullName()
        )

    except Exception:

        try:

            return str(
                obj.loc_name
            )

        except Exception:

            return str(
                obj
            )


def safe_text(
    value,
) -> str:

    if value is None:

        return ""

    try:

        if pd.isna(
            value
        ):

            return ""

    except Exception:

        pass

    return str(
        value
    )


def join_unique(
    values,
) -> str:

    output = []

    seen = set()

    for value in values:

        text = safe_text(
            value
        )

        if (
            text
            and text not in seen
        ):

            seen.add(
                text
            )

            output.append(
                text
            )

    return " | ".join(
        output
    )


def save_dataframe(
    frame: pd.DataFrame | None,
    path: Path,
    index: bool = True,
) -> None:

    if frame is None:

        return

    try:

        frame.to_csv(
            path,
            index=index,
            encoding="utf-8-sig",
        )

    except Exception:

        pass


def to_scalar_float(
    value,
) -> float:

    array = np.asarray(
        value
    )

    if array.size == 0:

        raise ValueError(
            "Cannot convert empty value to float."
        )

    return float(
        array.reshape(
            -1
        )[0]
    )


def table_in_service_mask(
    frame: pd.DataFrame,
) -> pd.Series:

    if frame is None:

        return pd.Series(
            dtype=bool
        )

    if (
        "in_service"
        in frame.columns
    ):

        return (
            frame[
                "in_service"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

    return pd.Series(
        True,
        index=frame.index,
        dtype=bool,
    )


# =============================================================================
# POWERFACTORY ElmSfilt SCAN
# =============================================================================

def scan_powerfactory_elmsfilt(
    app,
) -> list[dict]:

    records = []

    try:

        objects = (
            app.GetCalcRelevantObjects(
                "*.ElmSfilt"
            )
            or []
        )

    except Exception:

        objects = []

    for obj in objects:

        outserv = safe_pf_attribute(
            obj,
            "outserv",
            0,
        )

        try:

            in_service = not bool(
                outserv
            )

        except Exception:

            in_service = True

        records.append(
            {
                "name": safe_pf_attribute(
                    obj,
                    "loc_name",
                    "",
                ),
                "full_name": safe_pf_full_name(
                    obj
                ),
                "in_service": in_service,
            }
        )

    active = [
        record
        for record in records
        if record[
            "in_service"
        ]
    ]

    if active:

        pf_warn(
            app,
            (
                f"PowerFactory contains {len(active)} "
                "active ElmSfilt element(s)."
            ),
        )

        for record in active[
            :PRINT_TABLE_ROWS
        ]:

            pf_warn(
                app,
                (
                    "  ElmSfilt: "
                    f"{record['full_name']}"
                ),
            )

    return records


# =============================================================================
# SAFE ElmStactrl PATCH
# =============================================================================

def make_safe_create_stactrl(
    app,
    skipped_records: list[dict],
):

    original_function = (
        pf_import_functions.create_stactrl
    )

    def record_skip(
        item,
        reason: str,
    ) -> None:

        record = {
            "name": str(
                safe_pf_attribute(
                    item,
                    "loc_name",
                    "",
                )
            ),
            "full_name": safe_pf_full_name(
                item
            ),
            "i_ctrl": safe_pf_attribute(
                item,
                "i_ctrl",
                None,
            ),
            "selBus": safe_pf_attribute(
                item,
                "selBus",
                None,
            ),
            "reason": str(
                reason
            ),
        }

        skipped_records.append(
            record
        )

        pf_warn(
            app,
            (
                "Skipping unsupported ElmStactrl: "
                f"{record['name']} | "
                f"i_ctrl={record['i_ctrl']} | "
                f"selBus={record['selBus']} | "
                f"{record['reason']}"
            ),
        )

    def safe_create_stactrl(
        net,
        item,
        top,
        top_all,
    ):

        control_mode = safe_pf_attribute(
            item,
            "i_ctrl",
            None,
        )

        sel_bus = safe_pf_attribute(
            item,
            "selBus",
            None,
        )

        if (
            SKIP_UNSUPPORTED_STATION_CONTROLLERS
            and control_mode == 0
            and sel_bus not in (
                None,
                0,
            )
        ):

            record_skip(
                item,
                (
                    "controlled-node selection "
                    "not implemented by pandapower 3.5.4"
                ),
            )

            return None

        controller_indices_before = set()

        if (
            hasattr(
                net,
                "controller",
            )
            and net.controller is not None
            and not net.controller.empty
        ):

            controller_indices_before = set(
                net.controller.index.tolist()
            )

        try:

            return original_function(
                net=net,
                item=item,
                top=top,
                top_all=top_all,
            )

        except NotImplementedError as exc:

            if not SKIP_UNSUPPORTED_STATION_CONTROLLERS:

                raise

            if (
                hasattr(
                    net,
                    "controller",
                )
                and net.controller is not None
                and not net.controller.empty
            ):

                controller_indices_after = set(
                    net.controller.index.tolist()
                )

                added_indices = sorted(
                    controller_indices_after
                    - controller_indices_before
                )

                if added_indices:

                    try:

                        net.controller.drop(
                            index=added_indices,
                            inplace=True,
                        )

                    except Exception:

                        pass

            record_skip(
                item,
                str(
                    exc
                ),
            )

            return None

    return (
        original_function,
        safe_create_stactrl,
    )


# =============================================================================
# SAFE ext_grid Q-CAPABILITY PATCH
# =============================================================================

def make_safe_q_capability_helper(
    app,
    patch_records: list[dict],
):

    original_function = (
        pf_import_functions
        .get_min_max_q_mvar_from_characteristics_object
    )

    printed = {
        "count": 0
    }

    def fallback(
        eid: int,
        reason: str,
    ):

        patch_records.append(
            {
                "element": "ext_grid",
                "index": int(
                    eid
                ),
                "status": "fallback_infinite_limits",
                "reason": str(
                    reason
                ),
                "p_mw_pf": None,
                "q_min_mvar": -np.inf,
                "q_max_mvar": np.inf,
            }
        )

        pf_warn(
            app,
            (
                "ext_grid Q-capability fallback "
                f"index={eid}: {reason}; "
                "using [-inf, +inf]."
            ),
        )

        return (
            -np.inf,
            np.inf,
        )

    def evaluate_ext_grid(
        net,
        eid: int,
    ):

        eid = int(
            eid
        )

        if (
            eid
            not in net.ext_grid.index
        ):

            return fallback(
                eid,
                "ext_grid index does not exist",
            )

        row = net.ext_grid.loc[
            eid
        ]

        if (
            "reactive_capability_curve"
            not in net.ext_grid.columns
        ):

            return (
                -np.inf,
                np.inf,
            )

        try:

            enabled = bool(
                row[
                    "reactive_capability_curve"
                ]
            )

        except Exception:

            enabled = False

        if not enabled:

            return (
                -np.inf,
                np.inf,
            )

        if (
            "id_q_capability_characteristic"
            not in net.ext_grid.columns
        ):

            return fallback(
                eid,
                "id_q_capability_characteristic missing",
            )

        characteristic_id = row[
            "id_q_capability_characteristic"
        ]

        if pd.isna(
            characteristic_id
        ):

            return fallback(
                eid,
                "Q characteristic ID is NaN",
            )

        if (
            not hasattr(
                net,
                "res_ext_grid",
            )
            or net.res_ext_grid is None
            or "pf_p"
            not in net.res_ext_grid.columns
            or eid
            not in net.res_ext_grid.index
        ):

            return fallback(
                eid,
                "PF res_ext_grid.pf_p unavailable",
            )

        try:

            p_mw = to_scalar_float(
                net.res_ext_grid.at[
                    eid,
                    "pf_p",
                ]
            )

        except Exception as exc:

            return fallback(
                eid,
                (
                    "could not read "
                    f"res_ext_grid.pf_p: {exc}"
                ),
            )

        try:

            table = net[
                "q_capability_characteristic"
            ]

            q_min_function = table.loc[
                characteristic_id,
                "q_min_characteristic",
            ]

            q_max_function = table.loc[
                characteristic_id,
                "q_max_characteristic",
            ]

            q_min = to_scalar_float(
                q_min_function(
                    p_mw
                )
            )

            q_max = to_scalar_float(
                q_max_function(
                    p_mw
                )
            )

        except Exception as exc:

            return fallback(
                eid,
                (
                    "Q characteristic evaluation "
                    f"failed: {exc}"
                ),
            )

        if (
            not np.isfinite(
                q_min
            )
            or not np.isfinite(
                q_max
            )
        ):

            return fallback(
                eid,
                "Q characteristic returned NaN/Inf",
            )

        if (
            q_min
            > q_max
        ):

            q_min, q_max = (
                q_max,
                q_min,
            )

        patch_records.append(
            {
                "element": "ext_grid",
                "index": int(
                    eid
                ),
                "status": (
                    "evaluated_from_pf_operating_point"
                ),
                "characteristic_id": safe_text(
                    characteristic_id
                ),
                "p_mw_pf": float(
                    p_mw
                ),
                "q_min_mvar": float(
                    q_min
                ),
                "q_max_mvar": float(
                    q_max
                ),
            }
        )

        if (
            printed[
                "count"
            ]
            < PRINT_TABLE_ROWS
        ):

            pf_print(
                app,
                (
                    "ext_grid Q capability: "
                    f"index={eid}, "
                    f"P_pf={p_mw:.6f} MW, "
                    f"Qmin={q_min:.6f} Mvar, "
                    f"Qmax={q_max:.6f} Mvar"
                ),
            )

            printed[
                "count"
            ] += 1

        return (
            q_min,
            q_max,
        )

    def safe_get_min_max_q(
        net,
        element,
        element_index,
    ):

        if (
            element
            != "ext_grid"
        ):

            return original_function(
                net,
                element,
                element_index,
            )

        scalar_input = np.isscalar(
            element_index
        )

        if scalar_input:

            indices = [
                int(
                    element_index
                )
            ]

        else:

            indices = [
                int(
                    index
                )
                for index
                in np.asarray(
                    element_index
                ).reshape(
                    -1
                )
            ]

        q_min_values = []

        q_max_values = []

        for eid in indices:

            q_min, q_max = (
                evaluate_ext_grid(
                    net,
                    eid,
                )
            )

            q_min_values.append(
                q_min
            )

            q_max_values.append(
                q_max
            )

        if scalar_input:

            return (
                q_min_values[
                    0
                ],
                q_max_values[
                    0
                ],
            )

        return (
            np.asarray(
                q_min_values,
                dtype=float,
            ),
            np.asarray(
                q_max_values,
                dtype=float,
            ),
        )

    return (
        original_function,
        safe_get_min_max_q,
    )


# =============================================================================
# PF BUS VOLTAGE SNAPSHOT
# =============================================================================

def snapshot_pf_bus_results(
    net,
) -> pd.DataFrame | None:

    if (
        not hasattr(
            net,
            "res_bus",
        )
        or net.res_bus is None
        or net.res_bus.empty
    ):

        return None

    result = pd.DataFrame(
        index=net.bus.index
    )

    result[
        "vm_pu"
    ] = np.nan

    result[
        "va_degree"
    ] = np.nan

    if (
        "pf_vm_pu"
        in net.res_bus.columns
    ):

        result.loc[
            net.res_bus.index,
            "vm_pu",
        ] = pd.to_numeric(
            net.res_bus[
                "pf_vm_pu"
            ],
            errors="coerce",
        )

    elif (
        "vm_pu"
        in net.res_bus.columns
    ):

        result.loc[
            net.res_bus.index,
            "vm_pu",
        ] = pd.to_numeric(
            net.res_bus[
                "vm_pu"
            ],
            errors="coerce",
        )

    if (
        "pf_va_degree"
        in net.res_bus.columns
    ):

        result.loc[
            net.res_bus.index,
            "va_degree",
        ] = pd.to_numeric(
            net.res_bus[
                "pf_va_degree"
            ],
            errors="coerce",
        )

    elif (
        "va_degree"
        in net.res_bus.columns
    ):

        result.loc[
            net.res_bus.index,
            "va_degree",
        ] = pd.to_numeric(
            net.res_bus[
                "va_degree"
            ],
            errors="coerce",
        )

    result.index.name = (
        "pp_bus_index"
    )

    return result


# =============================================================================
# OPTIONAL PRIMSKOVO ElmSfilt
# =============================================================================

def get_unique_bus_by_name(
    net,
    name: str,
) -> int:

    mask = (
        net.bus[
            "name"
        ]
        .astype(
            str
        )
        .str.strip()
        == str(
            name
        ).strip()
    )

    candidates = net.bus.index[
        mask
    ].tolist()

    if (
        len(
            candidates
        )
        != 1
    ):

        raise RuntimeError(
            (
                f"Could not uniquely identify "
                f"bus '{name}'. "
                f"Candidates={candidates}"
            )
        )

    return int(
        candidates[
            0
        ]
    )


def add_primskovo_zks(
    app,
    net,
) -> dict:

    if not ADD_PRIMSKOVO_ZKS:

        return {
            "enabled": False
        }

    x_ohm = (
        2.0
        * math.pi
        * ZKS_F_HZ
        * ZKS_L_H
    )

    from_bus = get_unique_bus_by_name(
        net,
        ZKS_FROM_BUS_NAME,
    )

    to_bus = get_unique_bus_by_name(
        net,
        ZKS_TO_BUS_NAME,
    )

    vn_kv = float(
        net.bus.at[
            from_bus,
            "vn_kv",
        ]
    )

    sn_mva = float(
        net.sn_mva
    )

    z_base = (
        vn_kv ** 2
        / sn_mva
    )

    r_pu = (
        ZKS_R_OHM
        / z_base
    )

    x_pu = (
        x_ohm
        / z_base
    )

    index = pp.create_impedance(
        net,
        from_bus=from_bus,
        to_bus=to_bus,
        rft_pu=r_pu,
        xft_pu=x_pu,
        rtf_pu=r_pu,
        xtf_pu=x_pu,
        sn_mva=sn_mva,
        name=ZKS_PP_NAME,
        in_service=True,
    )

    pf_print(
        app,
        (
            "Created Primskovo Z_ks "
            f"impedance index={index}"
        ),
    )

    return {
        "enabled": True,
        "index": int(
            index
        ),
        "r_ohm": float(
            ZKS_R_OHM
        ),
        "x_ohm": float(
            x_ohm
        ),
    }


# =============================================================================
# SUPPLY / ISLAND DIAGNOSTICS
# =============================================================================

def get_slack_buses(
    net,
) -> set[int]:

    output = set()

    if (
        net.ext_grid is not None
        and not net.ext_grid.empty
    ):

        mask = table_in_service_mask(
            net.ext_grid
        )

        output.update(
            int(
                bus
            )
            for bus
            in net.ext_grid.loc[
                mask,
                "bus",
            ]
        )

    if (
        net.gen is not None
        and not net.gen.empty
        and "slack"
        in net.gen.columns
    ):

        mask = (
            table_in_service_mask(
                net.gen
            )
            & net.gen[
                "slack"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

        output.update(
            int(
                bus
            )
            for bus
            in net.gen.loc[
                mask,
                "bus",
            ]
        )

    return output


def sum_bus_element_power(
    frame: pd.DataFrame,
    buses: set[int],
) -> tuple[
    int,
    float,
    float,
]:

    if (
        frame is None
        or frame.empty
        or "bus"
        not in frame.columns
    ):

        return (
            0,
            0.0,
            0.0,
        )

    mask = (
        frame[
            "bus"
        ].isin(
            buses
        )
        & table_in_service_mask(
            frame
        )
    )

    selected = frame.loc[
        mask
    ]

    p = 0.0

    q = 0.0

    if (
        "p_mw"
        in selected.columns
    ):

        p = float(
            pd.to_numeric(
                selected[
                    "p_mw"
                ],
                errors="coerce",
            )
            .fillna(
                0.0
            )
            .sum()
        )

    if (
        "q_mvar"
        in selected.columns
    ):

        q = float(
            pd.to_numeric(
                selected[
                    "q_mvar"
                ],
                errors="coerce",
            )
            .fillna(
                0.0
            )
            .sum()
        )

    return (
        int(
            len(
                selected
            )
        ),
        p,
        q,
    )


def export_unsupplied_island_diagnostics(
    app,
    net,
    output_dir: Path,
):

    mg = create_nxgraph(
        net,
        respect_switches=(
            RESPECT_SWITCHES_FOR_SUPPLY
        ),
    )

    unsupplied = set(
        int(
            bus
        )
        for bus
        in unsupplied_buses(
            net,
            mg=mg,
            respect_switches=(
                RESPECT_SWITCHES_FOR_SUPPLY
            ),
        )
    )

    slack_buses = get_slack_buses(
        net
    )

    island_records = []

    bus_records = []

    island_id = 0

    for component in nx.connected_components(
        mg
    ):

        component = set(
            int(
                bus
            )
            for bus
            in component
        )

        if not (
            component
            & unsupplied
        ):

            continue

        island_id += 1

        buses = sorted(
            component
        )

        names = []

        for bus in buses:

            name = safe_text(
                net.bus.at[
                    bus,
                    "name",
                ]
            )

            names.append(
                name
            )

            bus_records.append(
                {
                    "island_id": island_id,
                    "bus": int(
                        bus
                    ),
                    "name": name,
                    "vn_kv": float(
                        net.bus.at[
                            bus,
                            "vn_kv",
                        ]
                    ),
                    "in_service": bool(
                        net.bus.at[
                            bus,
                            "in_service",
                        ]
                    ),
                }
            )

        bus_set = set(
            buses
        )

        (
            load_count,
            load_p,
            load_q,
        ) = sum_bus_element_power(
            net.load,
            bus_set,
        )

        (
            sgen_count,
            sgen_p,
            sgen_q,
        ) = sum_bus_element_power(
            net.sgen,
            bus_set,
        )

        (
            gen_count,
            gen_p,
            gen_q,
        ) = sum_bus_element_power(
            net.gen,
            bus_set,
        )

        island_records.append(
            {
                "island_id": island_id,
                "bus_count": len(
                    buses
                ),
                "bus_indices": ";".join(
                    str(
                        bus
                    )
                    for bus
                    in buses
                ),
                "bus_names": " | ".join(
                    names
                ),
                "has_slack": bool(
                    bus_set
                    & slack_buses
                ),
                "load_count": load_count,
                "load_p_mw": load_p,
                "load_q_mvar": load_q,
                "sgen_count": sgen_count,
                "sgen_p_mw": sgen_p,
                "sgen_q_mvar": sgen_q,
                "gen_count": gen_count,
                "gen_p_mw": gen_p,
                "gen_q_mvar": gen_q,
            }
        )

    island_df = pd.DataFrame(
        island_records
    )

    bus_df = pd.DataFrame(
        bus_records
    )

    save_dataframe(
        island_df,
        output_dir
        / "unsupplied_islands_before_deactivation.csv",
        index=False,
    )

    save_dataframe(
        bus_df,
        output_dir
        / "unsupplied_buses_before_deactivation.csv",
        index=False,
    )

    pf_print(
        app,
        (
            "Unsupplied electrical islands: "
            f"{len(island_df)}"
        ),
    )

    if not island_df.empty:

        print_columns = [
            "island_id",
            "bus_count",
            "bus_names",
            "load_count",
            "load_p_mw",
            "sgen_count",
            "sgen_p_mw",
            "gen_count",
            "gen_p_mw",
        ]

        pf_print(
            app,
            island_df[
                print_columns
            ].to_string(
                index=False
            ),
        )

    return (
        island_df,
        bus_df,
    )


def topology_diagnostics(
    app,
    net,
    stage: str,
):

    pf_print(
        app,
        "",
    )

    pf_print(
        app,
        "=" * 72,
    )

    pf_print(
        app,
        (
            "PANDAPOWER TOPOLOGY "
            f"DIAGNOSTICS: {stage}"
        ),
    )

    pf_print(
        app,
        "=" * 72,
    )

    unsupplied_all = sorted(
        int(
            bus
        )
        for bus
        in unsupplied_buses(
            net,
            respect_switches=(
                RESPECT_SWITCHES_FOR_SUPPLY
            ),
        )
    )

    active_unsupplied = [
        bus
        for bus
        in unsupplied_all
        if bool(
            net.bus.at[
                bus,
                "in_service",
            ]
        )
    ]

    pf_print(
        app,
        (
            f"buses={len(net.bus)}, "
            f"lines={len(net.line)}, "
            f"trafos={len(net.trafo)}, "
            f"loads={len(net.load)}, "
            f"gens={len(net.gen)}, "
            f"sgens={len(net.sgen)}, "
            f"ext_grids={len(net.ext_grid)}, "
            f"switches={len(net.switch)}"
        ),
    )

    pf_print(
        app,
        (
            "Unsupplied buses total: "
            f"{len(unsupplied_all)}"
        ),
    )

    pf_print(
        app,
        (
            "Unsupplied IN-SERVICE buses: "
            f"{len(active_unsupplied)}"
        ),
    )

    if active_unsupplied:

        columns = [
            column
            for column
            in (
                "name",
                "vn_kv",
                "in_service",
                "zone",
            )
            if column
            in net.bus.columns
        ]

        pf_print(
            app,
            net.bus.loc[
                active_unsupplied,
                columns,
            ].to_string(),
        )

    return (
        unsupplied_all,
        active_unsupplied,
    )


# =============================================================================
# SUPPLIED-ISLAND / SLACK DIAGNOSTICS
# =============================================================================

def supplied_island_diagnostics(
    app,
    net,
    output_dir: Path,
) -> pd.DataFrame:

    graph = create_nxgraph(
        net,
        respect_switches=True,
    )

    records = []

    island_id = 0

    for component in nx.connected_components(
        graph
    ):

        buses = set(
            int(
                bus
            )
            for bus
            in component
        )

        active_ext = pd.DataFrame()

        if (
            net.ext_grid is not None
            and not net.ext_grid.empty
        ):

            mask = (
                table_in_service_mask(
                    net.ext_grid
                )
                & net.ext_grid[
                    "bus"
                ].isin(
                    buses
                )
            )

            active_ext = net.ext_grid.loc[
                mask
            ]

        active_slack_gen = pd.DataFrame()

        if (
            net.gen is not None
            and not net.gen.empty
            and "slack"
            in net.gen.columns
        ):

            mask = (
                table_in_service_mask(
                    net.gen
                )
                & net.gen[
                    "slack"
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
                & net.gen[
                    "bus"
                ].isin(
                    buses
                )
            )

            active_slack_gen = net.gen.loc[
                mask
            ]

        if (
            active_ext.empty
            and active_slack_gen.empty
        ):

            continue

        island_id += 1

        records.append(
            {
                "island_id": island_id,
                "bus_count": int(
                    len(
                        buses
                    )
                ),
                "ext_grid_count": int(
                    len(
                        active_ext
                    )
                ),
                "slack_gen_count": int(
                    len(
                        active_slack_gen
                    )
                ),
                "ext_grid_names": (
                    join_unique(
                        active_ext[
                            "name"
                        ].tolist()
                    )
                    if (
                        not active_ext.empty
                        and "name"
                        in active_ext.columns
                    )
                    else ""
                ),
                "min_bus_index": int(
                    min(
                        buses
                    )
                ),
                "max_bus_index": int(
                    max(
                        buses
                    )
                ),
            }
        )

    frame = pd.DataFrame(
        records
    )

    save_dataframe(
        frame,
        output_dir
        / "supplied_islands.csv",
        index=False,
    )

    pf_print(
        app,
        "",
    )

    pf_print(
        app,
        (
            "Supplied electrical islands: "
            f"{len(frame)}"
        ),
    )

    if not frame.empty:

        pf_print(
            app,
            frame.to_string(
                index=False
            ),
        )

    return frame


# =============================================================================
# POWERFACTORY WARM START
# =============================================================================

def build_powerfactory_warm_start(
    app,
    net,
    pf_snapshot: pd.DataFrame | None,
    output_dir: Path,
) -> tuple[
    pd.Series,
    pd.Series,
    dict,
]:

    vm = pd.Series(
        1.0,
        index=net.bus.index,
        dtype=float,
    )

    va = pd.Series(
        0.0,
        index=net.bus.index,
        dtype=float,
    )

    metadata = {
        "available": False,
        "valid_pf_vm_count": 0,
        "valid_pf_va_count": 0,
        "island_angle_shifts": [],
    }

    if (
        not USE_POWERFACTORY_VOLTAGE_INITIALIZATION
        or pf_snapshot is None
    ):

        pf_warn(
            app,
            (
                "PowerFactory voltage warm-start "
                "is unavailable; using 1.0 pu / 0 deg."
            ),
        )

        return (
            vm,
            va,
            metadata,
        )

    pf_vm = pd.to_numeric(
        pf_snapshot[
            "vm_pu"
        ],
        errors="coerce",
    ).reindex(
        net.bus.index
    )

    pf_va = pd.to_numeric(
        pf_snapshot[
            "va_degree"
        ],
        errors="coerce",
    ).reindex(
        net.bus.index
    )

    valid_vm = (
        np.isfinite(
            pf_vm
        )
        & (
            pf_vm
            > 0.05
        )
        & (
            pf_vm
            < 2.0
        )
    )

    valid_va = np.isfinite(
        pf_va
    )

    vm.loc[
        valid_vm
    ] = pf_vm.loc[
        valid_vm
    ]

    va.loc[
        valid_va
    ] = pf_va.loc[
        valid_va
    ]

    metadata[
        "available"
    ] = True

    metadata[
        "valid_pf_vm_count"
    ] = int(
        valid_vm.sum()
    )

    metadata[
        "valid_pf_va_count"
    ] = int(
        valid_va.sum()
    )

    # -------------------------------------------------------------------------
    # Out-of-service buses are irrelevant to NR.
    # Keep them at harmless finite values.
    # -------------------------------------------------------------------------

    oos = ~net.bus[
        "in_service"
    ].fillna(
        False
    ).astype(
        bool
    )

    vm.loc[
        oos
    ] = 1.0

    va.loc[
        oos
    ] = 0.0

    # -------------------------------------------------------------------------
    # Align PF angle reference separately for each supplied island.
    #
    # PowerFactory and pandapower can choose independent zero-angle
    # references for disconnected electrical systems.
    # -------------------------------------------------------------------------

    if ALIGN_PF_ANGLE_PER_ISLAND:

        graph = create_nxgraph(
            net,
            respect_switches=True,
        )

        for component_number, component in enumerate(
            nx.connected_components(
                graph
            ),
            start=1,
        ):

            buses = set(
                int(
                    bus
                )
                for bus
                in component
            )

            ext_rows = net.ext_grid.loc[
                (
                    table_in_service_mask(
                        net.ext_grid
                    )
                    & net.ext_grid[
                        "bus"
                    ].isin(
                        buses
                    )
                )
            ]

            reference_bus = None

            target_angle = 0.0

            reference_name = ""

            if not ext_rows.empty:

                ext_index = ext_rows.index[
                    0
                ]

                reference_bus = int(
                    ext_rows.at[
                        ext_index,
                        "bus",
                    ]
                )

                target_angle = float(
                    ext_rows.at[
                        ext_index,
                        "va_degree",
                    ]
                )

                reference_name = safe_text(
                    ext_rows.at[
                        ext_index,
                        "name",
                    ]
                )

            else:

                if (
                    net.gen is not None
                    and not net.gen.empty
                    and "slack"
                    in net.gen.columns
                ):

                    slack_gen_rows = net.gen.loc[
                        (
                            table_in_service_mask(
                                net.gen
                            )
                            & net.gen[
                                "slack"
                            ]
                            .fillna(
                                False
                            )
                            .astype(
                                bool
                            )
                            & net.gen[
                                "bus"
                            ].isin(
                                buses
                            )
                        )
                    ]

                    if not slack_gen_rows.empty:

                        gen_index = slack_gen_rows.index[
                            0
                        ]

                        reference_bus = int(
                            slack_gen_rows.at[
                                gen_index,
                                "bus",
                            ]
                        )

                        target_angle = 0.0

                        reference_name = safe_text(
                            slack_gen_rows.at[
                                gen_index,
                                "name",
                            ]
                        )

            if reference_bus is None:

                continue

            pf_reference_angle = float(
                va.at[
                    reference_bus
                ]
            )

            shift = (
                target_angle
                - pf_reference_angle
            )

            component_index = pd.Index(
                sorted(
                    buses
                )
            )

            va.loc[
                component_index
            ] = (
                va.loc[
                    component_index
                ]
                + shift
            )

            metadata[
                "island_angle_shifts"
            ].append(
                {
                    "component_number": int(
                        component_number
                    ),
                    "reference_bus": int(
                        reference_bus
                    ),
                    "reference_name": (
                        reference_name
                    ),
                    "pf_reference_angle_degree": float(
                        pf_reference_angle
                    ),
                    "target_angle_degree": float(
                        target_angle
                    ),
                    "shift_degree": float(
                        shift
                    ),
                    "bus_count": int(
                        len(
                            buses
                        )
                    ),
                }
            )

    # -------------------------------------------------------------------------
    # Enforce known voltage-controlled bus magnitudes.
    # -------------------------------------------------------------------------

    if (
        net.ext_grid is not None
        and not net.ext_grid.empty
    ):

        for row in net.ext_grid.loc[
            table_in_service_mask(
                net.ext_grid
            )
        ].itertuples():

            bus = int(
                row.bus
            )

            vm.at[
                bus
            ] = float(
                row.vm_pu
            )

            va.at[
                bus
            ] = float(
                row.va_degree
            )

    if (
        net.gen is not None
        and not net.gen.empty
    ):

        for row in net.gen.loc[
            table_in_service_mask(
                net.gen
            )
        ].itertuples():

            bus = int(
                row.bus
            )

            vm.at[
                bus
            ] = float(
                row.vm_pu
            )

    # -------------------------------------------------------------------------
    # Final sanity
    # -------------------------------------------------------------------------

    vm = vm.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).fillna(
        1.0
    )

    va = va.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).fillna(
        0.0
    )

    warm_start_df = pd.DataFrame(
        {
            "bus": net.bus.index,
            "name": (
                net.bus[
                    "name"
                ].values
                if "name"
                in net.bus.columns
                else ""
            ),
            "in_service": (
                net.bus[
                    "in_service"
                ].values
            ),
            "init_vm_pu": vm.values,
            "init_va_degree": va.values,
        },
        index=net.bus.index,
    )

    save_dataframe(
        warm_start_df,
        output_dir
        / "pandapower_pf_warm_start.csv",
        index=True,
    )

    pf_print(
        app,
        "",
    )

    pf_print(
        app,
        (
            "PowerFactory warm-start prepared:"
        ),
    )

    pf_print(
        app,
        (
            "  valid PF Vm values = "
            f"{metadata['valid_pf_vm_count']}"
        ),
    )

    pf_print(
        app,
        (
            "  valid PF angle values = "
            f"{metadata['valid_pf_va_count']}"
        ),
    )

    pf_print(
        app,
        (
            "  island angle shifts = "
            f"{len(metadata['island_angle_shifts'])}"
        ),
    )

    pf_print(
        app,
        (
            "  active Vm range = "
            f"{vm.loc[~oos].min():.6f} ... "
            f"{vm.loc[~oos].max():.6f} pu"
        ),
    )

    pf_print(
        app,
        (
            "  active Va range = "
            f"{va.loc[~oos].min():.6f} ... "
            f"{va.loc[~oos].max():.6f} deg"
        ),
    )

    return (
        vm,
        va,
        metadata,
    )


# =============================================================================
# FACTS DIAGNOSTICS
# =============================================================================

def count_active_facts(
    net,
) -> dict:

    output = {}

    for element in (
        "svc",
        "tcsc",
        "ssc",
        "vsc",
        "vsc_stacked",
        "vsc_bipolar",
    ):

        if (
            element
            not in net
            or not isinstance(
                net[
                    element
                ],
                pd.DataFrame,
            )
            or net[
                element
            ].empty
        ):

            output[
                element
            ] = 0

            continue

        frame = net[
            element
        ]

        output[
            element
        ] = int(
            table_in_service_mask(
                frame
            ).sum()
        )

    return output


# =============================================================================
# POWER-FLOW INPUT REPAIR / FROZEN PF OPERATING POINT
# =============================================================================

def freeze_voltage_controls_to_pf_operating_point(
    app,
    net,
    pf_snapshot: pd.DataFrame | None,
    output_dir: Path,
) -> dict:
    """
    Freeze voltage-controlled elements to the already-converged PowerFactory
    bus voltages.

    This is deliberately a frozen-control representation. It avoids asking
    pandapower to reproduce remote/station-controller logic before a base
    operating point exists. For ext_grids both magnitude and angle are frozen;
    for gens the controlled voltage magnitude is frozen.
    """

    metadata = {
        "available": False,
        "ext_grid_updated": 0,
        "gen_updated": 0,
    }

    if pf_snapshot is None:
        pf_warn(
            app,
            "PF voltage snapshot unavailable; voltage-control setpoints were not repaired.",
        )
        return metadata

    records = []

    def pf_vm(bus: int) -> float | None:
        try:
            value = float(pf_snapshot.at[int(bus), "vm_pu"])
        except Exception:
            return None
        if not np.isfinite(value) or value <= 0.05 or value >= 2.0:
            return None
        return value

    def pf_va(bus: int) -> float | None:
        try:
            value = float(pf_snapshot.at[int(bus), "va_degree"])
        except Exception:
            return None
        return value if np.isfinite(value) else None

    if (
        hasattr(net, "ext_grid")
        and net.ext_grid is not None
        and not net.ext_grid.empty
    ):
        mask = table_in_service_mask(net.ext_grid)

        for idx, row in net.ext_grid.loc[mask].iterrows():
            bus = int(row["bus"])
            vm = pf_vm(bus)
            va = pf_va(bus)

            old_vm = row.get("vm_pu", np.nan)
            old_va = row.get("va_degree", np.nan)

            if vm is not None:
                net.ext_grid.at[idx, "vm_pu"] = vm

            if va is not None:
                net.ext_grid.at[idx, "va_degree"] = va

            if vm is not None or va is not None:
                metadata["ext_grid_updated"] += 1
                records.append(
                    {
                        "element": "ext_grid",
                        "index": int(idx),
                        "name": safe_text(row.get("name", "")),
                        "bus": bus,
                        "old_vm_pu": old_vm,
                        "new_vm_pu": vm,
                        "old_va_degree": old_va,
                        "new_va_degree": va,
                    }
                )

    if (
        hasattr(net, "gen")
        and net.gen is not None
        and not net.gen.empty
    ):
        mask = table_in_service_mask(net.gen)

        for idx, row in net.gen.loc[mask].iterrows():
            bus = int(row["bus"])
            vm = pf_vm(bus)

            if vm is None:
                continue

            old_vm = row.get("vm_pu", np.nan)
            net.gen.at[idx, "vm_pu"] = vm
            metadata["gen_updated"] += 1

            records.append(
                {
                    "element": "gen",
                    "index": int(idx),
                    "name": safe_text(row.get("name", "")),
                    "bus": bus,
                    "old_vm_pu": old_vm,
                    "new_vm_pu": vm,
                    "old_va_degree": np.nan,
                    "new_va_degree": np.nan,
                }
            )

    metadata["available"] = bool(records)

    frame = pd.DataFrame(records)
    save_dataframe(
        frame,
        output_dir / "pf_frozen_voltage_control_setpoints.csv",
        index=False,
    )

    pf_print(app, "")
    pf_print(app, "Frozen PF voltage-control setpoints applied:")
    pf_print(
        app,
        (
            f"  ext_grid updated = {metadata['ext_grid_updated']}, "
            f"gen updated = {metadata['gen_updated']}"
        ),
    )

    return metadata


def sanitize_powerflow_inputs(
    app,
    net,
    output_dir: Path,
) -> dict:
    """Repair only numerically invalid mandatory PF inputs (NaN/Inf)."""

    records = []

    # Fixed-power element columns where NaN/Inf is never a meaningful PF input.
    element_columns = {
        "load": ("p_mw", "q_mvar"),
        "sgen": ("p_mw", "q_mvar"),
        "gen": ("p_mw",),
        "storage": ("p_mw", "q_mvar"),
        "ward": ("ps_mw", "qs_mvar", "pz_mw", "qz_mvar"),
        "xward": ("ps_mw", "qs_mvar", "pz_mw", "qz_mvar"),
        "shunt": ("p_mw", "q_mvar"),
    }

    for element, columns in element_columns.items():
        if element not in net:
            continue

        frame = net[element]
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            continue

        active = table_in_service_mask(frame)

        for column in columns:
            if column not in frame.columns:
                continue

            values = pd.to_numeric(frame[column], errors="coerce")
            bad = active & ~np.isfinite(values)

            for idx in frame.index[bad]:
                records.append(
                    {
                        "element": element,
                        "index": int(idx),
                        "name": safe_text(frame.at[idx, "name"] if "name" in frame.columns else ""),
                        "column": column,
                        "old_value": safe_text(frame.at[idx, column]),
                        "new_value": 0.0,
                    }
                )
                frame.at[idx, column] = 0.0

    # Mandatory voltage setpoints must also be finite.
    for element, columns in {
        "ext_grid": ("vm_pu", "va_degree"),
        "gen": ("vm_pu",),
    }.items():
        if element not in net:
            continue

        frame = net[element]
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            continue

        active = table_in_service_mask(frame)

        for column in columns:
            if column not in frame.columns:
                continue

            values = pd.to_numeric(frame[column], errors="coerce")
            bad = active & ~np.isfinite(values)

            for idx in frame.index[bad]:
                replacement = 1.0 if column == "vm_pu" else 0.0
                records.append(
                    {
                        "element": element,
                        "index": int(idx),
                        "name": safe_text(frame.at[idx, "name"] if "name" in frame.columns else ""),
                        "column": column,
                        "old_value": safe_text(frame.at[idx, column]),
                        "new_value": replacement,
                    }
                )
                frame.at[idx, column] = replacement

    frame = pd.DataFrame(records)
    save_dataframe(
        frame,
        output_dir / "sanitized_powerflow_inputs.csv",
        index=False,
    )

    if records:
        pf_warn(
            app,
            f"Repaired {len(records)} non-finite pandapower input value(s).",
        )
    else:
        pf_print(app, "No non-finite mandatory pandapower PF inputs found.")

    return {
        "repaired_value_count": int(len(records)),
    }


def capture_homotopy_targets(net) -> dict[tuple[str, str], pd.Series]:
    """Capture active/fixed injection columns that can safely be continuation-scaled."""

    targets: dict[tuple[str, str], pd.Series] = {}

    # Scale only explicit P/Q injections. Do NOT scale shunts, wards or xwards:
    # they contribute admittance / internal equivalent-network terms and
    # scaling them to zero can create singular intermediate models.
    specs = {
        "load": ("p_mw", "q_mvar"),
        "sgen": ("p_mw", "q_mvar"),
        "gen": ("p_mw",),
        "storage": ("p_mw", "q_mvar"),
    }

    for element, columns in specs.items():
        if element not in net:
            continue

        frame = net[element]
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            continue

        for column in columns:
            if column not in frame.columns:
                continue

            targets[(element, column)] = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).fillna(0.0).copy()

    return targets


def apply_homotopy_factor(
    net,
    targets: dict[tuple[str, str], pd.Series],
    factor: float,
) -> None:
    factor = float(factor)

    for (element, column), target in targets.items():
        common = net[element].index.intersection(target.index)
        net[element].loc[common, column] = target.loc[common] * factor


def restore_homotopy_targets(
    net,
    targets: dict[tuple[str, str], pd.Series],
) -> None:
    for (element, column), target in targets.items():
        common = net[element].index.intersection(target.index)
        net[element].loc[common, column] = target.loc[common]


# =============================================================================
# ROBUST POWER-FLOW CONVERGENCE
# =============================================================================

def run_powerflow_attempt(
    app,
    net,
    description: str,
    algorithm: str,
    tolerance_mva: float,
    max_iteration: int,
    voltage_depend_loads: bool,
    init_mode: str,
    vm_init: pd.Series | None = None,
    va_init: pd.Series | None = None,
) -> bool:

    pf_print(app, "")
    pf_print(app, f"POWER FLOW ATTEMPT: {description}")
    pf_print(
        app,
        (
            f"  algorithm={algorithm}, "
            f"tolerance_mva={tolerance_mva}, "
            f"max_iteration={max_iteration}, "
            f"voltage_depend_loads={voltage_depend_loads}, "
            f"init={init_mode}"
        ),
    )

    kwargs = {
        "algorithm": algorithm,
        "calculate_voltage_angles": True,
        "max_iteration": max_iteration,
        "tolerance_mva": tolerance_mva,
        "enforce_q_lims": ENFORCE_Q_LIMS,
        "check_connectivity": CHECK_CONNECTIVITY,
        "voltage_depend_loads": voltage_depend_loads,
        "distributed_slack": False,
        "numba": USE_NUMBA,
        "lightsim2grid": USE_LIGHTSIM2GRID,
        "run_control": False,
    }

    try:
        if init_mode == "pf":
            pp.runpp(
                net,
                init="auto",
                init_vm_pu=vm_init,
                init_va_degree=va_init,
                **kwargs,
            )

        elif init_mode == "results":
            pp.runpp(
                net,
                init="results",
                **kwargs,
            )

        elif init_mode == "auto":
            pp.runpp(
                net,
                init="auto",
                **kwargs,
            )

        else:
            raise ValueError(f"Unsupported internal init_mode={init_mode}")

    except LoadflowNotConverged as exc:
        message = f"Power flow attempt FAILED: {description}: {exc}"
        pf_warn(app, message)
        pf_print(app, "WARNING: " + message)
        return False

    except Exception as exc:
        message = (
            f"Power flow attempt ERROR: {description}: "
            f"{type(exc).__name__}: {exc}"
        )
        pf_warn(app, message)
        pf_print(app, "WARNING: " + message)
        return False

    if not bool(net.converged):
        message = f"Power flow attempt did not converge: {description}"
        pf_warn(app, message)
        pf_print(app, "WARNING: " + message)
        return False

    iterations = None
    try:
        iterations = net._ppc.get("iterations", None)
    except Exception:
        pass

    pf_print(
        app,
        f"Power flow attempt CONVERGED: {description}; iterations={iterations}",
    )
    return True


def run_homotopy_powerflow(
    app,
    net,
    vm_init: pd.Series,
    va_init: pd.Series,
) -> bool:
    """
    Adaptive continuation from a lightly-loaded network to the full converted
    PF operating point. This is a numerical fallback, not a model change: all
    original values are restored at factor 1.0.
    """

    targets = capture_homotopy_targets(net)
    if not targets:
        return False

    pf_print(app, "")
    pf_print(app, "POWER FLOW ATTEMPT: D - adaptive homotopy continuation")

    # Never start at exactly zero injection. Some imported equivalent models
    # become singular at the artificial zero-power point even though the
    # actual operating point is well-defined.
    factor = float(HOMOTOPY_START_FACTOR)
    step = float(HOMOTOPY_INITIAL_STEP)
    have_solution = False

    try:
        apply_homotopy_factor(net, targets, factor)

        have_solution = run_powerflow_attempt(
            app,
            net,
            description=f"D0 - homotopy light-injection base ({factor:.4f})",
            algorithm="nr",
            tolerance_mva=BOOTSTRAP_TOLERANCE_MVA,
            max_iteration=BOOTSTRAP_MAX_ITERATION,
            voltage_depend_loads=False,
            init_mode="pf",
            vm_init=vm_init,
            va_init=va_init,
        )

        if not have_solution and USE_IWAMOTO_FALLBACK:
            have_solution = run_powerflow_attempt(
                app,
                net,
                description=f"D0b - Iwamoto homotopy light-injection base ({factor:.4f})",
                algorithm="iwamoto_nr",
                tolerance_mva=BOOTSTRAP_TOLERANCE_MVA,
                max_iteration=BOOTSTRAP_MAX_ITERATION,
                voltage_depend_loads=False,
                init_mode="pf",
                vm_init=vm_init,
                va_init=va_init,
            )

        if not have_solution:
            return False

        # Keep an explicit copy of the last *converged* voltage state. A failed
        # runpp() attempt may partially overwrite result tables, so using
        # init="results" after a failed continuation step is unsafe.
        last_vm = pd.to_numeric(
            net.res_bus["vm_pu"], errors="coerce"
        ).reindex(net.bus.index)
        last_va = pd.to_numeric(
            net.res_bus["va_degree"], errors="coerce"
        ).reindex(net.bus.index)
        last_vm = last_vm.where(np.isfinite(last_vm), vm_init).fillna(1.0)
        last_va = last_va.where(np.isfinite(last_va), va_init).fillna(0.0)

        while factor < 1.0 - 1e-12:
            trial = min(1.0, factor + step)
            apply_homotopy_factor(net, targets, trial)

            success = run_powerflow_attempt(
                app,
                net,
                description=f"D - homotopy factor {trial:.4f}",
                algorithm="nr",
                tolerance_mva=BOOTSTRAP_TOLERANCE_MVA,
                max_iteration=BOOTSTRAP_MAX_ITERATION,
                voltage_depend_loads=False,
                init_mode="pf",
                vm_init=last_vm,
                va_init=last_va,
            )

            if success:
                factor = trial
                last_vm = pd.to_numeric(
                    net.res_bus["vm_pu"], errors="coerce"
                ).reindex(net.bus.index)
                last_va = pd.to_numeric(
                    net.res_bus["va_degree"], errors="coerce"
                ).reindex(net.bus.index)
                last_vm = last_vm.where(np.isfinite(last_vm), vm_init).fillna(1.0)
                last_va = last_va.where(np.isfinite(last_va), va_init).fillna(0.0)
                step = min(max(step * 1.5, HOMOTOPY_MIN_STEP), 0.25)
                continue

            # Revert model parameters to the last solved continuation point.
            # The voltage seed remains the separately saved last successful state.
            apply_homotopy_factor(net, targets, factor)
            step *= 0.5

            if step < HOMOTOPY_MIN_STEP:
                pf_warn(
                    app,
                    (
                        "Homotopy continuation stalled at "
                        f"factor={factor:.6f}; next step={step:.6f}."
                    ),
                )
                return False

        restore_homotopy_targets(net, targets)

        return run_powerflow_attempt(
            app,
            net,
            description="D2 - final exact NR after homotopy",
            algorithm="nr",
            tolerance_mva=TOLERANCE_MVA,
            max_iteration=MAX_ITERATION,
            voltage_depend_loads=VOLTAGE_DEPEND_LOADS,
            init_mode="pf",
            vm_init=last_vm,
            va_init=last_va,
        )

    finally:
        restore_homotopy_targets(net, targets)


def solve_initial_operating_point(
    app,
    net,
    vm_init: pd.Series,
    va_init: pd.Series,
) -> dict:

    facts = count_active_facts(net)
    total_facts = int(sum(facts.values()))

    pf_print(app, "")
    pf_print(
        app,
        f"Active pandapower FACTS / converter elements: {facts}",
    )

    attempts = []

    # A) Exact NR from the repaired PowerFactory operating point.
    success = run_powerflow_attempt(
        app,
        net,
        description="A - exact NR from frozen PowerFactory operating point",
        algorithm="nr",
        tolerance_mva=TOLERANCE_MVA,
        max_iteration=MAX_ITERATION,
        voltage_depend_loads=VOLTAGE_DEPEND_LOADS,
        init_mode="pf",
        vm_init=vm_init,
        va_init=va_init,
    )
    attempts.append({"name": "exact_nr_from_pf", "success": bool(success)})

    if success:
        return {
            "success": True,
            "method": "exact_nr_from_pf",
            "attempts": attempts,
            "facts": facts,
        }

    # B) Iwamoto NR from PF voltages (supported here only without active FACTS).
    if USE_IWAMOTO_FALLBACK and total_facts == 0:
        success = run_powerflow_attempt(
            app,
            net,
            description="B - Iwamoto NR from PowerFactory voltages",
            algorithm="iwamoto_nr",
            tolerance_mva=BOOTSTRAP_TOLERANCE_MVA,
            max_iteration=BOOTSTRAP_MAX_ITERATION,
            voltage_depend_loads=VOLTAGE_DEPEND_LOADS,
            init_mode="pf",
            vm_init=vm_init,
            va_init=va_init,
        )
        attempts.append({"name": "iwamoto_from_pf", "success": bool(success)})

        if success:
            success_final = run_powerflow_attempt(
                app,
                net,
                description="B2 - exact NR from Iwamoto result",
                algorithm="nr",
                tolerance_mva=TOLERANCE_MVA,
                max_iteration=MAX_ITERATION,
                voltage_depend_loads=VOLTAGE_DEPEND_LOADS,
                init_mode="results",
            )
            attempts.append(
                {
                    "name": "exact_nr_after_iwamoto",
                    "success": bool(success_final),
                }
            )

            if success_final:
                return {
                    "success": True,
                    "method": "iwamoto_then_exact_nr",
                    "attempts": attempts,
                    "facts": facts,
                }

    elif USE_IWAMOTO_FALLBACK and total_facts > 0:
        pf_print(
            app,
            "Iwamoto fallback skipped because active pandapower FACTS elements exist.",
        )

    # C) Constant-power bootstrap. Kept for compatibility if the user later
    # re-enables voltage-dependent loads.
    if USE_CONSTANT_POWER_BOOTSTRAP:
        success = run_powerflow_attempt(
            app,
            net,
            description="C - constant-power NR bootstrap from PowerFactory voltages",
            algorithm="nr",
            tolerance_mva=BOOTSTRAP_TOLERANCE_MVA,
            max_iteration=BOOTSTRAP_MAX_ITERATION,
            voltage_depend_loads=False,
            init_mode="pf",
            vm_init=vm_init,
            va_init=va_init,
        )
        attempts.append(
            {"name": "constant_power_bootstrap", "success": bool(success)}
        )

        if success:
            success_final = run_powerflow_attempt(
                app,
                net,
                description="C2 - exact NR after constant-power bootstrap",
                algorithm="nr",
                tolerance_mva=TOLERANCE_MVA,
                max_iteration=MAX_ITERATION,
                voltage_depend_loads=VOLTAGE_DEPEND_LOADS,
                init_mode="results",
            )
            attempts.append(
                {
                    "name": "exact_nr_after_constant_power",
                    "success": bool(success_final),
                }
            )

            if success_final:
                return {
                    "success": True,
                    "method": "constant_power_bootstrap_then_exact_nr",
                    "attempts": attempts,
                    "facts": facts,
                }

    # D) Adaptive continuation. This can recover a difficult but valid solved
    # network without changing the final target model.
    if USE_HOMOTOPY_FALLBACK:
        success = run_homotopy_powerflow(
            app,
            net,
            vm_init,
            va_init,
        )
        attempts.append({"name": "homotopy", "success": bool(success)})

        if success:
            return {
                "success": True,
                "method": "adaptive_homotopy_then_exact_nr",
                "attempts": attempts,
                "facts": facts,
            }

    return {
        "success": False,
        "method": None,
        "attempts": attempts,
        "facts": facts,
    }


# =============================================================================
# CONTROLLER SUMMARY
# =============================================================================

def print_controller_summary(
    app,
    net,
) -> None:

    if (
        not hasattr(
            net,
            "controller",
        )
        or net.controller is None
        or net.controller.empty
    ):

        pf_print(
            app,
            "No pandapower controllers were exported.",
        )

        return

    pf_print(
        app,
        (
            "Converted pandapower controllers: "
            f"{len(net.controller)}"
        ),
    )

    columns = [
        column
        for column
        in (
            "object",
            "in_service",
            "order",
            "level",
            "initial_run",
        )
        if column
        in net.controller.columns
    ]

    frame = (
        net.controller[
            columns
        ]
        if columns
        else net.controller
    )

    pf_print(
        app,
        frame.head(
            PRINT_TABLE_ROWS
        ).to_string(),
    )


# =============================================================================
# POWERFACTORY-POINT JACOBIAN FALLBACK
# =============================================================================

def build_pf_point_internal_model(
    app,
    net,
    vm_init: pd.Series,
    va_init: pd.Series,
    output_dir: Path,
):
    """
    Build the pandapower/PYPOWER internal admittance model without requiring a
    converged pandapower Newton solve, then evaluate that model directly at the
    already-converged PowerFactory voltage vector.

    This is suitable for the classical frozen-control AC Jacobian because the
    Jacobian is a derivative of the network P/Q equations with respect to
    voltage state. Newton convergence is not a prerequisite for evaluating the
    derivative at a specified voltage vector.

    The mismatch between the converted pandapower model and the PowerFactory
    operating point is exported explicitly to pf_point_power_mismatch.csv.
    """

    facts = count_active_facts(net)
    if int(sum(facts.values())) != 0:
        raise RuntimeError(
            "PF-point Jacobian fallback currently requires zero active "
            f"pandapower FACTS/converter elements; found {facts}."
        )

    # runpp() attempts already populated net._options. _pd2ppc uses these
    # options to build the same internal model that NR would use.
    if not hasattr(net, "_options") or not isinstance(net._options, dict):
        raise RuntimeError(
            "pandapower _options are unavailable for PF-point fallback."
        )

    ppc, ppci = _pd2ppc(net)

    base_mva = float(ppci["baseMVA"])
    bus = np.asarray(ppci["bus"])
    branch = np.asarray(ppci["branch"])
    gen = np.asarray(ppci["gen"])

    Ybus, Yf, Yt = makeYbus(
        base_mva,
        bus,
        branch,
    )

    Ybus = to_local_csr(Ybus, dtype=np.complex128)
    Yf = to_local_csr(Yf, dtype=np.complex128)
    Yt = to_local_csr(Yt, dtype=np.complex128)

    ref, pv, pq = bustypes(bus, gen)
    ref = np.asarray(ref, dtype=np.int64).reshape(-1)
    pv = np.asarray(pv, dtype=np.int64).reshape(-1)
    pq = np.asarray(pq, dtype=np.int64).reshape(-1)

    nbus = int(bus.shape[0])

    # Start with the ppci voltage setpoints, then overwrite every mapped active
    # pandapower bus with its aligned PowerFactory result.
    vm = np.asarray(bus[:, VM], dtype=float).copy()
    va = np.asarray(bus[:, VA], dtype=float).copy()

    lookup = net._pd2ppc_lookups.get("bus", None)
    if lookup is None:
        raise RuntimeError("pandapower bus lookup is unavailable in PF-point fallback.")

    mapped_count = 0
    for pp_bus in net.bus.index:
        try:
            internal_bus = int(lookup[int(pp_bus)])
        except Exception:
            continue

        if internal_bus < 0 or internal_bus >= nbus:
            continue

        try:
            vm_value = float(vm_init.at[pp_bus])
            va_value = float(va_init.at[pp_bus])
        except Exception:
            continue

        if np.isfinite(vm_value) and vm_value > 0.05:
            vm[internal_bus] = vm_value

        if np.isfinite(va_value):
            va[internal_bus] = va_value

        mapped_count += 1

    if not np.all(np.isfinite(vm)) or np.any(vm <= 0.0):
        raise RuntimeError("PF-point fallback produced invalid voltage magnitudes.")

    if not np.all(np.isfinite(va)):
        raise RuntimeError("PF-point fallback produced invalid voltage angles.")

    V = vm * np.exp(1j * np.deg2rad(va))

    # Model residual at the PF voltage vector. A non-zero residual means the
    # converted pp model is not exactly equivalent to the PF model; it does NOT
    # prevent evaluation of the classical voltage Jacobian.
    S_calc = V * np.conj(Ybus.dot(V))
    S_spec = makeSbus(
        base_mva,
        bus,
        gen,
        vm=(vm if VOLTAGE_DEPEND_LOADS else None),
    )
    mismatch = S_calc - S_spec

    # Build a reverse ppci -> pandapower bus map for readable diagnostics.
    reverse: dict[int, list[int]] = {}
    for pp_bus in net.bus.index:
        try:
            internal_bus = int(lookup[int(pp_bus)])
        except Exception:
            continue
        if 0 <= internal_bus < nbus:
            reverse.setdefault(internal_bus, []).append(int(pp_bus))

    type_names = {
        int(REF): "REF",
        int(PV): "PV",
        int(PQ): "PQ",
    }

    records = []
    for internal_bus in range(nbus):
        pp_indices = sorted(reverse.get(internal_bus, []))
        names = []
        for pp_bus in pp_indices:
            try:
                names.append(safe_text(net.bus.at[pp_bus, "name"]))
            except Exception:
                pass

        bus_type_value = int(round(float(bus[internal_bus, BUS_TYPE])))
        records.append(
            {
                "ppci_bus": int(internal_bus),
                "bus_type": type_names.get(bus_type_value, f"TYPE_{bus_type_value}"),
                "pp_bus_indices": ";".join(str(x) for x in pp_indices),
                "bus_names": join_unique(names),
                "vm_pu_pf_point": float(vm[internal_bus]),
                "va_degree_pf_point": float(va[internal_bus]),
                "p_mismatch_pu": float(np.real(mismatch[internal_bus])),
                "q_mismatch_pu": float(np.imag(mismatch[internal_bus])),
                "p_mismatch_mw": float(np.real(mismatch[internal_bus]) * base_mva),
                "q_mismatch_mvar": float(np.imag(mismatch[internal_bus]) * base_mva),
            }
        )

    mismatch_df = pd.DataFrame(records)
    save_dataframe(
        mismatch_df,
        output_dir / "pf_point_power_mismatch.csv",
        index=False,
    )

    pvpq = np.concatenate((pv, pq)).astype(np.int64, copy=False)
    p_eq = np.abs(np.real(mismatch[pvpq])) if len(pvpq) else np.array([])
    q_eq = np.abs(np.imag(mismatch[pq])) if len(pq) else np.array([])

    max_p_pu = float(np.max(p_eq)) if p_eq.size else 0.0
    max_q_pu = float(np.max(q_eq)) if q_eq.size else 0.0
    max_eq_pu = max(max_p_pu, max_q_pu)

    metrics = {
        "used": True,
        "mapped_pf_bus_count": int(mapped_count),
        "internal_bus_count": int(nbus),
        "ref_count": int(len(ref)),
        "pv_count": int(len(pv)),
        "pq_count": int(len(pq)),
        "max_abs_p_equation_mismatch_pu": max_p_pu,
        "max_abs_q_equation_mismatch_pu": max_q_pu,
        "max_abs_equation_mismatch_pu": float(max_eq_pu),
        "max_abs_equation_mismatch_mva": float(max_eq_pu * base_mva),
    }

    pf_warn(
        app,
        (
            "pandapower NR did not converge; evaluating the classical AC "
            "Jacobian directly at the converged PowerFactory voltage point."
        ),
    )
    pf_print(
        app,
        (
            "PF-point model residual: "
            f"max |P/Q equation mismatch| = {metrics['max_abs_equation_mismatch_mva']:.6g} MVA"
        ),
    )

    if not mismatch_df.empty:
        worst = mismatch_df.copy()
        worst["equation_mismatch_abs_mva"] = np.maximum(
            worst["p_mismatch_mw"].abs(),
            worst["q_mismatch_mvar"].abs(),
        )
        worst = worst.sort_values(
            "equation_mismatch_abs_mva",
            ascending=False,
        ).head(PRINT_TABLE_ROWS)
        pf_print(
            app,
            "Worst PF-point converted-model bus residuals:",
        )
        pf_print(
            app,
            worst[
                [
                    "ppci_bus",
                    "bus_type",
                    "bus_names",
                    "p_mismatch_mw",
                    "q_mismatch_mvar",
                    "equation_mismatch_abs_mva",
                ]
            ].to_string(index=False),
        )

    internal = dict(ppci.get("internal", {}))
    internal.update(
        {
            "V": V,
            "bus": bus,
            "Ybus": Ybus,
            "Yf": Yf,
            "Yt": Yt,
            "ref": ref,
            "pv": pv,
            "pq": pq,
        }
    )

    # Make the fallback structure available to the existing Jacobian/mapping
    # export code through the same net._ppc interface used after a normal NR.
    ppc["internal"] = internal
    ppc["success"] = False
    ppc["iterations"] = None
    net["_ppc"] = ppc

    return internal, metrics, mismatch_df


# =============================================================================
# SPARSE MATRIX NORMALIZATION
# =============================================================================

def to_local_csr(
    matrix_raw,
    dtype=None,
) -> sparse.csr_matrix:

    if matrix_raw is None:

        raise RuntimeError(
            "Cannot convert None to CSR matrix."
        )

    if hasattr(
        matrix_raw,
        "tocsr",
    ):

        try:

            matrix = matrix_raw.tocsr(
                copy=False
            )

        except TypeError:

            matrix = matrix_raw.tocsr()

        if all(
            hasattr(
                matrix,
                attribute,
            )
            for attribute
            in (
                "data",
                "indices",
                "indptr",
                "shape",
            )
        ):

            data = np.asarray(
                matrix.data
            )

            if dtype is not None:

                data = np.asarray(
                    data,
                    dtype=dtype,
                )

            indices = np.asarray(
                matrix.indices,
                dtype=np.int64,
            )

            indptr = np.asarray(
                matrix.indptr,
                dtype=np.int64,
            )

            result = sparse.csr_matrix(
                (
                    data,
                    indices,
                    indptr,
                ),
                shape=tuple(
                    matrix.shape
                ),
            )

            result.sort_indices()

            return result

    array = np.asarray(
        matrix_raw
    )

    if (
        array.ndim
        != 2
    ):

        raise RuntimeError(
            (
                "Matrix is not 2-D: "
                f"type={type(matrix_raw)}, "
                f"shape={array.shape}, "
                f"dtype={array.dtype}"
            )
        )

    if dtype is not None:

        array = np.asarray(
            array,
            dtype=dtype,
        )

    return sparse.csr_matrix(
        array
    )


# =============================================================================
# CLASSICAL FINAL AC JACOBIAN
# =============================================================================

def build_effective_frozen_ybus(
    app,
    internal: dict,
):

    Ybus = to_local_csr(
        internal[
            "Ybus"
        ],
        dtype=np.complex128,
    )

    included = [
        "Ybus"
    ]

    for key in (
        "Ybus_svc",
        "Ybus_tcsc",
        "Ybus_ssc",
        "Ybus_vsc",
    ):

        matrix_raw = internal.get(
            key,
            None,
        )

        if matrix_raw is None:

            continue

        try:

            matrix = to_local_csr(
                matrix_raw,
                dtype=np.complex128,
            )

        except Exception as exc:

            pf_warn(
                app,
                (
                    f"Could not include "
                    f"{key}: {exc}"
                ),
            )

            continue

        if (
            matrix.shape
            != Ybus.shape
        ):

            continue

        if matrix.nnz:

            Ybus = (
                Ybus
                + matrix
            ).tocsr()

            included.append(
                key
            )

    Ybus.sort_indices()

    pf_print(
        app,
        (
            "Frozen-control effective Ybus: "
            f"shape={Ybus.shape}, "
            f"nnz={Ybus.nnz}, "
            f"components={included}"
        ),
    )

    return (
        Ybus,
        included,
    )


def build_classical_ac_jacobian(
    app,
    internal: dict,
):

    required = [
        "V",
        "bus",
        "Ybus",
        "ref",
        "pv",
        "pq",
    ]

    missing = [
        key
        for key
        in required
        if key
        not in internal
    ]

    if missing:

        raise RuntimeError(
            (
                "Missing pandapower internal data: "
                f"{missing}"
            )
        )

    V = np.asarray(
        internal[
            "V"
        ],
        dtype=np.complex128,
    ).reshape(
        -1
    )

    ref = np.asarray(
        internal[
            "ref"
        ],
        dtype=np.int64,
    ).reshape(
        -1
    )

    pv = np.asarray(
        internal[
            "pv"
        ],
        dtype=np.int64,
    ).reshape(
        -1
    )

    pq = np.asarray(
        internal[
            "pq"
        ],
        dtype=np.int64,
    ).reshape(
        -1
    )

    pvpq = np.concatenate(
        (
            pv,
            pq,
        )
    ).astype(
        np.int64,
        copy=False,
    )

    if (
        len(
            pvpq
        )
        == 0
    ):

        raise RuntimeError(
            "No PV/PQ state buses."
        )

    Ybus, included = (
        build_effective_frozen_ybus(
            app,
            internal,
        )
    )

    refpvpq = np.concatenate(
        (
            ref,
            pvpq,
        )
    ).astype(
        np.int64,
        copy=False,
    )

    bus_internal = np.asarray(
        internal[
            "bus"
        ]
    )

    if (
        bus_internal.shape[
            1
        ]
        > int(
            SL_FAC
        )
    ):

        slack_weights = np.asarray(
            bus_internal[
                :,
                SL_FAC,
            ],
            dtype=np.float64,
        )

    else:

        slack_weights = np.zeros(
            len(
                V
            ),
            dtype=np.float64,
        )

    dummy_lookup = np.zeros(
        len(
            V
        ),
        dtype=np.int64,
    )

    J = create_jacobian_matrix(
        Ybus,
        V,
        ref,
        refpvpq,
        pvpq,
        pq,
        None,
        dummy_lookup,
        len(
            ref
        ),
        len(
            pv
        ),
        len(
            pq
        ),
        False,
        slack_weights,
        False,
    )

    J = to_local_csr(
        J,
        dtype=np.float64,
    )

    J.sort_indices()

    expected_dimension = (
        len(
            pvpq
        )
        + len(
            pq
        )
    )

    expected_shape = (
        expected_dimension,
        expected_dimension,
    )

    if (
        J.shape
        != expected_shape
    ):

        raise RuntimeError(
            (
                "Unexpected Jacobian shape: "
                f"{J.shape}; "
                f"expected={expected_shape}"
            )
        )

    if (
        J.nnz == 0
    ):

        raise RuntimeError(
            "Jacobian is empty."
        )

    if not np.all(
        np.isfinite(
            J.data
        )
    ):

        raise RuntimeError(
            "Jacobian contains NaN or Inf."
        )

    pf_print(
        app,
        (
            "Classical AC Jacobian constructed:"
        ),
    )

    pf_print(
        app,
        (
            "  REF/PV/PQ = "
            f"{len(ref)}/"
            f"{len(pv)}/"
            f"{len(pq)}"
        ),
    )

    pf_print(
        app,
        (
            f"  shape = {J.shape}"
        ),
    )

    pf_print(
        app,
        (
            f"  nnz = {J.nnz}"
        ),
    )

    return (
        J,
        ref,
        pv,
        pq,
        included,
    )


# =============================================================================
# INTERNAL BUS MAPPING
# =============================================================================

def get_pp_to_internal_bus_lookup(
    net,
) -> dict[int, int]:

    lookups = getattr(
        net,
        "_pd2ppc_lookups",
        None,
    )

    if (
        not isinstance(
            lookups,
            dict,
        )
        or "bus"
        not in lookups
    ):

        raise RuntimeError(
            (
                "net._pd2ppc_lookups['bus'] "
                "is unavailable."
            )
        )

    lookup = lookups[
        "bus"
    ]

    mapping = {}

    for pp_idx in net.bus.index:

        try:

            internal_idx = int(
                lookup[
                    int(
                        pp_idx
                    )
                ]
            )

        except Exception:

            continue

        if (
            internal_idx
            >= 0
        ):

            mapping[
                int(
                    pp_idx
                )
            ] = internal_idx

    return mapping


def build_internal_bus_mapping(
    net,
    internal_bus: np.ndarray,
):

    pp_to_internal = (
        get_pp_to_internal_bus_lookup(
            net
        )
    )

    internal_count = int(
        internal_bus.shape[
            0
        ]
    )

    reverse = {}

    for (
        pp_idx,
        internal_idx,
    ) in pp_to_internal.items():

        if (
            0
            <= internal_idx
            < internal_count
        ):

            reverse.setdefault(
                internal_idx,
                [],
            ).append(
                pp_idx
            )

    records = []

    labels = {}

    type_names = {
        int(
            REF
        ): "REF",
        int(
            PV
        ): "PV",
        int(
            PQ
        ): "PQ",
    }

    for internal_idx in range(
        internal_count
    ):

        pp_indices = sorted(
            reverse.get(
                internal_idx,
                [],
            )
        )

        names = []

        for pp_idx in pp_indices:

            try:

                names.append(
                    safe_text(
                        net.bus.at[
                            pp_idx,
                            "name",
                        ]
                    )
                )

            except Exception:

                pass

        if names:

            core = join_unique(
                names
            )

        elif pp_indices:

            core = (
                "pp_bus_"
                + "_".join(
                    str(
                        index
                    )
                    for index
                    in pp_indices
                )
            )

        else:

            core = "AUX_INTERNAL_BUS"

        label = (
            f"{core} "
            f"[ppci={internal_idx}]"
        )

        labels[
            internal_idx
        ] = label

        bus_type = int(
            round(
                float(
                    internal_bus[
                        internal_idx,
                        BUS_TYPE,
                    ]
                )
            )
        )

        records.append(
            {
                "ppci_row": int(
                    internal_idx
                ),
                "ppc_bus_number": int(
                    round(
                        float(
                            internal_bus[
                                internal_idx,
                                BUS_I,
                            ]
                        )
                    )
                ),
                "bus_type": type_names.get(
                    bus_type,
                    f"TYPE_{bus_type}",
                ),
                "pp_bus_indices": ";".join(
                    str(
                        index
                    )
                    for index
                    in pp_indices
                ),
                "label": label,
            }
        )

    return (
        pd.DataFrame(
            records
        ),
        labels,
    )


def make_jacobian_labels(
    pv,
    pq,
    labels,
):

    pvpq = np.concatenate(
        (
            pv,
            pq,
        )
    ).astype(
        int
    )

    rows = []

    columns = []

    index = 0

    for bus in pvpq:

        label = labels.get(
            int(
                bus
            ),
            f"PPCI_BUS_{bus}",
        )

        rows.append(
            {
                "j_index": index,
                "quantity": "P",
                "unit": "p.u.",
                "ppci_bus": int(
                    bus
                ),
                "bus_label": label,
            }
        )

        columns.append(
            {
                "j_index": index,
                "quantity": "delta",
                "unit": "rad",
                "ppci_bus": int(
                    bus
                ),
                "bus_label": label,
            }
        )

        index += 1

    for bus in pq:

        label = labels.get(
            int(
                bus
            ),
            f"PPCI_BUS_{bus}",
        )

        rows.append(
            {
                "j_index": index,
                "quantity": "Q",
                "unit": "p.u.",
                "ppci_bus": int(
                    bus
                ),
                "bus_label": label,
            }
        )

        columns.append(
            {
                "j_index": index,
                "quantity": "U",
                "unit": "p.u.",
                "ppci_bus": int(
                    bus
                ),
                "bus_label": label,
            }
        )

        index += 1

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            columns
        ),
    )


# =============================================================================
# PF vs PANDAPOWER VOLTAGE VALIDATION
# =============================================================================

def make_voltage_validation(
    net,
    pf_snapshot,
):

    if (
        pf_snapshot is None
        or not pf_snapshot.index.equals(
            net.bus.index
        )
    ):

        return (
            None,
            {
                "available": False,
                "reason": (
                    "PF snapshot unavailable "
                    "or index mismatch."
                ),
            },
        )

    result = pd.DataFrame(
        index=net.bus.index
    )

    result[
        "name"
    ] = net.bus[
        "name"
    ]

    result[
        "in_service"
    ] = net.bus[
        "in_service"
    ]

    result[
        "vm_pu_pf"
    ] = pf_snapshot[
        "vm_pu"
    ]

    result[
        "va_degree_pf"
    ] = pf_snapshot[
        "va_degree"
    ]

    result[
        "vm_pu_pp"
    ] = net.res_bus[
        "vm_pu"
    ]

    result[
        "va_degree_pp"
    ] = net.res_bus[
        "va_degree"
    ]

    result[
        "vm_pu_diff"
    ] = (
        result[
            "vm_pu_pp"
        ]
        - result[
            "vm_pu_pf"
        ]
    )

    result[
        "va_degree_diff_raw"
    ] = (
        result[
            "va_degree_pp"
        ]
        - result[
            "va_degree_pf"
        ]
    )

    active = result.loc[
        result[
            "in_service"
        ].fillna(
            False
        ).astype(
            bool
        )
    ]

    vm_diff = (
        active[
            "vm_pu_diff"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    metrics = {
        "available": True,
        "active_bus_count": int(
            len(
                active
            )
        ),
        "max_abs_vm_pu_diff": (
            float(
                vm_diff.abs().max()
            )
            if not vm_diff.empty
            else None
        ),
        "rms_vm_pu_diff": (
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            vm_diff.to_numpy(
                                dtype=float
                            )
                        )
                    )
                )
            )
            if not vm_diff.empty
            else None
        ),
    }

    return (
        result.reset_index(),
        metrics,
    )


# =============================================================================
# SPARSE EXPORT
# =============================================================================

def export_sparse_coo_csv(
    path: Path,
    matrix,
):

    coo = matrix.tocoo(
        copy=False
    )

    pd.DataFrame(
        {
            "row": coo.row,
            "column": coo.col,
            "value": coo.data,
        }
    ).to_csv(
        path,
        index=False,
    )


def sparse_stats(
    matrix,
):

    matrix = matrix.tocsr(
        copy=False
    )

    total = (
        matrix.shape[
            0
        ]
        * matrix.shape[
            1
        ]
    )

    return {
        "shape": [
            int(
                matrix.shape[
                    0
                ]
            ),
            int(
                matrix.shape[
                    1
                ]
            ),
        ],
        "nnz": int(
            matrix.nnz
        ),
        "density": (
            float(
                matrix.nnz
                / total
            )
            if total
            else 0.0
        ),
        "max_abs": (
            float(
                np.max(
                    np.abs(
                        matrix.data
                    )
                )
            )
            if matrix.nnz
            else None
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    started = time.time()

    # =========================================================================
    # POWERFACTORY CONTEXT
    # =========================================================================

    app = powerfactory.GetApplication()

    if app is None:

        raise RuntimeError(
            "PowerFactory application is unavailable."
        )

    project = app.GetActiveProject()

    study_case = app.GetActiveStudyCase()

    if (
        project is None
        or study_case is None
    ):

        raise RuntimeError(
            (
                "Active PowerFactory project "
                "or Study Case is missing."
            )
        )

    project_name = str(
        project.loc_name
    )

    study_case_name = str(
        study_case.loc_name
    )

    output_dir = (
        OUTPUT_ROOT
        / safe_project_folder_name(
            project_name
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pf_print(
        app,
        "=" * 72,
    )

    pf_print(
        app,
        (
            "POWERFACTORY -> PANDAPOWER "
            "CLASSICAL AC JACOBIAN EXPORT"
        ),
    )

    pf_print(
        app,
        (
            f"Project: "
            f"{project_name}"
        ),
    )

    pf_print(
        app,
        (
            f"Study Case: "
            f"{study_case_name}"
        ),
    )

    pf_print(
        app,
        (
            f"Output: "
            f"{output_dir}"
        ),
    )

    pf_print(
        app,
        (
            "pandapower version: "
            f"{package_version('pandapower')}"
        ),
    )

    pf_print(
        app,
        (
            "scipy version: "
            f"{scipy.__version__}"
        ),
    )

    pf_print(
        app,
        (
            "numpy version: "
            f"{np.__version__}"
        ),
    )

    # =========================================================================
    # PRE-SCAN ElmSfilt
    # =========================================================================

    elm_sfilt_records = (
        scan_powerfactory_elmsfilt(
            app
        )
    )

    # =========================================================================
    # 1) POWERFACTORY LOAD FLOW
    # =========================================================================

    pf_print(
        app,
        (
            "1/6 Running PowerFactory "
            "base Load Flow ..."
        ),
    )

    ldf = app.GetFromStudyCase(
        "ComLdf"
    )

    if ldf is None:

        raise RuntimeError(
            "ComLdf is unavailable."
        )

    if int(
        ldf.Execute()
    ) != 0:

        raise RuntimeError(
            "PowerFactory load flow did not converge."
        )

    # =========================================================================
    # 2) CONVERSION
    # =========================================================================

    pf_print(
        app,
        (
            "2/6 Converting active "
            "PowerFactory model to pandapower ..."
        ),
    )

    try:

        project_ref = str(
            project.GetFullName()
        )

    except Exception:

        project_ref = project_name

    skipped_station_controllers = []

    q_capability_records = []

    (
        original_stactrl,
        safe_stactrl,
    ) = make_safe_create_stactrl(
        app,
        skipped_station_controllers,
    )

    original_q_helper = (
        pf_import_functions.get_min_max_q_mvar_from_characteristics_object
    )

    if PATCH_EXT_GRID_Q_CAPABILITY:
        (
            _,
            safe_q_helper,
        ) = make_safe_q_capability_helper(
            app,
            q_capability_records,
        )
    else:
        safe_q_helper = original_q_helper

    pf_import_functions.create_stactrl = (
        safe_stactrl
    )

    pf_import_functions.get_min_max_q_mvar_from_characteristics_object = (
        safe_q_helper
    )

    try:

        net = from_pfd(
            app,
            prj_name=project_ref,
            path_dst=None,
            pv_as_slack=False,
            pf_variable_p_loads=PF_VARIABLE_P_LOADS,
            pf_variable_p_gen=PF_VARIABLE_P_GEN,
            tap_opt=PF_TAP_OPTION,
            export_controller=EXPORT_CONTROLLERS,
            handle_us=PF_CONVERTER_HANDLE_US,
        )

    finally:

        pf_import_functions.create_stactrl = (
            original_stactrl
        )

        pf_import_functions.get_min_max_q_mvar_from_characteristics_object = (
            original_q_helper
        )

        try:

            app.ActivateProject(
                project_ref
            )

        except Exception:

            pass

        try:

            study_case.Activate()

        except Exception:

            pass

    if net is None:

        raise RuntimeError(
            "Conversion returned None."
        )

    pf_print(
        app,
        "Conversion completed.",
    )

    pf_print(
        app,
        (
            "Converted elements: "
            f"buses={len(net.bus)}, "
            f"lines={len(net.line)}, "
            f"trafos={len(net.trafo)}, "
            f"trafo3w={len(net.trafo3w)}, "
            f"loads={len(net.load)}, "
            f"gens={len(net.gen)}, "
            f"sgens={len(net.sgen)}, "
            f"ext_grids={len(net.ext_grid)}, "
            f"impedances={len(net.impedance)}"
        ),
    )

    pf_print(
        app,
        (
            "Skipped unsupported ElmStactrl: "
            f"{len(skipped_station_controllers)}"
        ),
    )

    pf_print(
        app,
        (
            "ext_grid Q-capability patch records: "
            f"{len(q_capability_records)}"
        ),
    )

    save_dataframe(
        pd.DataFrame(
            skipped_station_controllers
        ),
        output_dir
        / "skipped_station_controllers.csv",
        index=False,
    )

    save_dataframe(
        pd.DataFrame(
            q_capability_records
        ),
        output_dir
        / "ext_grid_q_capability_patch.csv",
        index=False,
    )

    # =========================================================================
    # SAVE POWERFACTORY VOLTAGES BEFORE ANY PANDAPOWER RUN
    # =========================================================================

    pf_snapshot = snapshot_pf_bus_results(
        net
    )

    save_dataframe(
        pf_snapshot,
        output_dir
        / "pf_exported_res_bus_raw.csv",
        index=True,
    )

    # =========================================================================
    # OPTIONAL PRIMSKOVO REPAIR
    # =========================================================================

    zks_metadata = add_primskovo_zks(
        app,
        net,
    )

    # =========================================================================
    # UNSUPPLIED ISLANDS
    # =========================================================================

    (
        unsupplied_before,
        active_unsupplied_before,
    ) = topology_diagnostics(
        app,
        net,
        "before_unsupplied_deactivation",
    )

    unsupplied_island_df = pd.DataFrame()

    if active_unsupplied_before:

        (
            unsupplied_island_df,
            _,
        ) = export_unsupplied_island_diagnostics(
            app,
            net,
            output_dir,
        )

        total_removed_load_mw = (
            float(
                unsupplied_island_df[
                    "load_p_mw"
                ].sum()
            )
            if not unsupplied_island_df.empty
            else 0.0
        )

        pf_warn(
            app,
            (
                "Unsupplied islands contain "
                f"{total_removed_load_mw:.6f} MW "
                "of converted load."
            ),
        )

        if AUTO_DEACTIVATE_UNSUPPLIED:

            pf_print(
                app,
                "",
            )

            pf_print(
                app,
                (
                    "Deactivating unsupplied "
                    "pandapower islands ..."
                ),
            )

            set_isolated_areas_out_of_service(
                net,
                respect_switches=(
                    RESPECT_SWITCHES_FOR_SUPPLY
                ),
            )

            pf_print(
                app,
                (
                    "Unsupplied island "
                    "deactivation completed."
                ),
            )

        else:

            raise RuntimeError(
                (
                    "Unsupplied buses exist and "
                    "AUTO_DEACTIVATE_UNSUPPLIED=False."
                )
            )

    (
        unsupplied_after,
        active_unsupplied_after,
    ) = topology_diagnostics(
        app,
        net,
        "after_unsupplied_deactivation",
    )

    if (
        active_unsupplied_after
        and STOP_IF_UNSUPPLIED_REMAIN
    ):

        raise RuntimeError(
            (
                "Unsupplied IN-SERVICE buses "
                "still remain after deactivation: "
                f"{active_unsupplied_after}"
            )
        )

    # =========================================================================
    # SUPPLIED ISLANDS
    # =========================================================================

    supplied_islands = (
        supplied_island_diagnostics(
            app,
            net,
            output_dir,
        )
    )

    # =========================================================================
    # FREEZE / REPAIR THE CONVERTED PF OPERATING POINT
    # =========================================================================

    frozen_voltage_metadata = (
        freeze_voltage_controls_to_pf_operating_point(
            app,
            net,
            pf_snapshot,
            output_dir,
        )
    )

    input_sanitize_metadata = sanitize_powerflow_inputs(
        app,
        net,
        output_dir,
    )

    # =========================================================================
    # POWERFACTORY WARM START
    # =========================================================================

    (
        pf_vm_init,
        pf_va_init,
        warm_start_metadata,
    ) = build_powerfactory_warm_start(
        app,
        net,
        pf_snapshot,
        output_dir,
    )

    # =========================================================================
    # 3) INITIAL PANDAPOWER OPERATING POINT
    # =========================================================================

    pf_print(
        app,
        "",
    )

    pf_print(
        app,
        (
            "3/6 pandapower calculation ..."
        ),
    )

    print_controller_summary(
        app,
        net,
    )

    initial_solution = (
        solve_initial_operating_point(
            app,
            net,
            pf_vm_init,
            pf_va_init,
        )
    )

    save_dataframe(
        pd.DataFrame(
            initial_solution[
                "attempts"
            ]
        ),
        output_dir
        / "powerflow_convergence_attempts.csv",
        index=False,
    )

    pf_point_fallback_metrics = {
        "used": False,
    }
    pf_point_mismatch_df = None

    if initial_solution[
        "success"
    ]:

        pf_print(
            app,
            (
                "Initial exact pandapower operating "
                "point obtained with method: "
                f"{initial_solution['method']}"
            ),
        )

        # =====================================================================
        # CONTROLLER LOOP
        # =====================================================================

        if RUN_CONTROLLERS is None:
            use_controllers = bool(
                hasattr(net, "controller")
                and net.controller is not None
                and not net.controller.empty
            )
        else:
            use_controllers = bool(RUN_CONTROLLERS)

        if use_controllers:
            pf_print(app, "")
            pf_print(app, "3b/6 Running supported pandapower controller loop ...")

            try:
                pp.runpp(
                    net,
                    algorithm="nr",
                    calculate_voltage_angles=True,
                    init="results",
                    max_iteration=MAX_ITERATION,
                    tolerance_mva=TOLERANCE_MVA,
                    enforce_q_lims=ENFORCE_Q_LIMS,
                    check_connectivity=CHECK_CONNECTIVITY,
                    voltage_depend_loads=VOLTAGE_DEPEND_LOADS,
                    run_control=True,
                    distributed_slack=False,
                    numba=USE_NUMBA,
                    lightsim2grid=USE_LIGHTSIM2GRID,
                )
            except Exception as exc:
                raise RuntimeError(
                    "pandapower controller loop failed after obtaining a "
                    f"converged base operating point:\n{exc}"
                ) from exc

            if not bool(net.converged):
                raise RuntimeError("pandapower controller loop did not converge.")

            pf_print(app, "Controller loop converged.")
        else:
            pf_print(app, "3b/6 Controller loop skipped.")

        # =====================================================================
        # FINAL EXACT STANDALONE NR
        # =====================================================================

        pf_print(app, "")
        pf_print(app, "3c/6 FINAL standalone NR ...")

        pp.runpp(
            net,
            algorithm="nr",
            calculate_voltage_angles=True,
            init="results",
            max_iteration=MAX_ITERATION,
            tolerance_mva=TOLERANCE_MVA,
            enforce_q_lims=ENFORCE_Q_LIMS,
            check_connectivity=CHECK_CONNECTIVITY,
            voltage_depend_loads=VOLTAGE_DEPEND_LOADS,
            run_control=False,
            distributed_slack=False,
            numba=USE_NUMBA,
            lightsim2grid=USE_LIGHTSIM2GRID,
        )

        if not bool(net.converged):
            raise RuntimeError("Final standalone pandapower NR did not converge.")

        pf_print(
            app,
            (
                "Final standalone NR converged; "
                f"iterations={net._ppc.get('iterations')}"
            ),
        )

    else:
        if not ALLOW_PF_POINT_JACOBIAN_FALLBACK:
            raise RuntimeError(
                "pandapower did not converge and "
                "ALLOW_PF_POINT_JACOBIAN_FALLBACK=False."
            )

        pf_warn(
            app,
            (
                "All pandapower power-flow attempts failed. Continuing with "
                "PF-point Jacobian fallback instead of aborting."
            ),
        )

        _, pf_point_fallback_metrics, pf_point_mismatch_df = (
            build_pf_point_internal_model(
                app,
                net,
                pf_vm_init,
                pf_va_init,
                output_dir,
            )
        )

    # =========================================================================
    # 4) JACOBIAN
    # =========================================================================

    pf_print(
        app,
        "",
    )

    pf_print(
        app,
        (
            "4/6 Constructing classical "
            "AC Jacobian ..."
        ),
    )

    ppc = net._ppc

    internal = ppc[
        "internal"
    ]

    (
        J,
        ref,
        pv,
        pq,
        ybus_components,
    ) = build_classical_ac_jacobian(
        app,
        internal,
    )

    pvpq = np.concatenate(
        (
            pv,
            pq,
        )
    )

    n_angle = len(
        pvpq
    )

    J_Pdelta = J[
        :n_angle,
        :n_angle,
    ].tocsr()

    J_PU = J[
        :n_angle,
        n_angle:,
    ].tocsr()

    J_Qdelta = J[
        n_angle:,
        :n_angle,
    ].tocsr()

    J_QU = J[
        n_angle:,
        n_angle:,
    ].tocsr()

    (
        bus_map,
        labels,
    ) = build_internal_bus_mapping(
        net,
        np.asarray(
            internal[
                "bus"
            ]
        ),
    )

    (
        row_map,
        column_map,
    ) = make_jacobian_labels(
        pv,
        pq,
        labels,
    )

    # =========================================================================
    # 5) PF vs PP VALIDATION
    # =========================================================================

    pf_print(
        app,
        "",
    )

    pf_print(
        app,
        (
            "5/6 Comparing PF "
            "and pandapower voltages ..."
        ),
    )

    if pf_point_fallback_metrics.get("used"):
        validation_df = None
        validation_metrics = {
            "available": False,
            "reason": (
                "pandapower NR did not converge; Jacobian was evaluated "
                "directly at the PowerFactory voltage point. See "
                "pf_point_power_mismatch.csv."
            ),
        }
    else:
        (
            validation_df,
            validation_metrics,
        ) = make_voltage_validation(
            net,
            pf_snapshot,
        )

    if validation_metrics.get(
        "available"
    ):

        pf_print(
            app,
            (
                "Max |PF - PP| Vm = "
                f"{validation_metrics.get('max_abs_vm_pu_diff')} "
                "p.u."
            ),
        )

        pf_print(
            app,
            (
                "RMS |PF - PP| Vm = "
                f"{validation_metrics.get('rms_vm_pu_diff')} "
                "p.u."
            ),
        )

    # =========================================================================
    # 6) SAVE
    # =========================================================================

    pf_print(
        app,
        "",
    )

    pf_print(
        app,
        "6/6 Saving outputs ...",
    )

    sparse.save_npz(
        output_dir
        / "jacobian_full.npz",
        J,
        compressed=True,
    )

    sparse.save_npz(
        output_dir
        / "J_Pdelta.npz",
        J_Pdelta,
        compressed=True,
    )

    sparse.save_npz(
        output_dir
        / "J_PU.npz",
        J_PU,
        compressed=True,
    )

    sparse.save_npz(
        output_dir
        / "J_Qdelta.npz",
        J_Qdelta,
        compressed=True,
    )

    sparse.save_npz(
        output_dir
        / "J_QU.npz",
        J_QU,
        compressed=True,
    )

    save_dataframe(
        row_map,
        output_dir
        / "jacobian_rows.csv",
        index=False,
    )

    save_dataframe(
        column_map,
        output_dir
        / "jacobian_columns.csv",
        index=False,
    )

    save_dataframe(
        bus_map,
        output_dir
        / "ppci_bus_mapping.csv",
        index=False,
    )

    if SAVE_JACOBIAN_COO_CSV:

        export_sparse_coo_csv(
            output_dir
            / "jacobian_nonzero_entries.csv",
            J,
        )

    if (
        SAVE_VOLTAGE_VALIDATION
        and validation_df is not None
    ):

        save_dataframe(
            validation_df,
            output_dir
            / "pf_vs_pandapower_bus_voltage.csv",
            index=False,
        )

    if SAVE_PANDAPOWER_NET_JSON:

        pp.to_json(
            net,
            str(
                output_dir
                / "pandapower_network.json"
            ),
        )

    if elm_sfilt_records:

        save_dataframe(
            pd.DataFrame(
                elm_sfilt_records
            ),
            output_dir
            / "powerfactory_ElmSfilt_elements.csv",
            index=False,
        )

    elapsed = (
        time.time()
        - started
    )

    total_unsupplied_load = (
        float(
            unsupplied_island_df[
                "load_p_mw"
            ].sum()
        )
        if not unsupplied_island_df.empty
        else 0.0
    )

    metadata = {
        "project": project_name,
        "study_case": study_case_name,
        "python_version": (
            sys.version
        ),
        "versions": {
            "pandapower": package_version(
                "pandapower"
            ),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "converter": {
            "handle_us": (
                PF_CONVERTER_HANDLE_US
            ),
            "tap_opt": (
                PF_TAP_OPTION
            ),
            "skipped_station_controllers": int(
                len(
                    skipped_station_controllers
                )
            ),
            "ext_grid_q_capability_patch_records": int(
                len(
                    q_capability_records
                )
            ),
        },
        "unsupplied": {
            "before_bus_count": int(
                len(
                    active_unsupplied_before
                )
            ),
            "island_count": int(
                len(
                    unsupplied_island_df
                )
            ),
            "removed_load_mw": float(
                total_unsupplied_load
            ),
            "remaining_after": int(
                len(
                    active_unsupplied_after
                )
            ),
        },
        "supplied_island_count": int(
            len(
                supplied_islands
            )
        ),
        "frozen_pf_voltage_controls": (
            frozen_voltage_metadata
        ),
        "sanitized_powerflow_inputs": (
            input_sanitize_metadata
        ),
        "converter_operating_point": {
            "pf_variable_p_loads": PF_VARIABLE_P_LOADS,
            "pf_variable_p_gen": PF_VARIABLE_P_GEN,
            "tap_opt": PF_TAP_OPTION,
            "voltage_depend_loads": VOLTAGE_DEPEND_LOADS,
            "run_controllers": RUN_CONTROLLERS,
        },
        "warm_start": (
            warm_start_metadata
        ),
        "initial_solution": (
            initial_solution
        ),
        "pf_point_jacobian_fallback": (
            pf_point_fallback_metrics
        ),
        "final_powerflow": {
            "algorithm": "nr",
            "tolerance_mva": (
                TOLERANCE_MVA
            ),
            "max_iteration": (
                MAX_ITERATION
            ),
            "voltage_depend_loads": (
                VOLTAGE_DEPEND_LOADS
            ),
            "iterations": (
                net._ppc.get("iterations")
                if hasattr(net, "_ppc") and net._ppc is not None
                else None
            ),
            "converged": bool(initial_solution.get("success", False)),
        },
        "primskovo_zks": (
            zks_metadata
        ),
        "jacobian_definition": (
            "frozen-control classical AC "
            "d[P_PV,P_PQ,Q_PQ] / "
            "d[delta_PV,delta_PQ,U_PQ]"
        ),
        "effective_ybus_components": (
            ybus_components
        ),
        "ref_count": int(
            len(
                ref
            )
        ),
        "pv_count": int(
            len(
                pv
            )
        ),
        "pq_count": int(
            len(
                pq
            )
        ),
        "jacobian": sparse_stats(
            J
        ),
        "J_Pdelta": sparse_stats(
            J_Pdelta
        ),
        "J_PU": sparse_stats(
            J_PU
        ),
        "J_Qdelta": sparse_stats(
            J_Qdelta
        ),
        "J_QU": sparse_stats(
            J_QU
        ),
        "voltage_validation": (
            validation_metrics
        ),
        "elapsed_seconds": float(
            elapsed
        ),
    }

    with (
        output_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    pf_print(
        app,
        "",
    )

    pf_print(
        app,
        "=" * 72,
    )

    pf_print(
        app,
        "SUCCESS",
    )

    pf_print(
        app,
        "=" * 72,
    )

    pf_print(
        app,
        (
            "Operating-point source: "
            + (
                str(initial_solution["method"])
                if initial_solution.get("success")
                else "PowerFactory voltage point (pandapower NR non-converged)"
            )
        ),
    )

    pf_print(
        app,
        (
            "Unsupplied buses removed: "
            f"{len(active_unsupplied_before)}"
        ),
    )

    pf_print(
        app,
        (
            "Load in removed islands: "
            f"{total_unsupplied_load:.6f} MW"
        ),
    )

    pf_print(
        app,
        (
            "REF/PV/PQ = "
            f"{len(ref)}/"
            f"{len(pv)}/"
            f"{len(pq)}"
        ),
    )

    pf_print(
        app,
        (
            "Jacobian: "
            f"{J.shape}, "
            f"nnz={J.nnz}"
        ),
    )

    if pf_point_fallback_metrics.get("used"):
        pf_print(
            app,
            (
                "PF-point max converted-model equation mismatch: "
                f"{pf_point_fallback_metrics.get('max_abs_equation_mismatch_mva')} MVA"
            ),
        )

    pf_print(
        app,
        (
            f"J_Pdelta = "
            f"{J_Pdelta.shape}"
        ),
    )

    pf_print(
        app,
        (
            f"J_PU = "
            f"{J_PU.shape}"
        ),
    )

    pf_print(
        app,
        (
            f"J_Qdelta = "
            f"{J_Qdelta.shape}"
        ),
    )

    pf_print(
        app,
        (
            f"J_QU = "
            f"{J_QU.shape}"
        ),
    )

    if validation_metrics.get(
        "available"
    ):

        pf_print(
            app,
            (
                "Max |PF-PP| Vm = "
                f"{validation_metrics.get('max_abs_vm_pu_diff')} "
                "p.u."
            ),
        )

        pf_print(
            app,
            (
                "RMS |PF-PP| Vm = "
                f"{validation_metrics.get('rms_vm_pu_diff')} "
                "p.u."
            ),
        )

    pf_print(
        app,
        (
            f"Saved to: "
            f"{output_dir}"
        ),
    )

    pf_print(
        app,
        (
            f"Elapsed: "
            f"{elapsed:.1f} s"
        ),
    )

    pf_print(
        app,
        "=" * 72,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()