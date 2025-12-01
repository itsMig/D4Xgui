#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pysotope_fork.py - Minimal fork containing only required functions from Pysotope
"""

import sys
import os
import json
import math
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.optimize import fsolve, broyden1, minimize_scalar, least_squares
import streamlit as st

sss = st.session_state

from tools.commons import color

# Block formatting for messages
WITHIN_BLOCK = f"{'#' * 80}\n{{msg}}\n{'#' * 80}"
WARNING_BLOCK = f"{color.WARNING}{WITHIN_BLOCK}{color.ENDC}"
ERROR_BLOCK = f"{color.ERROR}{WITHIN_BLOCK}{color.ENDC}"
INFO_BLOCK = f"{color.OKGREEN}{WITHIN_BLOCK}{color.ENDC}"


def f_poly4(poly, i):
    """Calculate 4th order polynomial."""
    return poly[0] * i + poly[1] * i**2 + poly[2] * i**3 + poly[3] * i**4


def K47_t(x_arr, scaling, offset):
    """Temperature calibration function for D47."""
    poly_63 = (-5.896755e00, -3.520888e03, 2.391274e07, -3.540693e09)
    poly_63_vals = f_poly4(poly_63, x_arr)
    poly_47_vals = (poly_63_vals * scaling) + offset
    return poly_47_vals


def K48_t(x_arr, scaling, offset):
    """Temperature calibration function for D48."""
    poly_64 = (6.001624e00, -1.298978e04, 8.995634e06, -7.422972e08)
    poly_64_vals = f_poly4(poly_64, x_arr)
    poly_48_vals = (poly_64_vals * scaling) + offset
    return poly_48_vals


def K49_t(x_arr, scaling, offset):
    """Temperature calibration function for D49."""
    poly_65 = (-6.741e00, -1.950e04, 5.845e07, -8.093e09)
    poly_65_vals = f_poly4(poly_65, x_arr)
    poly_49_vals = (poly_65_vals * scaling) + offset
    return poly_49_vals


class Pysotope:
    """Minimal Pysotope class with only required functionality for D4Xgui."""
    
    def __init__(self, level="replicate", level_etf="replicate", optimize="leastSquare"):
        """Initialize Pysotope with minimal required attributes."""
        self.level_etf = level_etf
        self.level = level
        self.optimize = optimize
        self.analyses = dict()
    
        self.isotopic_constants = {
            "R13_VPDB": 0.01118,
            "R17_VSMOW": 0.00038475,
            "R18_VSMOW": 0.0020052,
            "lambda_": 0.528,
            "R18_initial_guess": 0.002,
        }

        self.wg_ratios = {"d18O": 25.260, "d13C": -4.200}
        
        self.standards = {
            "d18O": {
                "ETH-1": 6.7 * 1.03092 + 30.92,
                "ETH-2": -10 * 1.03092 + 30.92,
                "ETH-3": 7.18 * 1.03092 + 30.92,
            },
            "d13C": {"ETH-1": 2.07, "ETH-2": -10.17, "ETH-3": 1.71},
            "D47": {
                "25C": 0.9196,  
                "1000C": 0.0266,
            },
            "D48": {
                "25C": 0.345,
                "1000C": 0.0,
            },
            "D49": {
                "25C": 2.228,
                "1000C": 0.0,
            },
        }
        
        self.half_mass_cup = '47.5'
        self.scaling_factors = dict()
        self.etf = dict()
        
        # Calculate working gas ratios
        self.calc_wg_ratios()

    def set_wg_ratios(self, wg_ratios: dict):
        """Set working gas ratios."""
        if not isinstance(wg_ratios, dict) or not all(
            (key in wg_ratios and isinstance(wg_ratios[key], float))
            for key in self.wg_ratios
        ):
            raise Exception(
                "\n".join([
                    "\n",
                    ERROR_BLOCK.format(msg=f"Error in {sys._getframe().f_code.co_name}():"),
                    WARNING_BLOCK.format(
                        msg=f"Please provide `wg_ratios` argument in the following format:\n"
                        + json.dumps(self.wg_ratios, indent=4)
                    ),
                ])
            )
        
        self.wg_ratios = wg_ratios
        '''print(
            INFO_BLOCK.format(
                msg=f"Working gas ratios (`wg_ratios`) set to:\n"
                + json.dumps(self.wg_ratios, indent=4)
            )
        )'''

    def add_session(self, session: dict):
        """Add a session to the object."""
        S = next(iter(session.items()))[0]
        self.etf.update({
            S: {
                "D47": {"m": np.nan, "b": np.nan},
                "D48": {"m": np.nan, "b": np.nan},
                "D49": {"m": np.nan, "b": np.nan},
            }
        })
        
        self.scaling_factors.update(
            {S: {"47b_47.5": -1, "48b_47.5": -1, "49b_47.5": -1}}
        )

    def set_standards(self, system, standards: dict):
        """Set standards for a given system."""
        if not isinstance(standards, dict) or not system in self.standards:
            raise Exception(
                ERROR_BLOCK.format(
                    msg=f"Please provide a `system`={', '.join([f'`{key}`' for key in self.standards.keys()])} and `standards`, e.g. {json.dumps(self.standards['D47'])}!"
                )
            )
        self.standards[system] = standards

    def add_data(self, session, df_session):
        """Add data to the object."""
        if isinstance(session, (list, tuple)):
            for session, df_session in df_session.groupby('Session'):
                self.analyses.update({session: df_session})
                self.add_session({session: df_session["Sample"].unique()})
        else:
            self.analyses.update({session: df_session})
            self.add_session({session: df_session["Sample"].unique()})

    def calc_wg_ratios(self):
        """Calculate isotopic ratios of the working gas."""
        self.K = self.isotopic_constants["R17_VSMOW"] * (
            self.isotopic_constants["R18_VSMOW"] ** (-self.isotopic_constants["lambda_"])
        )
        
        # Reference Gas / Working Gas Variables
        self.R13_Ref = self.isotopic_constants["R13_VPDB"] * (
            1 + self.wg_ratios["d13C"] / 1000
        )
        self.R18_Ref = self.isotopic_constants["R18_VSMOW"] * (
            1 + self.wg_ratios["d18O"] / 1000
        )
        self.R17_Ref = self.isotopic_constants["R17_VSMOW"] * (
            (self.R18_Ref / self.isotopic_constants["R18_VSMOW"]) ** self.isotopic_constants["lambda_"]
        )
        
        # Isotope abundances
        self.C12_Ref = 1 / (1 + self.R13_Ref)
        self.C13_Ref = self.R13_Ref / (1 + self.R13_Ref)
        self.C16_Ref = 1 / (1 + self.R17_Ref + self.R18_Ref)
        self.C17_Ref = self.R17_Ref / (1 + self.R17_Ref + self.R18_Ref)
        self.C18_Ref = self.R18_Ref / (1 + self.R17_Ref + self.R18_Ref)
        
        # 44 isotopologue
        C12_16_16_Ref = self.C12_Ref * self.C16_Ref * self.C16_Ref
        self.C44_Ref = C12_16_16_Ref
        
        # 45 isotopologues
        C12_17_16_Ref = (self.C12_Ref * self.C17_Ref * self.C16_Ref) * 2
        C13_16_16_Ref = self.C13_Ref * self.C16_Ref * self.C16_Ref
        self.C45_Ref = C12_17_16_Ref + C13_16_16_Ref
        
        # 46 isotopologues
        C12_18_16_Ref = (self.C12_Ref * self.C18_Ref * self.C16_Ref) * 2
        C13_17_16_Ref = (self.C13_Ref * self.C17_Ref * self.C16_Ref) * 2
        C12_17_17_Ref = self.C12_Ref * self.C17_Ref * self.C17_Ref
        self.C46_Ref = C12_18_16_Ref + C13_17_16_Ref + C12_17_17_Ref
        
        # 47 isotopologues
        C16_18_16_Ref = (self.C13_Ref * self.C18_Ref * self.C16_Ref) * 2
        C13_17_17_Ref = self.C13_Ref * self.C17_Ref * self.C17_Ref
        C12_18_17_Ref = (self.C12_Ref * self.C17_Ref * self.C18_Ref) * 2
        self.C47_Ref = C16_18_16_Ref + C13_17_17_Ref + C12_18_17_Ref
        
        # 48 isotopologues
        C13_18_17_Ref = (self.C13_Ref * self.C18_Ref * self.C17_Ref) * 2
        C12_18_18_Ref = self.C12_Ref * self.C18_Ref * self.C18_Ref
        self.C48_Ref = C13_18_17_Ref + C12_18_18_Ref
        
        # 49 isotopologue
        C13_18_18_Ref = self.C13_Ref * self.C18_Ref * self.C18_Ref
        self.C49_Ref = C13_18_18_Ref
        
        # Reference ratios
        self.Ref_Ratio_45_44 = self.C45_Ref / self.C44_Ref
        self.Ref_Ratio_46_44 = self.C46_Ref / self.C44_Ref
        self.Ref_Ratio_47_44 = self.C47_Ref / self.C44_Ref
        self.Ref_Ratio_48_44 = self.C48_Ref / self.C44_Ref
        self.Ref_Ratio_49_44 = self.C49_Ref / self.C44_Ref

    def calc_sample_ratios_1(self, session="all"):
        """Calculate sample ratios (first part - baseline independent)."""
        
        def _calc(df):
            df = df.reset_index(drop=True)
            
            # Calculate sample ratios
            df["Sample_Ratio_45_44"] = (
                self.Ref_Ratio_45_44 * df["raw_s45"] / df["raw_s44"] * df["raw_r44"] / df["raw_r45"]
            )
            df["Sample_Ratio_46_44"] = (
                self.Ref_Ratio_46_44 * df["raw_s46"] / df["raw_s44"] * df["raw_r44"] / df["raw_r46"]
            )
            
            # Calculate delta values
            df["d45"] = 1000 * (df["Sample_Ratio_45_44"] / self.Ref_Ratio_45_44 - 1)
            df["d46"] = 1000 * (df["Sample_Ratio_46_44"] / self.Ref_Ratio_46_44 - 1)
            
            # Equation solving for R18
            lambda_ = self.isotopic_constants["lambda_"]
            K_ = self.K
            eq_Part1 = -3 * K_**2
            eq_Part2 = 2 * lambda_
            eq_Part3 = 2 * K_
            
            def eq_R18_Sample_fsolve(init_guess, R45_44, R46_44):
                """Equation solver for R18_Sample."""
                RES = fsolve(
                    lambda R18_Sample: eq_Part1 * R18_Sample**eq_Part2
                    + (eq_Part3 * R45_44 * (R18_Sample**lambda_))
                    + (2 * R18_Sample - R46_44),
                    x0=init_guess,
                )
                return RES
            
            # Initialize R18
            try:
                init_R18 = eq_R18_Sample_fsolve(
                    self.isotopic_constants["R18_initial_guess"],
                    df["Sample_Ratio_45_44"].values[0],
                    df["Sample_Ratio_46_44"].values[0],
                )
            except:
                print(f'Couldn\'t solve R18! Using {self.isotopic_constants["R18_initial_guess"]}')
                init_R18 = self.isotopic_constants["R18_initial_guess"]
            
            # Solve R18 equations
            n_cycles = len(df)
            check = 33
            while True:
                if math.gcd(n_cycles, check) == check:
                    break
                else:
                    check -= 1
            
            step_fsolve = check
            df["lambda_"] = self.isotopic_constants["lambda_"]
            df["Sample_Ratio_18_16"] = [0.0] * n_cycles
            
            for _idx in range(step_fsolve, n_cycles + step_fsolve, step_fsolve):
                _series = eq_R18_Sample_fsolve(
                    [init_R18] * len(df[_idx - step_fsolve : _idx]),
                    df["Sample_Ratio_45_44"].values[_idx - step_fsolve : _idx],
                    df["Sample_Ratio_46_44"].values[_idx - step_fsolve : _idx],
                )
                df.loc[_idx - step_fsolve : _idx - 1, "Sample_Ratio_18_16"] = _series
            
            # Calculate other ratios
            df["Sample_Ratio_17_16"] = self.K * (df["Sample_Ratio_18_16"] ** df["lambda_"])
            df["Sample_Ratio_13_12"] = df["Sample_Ratio_45_44"] - (2 * df["Sample_Ratio_17_16"])
            
            # Calculate delta values
            df["d18O"] = (
                (df["Sample_Ratio_18_16"] - self.isotopic_constants["R18_VSMOW"])
                / self.isotopic_constants["R18_VSMOW"] * 1000
            )
            df["d13C"] = (
                (df["Sample_Ratio_13_12"] - self.isotopic_constants["R13_VPDB"])
                / self.isotopic_constants["R13_VPDB"] * 1000
            )
            
            # Calculate isotope abundances
            df["Sample_C_12"] = 1 / (1 + df["Sample_Ratio_13_12"])
            df["Sample_C_13"] = df["Sample_Ratio_13_12"] / (1 + df["Sample_Ratio_13_12"])
            df["Sample_C_16"] = 1 / (1 + df["Sample_Ratio_17_16"] + df["Sample_Ratio_18_16"])
            df["Sample_C_17"] = df["Sample_Ratio_17_16"] / (1 + df["Sample_Ratio_17_16"] + df["Sample_Ratio_18_16"])
            df["Sample_C_18"] = df["Sample_Ratio_18_16"] / (1 + df["Sample_Ratio_17_16"] + df["Sample_Ratio_18_16"])
            
            # Calculate isotopologue abundances
            df["Sample_C_266"] = df["Sample_C_12"] * (df["Sample_C_16"] ** 2)
            df["Sample_C_267"] = df["Sample_C_12"] * df["Sample_C_16"] * df["Sample_C_17"] * 2
            df["Sample_C_268"] = df["Sample_C_12"] * df["Sample_C_16"] * df["Sample_C_18"] * 2
            df["Sample_C_277"] = df["Sample_C_12"] * (df["Sample_C_17"] ** 2)
            df["Sample_C_278"] = df["Sample_C_12"] * df["Sample_C_17"] * df["Sample_C_18"] * 2
            df["Sample_C_288"] = df["Sample_C_12"] * (df["Sample_C_18"] ** 2)
            
            df["Sample_C_366"] = df["Sample_C_13"] * (df["Sample_C_16"] ** 2)
            df["Sample_C_367"] = df["Sample_C_13"] * df["Sample_C_16"] * df["Sample_C_17"] * 2
            df["Sample_C_368"] = df["Sample_C_13"] * df["Sample_C_16"] * df["Sample_C_18"] * 2
            df["Sample_C_377"] = df["Sample_C_13"] * (df["Sample_C_17"] ** 2)
            df["Sample_C_378"] = df["Sample_C_13"] * df["Sample_C_17"] * df["Sample_C_18"] * 2
            df["Sample_C_388"] = df["Sample_C_13"] * (df["Sample_C_18"] ** 2)
            
            # Calculate expected isotopologue ratios
            df["Sample_xC_44"] = df["Sample_C_266"]
            df["Sample_xC_45"] = df["Sample_C_366"] + df["Sample_C_267"]
            df["Sample_xC_46"] = df["Sample_C_367"] + df["Sample_C_268"] + df["Sample_C_277"]
            df["Sample_xC_47"] = df["Sample_C_368"] + df["Sample_C_278"] + df["Sample_C_377"]
            df["Sample_xC_48"] = df["Sample_C_378"] + df["Sample_C_288"]
            df["Sample_xC_49"] = df["Sample_C_388"]
            
            df["Sample_xRatio_45_44"] = df["Sample_xC_45"] / df["Sample_C_266"]
            df["Sample_xRatio_46_44"] = df["Sample_xC_46"] / df["Sample_C_266"]
            df["Sample_xRatio_47_44"] = df["Sample_xC_47"] / df["Sample_C_266"]
            df["Sample_xRatio_48_44"] = df["Sample_xC_48"] / df["Sample_C_266"]
            df["Sample_xRatio_49_44"] = df["Sample_xC_49"] / df["Sample_C_266"]
            
            return df
        
        # Process sessions
        if session == "all":
            for S in self.analyses:
                df_session = _calc(self.analyses[S])
                self.analyses[S] = df_session
        elif session in self.analyses:
            df_session = _calc(self.analyses[session])
            self.analyses[session] = df_session
        else:
            msg = "Please set `session` argument either to `all` to process every available session, or specify a session name accessible via `self.analyses`."
            raise Exception(f"{ERROR_BLOCK.format(msg=msg)}")

    def correctBaseline(self, scaling_mode: str = "scale", session="all", scaling_factors=None, 
                       D47std=None, D48std=None, D49std=None) -> None:
        """Correct for negative pressure baseline."""
        
        if D47std is None and f'D47' in self.standards:
            D47std = self.standards.get('D47', {})
        if D48std is None and f'D48' in self.standards:
            D48std = self.standards.get('D48', {})
        if D49std is None and f'D49' in self.standards:
            D49std = self.standards.get('D49', {})
        
        # Now update standards if provided (non-None)
        if D47std and isinstance(D47std, dict):
            self.standards['D47'] = D47std
        if D48std and isinstance(D48std, dict):
            self.standards['D48'] = D48std
        if D49std and isinstance(D49std, dict):
            self.standards['D49'] = D49std
        
        if scaling_mode == None or scaling_mode.lower() == "none":
            msg = "Please provide a legit `scaling_mode` argument like `'drift', 'scale', 'l_44p_t'`"
            raise Exception(f"{ERROR_BLOCK.format(msg=msg)}")
        
        def find_static_scalingfactor(session):
            """Find optimal scaling factors for baseline correction."""
            all_slopes = {}
            standards = set()
            MAPPING_MZ = {
                47: D47std,
                48: D48std,
                49: D49std
            }
            #if session == 'std':
            for mz in 47, 48, 49:
                if len(sss.get(f'bg_custom_{mz}_values', "")) == 0:
                    standards.update([*self.standards[f"D{mz}"]])
          
            # st.write(self.analyses[session])
            # st.write(standards)
            df = self.analyses[session].loc[self.analyses[session]["Sample"].isin(standards)]
            # st.write(df)
            

            def optimize_ETH12_leastSquares(scale, mz):
                """Optimize scaling factor for ETH-1 and ETH-2 standards using least squares."""
                stdP = Pysotope()
                stdP.scaling_factors["std"] = {
                    "47b_47.5": -1,
                    "48b_47.5": -1,
                    "49b_47.5": -1,
                }

                std_df = self.analyses.get('std', pd.DataFrame())
                if std_df.empty:
                    std_df = self.analyses.get(next(iter(self.analyses)), pd.DataFrame())

                std_in = std_df[std_df["Sample"].isin(["ETH-1", "ETH-2"])]
                stdP.add_data("std", std_in)
                stdP.scaling_factors["std"][f"{mz}b_47.5"] = scale
                stdP.calc_sample_ratios_1(session="std")

                stdP.correctBaseline(scaling_mode="std")
                stdP.calc_sample_ratios_2(mode="bg")

                std_df = stdP.analyses["std"]

                df_ETHs = std_df[std_df["Sample"].isin(["ETH-1", "ETH-2"])]
                results = linregress(df_ETHs[f"d{mz}"].values, df_ETHs[f"D{mz}"].values)
                return results.slope

            def customStds_targetValues(scale, mz, stds=dict()):
                stdP = Pysotope()
                #st.write(stds)
                # st.write(scale,'scale')
                stdP.scaling_factors["std"] = {
                    "47b_47.5": -1,
                    "48b_47.5": -1,
                    "49b_47.5": -1,
                }
                
                # {"Unrelated carbonate 06": 0.7158, "ETH-2" : 0.2119}
                # stds = json.loads(st.session_state.get(f'bg_custom_{mz}_values', '{}'))
                # ETH-2
                # 0.2119
                
                # st.write(stds)
                
                # std_df = df.copy()#self.analyses.get('std', pd.DataFrame())
                
                stds = {_:stds[_] for _ in stds if _ in self.analyses[session]["Sample"].values}
                df = self.analyses[session].loc[self.analyses[session]["Sample"].isin(stds)]
                # st.write('stds',stds)
                # st.write('@@@', std_df)
                stdP.add_data("std", df.copy())
                stdP.scaling_factors["std"][f"{mz}b_47.5"] = scale
                stdP.calc_sample_ratios_1(session="std")
                
                stdP.correctBaseline(scaling_mode="std", D47std=stds if mz==47 else None,
                     D48std=stds if mz==48 else None,
                     D49std=stds if mz==49 else None)
                stdP.calc_sample_ratios_2(mode="bg")
                
                std_df = stdP.analyses["std"]
                # std_df = stdP.analyses["std"]
                # st.write('@@@', std_df)
                all_res= 0
                PAIRS = [(x, y) for i, x in enumerate(stds) for y in [*stds.keys()][i+1:]]
                # st.write(stds)
                _="""
                ETH1-2 slope
                "47b_47.5":"array([-0.95692691])"
                "48b_47.5":"array([-0.89661941])"
                
                ETH1-2 differences
                "47b_47.5":"array([-0.98413416])"
                "48b_47.5":"array([-0.89242249])"
                """
                for S1, S2 in PAIRS:
                    #st.write('pair',S1, S2,)
                    REAL_DIFF = stds[S1] - stds[S2]
                    #st.write('REAL_DIFF',REAL_DIFF)
                    # st.write(std_df.loc[std_df['Sample'] == S1][f"D{mz}"].mean())
                    APPARENT_DIFF = std_df.loc[std_df['Sample'] == S1][f"D{mz}"].mean() - std_df.loc[std_df['Sample'] == S2][f"D{mz}"].mean()
                    #st.write('APPARENT_DIFF', APPARENT_DIFF)
                    all_res+= (APPARENT_DIFF - REAL_DIFF)**2
                #st.write('all_res',all_res)
                #st.write(std_df)
                return all_res
            
            
            def optimize_HGL_leastSquare(scale, mz):
                """Optimize heated gas line using least squares."""
                standards = [str(_) for _ in self.standards[f"D{mz}"].keys()]
                if len(standards) < 2:
                    raise Exception(
                        "\n".join([
                            "\n",
                            ERROR_BLOCK.format(
                                msg=f"Please provide at least 2 standards for D{mz}, which are also present in your dataset!"
                            ),
                            WARNING_BLOCK.format(
                                msg=f"Your actual D{mz} standards are:\n{json.dumps(self.standards[f'D{mz}'], indent=4)}"
                            ),
                        ])
                    )
                #df
                std_df = df[df["Sample"].isin(standards)]
                #std_df
                stdP = Pysotope()
                stdP.scaling_factors["std"] = {
                    "47b_47.5": -1,
                    "48b_47.5": -1,
                    "49b_47.5": -1,
                }
                
                stdP.add_data("std", std_df)
                stdP.scaling_factors["std"][f"{mz}b_47.5"] = scale
                stdP.calc_sample_ratios_1(session="std")
                stdP.correctBaseline(scaling_mode="std")
                stdP.calc_sample_ratios_2(mode="bg")
                
                for standard in standards:
                    std_analyses = stdP.analyses["std"]
                    this_std_df = std_analyses[std_analyses["Sample"] == standard]
                    
                    results = linregress(
                        this_std_df[f"d{mz}"].values, this_std_df[f"D{mz}"].values
                    )
                    all_slopes[standard] = results.slope
                
                return [_[1] for _ in all_slopes.items()]
            
            # Optimize scaling factors for each mass
            for mz in 47, 48, 49:
                if len(MAPPING_MZ[mz]) == 0:
                    #st.write(len(MAPPING_MZ[mz]),'len')
                    continue
                if self.optimize == 'customStds':
                    # result = least_squares(customStds_targetValues,
                    #                        -1.,
                    #                         args=(mz,self.standards[f'D{mz}']),
                    #                         #bounds=(-5, -1e-7),
                    #                        ftol=1e-9,
                    #                        verbose=0,
                    #                         )
                    result = least_squares(customStds_targetValues,
                                           -1.,
                                           args=(mz, MAPPING_MZ[mz]),
                                           ftol=1e-12,
                                           gtol=1e-12,
                                           verbose=0,
                                           )
                    #st.write(result)
                    sss['03_pbl_log'] = sss['03_pbl_log'] + f"\n ## Mass {mz} optimization results\n {result}"
                    self.scaling_factors[session].update({f"{mz}b_47.5": result.x})
                    continue
                elif self.optimize == "leastSquares":
                    result = least_squares(
                        optimize_HGL_leastSquare,
                        -1.0,
                        bounds=(-5, -1e-7),
                        args=([mz]),
                    )
                    sss['03_pbl_log'] = sss['03_pbl_log'] + f"\n ## Mass {mz} optimization results\n {result}"
                    self.scaling_factors[session].update({f"{mz}b_47.5": result.x})
                    continue
                elif self.optimize == "ETH":
                    # Simplified ETH optimization
                    result = least_squares(
                        optimize_ETH12_leastSquares,
                        -1.0,
                        args=(mz,),
                    )
                    sss['03_pbl_log'] = sss['03_pbl_log'] + f"\n ## Mass {mz} optimization results\n {result}"
                    self.scaling_factors[session].update({f"{mz}b_47.5": result.x})
                    continue
                else:
                    # Default to least squares
                    result = least_squares(
                        optimize_HGL_leastSquare,
                        -1.0,
                        bounds=(-5, -1e-7),
                        args=([mz]),
                    )
                    sss['03_pbl_log'] = sss['03_pbl_log'] + f"\n ## Mass {mz} optimization results\n {result}"
                    self.scaling_factors[session].update({f"{mz}b_47.5": result.x})
                    continue
                
                
        
        def _calc3(df, session, scaling_mode):
            """Apply baseline correction using scaling factors."""
            scaling_functions = self.scaling_factors[session]
            
            if "scale" in scaling_mode:
                for s_r in ("s", "r"):
                    for _c in ("7", "8", "9"):
                        df[f"bg_{s_r}4{_c}"] = (
                            df[f"raw_{s_r}4{_c}"]
                            + df[f"raw_{s_r}{self.half_mass_cup}"] * scaling_functions[f"4{_c}b_{self.half_mass_cup}"]
                        )
            
            return df
        
        # Apply baseline correction
        if scaling_mode == "std":
            self.analyses["std"] = _calc3(self.analyses["std"], "std", "scale")
            return
        elif isinstance(scaling_factors, dict):
            self.scaling_factors.update(scaling_factors)
            for S in scaling_factors:
                df_session = _calc3(self.analyses[S], S, scaling_mode)
                self.analyses[S] = df_session
            return
        elif session == "all":
            for S in self.analyses:
                find_static_scalingfactor(S)
                df_session = _calc3(self.analyses[S], S, scaling_mode)
                self.analyses[S] = df_session
            return
        elif session in self.analyses:
            find_static_scalingfactor(session)
            df_session = _calc3(self.analyses[session], session, scaling_mode)
            self.analyses[session] = df_session
            return
        else:
            msg = "Please set `session` argument either to `all` to process every available session, or specify a session name accessible via `self.analyses`."
            raise Exception(f"{ERROR_BLOCK.format(msg=msg)}")

    def calc_sample_ratios_2(self, mode="bg", session="all", data=None):
        """Calculate sample ratios (second part - baseline dependent)."""
        
        if not mode in ("raw", "bg"):
            msg = "Please set `mode` argument either to `'bg'` (background corrected) or `'raw'` (uncorrected raw intensities)."
            raise Exception(f"{ERROR_BLOCK.format(msg=msg)}")
        
        def _calc2(df, S):
            if mode == "bg" and not "bg_s47" in self.analyses[S]:
                _mode = "raw"
                msg = "\n".join([
                    "Background has not been corrected, using `raw_(sr)mz` values!",
                    "Processing results are based on non-baseline corrected raw intensities",
                ])
                f"{WARNING_BLOCK.format(msg)}"
            else:
                _mode = mode
            
            # Calculate sample ratios for masses 47, 48, 49
            df["Sample_Ratio_47_44"] = (
                self.Ref_Ratio_47_44 * df[f"{_mode}_s47"] / df[f"raw_s44"] * df[f"raw_r44"] / df[f"{_mode}_r47"]
            )
            df["Sample_Ratio_48_44"] = (
                self.Ref_Ratio_48_44 * df[f"{_mode}_s48"] / df[f"raw_s44"] * df[f"raw_r44"] / df[f"{_mode}_r48"]
            )
            df["Sample_Ratio_49_44"] = (
                self.Ref_Ratio_49_44 * df[f"{_mode}_s49"] / df[f"raw_s44"] * df[f"raw_r44"] / df[f"{_mode}_r49"]
            )
            
            # Calculate delta values
            df["d47"] = 1000 * (df["Sample_Ratio_47_44"] / self.Ref_Ratio_47_44 - 1)
            df["d48"] = 1000 * (df["Sample_Ratio_48_44"] / self.Ref_Ratio_48_44 - 1)
            df["d49"] = 1000 * (df["Sample_Ratio_49_44"] / self.Ref_Ratio_49_44 - 1)
            
            # Calculate Delta values (clumped isotope signatures)
            df["D47"] = 1000 * (
                (df["Sample_Ratio_47_44"] / df["Sample_xRatio_47_44"] - 1)
                - (df["Sample_Ratio_46_44"] / df["Sample_xRatio_46_44"] - 1)
                - (df["Sample_Ratio_45_44"] / df["Sample_xRatio_45_44"] - 1)
            )
            
            df["D48"] = 1000 * (
                (df["Sample_Ratio_48_44"] / df["Sample_xRatio_48_44"] - 1)
                - (2 * (df["Sample_Ratio_46_44"] / df["Sample_xRatio_46_44"] - 1))
            )
            
            df["D49"] = 1000 * (
                (df["Sample_Ratio_49_44"] / df["Sample_xRatio_49_44"] - 1)
                - (2 * (df["Sample_Ratio_46_44"] / df["Sample_xRatio_46_44"] - 1))
                - (df["Sample_Ratio_45_44"] / df["Sample_xRatio_45_44"] - 1)
            )
            
            return df
        
        # Process sessions
        if session == "all":
            for S in self.analyses:
                df_session = _calc2(self.analyses[S], S)
                self.analyses[S] = df_session
        elif session in self.analyses:
            df_session = _calc2(self.analyses[session], session)
            self.analyses[session] = df_session
        else:
            msg = "Please set `session` argument either to `all` to process every available session, or specify a session name accessible via `self.analyses`."
            raise Exception(f"{ERROR_BLOCK.format(msg=msg)}")