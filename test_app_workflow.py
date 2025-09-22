#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
import pandas as pd
import numpy as np

# Set default timeout for AppTest
DEFAULT_TIMEOUT = 30
LONG_TIMEOUT = 45
EXTRA_LONG_TIMEOUT = 60


def run_app_workflow():
    """Run the complete app workflow and perform assertions."""
    
    assertion_results = []
    
    try:
        at = AppTest.from_file("Welcome.py")
        at.run(timeout=DEFAULT_TIMEOUT)
       
        try:
            at.switch_page("pages/01_Data-IO.py")
            at.run(timeout=DEFAULT_TIMEOUT)
            assertion_results.append({
            "step": "Open Data IO page",
            "variable": "page_dataIO",
            "passed": True,
            "description": "App can resolve path `pages/01_Data-IO.py`"
             })


        except:
            assertion_results.append({
            "step": "Open Data IO page",
            "variable": "page_dataIO",
            "passed": False,
            "description": "App can not resolve path `pages/01_Data-IO.py`"
             })


        for i, button in enumerate(at.button):
            if button.key == "loadTestRaw":
                button.click()
                break
        at.run(timeout=DEFAULT_TIMEOUT)
        
        input_intensities = None
        if hasattr(at.session_state, "input_intensities"):
            input_intensities = at.session_state.input_intensities
        
        
        assertion_1 = input_intensities is not None and hasattr(input_intensities, 'empty') and not input_intensities.empty
        assertion_results.append({
            "step": "Data IO - Load intensities",
            "variable": "input_intensities",
            "passed": assertion_1,
            "description": "Check if intensity data was loaded successfully"
        })
        
    
        if assertion_1:
            expected_cols = ['UID', 'Sample', 'Session', 'Timetag', 'Replicate', 'raw_r44', 'raw_s46']
            has_expected_cols = all(col in input_intensities.columns for col in expected_cols)
            assertion_results.append({
                "step": "Data IO - Column validation",
                "variable": "input_intensities.columns",
                "passed": has_expected_cols,
                "description": "Check if loaded data has expected columns"
            })
            
            
        at.switch_page("pages/03_Baseline correction.py")
        at.run(timeout=DEFAULT_TIMEOUT)
        
        method_radio = None
        for radio in at.radio:
            if hasattr(radio, 'key') and radio.key == "bg_method":
                method_radio = radio
                break
        method_radio.set_value("Correct baseline using ETH-1 & ETH-2")
        at.run(timeout=DEFAULT_TIMEOUT)


        
        for i, button in enumerate(at.button):
            if button.key == "BUTTON1":
                button.click()
                break
        at.run(timeout=30)  
        
        bg_success = False
        raw_data = None
        if hasattr(at.session_state, "bg_success"):
            bg_success = at.session_state.bg_success
        
        if hasattr(at.session_state, "input_rep"):
            raw_data = at.session_state.input_rep
 
        selected_method = None
        if hasattr(at.session_state, "bg_method"):
            selected_method = at.session_state.bg_method
        

        assertion_results.append({
            "step": "Baseline Correction - Method selection",
            "variable": "bg_method",
            "passed": selected_method == "Correct baseline using ETH-1 & ETH-2",
            "description": f"Check if ETH correction method was selected. Current: {selected_method}"
        })
        
        assertion_results.append({
            "step": "Baseline Correction - Success flag",
            "variable": "bg_success",
            "passed": bg_success == True,
            "description": "Check if baseline correction completed successfully (any method)"
        })
        

        
        assertion_results.append({
            "step": "Baseline Correction - Raw data creation",
            "variable": "raw_data",
            "passed": raw_data is not None and hasattr(raw_data, 'empty') and not raw_data.empty,
            "description": "Check if raw_data DataFrame was created"
        })
        

        pbl_log = None
        if hasattr(at.session_state, "03_pbl_log"):
            pbl_log = at.session_state["03_pbl_log"]
        
        assertion_results.append({
            "step": "Baseline Correction - Optimization log",
            "variable": "03_pbl_log",
            "passed": pbl_log is not None and ("condition is satisfied." in str(pbl_log)),
            "description": f"Check if baseline correction log shows optimization. Log length: {len(str(pbl_log)) if pbl_log else 0}"
        })

        if hasattr(at.session_state, "scaling_factors"):
            sf = at.session_state.scaling_factors
            has_session_sf = "190405-190502" in sf
            assertion_results.append({
                "step": "Session State - Scaling factors",
                "variable": "scaling_factors",
                "passed": has_session_sf,
                "description": "Check if scaling factors exist for session '190405-190502'"
            })
            
            if has_session_sf:
                session_sf = sf["190405-190502"]
                expected_values = {
                    "47b_47.5": -1.15332033,
                    "48b_47.5": -0.38110352,
                    "49b_47.5": -1.53088379
                }
                
                for key, expected in expected_values.items():
                    if key in session_sf:
                        assertion_results.append({
                                "step": f"Scaling Factors - {key} value",
                                "variable": f"scaling_factors['190405-190502']['{key}']",
                                "passed": abs(session_sf[key] - expected_values[key]) < 0.00001,
                                "description": f"Actual ({session_sf[key]}) vs Expected ({expected_values[key]:.8f})"
                            })
                        
                        
                        
        
        at.switch_page("pages/04_Processing.py")
        at.run(timeout=DEFAULT_TIMEOUT)
        
        for checkbox in at.checkbox:
            if checkbox.key == "process_D48":
                checkbox.check()
                break
        at.run(timeout=DEFAULT_TIMEOUT)


        for multiselect in at.multiselect:
            if multiselect.key == "04_selected_calibs":
                multiselect.set_value(["Fiebig24 (original)"])
                break  
        at.run(timeout=DEFAULT_TIMEOUT)

        if hasattr(at.session_state, "04_used_calibs"):
            used_calibs = at.session_state["04_used_calibs"]
            expected_calib = "Fiebig24 (original)"
            assertion_results.append({
                "step": "Session State - Calibration used",
                "variable": "04_used_calibs",
                "passed": expected_calib in used_calibs,
                "description": f"Check if '{expected_calib}' calibration was used"
            })
        


        for i, button in enumerate(at.button):
            if button.key == "BUTTON1":
                button.click()
                break
        at.run(timeout=EXTRA_LONG_TIMEOUT)
        

        correction_output_summary = at.session_state.correction_output_summary
        assertion_results.append({
            "step": "Processing - Success flag",
            "variable": "processing_success",
            "passed": correction_output_summary is not None and hasattr(correction_output_summary, 'empty') and not correction_output_summary.empty,
            "description": "Check if D47/D48 processing completed successfully and output summary was created"
        })
        

        expected_cols = ["Sample", "N", "d13C_VPDB", "d18O_CO2_VSMOW", "D47", "D48", 
                        "2SE_D47", "SE_D48", "T(mean), Fiebig24 (original)"]
        has_cols = all(col in correction_output_summary.columns for col in expected_cols[:6])
        assertion_results.append({
            "step": "Processing - Output columns",
            "variable": "correction_output_summary.columns",
            "passed": has_cols,
            "description": "Check if output has expected columns"
        })
    

        TEMP = correction_output_summary.loc[correction_output_summary["Sample"] == "ETH-3", "T(mean), Fiebig24 (original)"].mean()
        assertion_results.append({
                "step": "Processing - ETH-3 temperature",
                "variable": "ETH-3 T(mean)",
                "passed": 0.001> abs(20.18-TEMP),
                "description": f"Check if ETH-3 temperature ({TEMP:.2f}°C) matches expected ({20.18}°C)"
            })
    

        if input_intensities is not None and hasattr(input_intensities, 'shape'):
            num_rows = input_intensities.shape[0]
            expected_rows = 15340 
            assertion_results.append({
                "step": "Data IO - Input data size",
                "variable": "input_intensities rows",
                "passed": num_rows == expected_rows,
                "description": f"Check if input has {expected_rows} rows (found {num_rows})"
            })
        
        if raw_data is not None and hasattr(raw_data, 'shape'):
            raw_rows = raw_data.shape[0]
            expected_raw_rows = 118 
            assertion_results.append({
                "step": "Baseline Correction - Output size",
                "variable": "raw_data rows",
                "passed": raw_rows == expected_raw_rows,
                "description": f"Check if raw_data has {expected_raw_rows} rows (found {raw_rows})"
            })
        

        if hasattr(at.session_state, "params_last_run"):
            params = at.session_state.params_last_run
            expected_params = {
                "process_D47": True,
                "process_D48": True,
                "process_D49": False,
                "scale": "CDES",
                "correction_method": "pooled"
            }
            params_match = all(params.get(k) == v for k, v in expected_params.items())
            assertion_results.append({
                "step": "Session State - Processing parameters",
                "variable": "params_last_run",
                "passed": params_match,
                "description": "Check if processing parameters match expected values"
            })

        if hasattr(at.session_state, "correction_output_r47Anchors"):
            r47_anchors = at.session_state.correction_output_r47Anchors
            expected_r47_anchors = 0.007620482431692819
            r47_anchors_correct = abs(r47_anchors - expected_r47_anchors) < 0.000001
            assertion_results.append({
                "step": "Correction Output - r47 Anchors",
                "variable": "correction_output_r47Anchors",
                "passed": r47_anchors_correct,
                "description": f"Check if r47 Anchors ({r47_anchors:.15f}) matches expected ({expected_r47_anchors})"
            })
        
        if hasattr(at.session_state, "correction_output_r47Sample"):
            r47_sample = at.session_state.correction_output_r47Sample
            expected_r47_sample = 0.008057696865376282
            r47_sample_correct = abs(r47_sample - expected_r47_sample) < 0.000001
            assertion_results.append({
                "step": "Correction Output - r47 Sample",
                "variable": "correction_output_r47Sample",
                "passed": r47_sample_correct,
                "description": f"Check if r47 Sample ({r47_sample:.15f}) matches expected ({expected_r47_sample})"
            })
        
        if hasattr(at.session_state, "correction_output_r47All"):
            r47_all = at.session_state.correction_output_r47All
            expected_r47_all = 0.007954791514131327
            r47_all_correct = abs(r47_all - expected_r47_all) < 0.000001
            assertion_results.append({
                "step": "Correction Output - r47 All",
                "variable": "correction_output_r47All",
                "passed": r47_all_correct,
                "description": f"Check if r47 All ({r47_all:.15f}) matches expected ({expected_r47_all})"
            })
        
        if hasattr(at.session_state, "correction_output_r48Anchors"):
            r48_anchors = at.session_state.correction_output_r48Anchors
            expected_r48_anchors = 0.029867145038988224
            r48_anchors_correct = abs(r48_anchors - expected_r48_anchors) < 0.000001
            assertion_results.append({
                "step": "Correction Output - r48 Anchors",
                "variable": "correction_output_r48Anchors",
                "passed": r48_anchors_correct,
                "description": f"Check if r48 Anchors ({r48_anchors:.15f}) matches expected ({expected_r48_anchors})"
            })
        
        if hasattr(at.session_state, "correction_output_r48Sample"):
            r48_sample = at.session_state.correction_output_r48Sample
            expected_r48_sample = 0.031045324737444587
            r48_sample_correct = abs(r48_sample - expected_r48_sample) < 0.000001
            assertion_results.append({
                "step": "Correction Output - r48 Sample",
                "variable": "correction_output_r48Sample",
                "passed": r48_sample_correct,
                "description": f"Check if r48 Sample ({r48_sample:.15f}) matches expected ({expected_r48_sample})"
            })
        
        if hasattr(at.session_state, "correction_output_r48All"):
            r48_all = at.session_state.correction_output_r48All
            expected_r48_all = 0.030766227999925982
            r48_all_correct = abs(r48_all - expected_r48_all) < 0.000001
            assertion_results.append({
                "step": "Correction Output - r48 All",
                "variable": "correction_output_r48All",
                "passed": r48_all_correct,
                "description": f"Check if r48 All ({r48_all:.15f}) matches expected ({expected_r48_all})"
            })
        
        if hasattr(at.session_state, "d13Cwg_VPDB"):
            d13c_wg = at.session_state.d13Cwg_VPDB
            expected_d13c_wg = -4.412053965607399
            d13c_wg_correct = abs(d13c_wg - expected_d13c_wg) < 0.000001
            assertion_results.append({
                "step": "Working Gas - d13C",
                "variable": "d13Cwg_VPDB",
                "passed": d13c_wg_correct,
                "description": f"Check if d13C working gas ({d13c_wg:.15f}) matches expected ({expected_d13c_wg})"
            })
        
        if hasattr(at.session_state, "d18Owg_VSMOW"):
            d18o_wg = at.session_state.d18Owg_VSMOW
            expected_d18o_wg = 25.208858121495766
            d18o_wg_correct = abs(d18o_wg - expected_d18o_wg) < 0.000001
            assertion_results.append({
                "step": "Working Gas - d18O",
                "variable": "d18Owg_VSMOW",
                "passed": d18o_wg_correct,
                "description": f"Check if d18O working gas ({d18o_wg:.15f}) matches expected ({expected_d18o_wg})"
            })
        

        if hasattr(at.session_state, "correction_output_full_dataset"):
            full_dataset = at.session_state.correction_output_full_dataset
            if isinstance(full_dataset, str):
                has_full_dataset = "UID" in full_dataset and "Session" in full_dataset and "D47" in full_dataset
                assertion_results.append({
                    "step": "Correction Output - Full dataset",
                    "variable": "correction_output_full_dataset",
                    "passed": has_full_dataset,
                    "description": "Check if full dataset contains expected columns"
                })
                
                assertion_results.append({
                        "step": "Correction Output - Full dataset rows",
                        "variable": "correction_output_full_dataset rows",
                        "passed": len(full_dataset) == 118,
                        "description": f"Check if full dataset has 118 rows (found {len(full_dataset)})"
                    })
        
        KEY = "D47_standardization_error_190405-190502_anchors_d"
        EXPECTED = [-26.000340355629554,-28.54796529749191,-27.222373118175298,-1.6740334117315425,17.013333326776092,-27.57129910649749,-2.9741172024557887,-25.04861901080073,-2.427765025781592,33.497752367310476,18.774766163735137,-26.78971717592915,-1.3674029388170135,-27.094088662831165,18.469678158598455,-3.5810140456915343,-25.74460438381227,-1.971580624700326,17.043323312647644,-21.647596951208342,32.255579561548174,32.64911245990828,-1.2050072563922483,-28.058807638430785,18.77101877309303]
        assertion_results.append({
                        "step": "Standardization Error - D47",
                        "variable": "standardization_error",
                        "passed": at.session_state[KEY] == EXPECTED if KEY in at.session_state else False,
                        "description": f"Check if {EXPECTED} matches (found {at.session_state[KEY]})"
                    })
        


           
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        
        assertion_results.append({
            "step": "Workflow execution",
            "variable": "Exception",
            "passed": False,
            "description": f"Workflow failed with error: {str(e)}"
        })
    
    write_assertion_results(assertion_results)
    
    return {
        "input_intensities": input_intensities if 'input_intensities' in locals() else None,
        "raw_data": raw_data if 'raw_data' in locals() else None,
        "correction_output_summary": correction_output_summary if 'correction_output_summary' in locals() else None,
        "bg_success": bg_success if 'bg_success' in locals() else False,
        "processing_success": processing_success if 'processing_success' in locals() else False,
        "assertion_results": assertion_results
    }


def write_assertion_results(results):
    """Write assertion results to test_assertion_results.txt."""
    
    all_passed = all(r["passed"] for r in results) if results else False
    header = "ALL ASSERTIONS PASSED" if all_passed else "ERRORS OCCURRED"
    
    with open("test_assertion_results.txt", "w") as f:
        f.write(f"{header}\n")
        f.write("=" * len(header) + "\n\n")
        
        if not results:
            f.write("No assertions were executed.\n")
        else:
            
            for i, result in enumerate(results, 1):
                TEXT = ""
                TEXT += str(f"Assertion {i}: {result['step']}\n")
                TEXT += str(f"Variable: {result['variable']}\n")
                TEXT += str(f"Description: {result['description']}\n")
                TEXT += str(f"Result: {'PASSED' if result['passed'] else 'FAILED'}\n")
                TEXT += str("-" * 50 + "\n")
                f.write(TEXT)
                #print(result['passed'], result['description'])
                if result['passed'] == False:
                    print(TEXT)
            f.write(f"\nSummary: {sum(r['passed'] for r in results)}/{len(results)} assertions passed\n")


if __name__ == "__main__":
    os.environ["PYTEST_CURRENT_TEST"] = "1"
    
    try:
        results = run_app_workflow()
        
        assertion_results = results["assertion_results"]
        passed = sum(r["passed"] for r in assertion_results)
        total = len(assertion_results)
        print(f"\nTest completed: {passed}/{total} assertions passed")
        print("Results written to test_assertion_results.txt")
        
        sys.exit(0 if passed == total else 1)
        
    except Exception as e:
        print(f"Error during test execution: {e}")
        import traceback
        traceback.print_exc()
        
        with open("test_assertion_results.txt", "w") as f:
            f.write("ERRORS OCCURRED\n")
            f.write("===============\n\n")
            f.write(f"Fatal error during test execution: {str(e)}\n")
            f.write("\nTraceback:\n")
            f.write(traceback.format_exc())
        
        sys.exit(1)