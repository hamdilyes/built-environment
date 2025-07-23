import streamlit as st
import pandas as pd

from aux_sensor import diagnostic_tools_collection


def tab_sensor(df: pd.DataFrame):  
    toolbox = diagnostic_tools_collection()
    
    END = df.index.max()
    
    # Build core series (using available columns)
    sensor_cols = ["BTU-01_SupTemp_degC", "BTU-01_RetTemp_degC", "BTU-01_ChwFlow_L/s"]
    existing_sensor_cols = [col for col in sensor_cols if col in df.columns]
    
    # Create slice function
    def _slice(obj, days):
        return obj.loc[END - pd.Timedelta(days=days): END]
    
    # Build payloads dict (adapted from your original code)
    payloads = {}
    TOOL_DEFAULTS = {
        "diagnostics_range_check": {"window_days": 1},
        "diagnostics_gap_check": {"window_days": 1},
        "diagnostics_spike_detection": {"window_days": 1},
        "diagnostics_flatline_detection": {"window_days": 1},
        "diagnostics_sensor_drift": {"window_days": 1},
        "diagnostics_calibration_check": {"window_days": 1}
    }
    
    for name, cfg in TOOL_DEFAULTS.items():
        days = cfg["window_days"]
        
        if name == "diagnostics_range_check":
            bounds = {
                "BTU-01_SupTemp_degC": (0, 60),
                "BTU-01_RetTemp_degC": (0, 60),
                "BTU-01_ChwFlow_L/s": (0, 50),
            }
            # Filter bounds to only include existing columns
            existing_bounds = {k: v for k, v in bounds.items() if k in df.columns}
            if existing_bounds:
                payloads[name] = {
                    "dataframe": _slice(df, days),
                    "columns": existing_bounds
                }
        
        elif name == "diagnostics_gap_check":
            payloads[name] = {
                "dataframe": _slice(df, days),
                "max_allowed_gap_min": 60
            }
        
        elif name == "diagnostics_spike_detection":
            if existing_sensor_cols:
                payloads[name] = {
                    "dataframe": _slice(df, days),
                    "sensor_cols": existing_sensor_cols
                }
        
        elif name == "diagnostics_flatline_detection":
            if existing_sensor_cols:
                payloads[name] = {
                    "dataframe": _slice(df, days),
                    "sensor_cols": existing_sensor_cols
                }
        
        elif name == "diagnostics_sensor_drift":
            if existing_sensor_cols:
                payloads[name] = {
                    "dataframe": _slice(df, days),
                    "sensor_cols": existing_sensor_cols
                }
        
        elif name == "diagnostics_calibration_check":
            # Check if setpoint and sensor columns exist
            if "CHW-SYS_ChwSetpt (°C)" in df.columns and "BTU-01_SupTemp_degC" in df.columns:
                payloads[name] = {
                    "df_setpoint": _slice(df["CHW-SYS_ChwSetpt (°C)"], days),
                    "df_sensor": _slice(df, days),
                    "setpoint_cols": ["CHW-SYS_ChwSetpt (°C)"],
                    "sensor_cols": ["BTU-01_SupTemp_degC"]
                }
    
    # Execute diagnostics
    results = {}
    for tool in toolbox:
        if tool.__name__ not in payloads:
            continue
        try:
            results[tool.__name__] = tool(**payloads[tool.__name__])
        except Exception as exc:
            results[tool.__name__] = {"success": False, "message": str(exc), "data": {}}
    
    # Create overview section
    with st.expander("OVERVIEW", expanded=True):
    
        # Count pass/fail
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get('success', False))
        failed_tests = total_tests - passed_tests
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tests", total_tests)
        with col2:
            st.metric("Passed", passed_tests)
        with col3:
            st.metric("Failed", failed_tests)
        # Compute health score
        health_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        with col4:
            st.metric("Health Rate", f"{health_score:.0f}%")
    
    # Detailed diagnostics with expanders
    st.subheader("🔬 Detailed Diagnostics")
    
    # Range Check
    if 'diagnostics_range_check' in results:
        result = results['diagnostics_range_check']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        with st.expander(f"Range Check - {status}"):
            st.write("")
    
    # Gap Check
    if 'diagnostics_gap_check' in results:
        result = results['diagnostics_gap_check']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        with st.expander(f"Gap Check - {status}"):
            st.write("")
    
    # Spike Detection
    if 'diagnostics_spike_detection' in results:
        result = results['diagnostics_spike_detection']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        with st.expander(f"Spike Detection - {status}"):
            st.write("")
    
    # Flatline Detection
    if 'diagnostics_flatline_detection' in results:
        result = results['diagnostics_flatline_detection']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        with st.expander(f"Flatline Detection - {status}"):
            st.write("")
    
    # Sensor Drift
    if 'diagnostics_sensor_drift' in results:
        result = results['diagnostics_sensor_drift']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        with st.expander(f"Sensor Drift - {status}"):
            st.write("")
    
    # Calibration Check
    if 'diagnostics_calibration_check' in results:
        result = results['diagnostics_calibration_check']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        with st.expander(f"Calibration Check - {status}"):
            st.write("")