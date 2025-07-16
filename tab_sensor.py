import streamlit as st
import pandas as pd

from aux_sensor_diagnostics import diagnostic_tools_collection


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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tests", total_tests)
        with col2:
            st.metric("Passed", passed_tests)
        with col3:
            st.metric("Failed", failed_tests)
        
        # Overall health indicator
        health_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        if health_score >= 90:
            st.success(f"✅ System Health: {health_score:.1f}% - Excellent")
        elif health_score >= 70:
            st.warning(f"⚠️ System Health: {health_score:.1f}% - Good")
        else:
            st.error(f"❌ System Health: {health_score:.1f}% - Poor")
    
    # Detailed diagnostics with expanders
    st.subheader("🔬 Detailed Diagnostics")
    
    # Range Check
    if 'diagnostics_range_check' in results:
        result = results['diagnostics_range_check']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        with st.expander(f"📏 Range Check - {status}"):
            if success and result.get('data'):
                st.write("**Out-of-range samples detected:**")
                for col, metrics in result['data'].items():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{col} - % Out of Range", f"{metrics['pct_out']*100:.2f}%")
                    with col2:
                        st.metric(f"{col} - Samples Out", metrics['n_out'])
            else:
                st.error(f"Error: {result.get('message', 'Unknown error')}")
    
    # Gap Check
    if 'diagnostics_gap_check' in results:
        result = results['diagnostics_gap_check']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        with st.expander(f"📊 Gap Check - {status}"):
            if success and result.get('data'):
                data = result['data']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Missing Data", f"{data['pct_missing']*100:.1f}%")
                with col2:
                    st.metric("Longest Gap", f"{data['longest_gap_min']} min")
            else:
                st.error(f"Error: {result.get('message', 'Unknown error')}")
    
    # Spike Detection
    if 'diagnostics_spike_detection' in results:
        result = results['diagnostics_spike_detection']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        with st.expander(f"⚡ Spike Detection - {status}"):
            if success and result.get('data'):
                st.write("**Spikes detected (Z-score > 3.0):**")
                for col, metrics in result['data'].items():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{col} - Number of Spikes", metrics['n_spikes'])
                    with col2:
                        last_spike = metrics['last_spike']
                        if last_spike and last_spike != 'None':
                            st.metric(f"{col} - Last Spike", last_spike[:19])
                        else:
                            st.metric(f"{col} - Last Spike", "None")
            else:
                st.error(f"Error: {result.get('message', 'Unknown error')}")
    
    # Flatline Detection
    if 'diagnostics_flatline_detection' in results:
        result = results['diagnostics_flatline_detection']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        with st.expander(f"📈 Flatline Detection - {status}"):
            if success and result.get('data'):
                st.write("**Flatline periods detected:**")
                for col, metrics in result['data'].items():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{col} - Flat Sections", metrics['n_flat_sections'])
                    with col2:
                        st.metric(f"{col} - Longest Flat", f"{metrics['longest_flat_min']} min")
            else:
                st.error(f"Error: {result.get('message', 'Unknown error')}")
    
    # Sensor Drift
    if 'diagnostics_sensor_drift' in results:
        result = results['diagnostics_sensor_drift']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        with st.expander(f"📉 Sensor Drift - {status}"):
            if success and result.get('data'):
                st.write("**Sensor drift analysis (slope per day):**")
                for col, metrics in result['data'].items():
                    if 'slope_per_day' in metrics:
                        slope = metrics['slope_per_day']
                        # Color code based on drift severity
                        if abs(slope) > 0.5:
                            delta_color = "inverse"
                        elif abs(slope) > 0.1:
                            delta_color = "normal"
                        else:
                            delta_color = "off"
                        
                        st.metric(
                            f"{col} - Drift Rate",
                            f"{slope:.4f} °C/day",
                            delta=f"{slope:.4f}",
                            delta_color=delta_color
                        )
                    else:
                        st.error(f"{col}: {metrics.get('error', 'Unknown error')}")
            else:
                st.error(f"Error: {result.get('message', 'Unknown error')}")
    
    # Calibration Check
    if 'diagnostics_calibration_check' in results:
        result = results['diagnostics_calibration_check']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        with st.expander(f"🎯 Calibration Check - {status}"):
            if success and result.get('data'):
                st.write("**Setpoint vs Sensor comparison:**")
                for comparison, offset in result['data'].items():
                    if offset is not None:
                        # Color code based on offset magnitude
                        if abs(offset) > 2.0:
                            delta_color = "inverse"
                        elif abs(offset) > 1.0:
                            delta_color = "normal"
                        else:
                            delta_color = "off"
                        
                        st.metric(
                            comparison,
                            f"{offset:.2f} °C",
                            delta=f"{offset:.2f}",
                            delta_color=delta_color
                        )
                    else:
                        st.error(f"{comparison}: Missing data")
            else:
                st.error(f"Error: {result.get('message', 'Unknown error')}")