import streamlit as st
import numpy as np

from aux_delta_t import plot_delta_t
from aux_over_pumping import plot_overpumping
from aux_short_cycling import plot_short_cycling


def tab_hvac(df):
    # Initial view
    if not st.session_state.get("root_causes", False) and \
       not st.session_state.get("over_pumping", False) and \
       not st.session_state.get("short_cycling", False):
        
        plot_delta_t(df)

    # Hidden tab for Root Causes
    if st.session_state.get("root_causes", False):
        tab1, = st.tabs(["Root Causes"])
        with tab1:
            tab_root_causes()

    # Hidden tab for Over-Pumping
    if st.session_state.get("over_pumping", False):
        tab2, = st.tabs(["Over Pumping"])
        with tab2:
            tab_over_pumping(df)

    # Hidden tab for Short-Cycling
    if st.session_state.get("short_cycling", False):
        tab3, = st.tabs(["Short Cycling"])
        with tab3:
            tab_short_cycling(df)


def tab_root_causes():
    # Hide button
    if st.button("🔙", key="hide_root_causes"):
        st.session_state.root_causes = False
        st.rerun()

    st.markdown("### Low ∆T Root Cause Diagnostics")

    # --- CSS Styling ---
    st.markdown("""
        <style>
        .root-btn {
            display: inline-block;
            width: 100%;
            padding: 1.2em;
            font-size: 1.2em;
            font-weight: 600;
            margin: 0.3em 0;
            text-align: center;
            border-radius: 10px;
            border: none;
            background-color: rgba(0, 255, 255, 0.1);
            transition: background-color 0.3s ease;
            cursor: pointer;
        }
        .root-btn:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        .blinking {
            animation: blink 1s infinite;
            background-color: rgba(255, 0, 0, 0.1);
        }
        @keyframes blink {
            0% {opacity: 1;}
            50% {opacity: 0.5;}
            100% {opacity: 1;}
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Define root causes ---
    interactive_causes = {
        "Over-Pumping": {
            "key": "form_over_pumping",
            "on_submit": lambda: (
                setattr(st.session_state, "over_pumping", True),
                setattr(st.session_state, "root_causes", False),
                st.rerun()
            )
        },
        "Short-Cycling": {
            "key": "form_short_cycling",
            "on_submit": lambda: (
                setattr(st.session_state, "short_cycling", True),
                setattr(st.session_state, "root_causes", False),
                st.rerun()
            )
        }
    }

    static_causes = [
        "Sensor Sanity", "Mixing / Bypass",
        "FAHU Dependency", "Chiller Staging",
        "Air Side", "Set-point / Weather Sensitivity"
    ]

    # --- Combine all causes for layout ---
    all_causes = list(interactive_causes.keys()) + static_causes
    np.random.shuffle(all_causes)

    # --- Display in a 2-column layout ---
    for i in range(0, len(all_causes), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(all_causes):
                cause = all_causes[i + j]
                col = cols[j]

                if cause in interactive_causes:
                    form_key = interactive_causes[cause]["key"]
                    on_submit = interactive_causes[cause]["on_submit"]

                    with col:
                        st.markdown(f'<div class="root-btn blinking">{cause}</div>', unsafe_allow_html=True)
                else:
                    with col:
                        st.markdown(f'<div class="root-btn">{cause}</div>', unsafe_allow_html=True)


    st.markdown("### Explore Details")

    cols = st.columns(len(interactive_causes))

    i = 0
    for cause in interactive_causes:
        form_key = interactive_causes[cause]["key"]
        on_submit = interactive_causes[cause]["on_submit"]

        with cols[i].form(key=form_key):
            submitted = st.form_submit_button(
                label=cause,
                use_container_width=True
            )
            if submitted:
                on_submit()

        # Style the button after it has been rendered
        cols[i].markdown(
            f"""<script>
                const btn = window.parent.document.querySelector('form[data-testid="stForm"][data-testid="{form_key}"] button');
                if (btn) {{
                    btn.className = "root-btn blinking";
                }}
            </script>""",
            unsafe_allow_html=True
        )
        i += 1


def tab_over_pumping(df):
    # --- Top Buttons ---
    col1, col2 = st.columns([1,25])
    with col1:
        if st.button("🔙", key="hide_over_pumping"):
            st.session_state.over_pumping = False
            st.session_state.root_causes = True
            st.rerun()

    with col2:
        if "show_over_pumping_info" not in st.session_state:
            st.session_state.show_over_pumping_info = False

        if st.button("ⓘ", key="show_info_btn"):
            st.session_state.show_over_pumping_info = not st.session_state.show_over_pumping_info

    # --- Information Section ---
    if st.session_state.show_over_pumping_info:
        with st.expander("", expanded=True):
            st.markdown("""
            **The Problem:**  
            When pumps operate at flow rates higher than design specifications, they reduce the temperature differential (ΔT) between supply and return water in HVAC systems.

            **Why It Happens:**
            - Excess flow rate doesn't allow sufficient time for heat transfer  
            - Water moves through coils too quickly to reach optimal temperature change  
            - Pump speed or system settings exceed design parameters  

            **Consequences:**
            - Reduced system efficiency and increased energy costs  
            - Poor temperature control and comfort issues  
            - Potential equipment strain and shortened lifespan  
            - Higher pumping energy consumption with diminished returns

            """)
            st.button("🔙", key="hide_info_btn", on_click=lambda: st.session_state.update({"show_over_pumping_info": False}))

    plot_overpumping(df)


def tab_short_cycling(df):
    hide_btn = st.button("🔙 ", key="hide_short_cycling")
    if hide_btn:
        st.session_state.short_cycling = False
        st.session_state.root_causes = True
        st.rerun()

    st.markdown("The set-point often isn’t met, and the chiller’s frequent cycling makes the system inefficient.")

    plot_short_cycling(df)