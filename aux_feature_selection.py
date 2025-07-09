import streamlit as st
import pandas as pd
import numpy as np

def select_columns(df: pd.DataFrame, search_terms: list[str]) -> list[str]:
    numeric_df = df.select_dtypes(include='number')
    return [
        col for col in numeric_df.columns
        if any(term.lower() in col.lower() for term in search_terms)
    ]


def columns_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Return a dictionary of categorized columns.
    """
    categories = {
        "Power Meter": ["actenergydlvd", "mdb"],
        "Power Consumption": ["kwh", "consumption"],
        "Delta T": ["delta"],
        "Flow": ["flow"],
        "Chiller Supply Temperature": ["suptemp", "suphdrtemp"],
        "Chiller Return Temperature": ["rettemp", "rethdrtemp"],
        "Chiller Set-point Temperature": ['set'],
        "Chiller Used Capacity": ["cap"],
        "Weather": ["oat"],
    }

    categorized = {}
    used_columns = set()

    for category, keywords in categories.items():
        cols = select_columns(df, keywords)
        cols = [col for col in cols if col not in used_columns]
        if cols:
            categorized[category] = cols
            used_columns.update(cols)

    return categorized


def get_category_features(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Display multiselect boxes for each category using all numeric columns as options,
    but default selections based on keyword matches.
    Returns a dictionary with selected columns per category and saves it in session state.
    """
    all_numeric_columns = df.select_dtypes(include='number').columns.tolist()
    categories = columns_categories(df)
    selections = {}

    st.markdown("## Feature Selection Setup")

    for category in categories:
        # Default selection using the keyword-based selector
        default_selection = select_columns(df, categories[category])
        
        selected = st.multiselect(
            label=f"{category}",
            options=all_numeric_columns,
            default=default_selection,
            key=f"select_{category}"
        )
        if selected:
            selections[category] = selected

    # Store in session state
    st.session_state["selections"] = selections

    return selections