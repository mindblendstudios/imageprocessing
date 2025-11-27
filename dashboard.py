# excel_dashboard_safe.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import io

def run_excel_dashboard():
    st.title("📊 Excel Data Visualization Dashboard")

    menu = st.sidebar.radio(
        "Navigate",
        ["📁 Upload Excel", "📊 Plot Graphs", "📋 Data Summary", "📥 Download Data"]
    )

    if "df" not in st.session_state:
        st.session_state.df = None

    # --- Upload Excel ---
    if menu == "📁 Upload Excel":
        st.header("Upload Excel File")
        uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
        if uploaded_file:
            try:
                excel_file = pd.ExcelFile(uploaded_file)
                sheet = st.selectbox("Choose Sheet", excel_file.sheet_names)
                df = pd.read_excel(excel_file, sheet_name=sheet)
                if df.empty:
                    st.warning("The selected sheet is empty.")
                else:
                    st.session_state.df = df
                    st.success("Data loaded successfully!")
                    st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error loading file: {e}")

    # --- Plot Graphs ---
    elif menu == "📊 Plot Graphs":
        if st.session_state.df is None:
            st.warning("Please upload a file first.")
        else:
            df = st.session_state.df
            all_cols = df.columns.tolist()
            num_cols = df.select_dtypes(include='number').columns.tolist()

            if not all_cols or not num_cols:
                st.error("No suitable columns available for plotting.")
                return

            x_axis = st.selectbox("X-axis", all_cols)
            y_axis = st.selectbox("Y-axis (numeric)", num_cols)

            # Safety check: ensure selected columns exist
            if x_axis not in df.columns or y_axis not in df.columns:
                st.error("Selected columns are not in the DataFrame.")
                return

            chart_type = st.radio("Chart Type", ["Line", "Bar", "Scatter"])

            plot_df = df[[x_axis, y_axis]].dropna()
            if plot_df.empty:
                st.warning("No data available after dropping NaN values.")
                return

            # --- Matplotlib Plot ---
            fig, ax = plt.subplots()
            if chart_type == "Line":
                ax.plot(plot_df[x_axis], plot_df[y_axis], marker='o')
            elif chart_type == "Bar":
                ax.bar(plot_df[x_axis], plot_df[y_axis])
            elif chart_type == "Scatter":
                ax.scatter(plot_df[x_axis], plot_df[y_axis])
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"{chart_type} Plot")
            st.pyplot(fig)

            # --- Optional Plotly Plot ---
            if st.checkbox("Show interactive Plotly chart"):
                if chart_type == "Line":
                    fig = px.line(plot_df, x=x_axis, y=y_axis)
                elif chart_type == "Bar":
                    fig = px.bar(plot_df, x=x_axis, y=y_axis)
                elif chart_type == "Scatter":
                    fig = px.scatter(plot_df, x=x_axis, y=y_axis)
                st.plotly_chart(fig)  # removed use_container_width

    # --- Data Summary ---
    elif menu == "📋 Data Summary":
        if st.session_state.df is None:
            st.warning("Upload data first.")
        else:
            df = st.session_state.df
            st.dataframe(df.head())
            st.write(df.describe())

    # --- Download Data ---
    elif menu == "📥 Download Data":
        if st.session_state.df is None:
            st.warning("No data to download.")
        else:
            df = st.session_state.df
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="ProcessedData")
            st.download_button(
                "📥 Download Excel",
                data=buffer.getvalue(),
                file_name="processed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
