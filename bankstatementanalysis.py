# app_bank_statement_analysis.py

import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

def load_statement(uploaded_file):
    """Load the bank statement (CSV or Excel) into a DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def detect_column(df, keywords):
    """Detect the first column that matches one of the keywords (case-insensitive)."""
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in col.lower():
                return col
    return None

def clean_transactions(df):
    """Standardize transaction data: Date, Description, Amount, Debit/Credit."""
    df = df.copy()
    df = df.rename(columns={col: col.strip() for col in df.columns})

    # Detect Date column
    date_col = detect_column(df, ['date', 'transaction date'])
    if date_col:
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        st.warning("No date column found; results may be limited.")
        df['Date'] = pd.NaT

    # Detect Amount column
    amount_col = detect_column(df, ['amount', 'amt', 'credit', 'debit'])
    if amount_col:
        df['Amount'] = pd.to_numeric(df[amount_col], errors='coerce')
    else:
        st.error("No Amount column found in uploaded file.")
        return pd.DataFrame()  # Return empty df to avoid further errors

    # Classify Debit/Credit
    df['Type'] = df['Amount'].apply(lambda x: 'Credit' if x >= 0 else 'Debit')

    return df.dropna(subset=['Date', 'Amount'])

def compute_metrics(df):
    total_credit = df[df['Type']=='Credit']['Amount'].sum()
    total_debit = df[df['Type']=='Debit']['Amount'].sum()
    avg_debit = df[df['Type']=='Debit']['Amount'].mean()
    avg_credit = df[df['Type']=='Credit']['Amount'].mean()
    return {
        'Total Credit': total_credit,
        'Total Debit': total_debit,
        'Average Debit': avg_debit,
        'Average Credit': avg_credit
    }

def plot_transactions(df):
    st.subheader("Transactions Over Time")
    df = df.set_index('Date').sort_index()
    monthly = df['Amount'].resample('M').sum()
    fig, ax = plt.subplots()
    monthly.plot(kind='bar', ax=ax)
    ax.set_ylabel("Amount")
    ax.set_title("Monthly Net Amount")
    st.pyplot(fig)

def run_statement_analysis_app():
    st.title("📄 Bank Statement Analysis Tool")

    uploaded_file = st.file_uploader("Upload bank statement (CSV or Excel)", type=["csv","xlsx","xls"])
    if uploaded_file:
        df = load_statement(uploaded_file)
        if df is not None and not df.empty:
            st.write("### Raw Data Preview")
            st.dataframe(df.head())

            df_clean = clean_transactions(df)
            if df_clean.empty:
                st.error("Could not detect Amount column. Please check your file.")
                return

            st.write("### Cleaned Data", df_clean.head())

            metrics = compute_metrics(df_clean)
            st.write("### Key Metrics")
            for k, v in metrics.items():
                st.write(f"**{k}:** {v:.2f}")

            plot_transactions(df_clean)

            st.write("### Transaction Type Distribution")
            st.bar_chart(df_clean['Type'].value_counts())

