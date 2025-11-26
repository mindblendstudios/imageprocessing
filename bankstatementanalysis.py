# app_bank_statement_analysis.py

import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

# --- Set page config at the very top ---
st.set_page_config(page_title="📄 Bank Statement Analysis", layout="wide")

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

def clean_transactions(df):
    """Standardize transaction data: Date, Description, Amount, Debit/Credit."""
    # Example: assume there are columns Date, Description, Amount
    df = df.copy()
    df = df.rename(columns={col: col.strip() for col in df.columns})
    # Ensure date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.warning("No 'Date' column found; results may be limited.")
    # Ensure Amount column
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    else:
        st.warning("No 'Amount' column found; results may be limited.")
    # Classify as Debit/Credit
    df['Type'] = df['Amount'].apply(lambda x: 'Credit' if x >= 0 else 'Debit')
    return df.dropna(subset=['Date','Amount'])

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
        if df is not None:
            st.write("### Raw Data Preview")
            st.dataframe(df.head())

            df_clean = clean_transactions(df)
            st.write("### Cleaned Data", df_clean.head())

            metrics = compute_metrics(df_clean)
            st.write("### Key Metrics")
            for k,v in metrics.items():
                st.write(f"**{k}:** {v:.2f}")

            plot_transactions(df_clean)

            st.write("### Transaction Type Distribution")
            st.bar_chart(df_clean['Type'].value_counts())

