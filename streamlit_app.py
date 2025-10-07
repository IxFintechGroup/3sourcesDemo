#!/usr/bin/env python3
"""
streamlit_app.py — GUI demo for multi-source **daily OHLCV** reconciliation (no API keys).
- Select assets and window (days)
- Fetch daily OHLCV from CMC public data-api, CryptoCompare, CoinGecko
- Show per-source tables
- Fuse per-day via median (close/volume), max/min (high/low), median (open)
- Visualize Close & Volume with matplotlib (no seaborn, one chart per plot)
"""

import io
import time
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import numpy as np
import os
import json
import pickle
from pathlib import Path
from sources_nokey import ASSETS, union_three_sources, fuse_daily, dispersion_daily
from auth_config import check_credentials, get_demo_credentials

# Data storage configuration
DATA_DIR = Path("data_storage")
DATA_DIR.mkdir(exist_ok=True)

def show_login_page():
    """Display the login page."""
    st.set_page_config(page_title="Login - Multi-Source OHLCV", layout="centered")
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1>🔐 Multi-Source OHLCV</h1>
            <h3>Please log in to continue</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if check_credentials(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        # # Demo credentials info
        # with st.expander("Demo Credentials", expanded=False):
        #     demo_creds = get_demo_credentials()
        #     creds_text = "**Available demo accounts:**\n"
        #     for username, password in demo_creds.items():
        #         if password:
        #             creds_text += f"- Username: `{username}` | Password: `{password}`\n"
        #         else:
        #             creds_text += f"- Username: `{username}` | Password: (empty)\n"
        #     st.markdown(creds_text)

def show_logout_button():
    """Display logout button in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"👤 Logged in as: **{st.session_state.username}**")
        if st.button("🚪 Logout", use_container_width=True):
            # Clear authentication state
            for key in list(st.session_state.keys()):
                if key.startswith('authenticated') or key.startswith('username'):
                    del st.session_state[key]
            st.rerun()

def get_run_id():
    """Generate a unique run ID based on current date and time."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_data(run_id, raw_all, fused, disp, assets, days):
    """Save all data for a run."""
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(exist_ok=True)
    
    # Save metadata
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "assets": assets,
        "days": days,
        "data_points": len(raw_all) if not raw_all.empty else 0
    }
    
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save dataframes
    if not raw_all.empty:
        raw_all.to_csv(run_dir / "raw_data.csv", index=False)
    if not fused.empty:
        fused.to_csv(run_dir / "fused_data.csv", index=False)
    if not disp.empty:
        disp.to_csv(run_dir / "dispersion_data.csv", index=False)
    
    return run_dir

def load_data(run_id):
    """Load data for a specific run."""
    run_dir = DATA_DIR / run_id
    
    if not run_dir.exists():
        return None, None, None, None
    
    # Load metadata
    metadata = None
    if (run_dir / "metadata.json").exists():
        with open(run_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
    
    # Load dataframes
    raw_all = pd.DataFrame()
    fused = pd.DataFrame()
    disp = pd.DataFrame()
    
    if (run_dir / "raw_data.csv").exists():
        raw_all = pd.read_csv(run_dir / "raw_data.csv")
    if (run_dir / "fused_data.csv").exists():
        fused = pd.read_csv(run_dir / "fused_data.csv")
    if (run_dir / "dispersion_data.csv").exists():
        disp = pd.read_csv(run_dir / "dispersion_data.csv")
    
    return raw_all, fused, disp, metadata

def get_available_runs():
    """Get list of available data runs."""
    if not DATA_DIR.exists():
        return []
    
    runs = []
    for run_dir in DATA_DIR.iterdir():
        if run_dir.is_dir() and (run_dir / "metadata.json").exists():
            try:
                with open(run_dir / "metadata.json", "r") as f:
                    metadata = json.load(f)
                runs.append({
                    "run_id": run_dir.name,
                    "metadata": metadata
                })
            except:
                continue
    
    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x["metadata"]["timestamp"], reverse=True)
    return runs

def format_x_axis(ax, dates, days_selected):
    """Format x-axis labels based on the number of days selected."""
    if days_selected <= 7:
        # For 7 days or less, show month and date
        ax.set_xticks(dates)
        ax.set_xticklabels([d.strftime('%m/%d') for d in dates], rotation=45)
    else:
        # For more than 7 days, show 9 evenly spaced points
        if len(dates) > 9:
            # Select 9 evenly spaced indices
            indices = np.linspace(0, len(dates) - 1, 9, dtype=int)
            selected_dates = [dates[i] for i in indices]
            ax.set_xticks(selected_dates)
            ax.set_xticklabels([d.strftime('%m/%d') for d in selected_dates], rotation=45)
        else:
            # If we have 9 or fewer dates, show all
            ax.set_xticks(dates)
            ax.set_xticklabels([d.strftime('%m/%d') for d in dates], rotation=45)

def create_chart(sym, raw_all, fused, chart_type, days_selected):
    """Create a properly formatted chart with consistent colors and x-axis."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get data for the selected symbol
    sdf = raw_all[raw_all["symbol"] == sym].copy()
    fdf = fused[fused["symbol"] == sym].copy()
    
    # Convert date strings to datetime for proper sorting and formatting
    sdf['date'] = pd.to_datetime(sdf['date'])
    sdf = sdf.sort_values('date')
    
    # Also convert fused data dates to datetime
    fdf['date'] = pd.to_datetime(fdf['date'])
    fdf = fdf.sort_values('date')
    
    # Create pivot table with proper date handling
    piv = sdf.pivot_table(index="date", columns="source", values=chart_type, aggfunc="last")
    
    # Define consistent colors for each source
    colors = {'CMC': 'green', 'CC': 'blue', 'CG': 'orange'}
    
    # Plot each source with consistent colors
    for source in piv.columns:
        if source in colors:
            piv[source].plot(ax=ax, label=source, color=colors[source], linewidth=2)
    
    # Plot fused data with a distinct color and style
    if chart_type in fdf.columns:
        # Set date as index for plotting
        fdf_plot = fdf.set_index('date')[chart_type]
        fdf_plot.plot(ax=ax, color='red', linewidth=2, linestyle='--', label='FUSED')
    
    # Format x-axis
    dates = piv.index
    format_x_axis(ax, dates, days_selected)
    
    # Set labels and title
    ax.set_xlabel("Date")
    if chart_type == "close":
        ax.set_ylabel("Close (USD)")
        ax.set_title(f"{sym} Close — Sources vs Fused")
    else:
        ax.set_ylabel("Volume (USD)")
        ax.set_title(f"{sym} Volume — Sources vs Fused")
    
    # Add legend
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_percentage_diff_chart(sym, raw_all, chart_type, days_selected):
    """Create a chart showing percentage differences between sources."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get data for the selected symbol
    sdf = raw_all[raw_all["symbol"] == sym].copy()
    sdf['date'] = pd.to_datetime(sdf['date'])
    sdf = sdf.sort_values('date')
    
    # Create pivot table
    piv = sdf.pivot_table(index="date", columns="source", values=chart_type, aggfunc="last")
    
    if len(piv.columns) < 2:
        ax.text(0.5, 0.5, f'Need at least 2 sources for comparison\nOnly {len(piv.columns)} source(s) available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f"{sym} {chart_type.title()} - Source Comparison (Insufficient Data)")
        return fig
    
    # Calculate percentage differences relative to the first source
    base_source = piv.columns[0]
    colors = ['orange', 'blue', 'green', 'red', 'purple']
    
    plotted_lines = 0
    for i, source in enumerate(piv.columns[1:]):
        # Calculate percentage difference
        pct_diff = ((piv[source] - piv[base_source]) / piv[base_source] * 100)
        
        # Check if the line has meaningful variation
        if pct_diff.std() > 0.01:  # Only plot if there's meaningful variation
            color = colors[i % len(colors)]
            pct_diff.plot(ax=ax, label=f"{source} vs {base_source} (%)", 
                         linewidth=2, color=color)
            plotted_lines += 1
        else:
            # Add text annotation for lines with no variation
            mean_diff = pct_diff.mean()
            ax.text(0.02, 0.95 - (i * 0.1), 
                   f"{source} vs {base_source}: {mean_diff:.3f}% (constant)", 
                   transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    # If no lines were plotted, show a message
    if plotted_lines == 0:
        ax.text(0.5, 0.5, 'All sources show identical values\n(0% difference)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    
    # Format x-axis
    dates = piv.index
    format_x_axis(ax, dates, days_selected)
    
    # Set labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Percentage Difference (%)")
    ax.set_title(f"{sym} {chart_type.title()} - Percentage Differences Between Sources")
    
    if plotted_lines > 0:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add summary statistics
    if len(piv.columns) >= 2:
        stats_text = f"Sources: {', '.join(piv.columns)}\n"
        stats_text += f"Data points: {len(piv)}\n"
        if plotted_lines > 0:
            stats_text += f"Active comparisons: {plotted_lines}"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_source_consistency_chart(sym, raw_all, chart_type, days_selected):
    """Create a chart showing source consistency (standard deviation)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get data for the selected symbol
    sdf = raw_all[raw_all["symbol"] == sym].copy()
    sdf['date'] = pd.to_datetime(sdf['date'])
    sdf = sdf.sort_values('date')
    
    # Create pivot table
    piv = sdf.pivot_table(index="date", columns="source", values=chart_type, aggfunc="last")
    
    if len(piv.columns) < 2:
        ax.text(0.5, 0.5, f'Need at least 2 sources for consistency analysis\nOnly {len(piv.columns)} source(s) available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f"{sym} {chart_type.title()} - Source Consistency (Insufficient Data)")
        return fig
    
    # Calculate coefficient of variation (std/mean * 100)
    mean_values = piv.mean(axis=1)
    std_values = piv.std(axis=1)
    cv_values = (std_values / mean_values * 100)
    
    cv_values.plot(ax=ax, color='purple', linewidth=2, label='Coefficient of Variation (%)')
    
    # Format x-axis
    dates = piv.index
    format_x_axis(ax, dates, days_selected)
    
    # Set labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_title(f"{sym} {chart_type.title()} - Source Consistency (Lower = More Consistent)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_source_ranking_chart(sym, raw_all, chart_type, days_selected):
    """Create a chart showing which source has the highest/lowest values over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get data for the selected symbol
    sdf = raw_all[raw_all["symbol"] == sym].copy()
    sdf['date'] = pd.to_datetime(sdf['date'])
    sdf = sdf.sort_values('date')
    
    # Create pivot table
    piv = sdf.pivot_table(index="date", columns="source", values=chart_type, aggfunc="last")
    
    if len(piv.columns) < 2:
        ax.text(0.5, 0.5, f'Need at least 2 sources for ranking analysis\nOnly {len(piv.columns)} source(s) available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f"{sym} {chart_type.title()} - Source Ranking (Insufficient Data)")
        return fig
    
    # Find which source has highest value each day
    highest_source = piv.idxmax(axis=1)
    lowest_source = piv.idxmin(axis=1)
    
    # Count occurrences
    highest_counts = highest_source.value_counts()
    lowest_counts = lowest_source.value_counts()
    
    # Create bar chart
    sources = piv.columns.tolist()
    x_pos = np.arange(len(sources))
    
    bars1 = ax.bar(x_pos - 0.2, [highest_counts.get(s, 0) for s in sources], 0.4, 
                   label='Highest Value Days', alpha=0.8)
    bars2 = ax.bar(x_pos + 0.2, [lowest_counts.get(s, 0) for s in sources], 0.4, 
                   label='Lowest Value Days', alpha=0.8)
    
    ax.set_xlabel("Data Source")
    ax.set_ylabel("Number of Days")
    ax.set_title(f"{sym} {chart_type.title()} - Source Ranking Analysis")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sources)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_correlation_heatmap(sym, raw_all, chart_type):
    """Create a correlation heatmap between sources."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get data for the selected symbol
    sdf = raw_all[raw_all["symbol"] == sym].copy()
    sdf['date'] = pd.to_datetime(sdf['date'])
    sdf = sdf.sort_values('date')
    
    # Create pivot table
    piv = sdf.pivot_table(index="date", columns="source", values=chart_type, aggfunc="last")
    
    if len(piv.columns) < 2:
        ax.text(0.5, 0.5, f'Need at least 2 sources for correlation analysis\nOnly {len(piv.columns)} source(s) available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f"{sym} {chart_type.title()} - Source Correlation (Insufficient Data)")
        return fig
    
    # Calculate correlation matrix
    corr_matrix = piv.corr()
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    # Set ticks and labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45)
    ax.set_yticklabels(corr_matrix.columns)
    
    # Add correlation values to cells
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title(f"{sym} {chart_type.title()} - Source Correlation Matrix")
    
    plt.tight_layout()
    return fig

def generate_price_volume_summary(sym, raw_all, fused):
    """Generate detailed summary for price and volume analysis."""
    sdf = raw_all[raw_all["symbol"] == sym].copy()
    sdf['date'] = pd.to_datetime(sdf['date'])
    fdf = fused[fused["symbol"] == sym].copy()
    fdf['date'] = pd.to_datetime(fdf['date'])
    
    summary = {
        "symbol": sym,
        "date_range": f"{sdf['date'].min().strftime('%Y-%m-%d')} to {sdf['date'].max().strftime('%Y-%m-%d')}",
        "total_days": len(sdf['date'].unique()),
        "sources": sdf['source'].unique().tolist(),
        "price_stats": {},
        "volume_stats": {}
    }
    
    # Price analysis
    piv_close = sdf.pivot_table(index="date", columns="source", values="close", aggfunc="last")
    if not piv_close.empty:
        summary["price_stats"] = {
            "mean_prices": piv_close.mean().to_dict(),
            "price_volatility": piv_close.std().to_dict(),
            "price_range": {
                "min": piv_close.min().min(),
                "max": piv_close.max().max(),
                "range_pct": ((piv_close.max().max() - piv_close.min().min()) / piv_close.min().min() * 100)
            }
        }
    
    # Volume analysis
    piv_volume = sdf.pivot_table(index="date", columns="source", values="volume", aggfunc="last")
    if not piv_volume.empty:
        summary["volume_stats"] = {
            "mean_volumes": piv_volume.mean().to_dict(),
            "volume_volatility": piv_volume.std().to_dict(),
            "volume_range": {
                "min": piv_volume.min().min(),
                "max": piv_volume.max().max(),
                "range_pct": ((piv_volume.max().max() - piv_volume.min().min()) / piv_volume.min().min() * 100)
            }
        }
    
    return summary

def generate_percentage_analysis_summary(sym, raw_all):
    """Generate detailed summary for percentage difference analysis."""
    sdf = raw_all[raw_all["symbol"] == sym].copy()
    sdf['date'] = pd.to_datetime(sdf['date'])
    
    summary = {
        "symbol": sym,
        "close_analysis": {},
        "volume_analysis": {}
    }
    
    # Close price percentage analysis
    piv_close = sdf.pivot_table(index="date", columns="source", values="close", aggfunc="last")
    if len(piv_close.columns) >= 2:
        base_source = piv_close.columns[0]
        close_diffs = {}
        for source in piv_close.columns[1:]:
            pct_diff = ((piv_close[source] - piv_close[base_source]) / piv_close[base_source] * 100)
            close_diffs[f"{source}_vs_{base_source}"] = {
                "mean_diff": pct_diff.mean(),
                "std_diff": pct_diff.std(),
                "min_diff": pct_diff.min(),
                "max_diff": pct_diff.max(),
                "abs_mean_diff": abs(pct_diff.mean()),
                "consistency_score": 100 - abs(pct_diff.mean())  # Higher = more consistent
            }
        summary["close_analysis"] = close_diffs
    
    # Volume percentage analysis
    piv_volume = sdf.pivot_table(index="date", columns="source", values="volume", aggfunc="last")
    if len(piv_volume.columns) >= 2:
        base_source = piv_volume.columns[0]
        volume_diffs = {}
        for source in piv_volume.columns[1:]:
            pct_diff = ((piv_volume[source] - piv_volume[base_source]) / piv_volume[base_source] * 100)
            volume_diffs[f"{source}_vs_{base_source}"] = {
                "mean_diff": pct_diff.mean(),
                "std_diff": pct_diff.std(),
                "min_diff": pct_diff.min(),
                "max_diff": pct_diff.max(),
                "abs_mean_diff": abs(pct_diff.mean()),
                "consistency_score": max(0, 100 - abs(pct_diff.mean()) / 10)  # Adjusted for volume scale
            }
        summary["volume_analysis"] = volume_diffs
    
    return summary

def generate_ranking_analysis_summary(sym, raw_all):
    """Generate detailed summary for source ranking analysis."""
    sdf = raw_all[raw_all["symbol"] == sym].copy()
    sdf['date'] = pd.to_datetime(sdf['date'])
    
    summary = {
        "symbol": sym,
        "close_ranking": {},
        "volume_ranking": {}
    }
    
    # Close price ranking analysis
    piv_close = sdf.pivot_table(index="date", columns="source", values="close", aggfunc="last")
    if len(piv_close.columns) >= 2:
        highest_source = piv_close.idxmax(axis=1)
        lowest_source = piv_close.idxmin(axis=1)
        
        highest_counts = highest_source.value_counts()
        lowest_counts = lowest_source.value_counts()
        
        total_days = len(piv_close)
        close_ranking = {}
        for source in piv_close.columns:
            close_ranking[source] = {
                "highest_days": highest_counts.get(source, 0),
                "lowest_days": lowest_counts.get(source, 0),
                "highest_pct": (highest_counts.get(source, 0) / total_days * 100),
                "lowest_pct": (lowest_counts.get(source, 0) / total_days * 100),
                "reliability_score": (highest_counts.get(source, 0) + lowest_counts.get(source, 0)) / total_days * 100
            }
        summary["close_ranking"] = close_ranking
    
    # Volume ranking analysis
    piv_volume = sdf.pivot_table(index="date", columns="source", values="volume", aggfunc="last")
    if len(piv_volume.columns) >= 2:
        highest_source = piv_volume.idxmax(axis=1)
        lowest_source = piv_volume.idxmin(axis=1)
        
        highest_counts = highest_source.value_counts()
        lowest_counts = lowest_source.value_counts()
        
        total_days = len(piv_volume)
        volume_ranking = {}
        for source in piv_volume.columns:
            volume_ranking[source] = {
                "highest_days": highest_counts.get(source, 0),
                "lowest_days": lowest_counts.get(source, 0),
                "highest_pct": (highest_counts.get(source, 0) / total_days * 100),
                "lowest_pct": (lowest_counts.get(source, 0) / total_days * 100),
                "reliability_score": (highest_counts.get(source, 0) + lowest_counts.get(source, 0)) / total_days * 100
            }
        summary["volume_ranking"] = volume_ranking
    
    return summary

def generate_correlation_analysis_summary(sym, raw_all):
    """Generate detailed summary for correlation analysis."""
    sdf = raw_all[raw_all["symbol"] == sym].copy()
    sdf['date'] = pd.to_datetime(sdf['date'])
    
    summary = {
        "symbol": sym,
        "close_correlation": {},
        "volume_correlation": {}
    }
    
    # Close price correlation analysis
    piv_close = sdf.pivot_table(index="date", columns="source", values="close", aggfunc="last")
    if len(piv_close.columns) >= 2:
        corr_matrix_close = piv_close.corr()
        close_corr = {}
        for i, source1 in enumerate(corr_matrix_close.columns):
            for j, source2 in enumerate(corr_matrix_close.columns):
                if i < j:  # Only upper triangle
                    corr_value = corr_matrix_close.loc[source1, source2]
                    close_corr[f"{source1}_vs_{source2}"] = {
                        "correlation": corr_value,
                        "strength": "Strong" if abs(corr_value) > 0.8 else "Moderate" if abs(corr_value) > 0.5 else "Weak",
                        "direction": "Positive" if corr_value > 0 else "Negative"
                    }
        summary["close_correlation"] = close_corr
    
    # Volume correlation analysis
    piv_volume = sdf.pivot_table(index="date", columns="source", values="volume", aggfunc="last")
    if len(piv_volume.columns) >= 2:
        corr_matrix_volume = piv_volume.corr()
        volume_corr = {}
        for i, source1 in enumerate(corr_matrix_volume.columns):
            for j, source2 in enumerate(corr_matrix_volume.columns):
                if i < j:  # Only upper triangle
                    corr_value = corr_matrix_volume.loc[source1, source2]
                    volume_corr[f"{source1}_vs_{source2}"] = {
                        "correlation": corr_value,
                        "strength": "Strong" if abs(corr_value) > 0.8 else "Moderate" if abs(corr_value) > 0.5 else "Weak",
                        "direction": "Positive" if corr_value > 0 else "Negative"
                    }
        summary["volume_correlation"] = volume_corr
    
    return summary

# Initialize authentication state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Check authentication
if not st.session_state.authenticated:
    show_login_page()
    st.stop()

# Main application starts here (only if authenticated)
st.set_page_config(page_title="Multi-Source Daily OHLCV (No-Key Demo)", layout="wide")

# Show logout button in sidebar
show_logout_button()

st.title("Multi-Source Daily OHLCV — No-Key Demo")
st.markdown("This demo pulls **daily OHLCV** for selected assets from *CMC public data-api*, *CryptoCompare*, and *CoinGecko*, then reconciles to a fused daily series.")

# Initialize session state for symbol selection - this must be outside any conditional blocks
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'BTC'  # Default to BTC

# Initialize a flag to track if data has been loaded
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Check for existing data runs
available_runs = get_available_runs()

with st.sidebar:
    st.header("Data Source")
    
    if available_runs:
        st.subheader("📁 Past Data Runs")
        run_options = ["🆕 Fetch New Data"] + [f"📊 {run['metadata']['timestamp'][:19].replace('T', ' ')} - {', '.join(run['metadata']['assets'])} ({run['metadata']['days']}d)" for run in available_runs]
        selected_run = st.selectbox("Choose data source:", options=run_options, index=0)
        
        use_existing = selected_run != "🆕 Fetch New Data"
        if use_existing:
            # Extract run_id from selection
            selected_run_id = available_runs[run_options.index(selected_run) - 1]["run_id"]
            st.info(f"📊 Using data from: {selected_run_id}")
        else:
            selected_run_id = None
            st.info("🆕 Will fetch fresh data")
    else:
        use_existing = False
        selected_run_id = None
        st.info("🆕 No past data found. Will fetch fresh data.")
    
    st.header("Controls")
    days = st.slider("Days of history", min_value=7, max_value=365, value=90, step=1)
    assets = st.multiselect("Assets", options=ASSETS, default=ASSETS)
    
    if use_existing:
        fetch_btn = st.button("📊 Load Selected Data")
    else:
        fetch_btn = st.button("🔄 Fetch & Reconcile")
    
    # Data management section
    st.header("🗂️ Data Management")
    if available_runs:
        st.write(f"**{len(available_runs)}** data runs available")
        
        # Show recent runs
        st.subheader("Recent Runs")
        for i, run in enumerate(available_runs[:3]):  # Show top 3
            metadata = run['metadata']
            timestamp = metadata['timestamp'][:19].replace('T', ' ')
            st.caption(f"• {timestamp} - {', '.join(metadata['assets'])} ({metadata['days']}d)")
        
        if len(available_runs) > 3:
            st.caption(f"... and {len(available_runs) - 3} more")
    else:
        st.write("No data runs found")

if fetch_btn:
    t0 = time.time()
    
    if use_existing:
        # Load existing data
        st.info(f"📊 Loading data from run: {selected_run_id}")
        raw_all, fused, disp, metadata = load_data(selected_run_id)
        
        if raw_all is None or raw_all.empty:
            st.error("❌ Failed to load data from selected run. Please try fetching new data.")
            st.session_state.data_loaded = False
        else:
            st.success(f"✅ Loaded data from {metadata['timestamp'][:19].replace('T', ' ')}")
            st.info(f"📈 Assets: {', '.join(metadata['assets'])} | Days: {metadata['days']} | Data points: {metadata['data_points']}")
            st.session_state.data_loaded = True
            # Store data in session state for persistence
            st.session_state.raw_all = raw_all
            st.session_state.fused = fused
            st.session_state.disp = disp
    else:
        # Fetch new data
        st.info("🔄 Fetching fresh data from APIs...")
        all_rows = []
        per_asset = {}

        progress = st.progress(0.0)
        status = st.empty()

        for i, sym in enumerate(assets):
            status.write(f"Fetching {sym} ...")
            df = union_three_sources(sym, days=days)
            per_asset[sym] = df.copy()
            all_rows.append(df)
            progress.progress((i+1)/len(assets))
        status.write("Done fetching.")
        progress.progress(1.0)

        if all_rows:
            raw_all = pd.concat(all_rows, ignore_index=True)
        else:
            raw_all = pd.DataFrame(columns=["date","open","high","low","close","volume","source","symbol"])

        # Process the data
        fused = fuse_daily(raw_all)
        disp = dispersion_daily(raw_all)
        
        # Save the data
        run_id = get_run_id()
        save_data(run_id, raw_all, fused, disp, assets, days)
        st.success(f"💾 Data saved with ID: {run_id}")
        st.session_state.data_loaded = True
        # Store data in session state for persistence
        st.session_state.raw_all = raw_all
        st.session_state.fused = fused
        st.session_state.disp = disp

    # Display the data (same for both new and existing)
    st.subheader("Per-Source Daily OHLCV (normalized)")
    st.dataframe(raw_all, width='stretch')

    st.subheader("Fused Daily OHLCV (median close/volume; max/min highs/lows)")
    st.dataframe(fused, width='stretch')

    st.subheader("Per-Day Dispersion Across Sources (% spread)")
    st.dataframe(disp, width='stretch')

# Check if we have data in session state (for when user switches currencies without reloading)
elif st.session_state.data_loaded and 'raw_all' in st.session_state:
    # Use data from session state
    raw_all = st.session_state.raw_all
    fused = st.session_state.fused
    disp = st.session_state.disp
    
    # Display the data
    st.subheader("Per-Source Daily OHLCV (normalized)")
    st.dataframe(raw_all, width='stretch')

    st.subheader("Fused Daily OHLCV (median close/volume; max/min highs/lows)")
    st.dataframe(fused, width='stretch')

    st.subheader("Per-Day Dispersion Across Sources (% spread)")
    st.dataframe(disp, width='stretch')

    # Downloads
    def to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Download per-source CSV", data=to_csv_bytes(raw_all), file_name="per_source_daily.csv", mime="text/csv")
    with c2:
        st.download_button("Download fused daily CSV", data=to_csv_bytes(fused), file_name="fused_daily.csv", mime="text/csv")
    with c3:
        st.download_button("Download dispersion CSV", data=to_csv_bytes(disp), file_name="dispersion_daily.csv", mime="text/csv")

# Show charts section if we have data (either from button click or session state)
if (fetch_btn and st.session_state.data_loaded) or (st.session_state.data_loaded and 'raw_all' in st.session_state):
    # Get data from session state if not already loaded
    if 'raw_all' not in locals():
        raw_all = st.session_state.raw_all
        fused = st.session_state.fused
        disp = st.session_state.disp

    # Visualization (matplotlib; one chart per figure; consistent colors)
    st.header("Charts")

    # Get available symbols from the data
    available_symbols = raw_all['symbol'].unique().tolist() if not raw_all.empty else assets
    
    # Debug information (can be removed later)
    with st.expander("🔍 Debug Info", expanded=False):
        st.write(f"Available symbols: {available_symbols}")
        st.write(f"Current session state symbol: {st.session_state.selected_symbol}")
        st.write(f"Session state keys: {list(st.session_state.keys())}")
        st.write(f"Data loaded flag: {st.session_state.data_loaded}")
    
    # Update session state if current selection is not in available symbols
    if available_symbols and st.session_state.selected_symbol not in available_symbols:
        st.session_state.selected_symbol = available_symbols[0]
    
    # Pick a symbol to visualize with session state
    if available_symbols:
        # Find the current index in the available symbols
        current_index = 0
        if st.session_state.selected_symbol in available_symbols:
            current_index = available_symbols.index(st.session_state.selected_symbol)
        
        # Use a simple, consistent key for the selectbox
        sym = st.selectbox(
            "Select symbol for charts", 
            options=available_symbols,
            index=current_index,
            key="symbol_selector"
        )
        
        # Always update session state to the selected symbol
        st.session_state.selected_symbol = sym
    else:
        sym = st.session_state.selected_symbol

    # Create tabs for different chart types
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price & Volume", "📊 Percentage Analysis", "🎯 Source Ranking", "🔗 Correlation"])
    
    with tab1:
        st.subheader("📈 Price & Volume Analysis")
        
        # Generate and display detailed summary
        price_volume_summary = generate_price_volume_summary(sym, raw_all, fused)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 Date Range", price_volume_summary["date_range"])
        with col2:
            st.metric("📊 Total Days", price_volume_summary["total_days"])
        with col3:
            st.metric("🔗 Data Sources", len(price_volume_summary["sources"]))
        
        # Detailed price statistics
        if price_volume_summary["price_stats"]:
            st.subheader("💰 Price Statistics")
            price_cols = st.columns(len(price_volume_summary["sources"]))
            for i, source in enumerate(price_volume_summary["sources"]):
                with price_cols[i]:
                    mean_price = price_volume_summary["price_stats"]["mean_prices"].get(source, 0)
                    volatility = price_volume_summary["price_stats"]["price_volatility"].get(source, 0)
                    st.metric(f"{source} Avg Price", f"${mean_price:,.2f}", f"±${volatility:,.2f}")
            
            # Price range analysis
            price_range = price_volume_summary["price_stats"]["price_range"]
            st.info(f"📈 **Price Range**: ${price_range['min']:,.2f} - ${price_range['max']:,.2f} ({price_range['range_pct']:.1f}% variation)")
        
        # Detailed volume statistics
        if price_volume_summary["volume_stats"]:
            st.subheader("📊 Volume Statistics")
            volume_cols = st.columns(len(price_volume_summary["sources"]))
            for i, source in enumerate(price_volume_summary["sources"]):
                with volume_cols[i]:
                    mean_volume = price_volume_summary["volume_stats"]["mean_volumes"].get(source, 0)
                    volatility = price_volume_summary["volume_stats"]["volume_volatility"].get(source, 0)
                    st.metric(f"{source} Avg Volume", f"${mean_volume:,.0f}", f"±${volatility:,.0f}")
            
            # Volume range analysis
            volume_range = price_volume_summary["volume_stats"]["volume_range"]
            st.info(f"📊 **Volume Range**: ${volume_range['min']:,.0f} - ${volume_range['max']:,.0f} ({volume_range['range_pct']:.1f}% variation)")
        
        # Charts
        st.subheader("📈 Price & Volume Charts")
        
        # Close plot: per source + fused
        st.subheader(f"{sym} — Close (per source vs fused)")
        fig1 = create_chart(sym, raw_all, fused, "close", days)
        st.pyplot(fig1)

        # Volume plot
        st.subheader(f"{sym} — 24h Volume (per source vs fused)")
        fig2 = create_chart(sym, raw_all, fused, "volume", days)
        st.pyplot(fig2)
    
    with tab2:
        st.subheader("📊 Percentage Difference Analysis")
        st.markdown("Shows how much each source differs from the first source (as percentage)")
        
        # Generate detailed percentage analysis summary
        percentage_summary = generate_percentage_analysis_summary(sym, raw_all)
        
        # Close price percentage analysis
        if percentage_summary["close_analysis"]:
            st.subheader("📈 Close Price Percentage Differences")
            
            # Create columns for each comparison
            close_comparisons = list(percentage_summary["close_analysis"].keys())
            if close_comparisons:
                close_cols = st.columns(len(close_comparisons))
                for i, comparison in enumerate(close_comparisons):
                    with close_cols[i]:
                        data = percentage_summary["close_analysis"][comparison]
                        st.metric(
                            f"{comparison.replace('_vs_', ' vs ')}",
                            f"{data['mean_diff']:.3f}%",
                            f"±{data['std_diff']:.3f}%"
                        )
                
                # Detailed breakdown
                st.subheader("📊 Close Price Detailed Analysis")
                for comparison, data in percentage_summary["close_analysis"].items():
                    with st.expander(f"📈 {comparison.replace('_vs_', ' vs ')} Details"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean Difference", f"{data['mean_diff']:.3f}%")
                        with col2:
                            st.metric("Std Deviation", f"{data['std_diff']:.3f}%")
                        with col3:
                            st.metric("Min Difference", f"{data['min_diff']:.3f}%")
                        with col4:
                            st.metric("Max Difference", f"{data['max_diff']:.3f}%")
                        
                        # Consistency score
                        consistency_color = "green" if data['consistency_score'] > 95 else "orange" if data['consistency_score'] > 90 else "red"
                        st.markdown(f"**Consistency Score**: <span style='color: {consistency_color}'>{data['consistency_score']:.1f}/100</span>", unsafe_allow_html=True)
        
        # Volume percentage analysis
        if percentage_summary["volume_analysis"]:
            st.subheader("📊 Volume Percentage Differences")
            
            # Create columns for each comparison
            volume_comparisons = list(percentage_summary["volume_analysis"].keys())
            if volume_comparisons:
                volume_cols = st.columns(len(volume_comparisons))
                for i, comparison in enumerate(volume_comparisons):
                    with volume_cols[i]:
                        data = percentage_summary["volume_analysis"][comparison]
                        st.metric(
                            f"{comparison.replace('_vs_', ' vs ')}",
                            f"{data['mean_diff']:.1f}%",
                            f"±{data['std_diff']:.1f}%"
                        )
                
                # Detailed breakdown
                st.subheader("📊 Volume Detailed Analysis")
                for comparison, data in percentage_summary["volume_analysis"].items():
                    with st.expander(f"📊 {comparison.replace('_vs_', ' vs ')} Details"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean Difference", f"{data['mean_diff']:.1f}%")
                        with col2:
                            st.metric("Std Deviation", f"{data['std_diff']:.1f}%")
                        with col3:
                            st.metric("Min Difference", f"{data['min_diff']:.1f}%")
                        with col4:
                            st.metric("Max Difference", f"{data['max_diff']:.1f}%")
                        
                        # Consistency score
                        consistency_color = "green" if data['consistency_score'] > 80 else "orange" if data['consistency_score'] > 60 else "red"
                        st.markdown(f"**Consistency Score**: <span style='color: {consistency_color}'>{data['consistency_score']:.1f}/100</span>", unsafe_allow_html=True)
        
        # Percentage difference charts
        st.subheader(f"{sym} — Close Price Percentage Differences")
        fig3 = create_percentage_diff_chart(sym, raw_all, "close", days)
        st.pyplot(fig3)
        
        st.subheader(f"{sym} — Volume Percentage Differences")
        fig4 = create_percentage_diff_chart(sym, raw_all, "volume", days)
        st.pyplot(fig4)
        
        # Source consistency charts
        st.subheader(f"{sym} — Close Price Source Consistency")
        st.markdown("Coefficient of Variation: Lower values = more consistent sources")
        fig5 = create_source_consistency_chart(sym, raw_all, "close", days)
        st.pyplot(fig5)
        
        st.subheader(f"{sym} — Volume Source Consistency")
        fig6 = create_source_consistency_chart(sym, raw_all, "volume", days)
        st.pyplot(fig6)
    
    with tab3:
        st.subheader("🎯 Source Ranking Analysis")
        st.markdown("Shows which source has the highest/lowest values most often")
        
        # Generate detailed ranking analysis summary
        ranking_summary = generate_ranking_analysis_summary(sym, raw_all)
        
        # Close price ranking analysis
        if ranking_summary["close_ranking"]:
            st.subheader("📈 Close Price Source Ranking")
            
            # Create ranking table
            close_ranking_data = []
            for source, data in ranking_summary["close_ranking"].items():
                close_ranking_data.append({
                    "Source": source,
                    "Highest Days": data["highest_days"],
                    "Lowest Days": data["lowest_days"],
                    "Highest %": f"{data['highest_pct']:.1f}%",
                    "Lowest %": f"{data['lowest_pct']:.1f}%",
                    "Reliability Score": f"{data['reliability_score']:.1f}%"
                })
            
            close_ranking_df = pd.DataFrame(close_ranking_data)
            close_ranking_df = close_ranking_df.sort_values("Reliability Score", ascending=False)
            st.dataframe(close_ranking_df, use_container_width=True)
            
            # Best and worst performers
            best_source = close_ranking_df.iloc[0]["Source"]
            worst_source = close_ranking_df.iloc[-1]["Source"]
            st.success(f"🏆 **Best Performer**: {best_source} (most reliable for close prices)")
            st.warning(f"⚠️ **Needs Attention**: {worst_source} (least reliable for close prices)")
        
        # Volume ranking analysis
        if ranking_summary["volume_ranking"]:
            st.subheader("📊 Volume Source Ranking")
            
            # Create ranking table
            volume_ranking_data = []
            for source, data in ranking_summary["volume_ranking"].items():
                volume_ranking_data.append({
                    "Source": source,
                    "Highest Days": data["highest_days"],
                    "Lowest Days": data["lowest_days"],
                    "Highest %": f"{data['highest_pct']:.1f}%",
                    "Lowest %": f"{data['lowest_pct']:.1f}%",
                    "Reliability Score": f"{data['reliability_score']:.1f}%"
                })
            
            volume_ranking_df = pd.DataFrame(volume_ranking_data)
            volume_ranking_df = volume_ranking_df.sort_values("Reliability Score", ascending=False)
            st.dataframe(volume_ranking_df, use_container_width=True)
            
            # Best and worst performers
            best_source = volume_ranking_df.iloc[0]["Source"]
            worst_source = volume_ranking_df.iloc[-1]["Source"]
            st.success(f"🏆 **Best Performer**: {best_source} (most reliable for volume)")
            st.warning(f"⚠️ **Needs Attention**: {worst_source} (least reliable for volume)")
        
        # Charts
        st.subheader("📊 Source Ranking Charts")
        
        # Source ranking charts
        st.subheader(f"{sym} — Close Price Source Ranking")
        fig7 = create_source_ranking_chart(sym, raw_all, "close", days)
        st.pyplot(fig7)
        
        st.subheader(f"{sym} — Volume Source Ranking")
        fig8 = create_source_ranking_chart(sym, raw_all, "volume", days)
        st.pyplot(fig8)
    
    with tab4:
        st.subheader("🔗 Source Correlation Analysis")
        st.markdown("Shows how correlated the data sources are with each other")
        
        # Generate detailed correlation analysis summary
        correlation_summary = generate_correlation_analysis_summary(sym, raw_all)
        
        # Close price correlation analysis
        if correlation_summary["close_correlation"]:
            st.subheader("📈 Close Price Correlation Analysis")
            
            # Create correlation table
            close_corr_data = []
            for comparison, data in correlation_summary["close_correlation"].items():
                close_corr_data.append({
                    "Comparison": comparison.replace("_vs_", " vs "),
                    "Correlation": f"{data['correlation']:.3f}",
                    "Strength": data["strength"],
                    "Direction": data["direction"]
                })
            
            close_corr_df = pd.DataFrame(close_corr_data)
            st.dataframe(close_corr_df, use_container_width=True)
            
            # Correlation insights
            strong_correlations = close_corr_df[close_corr_df["Strength"] == "Strong"]
            if not strong_correlations.empty:
                st.success(f"🔗 **Strong Correlations Found**: {len(strong_correlations)} source pairs show strong correlation (>0.8)")
            else:
                st.info("ℹ️ **Moderate Correlations**: No strong correlations found, sources show independent behavior")
        
        # Volume correlation analysis
        if correlation_summary["volume_correlation"]:
            st.subheader("📊 Volume Correlation Analysis")
            
            # Create correlation table
            volume_corr_data = []
            for comparison, data in correlation_summary["volume_correlation"].items():
                volume_corr_data.append({
                    "Comparison": comparison.replace("_vs_", " vs "),
                    "Correlation": f"{data['correlation']:.3f}",
                    "Strength": data["strength"],
                    "Direction": data["direction"]
                })
            
            volume_corr_df = pd.DataFrame(volume_corr_data)
            st.dataframe(volume_corr_df, use_container_width=True)
            
            # Correlation insights
            strong_correlations = volume_corr_df[volume_corr_df["Strength"] == "Strong"]
            if not strong_correlations.empty:
                st.success(f"🔗 **Strong Correlations Found**: {len(strong_correlations)} source pairs show strong correlation (>0.8)")
            else:
                st.info("ℹ️ **Moderate Correlations**: No strong correlations found, sources show independent behavior")
        
        # Overall correlation summary
        if correlation_summary["close_correlation"] and correlation_summary["volume_correlation"]:
            st.subheader("📊 Overall Correlation Summary")
            
            # Calculate average correlations
            close_correlations = [float(data["Correlation"]) for data in close_corr_data]
            volume_correlations = [float(data["Correlation"]) for data in volume_corr_data]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Close Correlation", f"{np.mean(close_correlations):.3f}")
            with col2:
                st.metric("Avg Volume Correlation", f"{np.mean(volume_correlations):.3f}")
            with col3:
                overall_avg = (np.mean(close_correlations) + np.mean(volume_correlations)) / 2
                st.metric("Overall Average", f"{overall_avg:.3f}")
        
        # Charts
        st.subheader("📊 Correlation Heatmaps")
        
        # Correlation heatmaps
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{sym} — Close Price Correlation")
            fig9 = create_correlation_heatmap(sym, raw_all, "close")
            st.pyplot(fig9)
        
        with col2:
            st.subheader(f"{sym} — Volume Correlation")
            fig10 = create_correlation_heatmap(sym, raw_all, "volume")
            st.pyplot(fig10)

    # Only show completion time if we have t0 defined
    if 't0' in locals():
        st.caption(f"Completed in {time.time()-t0:.2f}s")
else:
    st.info("Choose your assets and click **Fetch & Reconcile** to begin.")
