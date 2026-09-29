import streamlit as st
import pandas as pd
import io
import re
import logging
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

# ==========================================
# 0. Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 0a. Configuration Constants
# ==========================================
MAX_FILES = 50
MAX_FILE_SIZE_MB = 100
SMOOTHING_POLYORDER = 3
CLUSTERING_RANDOM_STATE = 42
DEFAULT_WINDOW_LENGTH = 11
MIN_WINDOW_LENGTH = 5
MAX_WINDOW_LENGTH = 51

# ==========================================
# 1. Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Peak Analysis & Clustering Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom header styling
st.markdown(
    """
    <style>
    .main-title { font-size: 38px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    .help-text { font-size: 12px; color: #7B8BA8; margin-top: 10px; font-style: italic; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. Sidebar: Logos & Developer Metadata
# ==========================================
def load_sidebar_images():
    """Load sidebar images with error handling."""
    images = [
        ("https://brand.umpsa.edu.my/images/logo-umpsa-full-color2.png", "UMPSA Logo"),
        ("https://www.majalahsains.com/wp-content/uploads/2012/05/Logo-Agensi-Nuklear-Malaysia.png", "Agensi Nuklear Malaysia Logo")
    ]
    
    for url, alt_text in images:
        try:
            st.sidebar.image(url, use_container_width=True)
        except Exception as e:
            logger.warning(f"Could not load {alt_text}: {str(e)}")
            st.sidebar.warning(f"⚠️ Could not load {alt_text}")

load_sidebar_images()

st.sidebar.markdown("## Data Analytics Sidebar")
st.sidebar.markdown("---")

# File Upload Panel
st.sidebar.header("📥 Upload Datasets")
uploaded_files = st.sidebar.file_uploader(
    "Upload Text (.txt) or CSV files",
    type=["txt", "csv"],
    accept_multiple_files=True,
    help="Supported formats: .txt, .csv. Files should contain two columns of numeric data (X, Y values)."
)

# Sidebar Hyperparameters Control
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Clustering Configuration")
cluster_features = st.sidebar.selectbox(
    "Select Clustering Features",
    ["Peak Y-Value Only", "Both Peak X and Y-Values"],
    help="Choose whether to cluster based on Y values alone or both X and Y peak coordinates."
)
n_clusters = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=5,
    value=2,
    step=1,
    help="Select the number of clusters for K-Means algorithm."
)

# Data Smoothing Configuration
st.sidebar.markdown("---")
st.sidebar.header("🔧 Data Processing Options")
apply_smoothing = st.sidebar.checkbox(
    "Apply Smoothing Filter (Savitzky-Golay)",
    value=False,
    help="Smooth data to reduce noise using Savitzky-Golay filter."
)
window_length = (
    st.sidebar.slider(
        "Smoothing Window Size",
        MIN_WINDOW_LENGTH,
        MAX_WINDOW_LENGTH,
        DEFAULT_WINDOW_LENGTH,
        step=2,
        help="Window size must be odd. Larger windows produce smoother results."
    )
    if apply_smoothing
    else DEFAULT_WINDOW_LENGTH
)

# Developer Info Block
st.sidebar.markdown("---")
st.sidebar.markdown("### Developers:")
st.sidebar.write("**Asyraf Arif Bin Abu Bakar**")
st.sidebar.caption("Leading Edge NDT Group\nAgensi Nuklear Malaysia\nEmail: asyrafarif@nm.gov.my")

st.sidebar.write("**Dr. Hanafi Ithnin**")
st.sidebar.caption("Bahagian Teknologi Industri (BTI)\nAgensi Nuklear Malaysia\nEmail: hanafi_i@nm.gov.my")

# ==========================================
# 3. Robust Data Parsing Engine
# ==========================================
@st.cache_data
def parse_raw_file(file_obj):
    """
    Cleans metadata headers/annotations and extracts numeric tabular data.
    
    Args:
        file_obj: Streamlit uploaded file object
        
    Returns:
        pd.DataFrame with columns ['x', 'y'], or None if parsing fails
    """
    try:
        content = file_obj.read().decode("utf-8")
        # Remove metadata text injected by logs or indexing tags like [source: XXX]
        clean_content = re.sub(r'\[source:\s*\d+\]', '', content)
        
        # Parse data using space separation, treating rows starting with % as comments
        df = pd.read_csv(io.StringIO(clean_content), comment='%', sep=r'\s+', header=None)
        
        if df.shape[1] >= 2:
            df = df.iloc[:, :2].astype(float, errors='coerce')
            df.columns = ['x', 'y']
            df = df.dropna()  # Remove rows with NaN values
            if not df.empty:
                logger.info(f"Successfully parsed {file_obj.name}: {len(df)} data points")
                return df
    except Exception as e:
        logger.warning(f"Primary parser failed for {file_obj.name}: {str(e)}")
        # Secondary tokenized fallback parser if formatting has irregular linebreaks
        try:
            tokens = []
            for line in clean_content.split('\n'):
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                tokens.extend(line.replace(',', ' ').split())
            numeric_tokens = [float(t) for t in tokens if t]
            if len(numeric_tokens) >= 2:
                x = numeric_tokens[0::2]
                y = numeric_tokens[1::2]
                min_len = min(len(x), len(y))
                df = pd.DataFrame({'x': x[:min_len], 'y': y[:min_len]})
                df = df.dropna()
                if not df.empty:
                    logger.info(f"Fallback parser succeeded for {file_obj.name}: {len(df)} data points")
                    return df
        except Exception as fallback_error:
            logger.error(f"Both parsers failed for {file_obj.name}: {str(fallback_error)}")
            return None
    return None


def apply_data_smoothing(df, window_length=DEFAULT_WINDOW_LENGTH, polyorder=SMOOTHING_POLYORDER):
    """
    Apply Savitzky-Golay smoothing filter to reduce noise.
    
    Args:
        df: DataFrame with 'x' and 'y' columns
        window_length: Window size for smoothing (must be odd)
        polyorder: Polynomial order
        
    Returns:
        Smoothed DataFrame or original DataFrame if conditions not met
    """
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    if len(df) <= window_length:
        logger.warning(f"Dataset too small ({len(df)} points) for smoothing window {window_length}")
        return df
    
    try:
        df_smooth = df.copy()
        df_smooth['y'] = savgol_filter(df['y'], window_length, polyorder)
        logger.info(f"Applied smoothing filter with window_length={window_length}")
        return df_smooth
    except Exception as e:
        logger.error(f"Smoothing filter failed: {str(e)}")
        return df


# ==========================================
# 4. Input Validation
# ==========================================
def validate_uploaded_files(files):
    """Validate uploaded files before processing."""
    if not files:
        return False, "No files uploaded"
    
    if len(files) > MAX_FILES:
        logger.warning(f"User uploaded {len(files)} files, limiting to {MAX_FILES}")
        st.warning(f"⚠️ Limiting to {MAX_FILES} files. You uploaded {len(files)} files.")
        files = files[:MAX_FILES]
    
    # Check file sizes
    for f in files:
        if f.size == 0:
            logger.error(f"Empty file detected: {f.name}")
            return False, f"File '{f.name}' is empty"
        
        size_mb = f.size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            logger.error(f"File too large: {f.name} ({size_mb:.2f}MB)")
            return False, f"File '{f.name}' exceeds {MAX_FILE_SIZE_MB}MB limit"
    
    return True, ""


# ==========================================
# 5. Help & Documentation Section
# ==========================================
def show_help_section():
    """Display help information about input file format."""
    with st.expander("📖 Help & File Format Guide", expanded=False):
        st.markdown(
            """
            ### Input File Format Requirements
            
            Your data files should contain **two columns** of numeric values separated by spaces or commas:
            
            **Example (.txt file):**
            ```
            0.5    10.2
            1.0    15.8
            1.5    20.1
            2.0    18.5
            ```
            
            **Supported Features:**
            - Comments: Lines starting with `%` are ignored
            - Headers: First few metadata lines are automatically cleaned
            - Separators: Spaces, tabs, or commas all work
            - Extensions: `.txt` or `.csv` files
            
            ### What the Dashboard Does
            
            1. **Peak Detection**: Finds the maximum Y-value in each curve
            2. **Smoothing** (optional): Reduces noise using Savitzky-Golay filter
            3. **Clustering**: Groups similar peak characteristics using K-Means
            4. **Visualization**: Interactive plots and downloadable reports
            """
        )


# ==========================================
# 6. Main Application Layout
# ==========================================
st.markdown('<div class="main-title">Peak Analysis & Clustering Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An interactive EDA platform for curve plotting, feature extraction, and K-Means clustering allocation</div>', unsafe_allow_html=True)

show_help_section()

if uploaded_files:
    # Validate files
    is_valid, error_msg = validate_uploaded_files(uploaded_files)
    if not is_valid:
        st.error(f"❌ {error_msg}")
        st.stop()
    
    all_data = {}
    peak_records = []
    total_data_points = 0
    failed_files = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process files sequentially
    for idx, f in enumerate(uploaded_files):
        status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {f.name}")
        progress_bar.progress((idx + 1) / len(uploaded_files))
        
        df_parsed = parse_raw_file(f)
        if df_parsed is not None and not df_parsed.empty:
            # Apply smoothing if enabled
            if apply_smoothing:
                df_parsed = apply_data_smoothing(df_parsed, window_length, polyorder=SMOOTHING_POLYORDER)
            
            all_data[f.name] = df_parsed
            total_data_points += len(df_parsed)
            
            # Extract Peak Features (Global Maxima per curve)
            max_idx = df_parsed['y'].idxmax()
            peak_x = df_parsed.loc[max_idx, 'x']
            peak_y = df_parsed.loc[max_idx, 'y']
            
            peak_records.append({
                "Filename": f.name,
                "Peak_X": peak_x,
                "Peak_Y": peak_y,
                "Total_Points": len(df_parsed)
            })
        else:
            failed_files.append(f.name)
            logger.warning(f"Failed to parse file: {f.name}")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show warning if some files failed
    if failed_files:
        st.warning(f"⚠️ Failed to parse {len(failed_files)} file(s): {', '.join(failed_files)}")
    
    if peak_records:
        df_peaks = pd.DataFrame(peak_records)
        
        # 6a. Summary High-Level KPI Cards
        st.subheader("📊 Executive Overview Metrics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Uploaded Files Count", len(all_data))
        m_col2.metric("Aggregated Data Points", total_data_points)
        m_col3.metric("Minimum Peak Y-Value", f"{df_peaks['Peak_Y'].min():.5f}")
        m_col4.metric("Maximum Peak Y-Value", f"{df_peaks['Peak_Y'].max():.5f}")
        st.markdown("---")
        
        # 6b. Multi-Tab Visualization & Analytics Layout
        tab1, tab2, tab3 = st.tabs(["📈 Combined Curve Plots", "🎯 Extracted Peak Features", "🧬 Peak Value Clustering"])
        
        with tab1:
            st.subheader("Interactive Multi-Curve Visualization")
            fig_curves = go.Figure()
            for name, df in all_data.items():
                fig_curves.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines', name=name))
            
            fig_curves.update_layout(
                xaxis_title="Independent Variable (x)",
                yaxis_title="Response Intensity (y)",
                legend_title="Datasets",
                template="plotly_white",
                height=550,
                hovermode='x unified'
            )
            st.plotly_chart(fig_curves, use_container_width=True)
            
            # Show a data preview sample from the first dataset
            st.subheader("Data Table Sample Preview")
            first_key = list(all_data.keys())[0]
            st.write(f"Displaying top 5 rows sample from: `{first_key}`")
            st.dataframe(all_data[first_key].head(), use_container_width=True)
            
        with tab2:
            st.subheader("Extracted Peak Values Matrix")
            st.markdown("The global maximum value pair $(x, y)$ is parsed and indexed for each unique file:")
            st.dataframe(df_peaks, use_container_width=True)
            
        with tab3:
            st.subheader("K-Means Peak Clustering Groupings")
            
            # Proper edge case handling
            if len(df_peaks) < n_clusters:
                st.error(
                    f"❌ Cannot perform clustering: You have {len(df_peaks)} dataset(s) "
                    f"but need at least {n_clusters} datasets for {n_clusters} clusters. "
                    f"Please upload more files or decrease the cluster slider value in the sidebar."
                )
                logger.error(f"Insufficient datasets: {len(df_peaks)} < {n_clusters} clusters requested")
            else:
                try:
                    # Select features based on sidebar choice
                    if cluster_features == "Peak Y-Value Only":
                        X_feat = df_peaks[['Peak_Y']].copy()
                    else:
                        X_feat = df_peaks[['Peak_X', 'Peak_Y']].copy()
                    
                    # Scale features for numerical convergence stability
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_feat)
                    
                    # Run K-Means Clustering Model
                    kmeans = KMeans(n_clusters=n_clusters, random_state=CLUSTERING_RANDOM_STATE, n_init=10)
                    df_peaks['Cluster_Label'] = kmeans.fit_predict(X_scaled)
                    df_peaks['Cluster_Label'] = df_peaks['Cluster_Label'].apply(lambda l: f"Cluster {l}")
                    
                    logger.info(f"K-Means clustering completed: {n_clusters} clusters, {len(df_peaks)} samples")
                    
                    # Plotly Scatter Plot for Clusters
                    if cluster_features == "Peak Y-Value Only":
                        # Use index for x-axis when only Y-value is used
                        x_col_data = [f"Dataset {i}" for i in range(len(df_peaks))]
                        fig_cluster = px.scatter(
                            df_peaks,
                            x=x_col_data,
                            y="Peak_Y",
                            color="Cluster_Label",
                            hover_data=["Filename", "Peak_X", "Peak_Y"],
                            title="Cluster Classification Map of Peaks",
                            labels={"Cluster_Label": "Assigned Cluster"},
                            size_max=12
                        )
                    else:
                        fig_cluster = px.scatter(
                            df_peaks,
                            x="Peak_X",
                            y="Peak_Y",
                            color="Cluster_Label",
                            hover_data=["Filename", "Peak_X", "Peak_Y"],
                            title="Cluster Classification Map of Peaks",
                            labels={"Cluster_Label": "Assigned Cluster"},
                            size_max=12
                        )
                    
                    fig_cluster.update_traces(marker=dict(size=14, line=dict(width=1.5, color='DarkSlateGrey')))
                    fig_cluster.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig_cluster, use_container_width=True)
                    
                    # Show organized Clustered Results Table
                    st.subheader("Clustered Dataset Allocation Summary")
                    st.dataframe(df_peaks.sort_values(by="Cluster_Label"), use_container_width=True)
                    
                    # Export and Download Section
                    st.subheader("💾 Export Analysis Results")
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    csv_buffer = df_peaks.to_csv(index=False)
                    st.download_button(
                        label="Download Peak Clusters Report as CSV",
                        data=csv_buffer,
                        file_name=f"peaks_clustering_report_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    logger.error(f"Clustering failed: {str(e)}")
                    st.error(f"❌ Clustering failed: {str(e)}")
    else:
        st.error("❌ Unable to parse any data from the uploaded files. Ensure files contain numeric columns.")
        logger.error("No valid peak records generated from uploaded files")
else:
    st.info(
        "💡 Welcome! Please upload one or multiple curve response dataset files (.txt or .csv) "
        "from the sidebar file management panel to start the automated curve peak analysis and clustering analysis."
    )
