import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import json
import io
from data_analyzer import DataAnalyzer
from visualization_generator import VisualizationGenerator
from utils import validate_dataframe, chunk_dataframe, optimize_dataframe_types, safe_dataframe_display

# Configure page
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise-grade CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Header Styling */
    .enterprise-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 0;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo-icon {
        background: rgba(255,255,255,0.2);
        padding: 0.8rem;
        border-radius: 12px;
        font-size: 1.5rem;
    }
    
    .brand-text h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .brand-text p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
        font-weight: 400;
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255,255,255,0.15);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10B981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Sidebar Styling */
    .sidebar-header {
        background: linear-gradient(135deg, #4F7CFF 0%, #6366F1 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(79, 124, 255, 0.2);
    }
    
    .sidebar-title {
        color: white;
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem 1rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 124, 255, 0.15);
        border-color: #4F7CFF;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4F7CFF, #6366F1);
    }
    
    .metric-label {
        color: #64748B;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
    }
    
    .metric-value {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0 0 0;
        line-height: 1;
    }
    
    /* Cards and Containers */
    .enterprise-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #F1F5F9;
    }
    
    .card-title {
        color: #0F172A;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Buttons */
    .enterprise-button {
        background: linear-gradient(135deg, #4F7CFF 0%, #6366F1 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.9rem;
    }
    
    .enterprise-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 124, 255, 0.3);
    }
    
    /* Status Messages */
    .success-alert {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-weight: 500;
    }
    
    .error-alert {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-weight: 500;
    }
    
    .info-alert {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-weight: 500;
    }
    
    /* Data Table Styling */
    .dataframe {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #F8FAFC;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 6px;
        border: 1px solid transparent;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F7CFF 0%, #6366F1 100%);
        color: white;
        border-color: #4F7CFF;
    }
    
    /* Chat Interface */
    .chat-container {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0;
        overflow: hidden;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #E2E8F0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .chat-messages {
        height: 400px;
        overflow-y: auto;
        padding: 1rem;
    }
    
    /* Upload Zone */
    .upload-zone {
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #F8FAFC;
        transition: all 0.2s ease;
        margin: 1rem 0;
    }
    
    .upload-zone:hover {
        border-color: #4F7CFF;
        background: #F0F4FF;
    }
    
    /* Progress Indicators */
    .progress-bar {
        background: #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
        height: 8px;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        background: linear-gradient(90deg, #4F7CFF, #6366F1);
        height: 100%;
        transition: width 0.3s ease;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .header-content {
            flex-direction: column;
            gap: 1rem;
            text-align: center;
        }
        
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_dataframe' not in st.session_state:
    st.session_state.current_dataframe = None
if 'data_analyzer' not in st.session_state:
    st.session_state.data_analyzer = DataAnalyzer()
if 'viz_generator' not in st.session_state:
    st.session_state.viz_generator = VisualizationGenerator()

def main():
    # Enterprise header
    st.markdown("""
    <div class="enterprise-header">
        <div class="header-content">
            <div class="logo-section">
                <div class="logo-icon">📊</div>
                <div class="brand-text">
                    <h1>DataIntel Pro</h1>
                    <p>Enterprise AI-Powered Analytics Platform</p>
                </div>
            </div>
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>AI Engine Active</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Professional sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3 class="sidebar-title">
                <span>📁</span>
                <span>Data Management</span>
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Supported formats: CSV, Excel (.xlsx, .xls), JSON"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading dataset..."):
                    df = load_dataset(uploaded_file)
                    
                if df is not None:
                    # Optimize data types for memory efficiency
                    df = optimize_dataframe_types(df)
                    
                    # Validate the dataframe
                    validation_result = validate_dataframe(df)
                    
                    if validation_result['is_valid']:
                        st.session_state.current_dataframe = df
                        st.markdown("""
                        <div class="success-alert">
                            <span>✅</span>
                            <span><strong>Dataset loaded successfully!</strong></span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Enterprise metrics grid
                        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Total Rows</div>
                                <div class="metric-value">{len(df):,}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Columns</div>
                                <div class="metric-value">{len(df.columns)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Data Size</div>
                                <div class="metric-value">{df.memory_usage(deep=True).sum() / 1024**2:.1f}MB</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Enterprise data overview cards
                        st.markdown("""
                        <div class="enterprise-card">
                            <div class="card-header">
                                <h4 class="card-title">📋 Column Overview</h4>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show column info in a more professional layout
                        col_info = []
                        for col in df.columns:
                            col_info.append({
                                "Column": col,
                                "Type": str(df[col].dtype),
                                "Non-null": f"{df[col].count():,}",
                                "Missing": f"{df[col].isnull().sum():,}",
                                "Unique": f"{df[col].nunique():,}"
                            })
                        st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)
                        
                        # Data preview with enterprise styling
                        st.markdown("""
                        <div class="enterprise-card">
                            <div class="card-header">
                                <h4 class="card-title">👁️ Data Preview</h4>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        safe_preview_df = safe_dataframe_display(df.head(10))
                        st.dataframe(safe_preview_df, use_container_width=True, hide_index=True)
                        
                        # Add visualization center to sidebar
                        st.markdown("---")
                        visualization_center()
                    else:
                        st.error(f"❌ Dataset validation failed: {validation_result['error']}")
                        
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")

    # Main content area with enterprise tabs
    if st.session_state.current_dataframe is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 AI Assistant", 
            "📊 Analytics Dashboard", 
            "🔍 Data Explorer", 
            "⚙️ Advanced Tools"
        ])
        
        with tab1:
            chat_interface()
        
        with tab2:
            analytics_dashboard()
            
        with tab3:
            data_explorer()
            
        with tab4:
            advanced_tools()
    else:
        # Professional welcome screen
        st.markdown("""
        <div class="enterprise-card">
            <div class="card-header">
                <h4 class="card-title">🚀 Getting Started</h4>
            </div>
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <h3 style="color: #4F7CFF; margin-bottom: 1rem;">Welcome to DataIntel Pro</h3>
                <p style="color: #64748B; font-size: 1.1rem; margin-bottom: 2rem;">
                    Upload your dataset to begin advanced AI-powered analysis
                </p>
                <div style="background: #F8FAFC; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #4F7CFF;">
                    <strong>Supported formats:</strong> CSV, Excel (.xlsx, .xls), JSON
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show example queries with visualization focus
        st.subheader("🔍 Example Queries to Try (with automatic visualizations):")
        examples = [
            "Show me the distribution of prices",
            "Display revenue by category",
            "What's the relationship between price and quantity?",
            "Create a pie chart of regions",
            "Show me trends over time",
            "Compare ratings across categories", 
            "Plot a histogram of quantities sold"
        ]
        
        for i, example in enumerate(examples, 1):
            st.write(f"{i}. {example}")

def load_dataset(uploaded_file):
    """Load dataset from uploaded file with chunked processing for large files"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Check file size for chunked processing
            file_size = uploaded_file.size
            
            if file_size > 50 * 1024 * 1024:  # 50MB threshold
                st.info("Large file detected. Using chunked processing...")
                chunks = []
                chunk_size = 10000
                
                # Reset file pointer
                uploaded_file.seek(0)
                
                for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
                    chunks.append(chunk)
                    
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(uploaded_file)
                
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
            
        elif file_extension == 'json':
            # Try different JSON orientations
            uploaded_file.seek(0)
            content = uploaded_file.read()
            
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    df = pd.json_normalize(data)
            except:
                df = pd.read_json(io.StringIO(content.decode('utf-8')), lines=True)
                
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def chat_interface():
    """Enterprise AI Assistant interface"""
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-header">
            <h4 class="card-title">🤖 AI Data Assistant</h4>
            <div style="color: #64748B; font-size: 0.9rem;">Ask questions in natural language</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.current_dataframe
    
    # Display chat history with elegant visualization styling
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            # Display the text content
            st.write(message["content"])
            
            # Display visualization if present with elegant styling
            if "visualization" in message and message["visualization"] is not None:
                st.markdown("""
                <div class="enterprise-card" style="margin-top: 1rem;">
                    <div class="card-header">
                        <h4 class="card-title">📊 AI-Generated Visualization</h4>
                        <div style="color: #64748B; font-size: 0.9rem;">Based on your query analysis</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Apply enterprise styling to the chart
                viz = message["visualization"]
                viz.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font=dict(size=16, color='#1E293B'),
                    font=dict(color='#1E293B'),
                    showlegend=True,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(viz, use_container_width=True, key=f"chart_history_{hash(str(message))}")
                
                # Add download option for the chart
                try:
                    img_data = pio.to_image(viz, format='png', width=1200, height=800)
                    st.download_button(
                        label="📥 Download Chart",
                        data=img_data,
                        file_name="ai_generated_chart.png",
                        mime="image/png",
                        help="Download this AI-generated visualization",
                        key=f"download_{hash(str(message))}"
                    )
                except:
                    pass  # Skip download if there's an issue
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your data..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                try:
                    # Get AI analysis
                    analysis_result = st.session_state.data_analyzer.analyze_query(df, prompt)
                    
                    # Display response
                    st.write(analysis_result["response"])
                    
                    # Generate and display visualization if suggested
                    viz = None
                    # Always try to generate visualization for better user experience
                    if analysis_result.get("needs_visualization") or any(word in prompt.lower() for word in ["show", "plot", "chart", "graph", "visualize", "display"]):
                        with st.spinner("Creating visualization..."):
                            try:
                                viz = st.session_state.viz_generator.generate_visualization(
                                    df, 
                                    prompt, 
                                    analysis_result
                                )
                                print(f"Debug - Generated viz: {viz is not None}")
                            except Exception as e:
                                print(f"Debug - Viz generation error: {e}")
                                st.error(f"Visualization generation error: {e}")
                            
                            if viz is not None:
                                # Display visualization elegantly
                                st.markdown("""
                                <div class="enterprise-card" style="margin-top: 1rem;">
                                    <div class="card-header">
                                        <h4 class="card-title">📊 AI-Generated Visualization</h4>
                                        <div style="color: #64748B; font-size: 0.9rem;">Based on your query analysis</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Apply enterprise styling
                                viz.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    title_font=dict(size=16, color='#1E293B'),
                                    font=dict(color='#1E293B'),
                                    showlegend=True,
                                    margin=dict(l=10, r=10, t=40, b=10)
                                )
                                st.plotly_chart(viz, use_container_width=True, key=f"chart_current_{len(st.session_state.chat_history)}")
                                
                                # Add download option
                                try:
                                    img_data = pio.to_image(viz, format='png', width=1200, height=800)
                                    st.download_button(
                                        label="📥 Download Chart",
                                        data=img_data,
                                        file_name="ai_generated_chart.png",
                                        mime="image/png",
                                        help="Download this AI-generated visualization",
                                        key=f"download_current_{len(st.session_state.chat_history)}"
                                    )
                                except:
                                    pass
                            else:
                                # If no specific viz generated, try to create a simple automatic one
                                st.info("💡 Let me generate an automatic visualization based on your data...")
                                auto_vizs = st.session_state.viz_generator.generate_automatic_visualizations(df)
                                if auto_vizs:
                                    viz = auto_vizs[0]  # Use the first auto-generated visualization
                                    st.plotly_chart(viz, use_container_width=True, key=f"chart_fallback_{len(st.session_state.chat_history)}")
                    
                    # Add to chat history with or without visualization
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": analysis_result["response"],
                        "visualization": viz
                    })
                        
                except Exception as e:
                    error_msg = f"I encountered an error while analyzing your data: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })

def analytics_dashboard():
    """Enterprise analytics dashboard"""
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-header">
            <h4 class="card-title">📊 Analytics Dashboard</h4>
            <div style="color: #64748B; font-size: 0.9rem;">Comprehensive data insights and automated analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.current_dataframe
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="enterprise-card">
            <div class="card-header">
                <h4 class="card-title">📈 Statistical Summary</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate Statistics", key="stats_btn", help="Generate comprehensive statistical analysis"):
            with st.spinner("Generating statistics..."):
                stats = df.describe(include='all')
                safe_df = safe_dataframe_display(stats)
                st.dataframe(safe_df, use_container_width=True, hide_index=False)
    
    with col2:
        st.markdown("""
        <div class="enterprise-card">
            <div class="card-header">
                <h4 class="card-title">🔍 Data Quality Assessment</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Analyze Data Quality", key="quality_btn", help="Comprehensive data quality analysis"):
            with st.spinner("Analyzing data quality..."):
                missing_count = df.isnull().sum().sum()
                duplicate_count = df.duplicated().sum()
                memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                
                # Quality metrics in enterprise style
                quality_metrics = f"""
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Completeness</div>
                        <div class="metric-value">{((1 - missing_count/(len(df)*len(df.columns)))*100):.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Duplicates</div>
                        <div class="metric-value">{duplicate_count:,}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Memory</div>
                        <div class="metric-value">{memory_usage:.1f}MB</div>
                    </div>
                </div>
                """
                st.markdown(quality_metrics, unsafe_allow_html=True)

    # AI-Generated Insights with enterprise styling
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-header">
            <h4 class="card-title">🧠 AI-Generated Insights</h4>
            <div style="color: #64748B; font-size: 0.9rem;">Automated analysis and pattern detection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Generate Comprehensive Analysis", key="insights_btn", help="AI-powered insights and visualizations"):
        with st.spinner("Analyzing data with AI..."):
            try:
                insights = st.session_state.data_analyzer.generate_automatic_insights(df)
                
                # Display insights in enterprise card
                st.markdown(f"""
                <div class="enterprise-card">
                    <div class="card-header">
                        <h4 class="card-title">📋 Analysis Results</h4>
                    </div>
                    <div style="line-height: 1.6; color: #374151;">
                        {insights}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate automatic visualizations
                st.markdown("""
                <div class="enterprise-card">
                    <div class="card-header">
                        <h4 class="card-title">📊 Automated Visualizations</h4>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                auto_viz = st.session_state.viz_generator.generate_automatic_visualizations(df)
                for i, viz in enumerate(auto_viz):
                    st.plotly_chart(viz, use_container_width=True, key=f"auto_viz_{i}")
                    
            except Exception as e:
                st.markdown(f"""
                <div class="error-alert">
                    <span>⚠️</span>
                    <span>Error generating insights: {str(e)}</span>
                </div>
                """, unsafe_allow_html=True)

def advanced_tools():
    """Advanced enterprise tools"""
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-header">
            <h4 class="card-title">⚙️ Advanced Analytics Tools</h4>
            <div style="color: #64748B; font-size: 0.9rem;">Professional data science capabilities</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.current_dataframe
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="enterprise-card">
            <div class="card-header">
                <h4 class="card-title">🔍 Outlier Detection</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select column for outlier analysis", numeric_cols, key="outlier_col")
            if st.button("Detect Outliers", key="outlier_btn"):
                outlier_info = st.session_state.data_analyzer.detect_outliers(df, selected_col)
                if outlier_info:
                    st.markdown(f"""
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-label">Outliers Found</div>
                            <div class="metric-value">{outlier_info['count']}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Percentage</div>
                            <div class="metric-value">{outlier_info['percentage']:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="enterprise-card">
            <div class="card-header">
                <h4 class="card-title">📈 Correlation Analysis</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Analyze Correlations", key="corr_btn"):
            corr_info = st.session_state.data_analyzer.analyze_correlations(df)
            if corr_info and corr_info['strong_correlations']:
                st.write("**Strong Correlations Found:**")
                for corr in corr_info['strong_correlations']:
                    st.write(f"• {corr['column1']} ↔ {corr['column2']}: {corr['correlation']:.3f}")
            else:
                st.info("No strong correlations detected (threshold: 0.7)")

def data_explorer():
    """Enterprise data explorer"""
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-header">
            <h4 class="card-title">🔍 Data Explorer</h4>
            <div style="color: #64748B; font-size: 0.9rem;">Interactive data filtering and export</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.current_dataframe
    
    # Professional filtering interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="enterprise-card">
            <div class="card-header">
                <h4 class="card-title">🔧 Data Filters</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Column selector
        selected_columns = st.multiselect(
            "Select columns to display",
            df.columns.tolist(),
            default=df.columns.tolist()[:10],
            help="Choose which columns to include in the filtered view"
        )
        
        if selected_columns:
            filtered_df = df[selected_columns]
            
            # Row range selector
            max_rows = len(filtered_df)
            row_range = st.slider(
                "Select row range",
                0, max_rows,
                (0, min(100, max_rows)),
                step=1,
                help="Choose the range of rows to display"
            )
    
    with col2:
        st.markdown("""
        <div class="enterprise-card">
            <div class="card-header">
                <h4 class="card-title">📊 Filter Summary</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if selected_columns:
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Selected Columns</div>
                    <div class="metric-value">{len(selected_columns)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Rows Shown</div>
                    <div class="metric-value">{row_range[1] - row_range[0]:,}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    if selected_columns:
        # Display filtered data
        st.markdown("""
        <div class="enterprise-card">
            <div class="card-header">
                <h4 class="card-title">📋 Filtered Dataset</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        display_df = filtered_df.iloc[row_range[0]:row_range[1]]
        safe_df = safe_dataframe_display(display_df)
        st.dataframe(safe_df, use_container_width=True, hide_index=True)
        
        # Professional download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Filtered Data",
            data=csv,
            file_name=f"filtered_data_{len(display_df)}_rows.csv",
            mime="text/csv",
            help="Download the current filtered view as CSV"
        )

def visualization_center():
    """Advanced visualization center with comprehensive chart options"""
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-header">
            <h4 class="card-title">📊 Visualization Center</h4>
            <div style="color: #64748B; font-size: 0.9rem;">Create professional charts and graphs</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.current_dataframe
    
    # Chart type selection
    chart_types = {
        "📊 Bar Chart": "bar",
        "📈 Line Chart": "line", 
        "🔵 Scatter Plot": "scatter",
        "🥧 Pie Chart": "pie",
        "📊 Histogram": "histogram",
        "📦 Box Plot": "box",
        "🔥 Heatmap": "heatmap",
        "🎯 Auto-Generate": "auto"
    }
    
    selected_chart = st.selectbox(
        "Select Chart Type",
        options=list(chart_types.keys()),
        help="Choose the type of visualization to create"
    )
    
    chart_type = chart_types[selected_chart]
    
    # Column selection based on chart type
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = df.columns.tolist()
    
    if chart_type in ["bar", "scatter", "line", "box"]:
        col1, col2 = st.columns(2)
        with col1:
            if chart_type == "bar":
                x_col = st.selectbox("X-axis (Categories)", categorical_cols if categorical_cols else all_cols)
                y_col = st.selectbox("Y-axis (Values)", numeric_cols if numeric_cols else all_cols)
            elif chart_type in ["scatter", "line"]:
                x_col = st.selectbox("X-axis", numeric_cols if numeric_cols else all_cols)
                y_col = st.selectbox("Y-axis", numeric_cols if numeric_cols else all_cols)
            elif chart_type == "box":
                x_col = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
                y_col = st.selectbox("Values", numeric_cols if numeric_cols else all_cols)
                x_col = None if x_col == "None" else x_col
        
        with col2:
            color_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
            color_col = None if color_col == "None" else color_col
            
            size_col = None
            if chart_type == "scatter" and numeric_cols:
                size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols)
                size_col = None if size_col == "None" else size_col
    
    elif chart_type in ["pie", "histogram"]:
        if chart_type == "pie":
            target_col = st.selectbox("Column to analyze", categorical_cols if categorical_cols else all_cols)
        else:  # histogram
            target_col = st.selectbox("Column to analyze", numeric_cols if numeric_cols else all_cols)
    
    # Generate visualization button
    if st.button("🎨 Generate Visualization", key="viz_btn", help="Create the selected visualization"):
        with st.spinner("Creating visualization..."):
            try:
                fig = None
                
                if chart_type == "bar":
                    if x_col and y_col:
                        fig = px.bar(
                            df, x=x_col, y=y_col, color=color_col,
                            title=f"{y_col} by {x_col}",
                            color_discrete_sequence=st.session_state.viz_generator.color_palette
                        )
                
                elif chart_type == "line":
                    if x_col and y_col:
                        fig = px.line(
                            df, x=x_col, y=y_col, color=color_col,
                            title=f"{y_col} vs {x_col}",
                            color_discrete_sequence=st.session_state.viz_generator.color_palette
                        )
                
                elif chart_type == "scatter":
                    if x_col and y_col:
                        fig = px.scatter(
                            df, x=x_col, y=y_col, color=color_col, size=size_col,
                            title=f"{y_col} vs {x_col}",
                            color_discrete_sequence=st.session_state.viz_generator.color_palette
                        )
                
                elif chart_type == "pie":
                    value_counts = df[target_col].value_counts().head(10)
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Distribution of {target_col}",
                        color_discrete_sequence=st.session_state.viz_generator.color_palette
                    )
                
                elif chart_type == "histogram":
                    fig = px.histogram(
                        df, x=target_col,
                        title=f"Distribution of {target_col}",
                        color_discrete_sequence=st.session_state.viz_generator.color_palette
                    )
                
                elif chart_type == "box":
                    if x_col:
                        fig = px.box(
                            df, x=x_col, y=y_col,
                            title=f"Box Plot of {y_col} by {x_col}",
                            color_discrete_sequence=st.session_state.viz_generator.color_palette
                        )
                    else:
                        fig = px.box(
                            df, y=y_col,
                            title=f"Box Plot of {y_col}",
                            color_discrete_sequence=st.session_state.viz_generator.color_palette
                        )
                
                elif chart_type == "heatmap":
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) >= 2:
                        corr_matrix = numeric_df.corr()
                        fig = px.imshow(
                            corr_matrix,
                            title="Correlation Heatmap",
                            color_continuous_scale="RdBu_r",
                            aspect="auto"
                        )
                
                elif chart_type == "auto":
                    # Generate multiple automatic visualizations
                    auto_vizs = st.session_state.viz_generator.generate_automatic_visualizations(df)
                    for i, viz in enumerate(auto_vizs):
                        st.plotly_chart(viz, use_container_width=True, key=f"sidebar_auto_viz_{i}")
                    return
                
                # Display the generated chart
                if fig:
                    # Apply enterprise styling
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        title_font=dict(size=16, color='#1E293B'),
                        font=dict(color='#1E293B'),
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"sidebar_viz_{len(st.session_state.chat_history)}_{chart_type}")
                    
                    # Offer download option
                    img_data = pio.to_image(fig, format='png', width=1200, height=800)
                    st.download_button(
                        label="📥 Download Chart as PNG",
                        data=img_data,
                        file_name=f"{chart_type}_chart.png",
                        mime="image/png",
                        help="Download the chart as a high-quality PNG image"
                    )
                else:
                    st.warning("Could not generate visualization. Please check your column selections.")
                    
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

if __name__ == "__main__":
    main()
