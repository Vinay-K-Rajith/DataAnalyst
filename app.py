import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import io
from data_analyzer import DataAnalyzer
from visualization_generator import VisualizationGenerator
from utils import validate_dataframe, chunk_dataframe, optimize_dataframe_types

# Configure page
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.title("🤖 AI-Powered Data Analyst")
    st.markdown("Upload your dataset and ask questions in natural language. The AI will analyze your data and create visualizations automatically.")

    # Sidebar for file upload and dataset info
    with st.sidebar:
        st.header("📁 Dataset Upload")
        
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
                        st.success(f"✅ Dataset loaded successfully!")
                        
                        # Display dataset info
                        st.subheader("📊 Dataset Information")
                        st.write(f"**Rows:** {len(df):,}")
                        st.write(f"**Columns:** {len(df.columns)}")
                        st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                        
                        # Show column info
                        with st.expander("Column Details"):
                            col_info = []
                            for col in df.columns:
                                col_info.append({
                                    "Column": col,
                                    "Type": str(df[col].dtype),
                                    "Non-null": df[col].count(),
                                    "Null": df[col].isnull().sum()
                                })
                            st.dataframe(pd.DataFrame(col_info), use_container_width=True)
                        
                        # Data preview
                        with st.expander("Data Preview"):
                            st.dataframe(df.head(10), use_container_width=True)
                    else:
                        st.error(f"❌ Dataset validation failed: {validation_result['error']}")
                        
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")

    # Main content area
    if st.session_state.current_dataframe is not None:
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["💬 Chat Analysis", "📈 Quick Insights", "📊 Data Explorer"])
        
        with tab1:
            chat_interface()
        
        with tab2:
            quick_insights()
            
        with tab3:
            data_explorer()
    else:
        st.info("👆 Please upload a dataset using the sidebar to get started.")
        
        # Show example queries
        st.subheader("🔍 Example Queries You Can Ask:")
        examples = [
            "Show me the distribution of values in the sales column",
            "Create a correlation heatmap of all numeric columns",
            "What are the top 10 categories by revenue?",
            "Show trends over time for the date column",
            "Find outliers in the dataset",
            "Create a scatter plot comparing price and quantity",
            "Generate summary statistics for all columns"
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
    """Interactive chat interface for data analysis"""
    st.header("💬 Chat with Your Data")
    
    df = st.session_state.current_dataframe
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "visualization" in message:
                st.plotly_chart(message["visualization"], use_container_width=True)
    
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
                    if analysis_result.get("needs_visualization"):
                        viz = st.session_state.viz_generator.generate_visualization(
                            df, 
                            prompt, 
                            analysis_result
                        )
                        
                        if viz is not None:
                            st.plotly_chart(viz, use_container_width=True)
                            
                            # Add to chat history with visualization
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": analysis_result["response"],
                                "visualization": viz
                            })
                        else:
                            # Add to chat history without visualization
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": analysis_result["response"]
                            })
                    else:
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": analysis_result["response"]
                        })
                        
                except Exception as e:
                    error_msg = f"I encountered an error while analyzing your data: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })

def quick_insights():
    """Generate quick insights about the dataset"""
    st.header("📈 Quick Insights")
    
    df = st.session_state.current_dataframe
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Statistics")
        if st.button("Generate Statistics"):
            with st.spinner("Generating statistics..."):
                stats = df.describe(include='all')
                st.dataframe(stats, use_container_width=True)
    
    with col2:
        st.subheader("Data Quality Report")
        if st.button("Analyze Data Quality"):
            with st.spinner("Analyzing data quality..."):
                quality_report = {
                    "Total Rows": len(df),
                    "Total Columns": len(df.columns),
                    "Missing Values": df.isnull().sum().sum(),
                    "Duplicate Rows": df.duplicated().sum(),
                    "Memory Usage (MB)": df.memory_usage(deep=True).sum() / 1024**2
                }
                
                for key, value in quality_report.items():
                    if isinstance(value, float):
                        st.metric(key, f"{value:.2f}")
                    else:
                        st.metric(key, f"{value:,}")

    # Auto-generate insights
    st.subheader("🔍 AI-Generated Insights")
    if st.button("Generate Automatic Insights"):
        with st.spinner("Generating insights..."):
            try:
                insights = st.session_state.data_analyzer.generate_automatic_insights(df)
                st.write(insights)
                
                # Generate automatic visualizations
                auto_viz = st.session_state.viz_generator.generate_automatic_visualizations(df)
                for viz in auto_viz:
                    st.plotly_chart(viz, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")

def data_explorer():
    """Interactive data explorer"""
    st.header("📊 Data Explorer")
    
    df = st.session_state.current_dataframe
    
    # Data filtering
    st.subheader("🔧 Data Filters")
    
    # Column selector
    selected_columns = st.multiselect(
        "Select columns to display",
        df.columns.tolist(),
        default=df.columns.tolist()[:10]  # Show first 10 columns by default
    )
    
    if selected_columns:
        filtered_df = df[selected_columns]
        
        # Row range selector
        max_rows = len(filtered_df)
        row_range = st.slider(
            "Select row range",
            0, max_rows,
            (0, min(100, max_rows)),
            step=1
        )
        
        # Display filtered data
        st.subheader("📋 Filtered Data")
        display_df = filtered_df.iloc[row_range[0]:row_range[1]]
        st.dataframe(display_df, use_container_width=True)
        
        # Download filtered data
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
