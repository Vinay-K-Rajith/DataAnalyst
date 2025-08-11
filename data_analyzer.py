import os
import json
import pandas as pd
import numpy as np
from google import genai
from google.genai import types

class DataAnalyzer:
    def __init__(self):
        """Initialize the DataAnalyzer with Gemini API client"""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.client = genai.Client(api_key=api_key)
        
    def get_dataframe_summary(self, df):
        """Generate a comprehensive summary of the dataframe"""
        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        # Add basic statistics for numeric columns
        if summary["numeric_columns"]:
            numeric_stats = df[summary["numeric_columns"]].describe()
            summary["numeric_stats"] = numeric_stats.to_dict()
            
        # Add unique value counts for categorical columns (limited to top 10)
        categorical_info = {}
        for col in summary["categorical_columns"]:
            if df[col].nunique() <= 50:  # Only for columns with reasonable unique values
                value_counts = df[col].value_counts().head(10).to_dict()
                categorical_info[col] = {
                    "unique_count": df[col].nunique(),
                    "top_values": value_counts
                }
        summary["categorical_info"] = categorical_info
        
        return summary
    
    def analyze_query(self, df, query):
        """Analyze user query and provide insights about the data"""
        try:
            # Get dataframe summary
            df_summary = self.get_dataframe_summary(df)
            
            # Create context for the AI
            context = f"""
            You are an expert data analyst. Analyze the following dataset and answer the user's query.
            
            Dataset Summary:
            - Shape: {df_summary['shape'][0]} rows, {df_summary['shape'][1]} columns
            - Columns: {', '.join(df_summary['columns'])}
            - Numeric columns: {', '.join(df_summary['numeric_columns'])}
            - Categorical columns: {', '.join(df_summary['categorical_columns'])}
            - Missing values: {df_summary['missing_values']}
            
            Sample data (first 5 rows):
            {df.head().to_string()}
            
            User Query: {query}
            
            Please provide:
            1. A detailed analysis answering the user's query
            2. Specific insights based on the actual data
            3. Whether a visualization would be helpful (respond with "VISUALIZATION_NEEDED: true/false")
            4. If visualization is needed, suggest the type of chart/plot
            
            Be specific and reference actual values from the data when possible.
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=context
            )
            
            ai_response = response.text if response.text else "I couldn't analyze your query. Please try rephrasing it."
            
            # Check if visualization is needed
            needs_visualization = "VISUALIZATION_NEEDED: true" in ai_response.upper()
            
            # Extract visualization suggestion
            viz_suggestion = self._extract_visualization_suggestion(ai_response)
            
            return {
                "response": ai_response,
                "needs_visualization": needs_visualization,
                "visualization_suggestion": viz_suggestion,
                "dataframe_summary": df_summary
            }
            
        except Exception as e:
            return {
                "response": f"I encountered an error while analyzing your query: {str(e)}. Please try again with a different question.",
                "needs_visualization": False,
                "visualization_suggestion": None,
                "dataframe_summary": None
            }
    
    def _extract_visualization_suggestion(self, response_text):
        """Extract visualization suggestions from AI response"""
        viz_keywords = {
            "histogram": "histogram",
            "bar chart": "bar",
            "scatter plot": "scatter",
            "line chart": "line",
            "pie chart": "pie",
            "box plot": "box",
            "heatmap": "heatmap",
            "correlation": "correlation"
        }
        
        response_lower = response_text.lower()
        for keyword, viz_type in viz_keywords.items():
            if keyword in response_lower:
                return viz_type
                
        return "auto"  # Let the visualization generator decide
    
    def generate_automatic_insights(self, df):
        """Generate automatic insights about the dataset"""
        try:
            df_summary = self.get_dataframe_summary(df)
            
            # Create a comprehensive analysis prompt
            analysis_prompt = f"""
            As an expert data analyst, provide comprehensive insights about this dataset:
            
            Dataset Overview:
            - {df_summary['shape'][0]} rows and {df_summary['shape'][1]} columns
            - Columns: {', '.join(df_summary['columns'])}
            - Data types: {df_summary['dtypes']}
            - Missing values: {df_summary['missing_values']}
            
            Numeric Columns Statistics:
            {json.dumps(df_summary.get('numeric_stats', {}), indent=2, default=str)}
            
            Categorical Information:
            {json.dumps(df_summary.get('categorical_info', {}), indent=2, default=str)}
            
            Sample Data:
            {df.head(10).to_string()}
            
            Please provide:
            1. Key insights and patterns in the data
            2. Data quality assessment
            3. Interesting observations or anomalies
            4. Recommendations for further analysis
            5. Potential business insights (if applicable)
            
            Be specific and reference actual values from the data.
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=analysis_prompt
            )
            
            return response.text if response.text else "Unable to generate automatic insights at this time."
            
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def detect_outliers(self, df, column):
        """Detect outliers in a numeric column using IQR method"""
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return None
            
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        return {
            "count": len(outliers),
            "percentage": (len(outliers) / len(df)) * 100,
            "outlier_indices": outliers.index.tolist(),
            "bounds": {"lower": lower_bound, "upper": upper_bound}
        }
    
    def analyze_correlations(self, df):
        """Analyze correlations between numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return None
            
        correlation_matrix = numeric_df.corr()
        
        # Find strong correlations (> 0.7 or < -0.7)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        "column1": correlation_matrix.columns[i],
                        "column2": correlation_matrix.columns[j],
                        "correlation": corr_value
                    })
        
        return {
            "correlation_matrix": correlation_matrix,
            "strong_correlations": strong_correlations
        }
