import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

class VisualizationGenerator:
    def __init__(self):
        """Initialize the VisualizationGenerator with blue theme colors"""
        # Custom blue color palette matching the dashboard theme
        self.color_palette = [
            '#4F7CFF',  # Primary blue
            '#6B9AFF',  # Light blue
            '#3B5CCC',  # Dark blue
            '#8FB2FF',  # Lighter blue
            '#2A4BB7',  # Darker blue
            '#B3CBFF',  # Very light blue
            '#1E3A8A',  # Navy blue
            '#DBEAFE'   # Very light blue accent
        ]
        
    def generate_visualization(self, df, query, analysis_result):
        """Generate visualization based on query and analysis result"""
        try:
            viz_type = analysis_result.get("visualization_suggestion", "auto")
            
            # Determine the best visualization based on query keywords
            if viz_type == "auto":
                viz_type = self._determine_viz_type(query, df)
            
            # Generate the appropriate visualization
            if viz_type == "histogram":
                return self._create_histogram(df, query)
            elif viz_type == "bar":
                return self._create_bar_chart(df, query)
            elif viz_type == "scatter":
                return self._create_scatter_plot(df, query)
            elif viz_type == "line":
                return self._create_line_chart(df, query)
            elif viz_type == "pie":
                return self._create_pie_chart(df, query)
            elif viz_type == "box":
                return self._create_box_plot(df, query)
            elif viz_type == "heatmap" or viz_type == "correlation":
                return self._create_correlation_heatmap(df)
            else:
                return self._create_auto_visualization(df, query)
                
        except Exception as e:
            print(f"Error generating visualization: {e}")
            return None
    
    def _determine_viz_type(self, query, df):
        """Determine visualization type based on query keywords"""
        query_lower = query.lower()
        
        # Enhanced keyword mapping with more specific detection
        if any(word in query_lower for word in ["distribution", "histogram", "frequency", "spread", "range"]):
            return "histogram"
        elif any(word in query_lower for word in ["correlation", "heatmap", "relationship between all", "correlations", "related"]):
            return "correlation"
        elif any(word in query_lower for word in ["scatter", "relationship", " vs ", "compare", "correlation between"]):
            return "scatter"
        elif any(word in query_lower for word in ["trend", "time", "over time", "line", "timeline", "progression"]):
            return "line"
        elif any(word in query_lower for word in ["proportion", "percentage", "pie", "composition", "breakdown", "share"]):
            return "pie"
        elif any(word in query_lower for word in ["top", "bottom", "highest", "lowest", "ranking", "best", "worst", "most", "least"]):
            return "bar"
        elif any(word in query_lower for word in ["outlier", "box plot", "quartile", "median", "whisker"]):
            return "box"
        elif any(word in query_lower for word in ["show", "display", "visualize", "plot", "chart", "graph"]):
            # If general visualization request, use auto-detection
            return "auto"
        else:
            return "auto"
    
    def _create_histogram(self, df, query):
        """Create histogram for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        # Try to find the most relevant column based on query
        target_col = self._find_relevant_column(query, numeric_cols.tolist())
        if not target_col:
            target_col = numeric_cols[0]
        
        fig = px.histogram(
            df, 
            x=target_col, 
            title=f"Distribution of {target_col}",
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title=target_col,
            yaxis_title="Frequency",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=16, color='#1E293B'),
            font=dict(color='#1E293B'),
            margin=dict(l=10, r=10, t=60, b=10)
        )
        
        return fig
    
    def _create_bar_chart(self, df, query):
        """Create bar chart for categorical data or top/bottom analysis"""
        # Look for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return None
        
        # Find relevant columns
        cat_col = self._find_relevant_column(query, categorical_cols.tolist())
        num_col = self._find_relevant_column(query, numeric_cols.tolist())
        
        if not cat_col:
            cat_col = categorical_cols[0]
        if not num_col:
            num_col = numeric_cols[0]
        
        # Check if we need top/bottom analysis
        if any(word in query.lower() for word in ["top", "bottom", "highest", "lowest"]):
            # Aggregate data and get top/bottom values
            agg_data = df.groupby(cat_col)[num_col].agg(['sum', 'mean', 'count']).reset_index()
            
            # Determine which aggregation to use
            if "count" in query.lower() or "number" in query.lower():
                agg_col = 'count'
            elif "average" in query.lower() or "mean" in query.lower():
                agg_col = 'mean'
            else:
                agg_col = 'sum'
            
            # Sort and get top 10
            agg_data = agg_data.sort_values(agg_col, ascending=False).head(10)
            
            fig = px.bar(
                agg_data,
                x=cat_col,
                y=agg_col,
                title=f"Top 10 {cat_col} by {agg_col.title()} of {num_col}",
                color_discrete_sequence=self.color_palette
            )
        else:
            # Regular bar chart
            fig = px.bar(
                df,
                x=cat_col,
                y=num_col,
                title=f"{num_col} by {cat_col}",
                color_discrete_sequence=self.color_palette
            )
        
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    def _create_scatter_plot(self, df, query):
        """Create scatter plot for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        # Try to find relevant columns from query
        x_col = self._find_relevant_column(query, numeric_cols.tolist())
        y_col = self._find_relevant_column(query, numeric_cols.tolist(), exclude=[x_col])
        
        if not x_col:
            x_col = numeric_cols[0]
        if not y_col:
            y_col = numeric_cols[1]
        
        # Look for categorical column for color coding
        categorical_cols = df.select_dtypes(include=['object']).columns
        color_col = None
        if len(categorical_cols) > 0:
            color_col = categorical_cols[0]
            # Limit categories to avoid overcrowding
            if df[color_col].nunique() > 20:
                color_col = None
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"{y_col} vs {x_col}",
            color_discrete_sequence=self.color_palette
        )
        
        return fig
    
    def _create_line_chart(self, df, query):
        """Create line chart for time series or trend analysis"""
        # Look for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # If no datetime columns, try to find date-like columns
        if len(datetime_cols) == 0:
            for col in df.columns:
                if any(word in col.lower() for word in ['date', 'time', 'year', 'month']):
                    try:
                        df[col] = pd.to_datetime(df[col])
                        datetime_cols = [col]
                        break
                    except:
                        continue
        
        if len(datetime_cols) == 0 or len(numeric_cols) == 0:
            return None
        
        date_col = datetime_cols[0]
        value_col = self._find_relevant_column(query, numeric_cols.tolist())
        if not value_col:
            value_col = numeric_cols[0]
        
        # Sort by date
        df_sorted = df.sort_values(date_col)
        
        fig = px.line(
            df_sorted,
            x=date_col,
            y=value_col,
            title=f"{value_col} Over Time",
            color_discrete_sequence=self.color_palette
        )
        
        return fig
    
    def _create_pie_chart(self, df, query):
        """Create pie chart for categorical proportions"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            return None
        
        target_col = self._find_relevant_column(query, categorical_cols.tolist())
        if not target_col:
            target_col = categorical_cols[0]
        
        # Count values and get top categories to avoid cluttered pie chart
        value_counts = df[target_col].value_counts().head(10)
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Distribution of {target_col}",
            color_discrete_sequence=self.color_palette
        )
        
        return fig
    
    def _create_box_plot(self, df, query):
        """Create box plot for outlier analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) == 0:
            return None
        
        target_col = self._find_relevant_column(query, numeric_cols.tolist())
        if not target_col:
            target_col = numeric_cols[0]
        
        # Use categorical column for grouping if available
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            # Limit categories
            if df[cat_col].nunique() <= 20:
                fig = px.box(
                    df,
                    x=cat_col,
                    y=target_col,
                    title=f"Box Plot of {target_col} by {cat_col}",
                    color_discrete_sequence=self.color_palette
                )
            else:
                fig = px.box(
                    df,
                    y=target_col,
                    title=f"Box Plot of {target_col}",
                    color_discrete_sequence=self.color_palette
                )
        else:
            fig = px.box(
                df,
                y=target_col,
                title=f"Box Plot of {target_col}",
                color_discrete_sequence=self.color_palette
            )
        
        return fig
    
    def _create_correlation_heatmap(self, df):
        """Create correlation heatmap for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return None
        
        correlation_matrix = numeric_df.corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        
        fig.update_layout(
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig
    
    def _create_auto_visualization(self, df, query):
        """Create automatic visualization based on data characteristics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Decision tree for automatic visualization
        if len(numeric_cols) >= 2:
            # Create scatter plot for numeric data
            return self._create_scatter_plot(df, query)
        elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            # Create bar chart
            return self._create_bar_chart(df, query)
        elif len(numeric_cols) == 1:
            # Create histogram
            return self._create_histogram(df, query)
        elif len(categorical_cols) >= 1:
            # Create pie chart
            return self._create_pie_chart(df, query)
        else:
            return None
    
    def _find_relevant_column(self, query, columns, exclude=None):
        """Find the most relevant column based on query keywords"""
        if exclude is None:
            exclude = []
            
        query_lower = query.lower()
        
        # Check for exact matches first
        for col in columns:
            if col not in exclude and col.lower() in query_lower:
                return col
        
        # Check for partial matches
        for col in columns:
            if col not in exclude:
                col_words = col.lower().split('_')
                for word in col_words:
                    if len(word) > 2 and word in query_lower:
                        return col
        
        return None
    
    def generate_automatic_visualizations(self, df):
        """Generate a set of automatic visualizations for the dataset"""
        visualizations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        try:
            # 1. Correlation heatmap if multiple numeric columns
            if len(numeric_cols) >= 2:
                corr_viz = self._create_correlation_heatmap(df)
                if corr_viz:
                    visualizations.append(corr_viz)
            
            # 2. Distribution of first numeric column
            if len(numeric_cols) >= 1:
                hist_viz = px.histogram(
                    df,
                    x=numeric_cols[0],
                    title=f"Distribution of {numeric_cols[0]}",
                    color_discrete_sequence=self.color_palette
                )
                visualizations.append(hist_viz)
            
            # 3. Categorical distribution
            if len(categorical_cols) >= 1:
                cat_col = categorical_cols[0]
                if df[cat_col].nunique() <= 20:  # Avoid overcrowded charts
                    pie_viz = self._create_pie_chart(df, f"distribution of {cat_col}")
                    if pie_viz:
                        visualizations.append(pie_viz)
            
            # 4. Numeric vs Categorical if both exist
            if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                bar_viz = self._create_bar_chart(df, f"{numeric_cols[0]} by {categorical_cols[0]}")
                if bar_viz:
                    visualizations.append(bar_viz)
        
        except Exception as e:
            print(f"Error generating automatic visualizations: {e}")
        
        return visualizations
