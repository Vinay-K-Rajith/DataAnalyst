import os
import json
import pandas as pd
import numpy as np
import google.generativeai as genai
from typing import Dict, List, Optional, Any, Union, Tuple
import warnings
from dataclasses import dataclass
from enum import Enum
import re

class ChartType(Enum):
    """Enumeration of supported chart types"""
    HISTOGRAM = "histogram"
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    BOX = "box"
    HEATMAP = "heatmap"
    CORRELATION_MATRIX = "correlation_matrix"
    VIOLIN = "violin"
    AREA = "area"
    STACKED_BAR = "stacked_bar"
    GROUPED_BAR = "grouped_bar"
    BUBBLE = "bubble"
    TREEMAP = "treemap"

@dataclass
class VisualizationSuggestion:
    """Enhanced data class for visualization suggestions"""
    chart_type: ChartType
    columns: List[str]
    title: str
    description: str
    priority: int  # 1 = highest, 5 = lowest
    rationale: str
    additional_params: Dict[str, Any] = None

@dataclass
class QueryAnalysis:
    """Data class for query analysis results"""
    intent: str  # 'comparison', 'distribution', 'trend', 'relationship', 'composition'
    data_types_needed: List[str]  # 'numeric', 'categorical', 'datetime'
    aggregation_needed: bool
    time_series: bool
    keywords: List[str]

class DataAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the DataAnalyzer with Gemini API client"""
        self.api_key = "AIzaSyDjqSU7pZa44w7FyYSVZlsAw3nBEPixdM0" or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Provide it as parameter or set GEMINI_API_KEY environment variable"
            )
        
        genai.configure(api_key=self.api_key)
        # Use Pro as primary and Flash as fallback
        self.primary_model_name = 'gemini-2.5-pro'
        self.fallback_model_name = 'gemini-2.5-flash'
        self.model_primary = genai.GenerativeModel(self.primary_model_name)
        self.model_fallback = genai.GenerativeModel(self.fallback_model_name)
        
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.2,  # Lower for more consistent analysis
            top_p=0.95,
            top_k=40,
            max_output_tokens=4096,
        )
        
        # Define visualization rules based on data characteristics
        self._init_visualization_rules()
        
    def _init_visualization_rules(self):
        """Initialize rules for automatic visualization suggestions"""
        self.viz_rules = {
            'distribution': {
                'numeric_single': [ChartType.HISTOGRAM, ChartType.BOX, ChartType.VIOLIN],
                'categorical_single': [ChartType.BAR, ChartType.PIE],
                'multiple_numeric': [ChartType.HISTOGRAM, ChartType.BOX]
            },
            'comparison': {
                'numeric_vs_categorical': [ChartType.BAR, ChartType.BOX, ChartType.VIOLIN],
                'categorical_vs_categorical': [ChartType.STACKED_BAR, ChartType.GROUPED_BAR],
                'numeric_vs_numeric': [ChartType.SCATTER, ChartType.BUBBLE]
            },
            'trend': {
                'time_series': [ChartType.LINE, ChartType.AREA],
                'sequential': [ChartType.LINE, ChartType.BAR]
            },
            'relationship': {
                'correlation': [ChartType.SCATTER, ChartType.HEATMAP, ChartType.CORRELATION_MATRIX],
                'multiple_variables': [ChartType.SCATTER, ChartType.BUBBLE, ChartType.HEATMAP]
            },
            'composition': {
                'parts_of_whole': [ChartType.PIE, ChartType.TREEMAP, ChartType.STACKED_BAR],
                'hierarchical': [ChartType.TREEMAP, ChartType.STACKED_BAR]
            }
        }
    
    def get_dataframe_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary with enhanced analysis"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
                "boolean_columns": df.select_dtypes(include=['bool']).columns.tolist()
            }
            
            # Enhanced numeric analysis
            if summary["numeric_columns"]:
                numeric_stats = df[summary["numeric_columns"]].describe(include='all')
                summary["numeric_stats"] = numeric_stats.to_dict()
                
                # Add distribution characteristics
                distribution_info = {}
                for col in summary["numeric_columns"]:
                    skewness = df[col].skew()
                    kurtosis = df[col].kurtosis()
                    distribution_info[col] = {
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'distribution_type': self._classify_distribution(skewness, kurtosis),
                        'outlier_count': self._quick_outlier_count(df[col])
                    }
                summary["distribution_info"] = distribution_info
            
            # Enhanced categorical analysis
            categorical_info = {}
            for col in summary["categorical_columns"]:
                unique_count = df[col].nunique()
                value_counts = df[col].value_counts()
                
                categorical_info[col] = {
                    "unique_count": unique_count,
                    "null_count": df[col].isnull().sum(),
                    "cardinality": self._classify_cardinality(unique_count, len(df)),
                    "top_values": value_counts.head(10).to_dict() if unique_count <= 100 else {},
                    "entropy": self._calculate_entropy(value_counts) if unique_count <= 100 else None
                }
            summary["categorical_info"] = categorical_info
            
            # DateTime analysis
            datetime_info = {}
            for col in summary["datetime_columns"]:
                datetime_info[col] = {
                    "min_date": str(df[col].min()),
                    "max_date": str(df[col].max()),
                    "null_count": df[col].isnull().sum(),
                    "frequency": self._estimate_datetime_frequency(df[col]),
                    "span_days": (df[col].max() - df[col].min()).days if df[col].notna().any() else 0
                }
            summary["datetime_info"] = datetime_info
            
            # Data relationships
            summary["potential_relationships"] = self._identify_potential_relationships(df, summary)
            
        return summary
    
    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution type based on skewness and kurtosis"""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "normal"
        elif skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        elif kurtosis > 3:
            return "heavy_tailed"
        elif kurtosis < -1:
            return "light_tailed"
        else:
            return "moderate_skew"
    
    def _quick_outlier_count(self, series: pd.Series) -> int:
        """Quick outlier count using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
        return len(outliers)
    
    def _classify_cardinality(self, unique_count: int, total_count: int) -> str:
        """Classify cardinality of categorical variables"""
        ratio = unique_count / total_count
        if ratio > 0.95:
            return "very_high"
        elif ratio > 0.5:
            return "high"
        elif ratio > 0.1:
            return "medium"
        else:
            return "low"
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of categorical distribution"""
        probabilities = value_counts / value_counts.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _estimate_datetime_frequency(self, datetime_series: pd.Series) -> str:
        """Estimate the frequency of datetime data"""
        if datetime_series.isna().all():
            return "unknown"
        
        clean_series = datetime_series.dropna().sort_values()
        if len(clean_series) < 2:
            return "insufficient_data"
        
        diffs = clean_series.diff().dropna()
        median_diff = diffs.median()
        
        if median_diff <= pd.Timedelta(hours=1):
            return "hourly_or_less"
        elif median_diff <= pd.Timedelta(days=1):
            return "daily"
        elif median_diff <= pd.Timedelta(days=7):
            return "weekly"
        elif median_diff <= pd.Timedelta(days=31):
            return "monthly"
        else:
            return "yearly_or_more"
    
    def _identify_potential_relationships(self, df: pd.DataFrame, summary: Dict) -> Dict[str, List[str]]:
        """Identify potential relationships between columns"""
        relationships = {
            "time_series_candidates": [],
            "categorical_groupings": [],
            "numeric_correlations": [],
            "hierarchical_candidates": []
        }
        
        # Time series candidates
        datetime_cols = summary["datetime_columns"]
        numeric_cols = summary["numeric_columns"]
        
        for dt_col in datetime_cols:
            for num_col in numeric_cols:
                relationships["time_series_candidates"].append([dt_col, num_col])
        
        # Categorical groupings
        categorical_cols = summary["categorical_columns"]
        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                relationships["categorical_groupings"].append([cat_col, num_col])
        
        # Potential correlations (if numeric columns exist)
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    relationships["numeric_correlations"].append([col1, col2])
        
        return relationships
    
    def analyze_query_intent(self, query: str, df_summary: Dict) -> QueryAnalysis:
        """Analyze query to understand user intent and suggest appropriate visualizations"""
        query_lower = query.lower()
        
        # Intent keywords
        intent_patterns = {
            'distribution': ['distribution', 'spread', 'histogram', 'range', 'frequency', 'how many', 'count'],
            'comparison': ['compare', 'vs', 'versus', 'difference', 'between', 'across', 'by group'],
            'trend': ['trend', 'over time', 'change', 'growth', 'decline', 'pattern', 'time series'],
            'relationship': ['relationship', 'correlation', 'association', 'connected', 'related', 'depends'],
            'composition': ['composition', 'proportion', 'percentage', 'share', 'part of', 'breakdown']
        }
        
        # Determine primary intent
        intent_scores = {}
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        primary_intent = max(intent_scores.keys(), key=intent_scores.get) if intent_scores else 'exploration'
        
        # Analyze data types needed
        data_types_needed = []
        if any(word in query_lower for word in ['number', 'amount', 'value', 'count', 'sum', 'average']):
            data_types_needed.append('numeric')
        if any(word in query_lower for word in ['category', 'type', 'group', 'region', 'class']):
            data_types_needed.append('categorical')
        if any(word in query_lower for word in ['time', 'date', 'year', 'month', 'day', 'period']):
            data_types_needed.append('datetime')
        
        # Check for aggregation needs
        aggregation_needed = any(word in query_lower for word in ['total', 'sum', 'average', 'mean', 'count', 'max', 'min'])
        
        # Check for time series
        time_series = any(word in query_lower for word in ['over time', 'trend', 'time series', 'temporal', 'chronological'])
        
        # Extract key terms
        keywords = re.findall(r'\b\w+\b', query_lower)
        
        return QueryAnalysis(
            intent=primary_intent,
            data_types_needed=data_types_needed,
            aggregation_needed=aggregation_needed,
            time_series=time_series,
            keywords=keywords
        )
    
    def suggest_multiple_visualizations(self, df_summary: Dict, query_analysis: QueryAnalysis, max_suggestions: int = 5) -> List[VisualizationSuggestion]:
        """Suggest multiple appropriate visualizations based on data and query"""
        suggestions = []
        
        # Get available columns by type
        numeric_cols = df_summary["numeric_columns"]
        categorical_cols = df_summary["categorical_columns"] 
        datetime_cols = df_summary["datetime_columns"]
        
        intent = query_analysis.intent
        
        # Generate suggestions based on intent and available data
        if intent == 'distribution':
            suggestions.extend(self._suggest_distribution_charts(numeric_cols, categorical_cols))
            
        elif intent == 'comparison':
            suggestions.extend(self._suggest_comparison_charts(numeric_cols, categorical_cols))
            
        elif intent == 'trend':
            suggestions.extend(self._suggest_trend_charts(datetime_cols, numeric_cols))
            
        elif intent == 'relationship':
            suggestions.extend(self._suggest_relationship_charts(numeric_cols, categorical_cols))
            
        elif intent == 'composition':
            suggestions.extend(self._suggest_composition_charts(categorical_cols, numeric_cols))
            
        else:  # exploration
            suggestions.extend(self._suggest_exploratory_charts(df_summary))
        
        # Sort by priority and return top suggestions
        suggestions.sort(key=lambda x: x.priority)
        return suggestions[:max_suggestions]
    
    def _suggest_distribution_charts(self, numeric_cols: List[str], categorical_cols: List[str]) -> List[VisualizationSuggestion]:
        """Suggest charts for distribution analysis"""
        suggestions = []
        
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            suggestions.append(VisualizationSuggestion(
                chart_type=ChartType.HISTOGRAM,
                columns=[col],
                title=f"Distribution of {col}",
                description=f"Shows the frequency distribution of {col} values",
                priority=1,
                rationale="Histogram is ideal for showing numeric distribution patterns"
            ))
            
            suggestions.append(VisualizationSuggestion(
                chart_type=ChartType.BOX,
                columns=[col],
                title=f"Box Plot of {col}",
                description=f"Shows quartiles, median, and outliers for {col}",
                priority=2,
                rationale="Box plot reveals statistical distribution and outliers"
            ))
        
        for col in categorical_cols[:2]:  # Top 2 categorical columns
            suggestions.append(VisualizationSuggestion(
                chart_type=ChartType.BAR,
                columns=[col],
                title=f"Distribution of {col}",
                description=f"Shows count of each category in {col}",
                priority=1,
                rationale="Bar chart is perfect for categorical frequency distribution"
            ))
        
        return suggestions
    
    def _suggest_comparison_charts(self, numeric_cols: List[str], categorical_cols: List[str]) -> List[VisualizationSuggestion]:
        """Suggest charts for comparison analysis"""
        suggestions = []
        
        # Categorical vs Numeric comparisons
        for cat_col in categorical_cols[:2]:
            for num_col in numeric_cols[:2]:
                suggestions.append(VisualizationSuggestion(
                    chart_type=ChartType.BAR,
                    columns=[cat_col, num_col],
                    title=f"{num_col} by {cat_col}",
                    description=f"Compare {num_col} values across different {cat_col} categories",
                    priority=1,
                    rationale="Bar chart excels at comparing numeric values across categories"
                ))
                
                suggestions.append(VisualizationSuggestion(
                    chart_type=ChartType.BOX,
                    columns=[cat_col, num_col],
                    title=f"{num_col} Distribution by {cat_col}",
                    description=f"Compare {num_col} distributions across {cat_col} groups",
                    priority=2,
                    rationale="Box plot shows distribution differences between groups"
                ))
        
        # Numeric vs Numeric comparisons
        for i, col1 in enumerate(numeric_cols[:3]):
            for col2 in numeric_cols[i+1:4]:
                suggestions.append(VisualizationSuggestion(
                    chart_type=ChartType.SCATTER,
                    columns=[col1, col2],
                    title=f"{col1} vs {col2}",
                    description=f"Shows relationship between {col1} and {col2}",
                    priority=2,
                    rationale="Scatter plot reveals correlations between numeric variables"
                ))
        
        return suggestions
    
    def _suggest_trend_charts(self, datetime_cols: List[str], numeric_cols: List[str]) -> List[VisualizationSuggestion]:
        """Suggest charts for trend analysis"""
        suggestions = []
        
        for dt_col in datetime_cols[:2]:
            for num_col in numeric_cols[:3]:
                suggestions.append(VisualizationSuggestion(
                    chart_type=ChartType.LINE,
                    columns=[dt_col, num_col],
                    title=f"{num_col} Over Time",
                    description=f"Shows how {num_col} changes over {dt_col}",
                    priority=1,
                    rationale="Line chart is ideal for showing trends over time"
                ))
                
                suggestions.append(VisualizationSuggestion(
                    chart_type=ChartType.AREA,
                    columns=[dt_col, num_col],
                    title=f"{num_col} Area Chart",
                    description=f"Shows {num_col} trend with filled area for emphasis",
                    priority=2,
                    rationale="Area chart emphasizes magnitude of change over time"
                ))
        
        return suggestions
    
    def _suggest_relationship_charts(self, numeric_cols: List[str], categorical_cols: List[str]) -> List[VisualizationSuggestion]:
        """Suggest charts for relationship analysis"""
        suggestions = []
        
        # Correlation matrix for multiple numeric columns
        if len(numeric_cols) >= 3:
            suggestions.append(VisualizationSuggestion(
                chart_type=ChartType.CORRELATION_MATRIX,
                columns=numeric_cols[:6],
                title="Correlation Matrix",
                description="Shows correlations between all numeric variables",
                priority=1,
                rationale="Correlation matrix reveals relationships between multiple variables"
            ))
        
        # Scatter plots for pairs
        for i, col1 in enumerate(numeric_cols[:3]):
            for col2 in numeric_cols[i+1:4]:
                suggestions.append(VisualizationSuggestion(
                    chart_type=ChartType.SCATTER,
                    columns=[col1, col2],
                    title=f"{col1} vs {col2} Relationship",
                    description=f"Scatter plot showing relationship between {col1} and {col2}",
                    priority=2,
                    rationale="Scatter plot shows direct relationship between two variables"
                ))
        
        # Bubble chart if we have 3+ numeric columns
        if len(numeric_cols) >= 3:
            suggestions.append(VisualizationSuggestion(
                chart_type=ChartType.BUBBLE,
                columns=numeric_cols[:3],
                title="Multi-variable Relationship",
                description=f"Bubble chart showing relationships between {', '.join(numeric_cols[:3])}",
                priority=3,
                rationale="Bubble chart can show relationships between three variables"
            ))
        
        return suggestions
    
    def _suggest_composition_charts(self, categorical_cols: List[str], numeric_cols: List[str]) -> List[VisualizationSuggestion]:
        """Suggest charts for composition analysis"""
        suggestions = []
        
        for cat_col in categorical_cols[:2]:
            # Pie charts for categorical composition
            suggestions.append(VisualizationSuggestion(
                chart_type=ChartType.PIE,
                columns=[cat_col],
                title=f"Composition of {cat_col}",
                description=f"Shows proportion of each category in {cat_col}",
                priority=1,
                rationale="Pie chart clearly shows parts of a whole"
            ))
            
            # Stacked bar with numeric value
            if numeric_cols:
                suggestions.append(VisualizationSuggestion(
                    chart_type=ChartType.STACKED_BAR,
                    columns=[cat_col, numeric_cols[0]],
                    title=f"{numeric_cols[0]} Composition by {cat_col}",
                    description=f"Shows how {numeric_cols[0]} is composed across {cat_col} categories",
                    priority=2,
                    rationale="Stacked bar chart shows both individual values and total composition"
                ))
        
        return suggestions
    
    def _suggest_exploratory_charts(self, df_summary: Dict) -> List[VisualizationSuggestion]:
        """Suggest charts for general data exploration"""
        suggestions = []
        
        numeric_cols = df_summary["numeric_columns"]
        categorical_cols = df_summary["categorical_columns"]
        datetime_cols = df_summary["datetime_columns"]
        
        # Overview charts
        if numeric_cols:
            suggestions.append(VisualizationSuggestion(
                chart_type=ChartType.HISTOGRAM,
                columns=[numeric_cols[0]],
                title=f"Distribution Overview - {numeric_cols[0]}",
                description=f"Explore the distribution of {numeric_cols[0]}",
                priority=2,
                rationale="Start exploration with distribution of main numeric variable"
            ))
        
        if categorical_cols:
            suggestions.append(VisualizationSuggestion(
                chart_type=ChartType.BAR,
                columns=[categorical_cols[0]],
                title=f"Category Overview - {categorical_cols[0]}",
                description=f"Explore the categories in {categorical_cols[0]}",
                priority=2,
                rationale="Understand categorical distribution for exploration"
            ))
        
        if len(numeric_cols) >= 2:
            suggestions.append(VisualizationSuggestion(
                chart_type=ChartType.SCATTER,
                columns=numeric_cols[:2],
                title="Variable Relationship Exploration",
                description=f"Explore relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                priority=3,
                rationale="Scatter plot helps discover unexpected relationships"
            ))
        
        return suggestions
    
    def analyze_query(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Enhanced query analysis with multiple visualization suggestions"""
        try:
            # Get dataframe summary
            df_summary = self.get_dataframe_summary(df)
            
            # Analyze query intent
            query_analysis = self.analyze_query_intent(query, df_summary)
            
            # Create enhanced context for AI
            context = self._create_enhanced_context(df, df_summary, query, query_analysis)
            
            # Generate AI response with fallback
            ai_response = self._generate_with_fallback(context)
            
            # Parse visualization instructions directly from AI text (structured parsing)
            parsed_viz = self._parse_visualization_instructions(ai_response, df_summary.get("columns", []))
            
            # Generate multiple visualization suggestions (heuristics fallback)
            viz_suggestions = self.suggest_multiple_visualizations(df_summary, query_analysis)
            
            # Enhanced visualization detection (more inclusive)
            viz_keywords = ['show', 'plot', 'chart', 'graph', 'visualiz', 'display', 'distribution', 'compare', 'trend']
            query_has_viz_keyword = any(k in (query or '').lower() for k in viz_keywords)
            needs_visualization = (
                bool(parsed_viz) or
                len(viz_suggestions) > 0 or
                query_has_viz_keyword or
                self._detect_visualization_need(ai_response, query)
            )
            
            # Debug logs
            try:
                print(f"[DEBUG] needs_visualization={needs_visualization} parsed={len(parsed_viz) if parsed_viz else 0} sugg={len(viz_suggestions)} keyword={query_has_viz_keyword}")
            except Exception:
                pass
            
            return {
                "response": ai_response,
                "needs_visualization": needs_visualization,
                "visualization_suggestions": [
                    {
                        "chart_type": viz.chart_type.value,
                        "columns": viz.columns,
                        "title": viz.title,
                        "description": viz.description,
                        "priority": viz.priority,
                        "rationale": viz.rationale,
                        "additional_params": viz.additional_params
                    } for viz in viz_suggestions
                ],
                "parsed_visualizations": parsed_viz,
                "query_analysis": {
                    "intent": query_analysis.intent,
                    "data_types_needed": query_analysis.data_types_needed,
                    "aggregation_needed": query_analysis.aggregation_needed,
                    "time_series": query_analysis.time_series
                },
                "dataframe_summary": df_summary,
                "query": query
            }
            
        except Exception as e:
            return {
                "response": f"Analysis error: {str(e)}",
                "needs_visualization": False,
                "visualization_suggestions": [],
                "query_analysis": None,
                "dataframe_summary": None,
                "error": str(e)
            }
    
    def _create_enhanced_context(self, df: pd.DataFrame, df_summary: Dict, query: str, query_analysis: QueryAnalysis) -> str:
        """Create enhanced context for AI analysis"""
        sample_size = min(8, len(df))
        sample_data = df.head(sample_size)
        
        context = f"""
        You are a senior data analyst. Analyze this dataset and provide comprehensive insights for the user's query.
        
        📊 DATASET OVERVIEW:
        • Shape: {df_summary['shape'][0]:,} rows × {df_summary['shape'][1]} columns
        • Memory: {df_summary['memory_usage'] / (1024**2):.2f} MB
        
        🔢 COLUMN BREAKDOWN:
        • Numeric ({len(df_summary['numeric_columns'])}): {', '.join(df_summary['numeric_columns'][:5])}{'...' if len(df_summary['numeric_columns']) > 5 else ''}
        • Categorical ({len(df_summary['categorical_columns'])}): {', '.join(df_summary['categorical_columns'][:3])}{'...' if len(df_summary['categorical_columns']) > 3 else ''}
        • DateTime ({len(df_summary['datetime_columns'])}): {', '.join(df_summary['datetime_columns'])}
        
        🎯 QUERY ANALYSIS:
        • Intent: {query_analysis.intent}
        • Data types needed: {', '.join(query_analysis.data_types_needed)}
        • Time series: {query_analysis.time_series}
        
        📋 SAMPLE DATA:
        {sample_data.to_string(max_cols=8)}
        
        📈 DATA INSIGHTS:
        {self._format_data_insights(df_summary)}
        
        ❓ USER QUERY: "{query}"
        
        🎯 INSTRUCTIONS:
        1. Provide a comprehensive analysis addressing the user's specific query
        2. Reference actual data values and patterns from the dataset
        3. If multiple visualization approaches would be helpful, mention them
        4. Suggest specific columns and chart types that would best answer the query
        5. Explain the reasoning behind your visualization recommendations
        6. Include any data quality considerations or limitations
        
        Focus on being specific, actionable, and directly addressing what the user wants to know.
        """
        
        return context
    
    def _format_data_insights(self, df_summary: Dict) -> str:
        """Format key data insights for context"""
        insights = []
        
        # Distribution insights
        if 'distribution_info' in df_summary:
            for col, info in list(df_summary['distribution_info'].items())[:3]:
                insights.append(f"• {col}: {info['distribution_type']} distribution")
        
        # Missing data insights
        missing_cols = [col for col, count in df_summary['missing_values'].items() if count > 0]
        if missing_cols:
            insights.append(f"• Missing data in: {', '.join(missing_cols[:3])}")
        
        # High cardinality warnings
        if 'categorical_info' in df_summary:
            high_card_cols = [col for col, info in df_summary['categorical_info'].items() 
                            if info.get('cardinality') == 'very_high']
            if high_card_cols:
                insights.append(f"• High cardinality: {', '.join(high_card_cols[:2])}")
        
        return '\n'.join(insights[:5]) if insights else "Data appears well-structured"
    
    def _detect_visualization_need(self, response: str, query: str) -> bool:
        """Enhanced visualization need detection"""
        viz_indicators = [
            'chart', 'graph', 'plot', 'visualiz', 'show', 'display',
            'trend', 'pattern', 'distribution', 'relationship', 'correlation',
            'compare', 'comparison', 'over time'
        ]
        
        text = (response + " " + query).lower()
        return any(indicator in text for indicator in viz_indicators)
    
    def generate_automatic_insights(self, df: pd.DataFrame) -> str:
        """Generate comprehensive automatic insights about the dataset"""
        try:
            df_summary = self.get_dataframe_summary(df)
            
            # Detect data quality issues
            quality_issues = self._assess_data_quality(df, df_summary)
            
            # Generate multiple visualization suggestions for overview
            overview_query_analysis = QueryAnalysis(
                intent='exploration',
                data_types_needed=['numeric', 'categorical'],
                aggregation_needed=False,
                time_series=False,
                keywords=['overview', 'explore']
            )
            
            viz_suggestions = self.suggest_multiple_visualizations(df_summary, overview_query_analysis)
            
            # Create comprehensive analysis prompt
            analysis_prompt = f"""
            As a senior data scientist, provide comprehensive insights about this dataset:
            
            📊 DATASET OVERVIEW:
            • {df_summary['shape'][0]:,} rows × {df_summary['shape'][1]} columns
            • Memory: {df_summary['memory_usage'] / (1024**2):.2f} MB
            • Data Types: {len(df_summary['numeric_columns'])} numeric, {len(df_summary['categorical_columns'])} categorical, {len(df_summary['datetime_columns'])} datetime
            
            🔍 DATA QUALITY ASSESSMENT:
            {quality_issues}
            
            📈 DISTRIBUTION INSIGHTS:
            {self._format_distribution_insights(df_summary)}
            
            🔗 POTENTIAL RELATIONSHIPS:
            {self._format_relationship_insights(df_summary)}
            
            📋 SAMPLE DATA:
            {df.head(3).to_string()}
            
            🎯 RECOMMENDED VISUALIZATIONS:
            {self._format_viz_recommendations(viz_suggestions)}
            
            Please provide:
            1. 🔍 KEY PATTERNS & INSIGHTS discovered in the data
            2. 📊 DATA QUALITY & COMPLETENESS assessment
            3. 🚨 NOTABLE ANOMALIES or unusual patterns
            4. 💡 RECOMMENDED ANALYSIS DIRECTIONS
            5. 🎯 POTENTIAL BUSINESS VALUE & use cases
            6. 📈 VISUALIZATION STRATEGY for exploring this data
            
            Be specific, reference actual values, and focus on actionable insights.
            """
            
            return self._generate_with_fallback(analysis_prompt)
            
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def _format_distribution_insights(self, df_summary: Dict) -> str:
        """Format distribution insights for better readability"""
        insights = []
        
        if 'distribution_info' in df_summary:
            for col, info in list(df_summary['distribution_info'].items())[:4]:
                dist_type = info['distribution_type']
                outlier_count = info['outlier_count']
                insights.append(f"• {col}: {dist_type} distribution ({outlier_count} outliers)")
        
        return '\n'.join(insights) if insights else "No numeric columns for distribution analysis"
    
    def _format_relationship_insights(self, df_summary: Dict) -> str:
        """Format potential relationships for context"""
        relationships = df_summary.get('potential_relationships', {})
        insights = []
        
        time_series = relationships.get('time_series_candidates', [])
        if time_series:
            insights.append(f"• Time series potential: {len(time_series)} datetime-numeric pairs")
        
        correlations = relationships.get('numeric_correlations', [])
        if correlations:
            insights.append(f"• Correlation analysis: {len(correlations)} numeric variable pairs")
        
        groupings = relationships.get('categorical_groupings', [])
        if groupings:
            insights.append(f"• Group analysis: {len(groupings)} categorical-numeric combinations")
        
        return '\n'.join(insights) if insights else "Limited relationship analysis possible"
    
    def _format_viz_recommendations(self, viz_suggestions: List[VisualizationSuggestion]) -> str:
        """Format visualization recommendations"""
        if not viz_suggestions:
            return "No specific visualizations recommended"
        
        formatted = []
        for i, viz in enumerate(viz_suggestions[:3], 1):
            formatted.append(f"{i}. {viz.chart_type.value.title()}: {viz.title} - {viz.rationale}")
        
        return '\n'.join(formatted)
    
    def _assess_data_quality(self, df: pd.DataFrame, df_summary: Dict) -> str:
        """Assess data quality and return formatted string"""
        issues = []
        
        # Check missing values
        missing_data = {col: count for col, count in df_summary['missing_values'].items() if count > 0}
        if missing_data:
            total_missing = sum(missing_data.values())
            issues.append(f"Missing data: {total_missing:,} values across {len(missing_data)} columns")
        
        # Check for potential duplicates
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            issues.append(f"Duplicate rows: {duplicate_rows:,} ({duplicate_rows/len(df)*100:.1f}%)")
        
        # Check for high cardinality categorical columns
        if 'categorical_info' in df_summary:
            high_cardinality = [col for col, info in df_summary['categorical_info'].items() 
                              if info.get('cardinality') in ['high', 'very_high']]
            if high_cardinality:
                issues.append(f"High cardinality categories: {', '.join(high_cardinality[:3])}")
        
        # Check for outliers in numeric columns
        if 'distribution_info' in df_summary:
            outlier_cols = [col for col, info in df_summary['distribution_info'].items() 
                           if info.get('outlier_count', 0) > len(df) * 0.05]  # > 5% outliers
            if outlier_cols:
                issues.append(f"Significant outliers in: {', '.join(outlier_cols[:3])}")
        
        return '; '.join(issues) if issues else "Data quality appears good - no major issues detected"
    
    def detect_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr') -> Optional[Dict[str, Any]]:
        """Detect outliers in a numeric column using various methods"""
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return None
        
        series = df[column].dropna()
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask = (series < lower_bound) | (series > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers_mask = z_scores > 3
            lower_bound = series.mean() - 3 * series.std()
            upper_bound = series.mean() + 3 * series.std()
            
        elif method == 'modified_zscore':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            outliers_mask = np.abs(modified_z_scores) > 3.5
            lower_bound = median - 3.5 * mad / 0.6745
            upper_bound = median + 3.5 * mad / 0.6745
            
        else:
            raise ValueError("Method must be 'iqr', 'zscore', or 'modified_zscore'")
        
        outlier_indices = series[outliers_mask].index.tolist()
        
        return {
            "method": method,
            "count": len(outlier_indices),
            "percentage": (len(outlier_indices) / len(series)) * 100,
            "outlier_indices": outlier_indices,
            "outlier_values": series[outliers_mask].tolist(),
            "bounds": {"lower": lower_bound, "upper": upper_bound},
            "statistics": {
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std()
            }
        }
    
    def analyze_correlations(self, df: pd.DataFrame, threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """Analyze correlations between numeric columns with enhanced analysis"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return None
        
        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if not pd.isna(corr_value) and abs(corr_value) >= threshold:
                    strong_correlations.append({
                        "column1": correlation_matrix.columns[i],
                        "column2": correlation_matrix.columns[j],
                        "correlation": round(corr_value, 4),
                        "strength": self._classify_correlation_strength(corr_value)
                    })
        
        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            "correlation_matrix": correlation_matrix,
            "strong_correlations": strong_correlations,
            "summary": {
                "total_pairs": len(correlation_matrix.columns) * (len(correlation_matrix.columns) - 1) // 2,
                "strong_correlations_count": len(strong_correlations),
                "threshold_used": threshold
            }
        }
    
    def _generate_with_fallback(self, content: str) -> str:
        """Try generating with Pro; on failure, fall back to Flash."""
        try:
            resp = self.model_primary.generate_content(content, generation_config=self.generation_config)
            if resp and getattr(resp, 'text', None):
                return resp.text
        except Exception as _:
            pass
        try:
            resp = self.model_fallback.generate_content(content, generation_config=self.generation_config)
            if resp and getattr(resp, 'text', None):
                return resp.text
        except Exception as e:
            return f"Analysis error: {str(e)}"
        return "Unable to generate a response at this time."
    
    def generate_plotly_code(self, df: pd.DataFrame, ai_text: str, extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ask Gemini to synthesize Plotly code based on its own recommendations text.
        Returns {'code': str} or {'error': str}.
        """
        try:
            summary = {
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "sample": df.head(5).to_dict(orient='list')
            }
            extras = extras or {}
            suggestions = extras.get('suggestions', [])
            num_charts = len(suggestions) if suggestions else 1
            
            system_prompt = f"""
You are a senior data visualization engineer. Write clean, executable Python code that builds Plotly charts.

TASK: Create {num_charts} visualization(s) based on the RECOMMENDATION TEXT below.

CONSTRAINTS:
- Use ONLY these pre-imported modules: pd (pandas), np (numpy), px (plotly.express), go (plotly.graph_objects), make_subplots
- Do NOT include any import statements
- DataFrame 'df' is already available in scope
- Define a function: def build_figure(df): that returns a LIST of figures
- Even if one chart, return a list: [fig]
- Use column names EXACTLY as shown in DATAFRAME SCHEMA
- Keep code simple and avoid complex string formatting that could break

DATAFRAME SCHEMA:
Columns: {summary['columns']}
Types: {summary['dtypes']}

RECOMMENDATION TEXT:
{ai_text}

EXAMPLE OUTPUT FORMAT:
def build_figure(df):
    fig1 = px.bar(df, x='column_a', y='column_b', title='Chart 1')
    fig2 = px.line(df, x='column_a', y='column_c', title='Chart 2')
    return [fig1, fig2]

Generate ONLY the build_figure function. No explanations, no markdown formatting.
"""
            resp_text = self._generate_with_fallback(system_prompt)
            
            # Extract code block if wrapped in backticks
            code = resp_text.strip()
            fence = '```'
            
            if fence in resp_text:
                parts = resp_text.split(fence)
                # pick the first python block if present
                for i in range(len(parts)-1):
                    block = parts[i+1].strip()
                    if block.startswith('python'):
                        # Remove 'python' prefix and get code
                        code = '\n'.join(block.split('\n')[1:])
                        break
                else:
                    # fallback to the first fenced segment
                    if len(parts) > 1:
                        code = parts[1].strip()
            
            # Remove common text artifacts
            if code.startswith('python'):
                code = '\n'.join(code.split('\n')[1:])
            
            # Strip any leading/trailing whitespace
            code = code.strip()
            
            print(f"[GEMINI CODE] Generated {len(code)} chars")
            if len(code) < 500:
                print(f"[GEMINI CODE PREVIEW]\n{code}")
            else:
                print(f"[GEMINI CODE PREVIEW]\n{code[:500]}...")
            
            return {"code": code}
        except Exception as e:
            return {"error": str(e)}
    
    def _parse_visualization_instructions(self, text: str, df_columns: List[str]) -> List[Dict[str, Any]]:
        """Parse AI-written recommendations into structured viz instructions.
        Supports patterns like:
        - Chart Type: Dual-Axis Combo Chart / Combo / Dual Axis
          Columns: X-Axis: <x>, Y1-Axis (Bars): <y1>, Y2-Axis (Line): <y2>
        - Chart Type: Stacked Area Chart / Area Chart
          Columns: X-Axis: <x>, Y-Axis: <y>, or category/value forms
        Returns a list of {chart_type, columns, params}.
        """
        if not text:
            return []
        lines = text.splitlines()
        results = []
        current = None
        # Normalize helper
        def norm_col(name: str) -> Optional[str]:
            if not name:
                return None
            n = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip()).strip("_").lower()
            # Try exact/loose match against df columns
            for c in df_columns:
                if c.lower() == n:
                    return c
            for c in df_columns:
                if n in c.lower() or c.lower() in n:
                    return c
            return name.strip()
        
        # Regex patterns
        ct_re = re.compile(r"chart\s*type\s*:\s*(.+)", re.I)
        x_re = re.compile(r"x-?axis(?:\s*\([^)]+\))?\s*:\s*`?([A-Za-z0-9_\- ]+)`?", re.I)
        y1_re = re.compile(r"y1-?axis.*:\s*`?([A-Za-z0-9_\- ]+)`?", re.I)
        y2_re = re.compile(r"y2-?axis.*:\s*`?([A-Za-z0-9_\- ]+)`?", re.I)
        y_re = re.compile(r"y-?axis.*:\s*`?([A-Za-z0-9_\- ]+)`?", re.I)
        from_col_re = re.compile(r"from\s+the\s+([A-Za-z0-9_]+)\s+column", re.I)
        series_list_re = re.compile(r"for\s+(.+?)\s+from\s+the", re.I)
        
        def flush():
            nonlocal current
            if current:
                # Map chart type synonyms
                ct = current.get('chart_type', '').lower()
                if any(k in ct for k in ["dual-axis", "dual axis", "combo"]):
                    current['chart_type'] = 'combo_dual_axis'
                elif "stacked area" in ct or ("area" in ct and "stack" in ct):
                    current['chart_type'] = 'stacked_area'
                elif "area" in ct:
                    current['chart_type'] = 'area'
                elif "line" in ct:
                    current['chart_type'] = 'line'
                elif "bar" in ct and "stack" in ct:
                    current['chart_type'] = 'stacked_bar'
                elif "bar" in ct:
                    current['chart_type'] = 'bar'
                elif "pie" in ct:
                    current['chart_type'] = 'pie'
                results.append(current)
                current = None
        
        for ln in lines:
            m = ct_re.search(ln)
            if m:
                flush()
                current = {"chart_type": m.group(1).strip(), "columns": {}, "params": {}}
                continue
            if current is None:
                continue
            mx = x_re.search(ln)
            if mx:
                current['columns']['x'] = norm_col(mx.group(1))
                continue
            my1 = y1_re.search(ln)
            if my1:
                current['columns']['y1'] = norm_col(my1.group(1))
                continue
            my2 = y2_re.search(ln)
            if my2:
                current['columns']['y2'] = norm_col(my2.group(1))
                continue
            my = y_re.search(ln)
            if my:
                current['columns'].setdefault('y', norm_col(my.group(1)))
                continue
            mf = from_col_re.search(ln)
            if mf:
                current['params']['from_column'] = norm_col(mf.group(1))
                # Try to capture series names list before 'from the'
                ms = series_list_re.search(ln)
                if ms:
                    # split by comma/and
                    parts = re.split(r",|and", ms.group(1))
                    current['params']['series_labels'] = [p.strip().strip('`').lower() for p in parts if p.strip()]
        flush()
        return results
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.9:
            return "Very Strong"
        elif abs_corr >= 0.7:
            return "Strong" 
        elif abs_corr >= 0.5:
            return "Moderate"
        elif abs_corr >= 0.3:
            return "Weak"
        else:
            return "Very Weak"
    
    def get_visualization_recommendations_summary(self, df: pd.DataFrame, query: str = None) -> Dict[str, Any]:
        """Get a comprehensive summary of visualization recommendations"""
        df_summary = self.get_dataframe_summary(df)
        
        if query:
            query_analysis = self.analyze_query_intent(query, df_summary)
        else:
            # Default exploratory analysis
            query_analysis = QueryAnalysis(
                intent='exploration',
                data_types_needed=['numeric', 'categorical', 'datetime'],
                aggregation_needed=False,
                time_series=bool(df_summary['datetime_columns']),
                keywords=['explore', 'overview']
            )
        
        # Get all possible visualization suggestions
        all_suggestions = self.suggest_multiple_visualizations(df_summary, query_analysis, max_suggestions=10)
        
        # Group by chart type for summary
        chart_type_counts = {}
        for suggestion in all_suggestions:
            chart_type = suggestion.chart_type.value
            if chart_type not in chart_type_counts:
                chart_type_counts[chart_type] = []
            chart_type_counts[chart_type].append(suggestion)
        
        return {
            "total_suggestions": len(all_suggestions),
            "chart_type_breakdown": {
                chart_type: len(suggestions) 
                for chart_type, suggestions in chart_type_counts.items()
            },
            "top_recommendations": [
                {
                    "rank": i + 1,
                    "chart_type": suggestion.chart_type.value,
                    "title": suggestion.title,
                    "columns": suggestion.columns,
                    "priority": suggestion.priority,
                    "rationale": suggestion.rationale
                }
                for i, suggestion in enumerate(all_suggestions[:5])
            ],
            "query_analysis": {
                "intent": query_analysis.intent,
                "data_types_available": {
                    "numeric": len(df_summary['numeric_columns']),
                    "categorical": len(df_summary['categorical_columns']),
                    "datetime": len(df_summary['datetime_columns'])
                },
                "recommended_analysis_paths": self._get_analysis_paths(df_summary, query_analysis)
            }
        }
    
    def _get_analysis_paths(self, df_summary: Dict, query_analysis: QueryAnalysis) -> List[str]:
        """Get recommended analysis paths based on data characteristics"""
        paths = []
        
        numeric_cols = len(df_summary['numeric_columns'])
        categorical_cols = len(df_summary['categorical_columns'])
        datetime_cols = len(df_summary['datetime_columns'])
        
        if datetime_cols > 0 and numeric_cols > 0:
            paths.append("Time series analysis - explore trends over time")
        
        if numeric_cols >= 2:
            paths.append("Correlation analysis - find relationships between variables")
        
        if categorical_cols > 0 and numeric_cols > 0:
            paths.append("Segmentation analysis - compare groups and categories")
        
        if numeric_cols > 0:
            paths.append("Distribution analysis - understand data spread and outliers")
        
        if categorical_cols > 0:
            paths.append("Composition analysis - understand categorical breakdowns")
        
        return paths

# Example usage and comprehensive testing
if __name__ == "__main__":
    try:
        # Initialize analyzer
        analyzer = DataAnalyzer()
        
        # Create more comprehensive sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'sales': np.random.normal(1000, 200, 200),
            'marketing_spend': np.random.normal(500, 100, 200),
            'customer_satisfaction': np.random.uniform(1, 5, 200),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 200),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 200),
            'date': pd.date_range('2024-01-01', periods=200, freq='D'),
            'is_premium': np.random.choice([True, False], 200),
            'revenue': np.random.gamma(2, 500, 200)
        })
        
        print("🔍 Testing Enhanced Multi-Visualization DataAnalyzer...")
        print(f"📊 Sample data shape: {sample_data.shape}")
        
        # Test 1: Basic summary
        summary = analyzer.get_dataframe_summary(sample_data)
        print(f"✅ Enhanced summary generated - {len(summary)} components")
        
        # Test 2: Multiple visualization suggestions
        viz_summary = analyzer.get_visualization_recommendations_summary(sample_data)
        print(f"✅ Visualization recommendations: {viz_summary['total_suggestions']} total suggestions")
        print(f"📈 Chart types suggested: {list(viz_summary['chart_type_breakdown'].keys())}")
        
        # Test 3: Query-specific analysis with multiple visualizations
        queries = [
            "Show me sales trends by region",
            "What's the distribution of customer satisfaction?",
            "How do sales relate to marketing spend?",
            "Compare revenue across product categories"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n🎯 Test Query {i}: {query}")
            result = analyzer.analyze_query(sample_data, query)
            viz_suggestions = result.get('visualization_suggestions', [])
            print(f"   📊 Visualizations suggested: {len(viz_suggestions)}")
            for j, viz in enumerate(viz_suggestions[:3], 1):
                print(f"   {j}. {viz['chart_type']}: {viz['title']}")
        
        # Test 4: Automatic insights
        print(f"\n🔍 Generating automatic insights...")
        insights = analyzer.generate_automatic_insights(sample_data)
        print(f"✅ Insights generated: {len(insights)} characters")
        
        print(f"\n🎉 Enhanced DataAnalyzer is ready with multiple visualization support!")
        print(f"📈 Now suggests multiple chart types per query for comprehensive analysis")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Make sure to set your GEMINI_API_KEY environment variable")