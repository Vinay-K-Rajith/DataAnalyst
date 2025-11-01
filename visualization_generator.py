import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import textwrap

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
        """Generate visualization based on query and analysis result.
        Leverages AI recommendations if provided in analysis_result["visualization_suggestions"]
        or parsed from analysis_result["parsed_visualizations"]. Always returns Plotly figures when possible.
        """
        try:
            # 1) Prefer parsed, explicit AI instructions
            parsed = analysis_result.get("parsed_visualizations") or []
            if isinstance(parsed, list) and len(parsed) > 0:
                instr = parsed[0]
                ct = (instr.get("chart_type") or "auto").lower()
                cols = instr.get("columns") or {}
                params = instr.get("params") or {}
                if ct == "combo_dual_axis":
                    return self._create_combo_dual_axis(
                        df,
                        x_col=cols.get("x"),
                        y1_col=cols.get("y1"),
                        y2_col=cols.get("y2"),
                        title=instr.get("title") or "Dual-Axis Chart"
                    )
                if ct == "stacked_area":
                    return self._create_stacked_area(
                        df,
                        x_col=cols.get("x"),
                        series_cols=None,
                        category_col=params.get("category_col"),
                        value_col=params.get("from_column"),
                        title=instr.get("title") or "Stacked Area"
                    )
                # Fallthrough to generic mapping if type is standard
                best = {"chart_type": ct, "columns": list(cols.values()), "additional_params": params, "priority": 1}
                query = f"{query} {' '.join(str(c) for c in cols.values() if c)}"
            else:
                best = None

            # 2) Otherwise, use AI-provided structured suggestions (priority order)
            if best is None:
                suggestions = analysis_result.get("visualization_suggestions") or []
                if isinstance(suggestions, list) and len(suggestions) > 0:
                    try:
                        suggestions = sorted(suggestions, key=lambda s: s.get("priority", 999))
                    except Exception:
                        pass
                    best = suggestions[0]

            # 3) Fallback to keyword detection
            viz_type = (best.get("chart_type") if best else analysis_result.get("visualization_suggestion", "auto")) or "auto"
            viz_type = "heatmap" if viz_type in ["correlation_matrix", "correlation", "heatmap"] else viz_type
            enriched_query = query
            cols = (best.get("columns") if best else None) or []
            params = (best.get("additional_params") if best else None) or {}
            if cols:
                enriched_query = f"{query} {' '.join(str(c) for c in cols)}"

            if viz_type == "auto":
                viz_type = self._determine_viz_type(enriched_query, df)
            
            # 4) Generate the appropriate visualization using Plotly
            if viz_type == "histogram":
                return self._create_histogram(df, enriched_query, columns=cols)
            elif viz_type in ("bar", "stacked_bar", "grouped_bar"):
                mode = "stack" if viz_type == "stacked_bar" else ("group" if viz_type == "grouped_bar" else None)
                return self._create_bar_chart(df, enriched_query, columns=cols, mode=mode)
            elif viz_type in ("scatter", "bubble"):
                bubble = viz_type == "bubble"
                return self._create_scatter_plot(df, enriched_query, columns=cols, bubble=bubble)
            elif viz_type in ("line", "area"):
                return self._create_line_chart(df, enriched_query, columns=cols, area=(viz_type == "area"))
            elif viz_type == "pie":
                return self._create_pie_chart(df, enriched_query, columns=cols)
            elif viz_type == "box":
                return self._create_box_plot(df, enriched_query, columns=cols)
            elif viz_type in ("heatmap", "correlation"):
                return self._create_correlation_heatmap(df)
            elif viz_type == "violin":
                return self._create_violin_plot(df, enriched_query, columns=cols)
            elif viz_type == "treemap":
                return self._create_treemap(df, enriched_query, columns=cols)
            elif viz_type in ("combo_dual_axis", "dual_axis", "combo"):
                # Attempt generic combo if passed through
                x = None; y1=None; y2=None
                if isinstance(cols, list) and len(cols) >= 3:
                    x, y1, y2 = cols[0], cols[1], cols[2]
                return self._create_combo_dual_axis(df, x, y1, y2, title="Dual-Axis Chart")
            else:
                return self._create_auto_visualization(df, enriched_query)
                
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
        elif any(word in query_lower for word in ["violin"]):
            return "violin"
        elif any(word in query_lower for word in ["treemap", "hierarch", "tree"]):
            return "treemap"
        elif any(word in query_lower for word in ["combo", "dual-axis", "dual axis", "secondary y", "two axes"]):
            return "combo_dual_axis"
        elif any(word in query_lower for word in ["stacked area"]):
            return "stacked_area"
        elif any(word in query_lower for word in ["show", "display", "visualize", "plot", "chart", "graph"]):
            # If general visualization request, use auto-detection
            return "auto"
        else:
            return "auto"
    
    
    def _create_histogram(self, df, query, columns=None):
        """Create histogram for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        # Prefer suggested column
        target_col = None
        if columns:
            for c in columns:
                if c in df.columns and c in numeric_cols:
                    target_col = c
                    break
        if not target_col:
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
    
    def _create_bar_chart(self, df, query, columns=None, mode=None):
        """Create bar chart for categorical data or top/bottom analysis"""
        # Look for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return None
        
        # Find relevant columns; prefer AI-suggested columns
        cat_col = None
        num_col = None
        color_col = None
        if columns:
            # Try to map provided columns intelligently
            for c in columns:
                if c in categorical_cols and cat_col is None:
                    cat_col = c
                elif c in numeric_cols and num_col is None:
                    num_col = c
                elif c in categorical_cols and color_col is None:
                    color_col = c
        if not cat_col:
            cat_col = self._find_relevant_column(query, categorical_cols.tolist())
        if not num_col:
            num_col = self._find_relevant_column(query, numeric_cols.tolist())
        
        if not cat_col:
            cat_col = categorical_cols[0]
        if not num_col:
            num_col = numeric_cols[0]
        
        # Decide color/stacking/group mode
        if color_col is None and len(categorical_cols) > 1:
            # pick another categorical as color for grouped/stacked bars
            for c in categorical_cols:
                if c != cat_col and df[c].nunique() <= 20:
                    color_col = c
                    break
        
        # Check if we need top/bottom analysis
        if any(word in query.lower() for word in ["top", "bottom", "highest", "lowest"]):
            # Aggregate data and get top/bottom values
            agg_data = df.groupby(cat_col, observed=True)[num_col].agg(['sum', 'mean', 'count']).reset_index()
            
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
                color=color_col,
                color_discrete_sequence=self.color_palette
            )
        else:
            # Regular/stacked/grouped bar chart
            fig = px.bar(
                df,
                x=cat_col,
                y=num_col,
                color=color_col,
                title=f"{num_col} by {cat_col}",
                color_discrete_sequence=self.color_palette
            )
            if mode == "stack":
                fig.update_layout(barmode='stack')
            elif mode == "group":
                fig.update_layout(barmode='group')
        
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    def _create_scatter_plot(self, df, query, columns=None, bubble=False):
        """Create scatter plot for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        # Prefer suggested columns
        x_col = None
        y_col = None
        size_col = None
        if columns:
            for c in columns:
                if c in df.columns and c in numeric_cols and x_col is None:
                    x_col = c
                elif c in df.columns and c in numeric_cols and y_col is None:
                    y_col = c
                elif c in df.columns and c in numeric_cols and size_col is None:
                    size_col = c
        # Try to find relevant columns from query if still missing
        if not x_col:
            x_col = self._find_relevant_column(query, numeric_cols.tolist())
        if not y_col:
            y_col = self._find_relevant_column(query, numeric_cols.tolist(), exclude=[x_col])
        
        if not x_col:
            x_col = numeric_cols[0]
        if not y_col:
            y_col = numeric_cols[1]
        
        # Look for categorical column for color coding
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        color_col = None
        if len(categorical_cols) > 0:
            color_col = categorical_cols[0]
            # Limit categories to avoid overcrowding
            if df[color_col].nunique() > 20:
                color_col = None
        
        # Decide bubble size
        if bubble:
            if size_col is None and len(numeric_cols) > 2:
                # pick a third numeric column for bubble size
                for c in numeric_cols:
                    if c not in [x_col, y_col]:
                        size_col = c
                        break
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col if bubble else None,
            title=f"{y_col} vs {x_col}",
            color_discrete_sequence=self.color_palette
        )
        
        return fig
    
    def _create_line_chart(self, df, query, columns=None, area=False):
        """Create line chart for time series or trend analysis"""
        # Look for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
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
        
        date_col = None
        value_col = None
        # Prefer suggested columns
        if columns:
            for c in columns:
                if c in df.columns and c in datetime_cols and date_col is None:
                    date_col = c
                elif c in df.columns and c in numeric_cols and value_col is None:
                    value_col = c
        if date_col is None:
            date_col = datetime_cols[0]
        if value_col is None:
            value_col = self._find_relevant_column(query, numeric_cols.tolist()) or (numeric_cols[0] if len(numeric_cols) else None)
        if value_col is None:
            return None
        
        # Sort by date
        df_sorted = df.sort_values(date_col)
        
        if area:
            fig = px.area(
                df_sorted,
                x=date_col,
                y=value_col,
                title=f"{value_col} Over Time",
                color_discrete_sequence=self.color_palette
            )
        else:
            fig = px.line(
                df_sorted,
                x=date_col,
                y=value_col,
                title=f"{value_col} Over Time",
                color_discrete_sequence=self.color_palette
            )
        
        return fig
    
    def _create_pie_chart(self, df, query, columns=None):
        """Create pie chart for categorical proportions"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return None
        
        target_col = None
        if columns:
            for c in columns:
                if c in df.columns and c in categorical_cols:
                    target_col = c
                    break
        if not target_col:
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
    
    def _create_box_plot(self, df, query, columns=None):
        """Create box plot for outlier analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) == 0:
            return None
        
        target_col = None
        cat_col = None
        if columns:
            for c in columns:
                if c in df.columns and c in numeric_cols and target_col is None:
                    target_col = c
                elif c in df.columns and c in categorical_cols and cat_col is None:
                    cat_col = c
        if not target_col:
            target_col = self._find_relevant_column(query, numeric_cols.tolist())
        if not target_col:
            target_col = numeric_cols[0]
        
        # Use categorical column for grouping if available
        if cat_col is None and len(categorical_cols) > 0:
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
    
    def _create_violin_plot(self, df, query, columns=None):
        """Create violin plot for distribution with optional grouping."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(numeric_cols) == 0:
            return None
        y_col = None
        x_col = None
        if columns:
            for c in columns:
                if c in df.columns and c in numeric_cols and y_col is None:
                    y_col = c
                elif c in df.columns and c in categorical_cols and x_col is None:
                    x_col = c
        if y_col is None:
            y_col = self._find_relevant_column(query, numeric_cols.tolist()) or numeric_cols[0]
        # Limit categories for readability
        if x_col is None and len(categorical_cols) > 0 and df[categorical_cols[0]].nunique() <= 20:
            x_col = categorical_cols[0]
        fig = px.violin(df, x=x_col, y=y_col, box=True, points='outliers', color_discrete_sequence=self.color_palette)
        fig.update_layout(title=f"Distribution of {y_col}" if x_col is None else f"{y_col} by {x_col}")
        return fig

    def _create_treemap(self, df, query, columns=None):
        """Create treemap for hierarchical composition.
        Requires 1-2 categorical columns; may use a numeric column for values.
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(categorical_cols) == 0:
            return None
        path_cols = []
        value_col = None
        if columns:
            for c in columns:
                if c in df.columns and c in categorical_cols and len(path_cols) < 2:
                    path_cols.append(c)
                elif c in df.columns and c in numeric_cols and value_col is None:
                    value_col = c
        if not path_cols:
            path_cols = [categorical_cols[0]]
            if len(categorical_cols) > 1:
                path_cols.append(categorical_cols[1])
        if value_col is None and len(numeric_cols) > 0:
            value_col = numeric_cols[0]
        # Aggregate values by path to avoid too many rectangles
        agg = df.groupby(path_cols, observed=True)[value_col].sum().reset_index() if value_col else df[path_cols].assign(value=1)
        if value_col:
            fig = px.treemap(agg, path=path_cols, values=value_col, color_discrete_sequence=self.color_palette)
        else:
            fig = px.treemap(agg, path=path_cols, values='value', color_discrete_sequence=self.color_palette)
        return fig

    def _create_combo_dual_axis(self, df, x_col, y1_col, y2_col, title="Dual-Axis Chart"):
        """Create a bar+line dual-axis combo chart using Plotly subplots."""
        if not x_col or not y1_col or not y2_col or x_col not in df.columns or y1_col not in df.columns or y2_col not in df.columns:
            return None
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=df[x_col], y=df[y1_col], name=y1_col, marker_color=self.color_palette[0]), secondary_y=False)
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y2_col], name=y2_col, mode='lines+markers', line=dict(color=self.color_palette[2], width=2)), secondary_y=True)
        fig.update_layout(title=title)
        fig.update_xaxes(title_text=x_col)
        fig.update_yaxes(title_text=y1_col, secondary_y=False)
        fig.update_yaxes(title_text=y2_col, secondary_y=True)
        return fig

    def _create_stacked_area(self, df, x_col, series_cols=None, category_col=None, value_col=None, title="Stacked Area"):
        """Create stacked area chart either from multiple series columns or category+value long-form."""
        if x_col is None or x_col not in df.columns:
            return None
        # Case 1: multiple numeric series columns provided
        if series_cols:
            valid = [c for c in series_cols if c in df.columns]
            if not valid:
                return None
            fig = px.area(df, x=x_col, y=valid, title=title, color_discrete_sequence=self.color_palette)
            return fig
        # Case 2: category + value
        if category_col and category_col in df.columns and value_col and value_col in df.columns:
            agg = df.groupby([x_col, category_col], observed=True)[value_col].sum().reset_index()
            fig = px.area(agg, x=x_col, y=value_col, color=category_col, title=title, color_discrete_sequence=self.color_palette)
            return fig
        # Heuristic fallback: try first categorical as color with first numeric as value
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols and numeric_cols:
            agg = df.groupby([x_col, categorical_cols[0]], observed=True)[numeric_cols[0]].sum().reset_index()
            fig = px.area(agg, x=x_col, y=numeric_cols[0], color=categorical_cols[0], title=title, color_discrete_sequence=self.color_palette)
            return fig
        return None

    def build_figure_from_code(self, code: str, df: pd.DataFrame):
        """Execute LLM-generated code in a restricted namespace and return Plotly Figure(s).
        Returns a single Figure or a list of Figures depending on what the code generates.
        """
        try:
            # Strip imports (we provide modules in globals)
            sanitized_lines = []
            for ln in textwrap.dedent(code).splitlines():
                lnl = ln.lower().strip()
                # Drop imports and deprecated/invalid layout props that often break (e.g., titlefont)
                if lnl.startswith('import ') or lnl.startswith('from '):
                    continue
                if 'titlefont' in lnl:
                    continue
                sanitized_lines.append(ln)
            code_clean = '\n'.join(sanitized_lines)
            
            # Validate syntax before executing
            try:
                compile(code_clean, '<syntax_check>', 'exec')
            except SyntaxError as se:
                print(f"[VIZ ERROR] Syntax error in generated code at line {se.lineno}: {se.msg}")
                print(f"[VIZ ERROR] Problematic code snippet:")
                lines = code_clean.splitlines()
                start = max(0, se.lineno - 3) if se.lineno else 0
                end = min(len(lines), (se.lineno + 2) if se.lineno else 5)
                for i in range(start, end):
                    marker = ">>>" if i == (se.lineno - 1) else "   "
                    print(f"{marker} {i+1}: {lines[i]}")
                return None

            safe_globals = {
                '__builtins__': {
                    'len': len,
                    'range': range,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'abs': abs,
                    'float': float,
                    'int': int,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'zip': zip,
                    'enumerate': enumerate,
                },
                'pd': pd,
                'np': np,
                'px': px,
                'go': go,
                'make_subplots': make_subplots,
            }
            compiled = compile(code_clean, '<llm_plotly>', 'exec')
            local_ns = {}
            exec(compiled, safe_globals, local_ns)
            build_fn = local_ns.get('build_figure') or safe_globals.get('build_figure')
            if callable(build_fn):
                result = build_fn(df)
                # Check if result is a list of figures
                if isinstance(result, list):
                    valid_figs = []
                    for item in result:
                        if isinstance(item, go.Figure) or hasattr(item, 'to_dict'):
                            valid_figs.append(item)
                    if valid_figs:
                        return valid_figs if len(valid_figs) > 1 else valid_figs[0]
                # Single figure
                elif isinstance(result, go.Figure) or hasattr(result, 'to_dict'):
                    return result
            # Fallbacks: direct 'fig' variable
            if 'fig' in local_ns and (isinstance(local_ns['fig'], go.Figure) or hasattr(local_ns['fig'], 'to_dict')):
                return local_ns['fig']
            # Try any build-like function
            for name, obj in list(local_ns.items()):
                if callable(obj) and any(k in name.lower() for k in ['build', 'make', 'create']):
                    try:
                        f = obj(df)
                        if isinstance(f, go.Figure) or hasattr(f, 'to_dict'):
                            return f
                    except Exception:
                        continue
            return None
        except Exception as e:
            print(f"Code execution error: {e}")
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
