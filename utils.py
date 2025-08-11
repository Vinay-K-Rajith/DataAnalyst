import pandas as pd
import numpy as np

def validate_dataframe(df):
    """Validate dataframe for common issues"""
    try:
        issues = []
        
        # Check if dataframe is empty
        if df.empty:
            return {"is_valid": False, "error": "Dataset is empty"}
        
        # Check for minimum dimensions
        if df.shape[0] < 1:
            return {"is_valid": False, "error": "Dataset has no rows"}
        
        if df.shape[1] < 1:
            return {"is_valid": False, "error": "Dataset has no columns"}
        
        # Check for excessive missing data
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_percentage > 90:
            issues.append(f"Dataset has {missing_percentage:.1f}% missing values")
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            issues.append(f"Duplicate column names found: {duplicate_cols}")
        
        # Check for extremely large datasets (memory warning)
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024**2
        if memory_usage_mb > 500:  # 500MB threshold
            issues.append(f"Large dataset detected ({memory_usage_mb:.1f} MB). Processing may be slow.")
        
        return {
            "is_valid": True,
            "issues": issues,
            "memory_usage_mb": memory_usage_mb,
            "missing_percentage": missing_percentage
        }
        
    except Exception as e:
        return {"is_valid": False, "error": f"Validation error: {str(e)}"}

def optimize_dataframe_types(df):
    """Optimize dataframe memory usage by converting data types"""
    try:
        original_memory = df.memory_usage(deep=True).sum()
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # Handle object columns carefully to avoid Arrow serialization issues
            if col_type == 'object':
                # Check if column contains complex objects (like lists, dicts)
                try:
                    # Test if first non-null value is a complex object
                    first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if isinstance(first_val, (list, dict)):
                        # Convert complex objects to string representation
                        df[col] = df[col].astype(str)
                        continue
                except:
                    pass
                
                # Convert to category if low cardinality and not complex objects
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    try:
                        df[col] = df[col].astype('category')
                    except:
                        # If category conversion fails, keep as string
                        df[col] = df[col].astype(str)
            
            # Optimize numeric types
            elif col_type in ['int64', 'int32']:
                # Try to downcast integer types
                try:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                except:
                    pass  # Keep original type if optimization fails
                    
            elif col_type in ['float64', 'float32']:
                # Try to downcast float types
                try:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                except:
                    pass  # Keep original type if optimization fails
        
        optimized_memory = df.memory_usage(deep=True).sum()
        print(f"Memory usage reduced from {original_memory/1024**2:.2f} MB to {optimized_memory/1024**2:.2f} MB")
        
        return df
        
    except Exception as e:
        print(f"Error optimizing dataframe: {e}")
        return df

def safe_dataframe_display(df, max_rows=1000):
    """Safely prepare dataframe for display by handling complex objects"""
    try:
        display_df = df.copy()
        
        # Convert complex object columns to string representation
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                try:
                    # Check if any values are complex objects
                    sample_vals = display_df[col].dropna().head(5)
                    has_complex = any(isinstance(val, (list, dict, tuple)) for val in sample_vals)
                    
                    if has_complex:
                        # Convert to string representation
                        display_df[col] = display_df[col].apply(
                            lambda x: str(x) if pd.notna(x) else x
                        )
                except:
                    # If there's any issue, convert entire column to string
                    display_df[col] = display_df[col].astype(str)
        
        # Limit rows for performance
        if len(display_df) > max_rows:
            display_df = display_df.head(max_rows)
            
        return display_df
        
    except Exception as e:
        print(f"Error preparing dataframe for display: {e}")
        # Return a simple version as fallback
        return df.head(100).astype(str)

def chunk_dataframe(df, chunk_size=10000):
    """Split dataframe into chunks for processing large datasets"""
    try:
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            chunks.append(chunk)
        return chunks
    except Exception as e:
        print(f"Error chunking dataframe: {e}")
        return [df]

def detect_data_types(df):
    """Detect and suggest better data types for columns"""
    suggestions = {}
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Check if object column could be datetime
        if col_type == 'object':
            # Try to parse as datetime
            try:
                pd.to_datetime(df[col].dropna().iloc[:100])  # Test with first 100 non-null values
                suggestions[col] = 'datetime'
                continue
            except:
                pass
            
            # Check if it could be numeric
            try:
                pd.to_numeric(df[col].dropna().iloc[:100])
                suggestions[col] = 'numeric'
                continue
            except:
                pass
            
            # Check if it should be categorical
            unique_ratio = df[col].nunique() / len(df[col].dropna())
            if unique_ratio < 0.5:
                suggestions[col] = 'category'
    
    return suggestions

def handle_missing_values(df, strategy='auto'):
    """Handle missing values in the dataframe"""
    try:
        df_clean = df.copy()
        
        for col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            
            if missing_count == 0:
                continue
                
            missing_percentage = (missing_count / len(df_clean)) * 100
            
            if strategy == 'auto':
                # Automatic strategy based on data type and missing percentage
                if missing_percentage > 50:
                    # Drop columns with too many missing values
                    df_clean = df_clean.drop(columns=[col])
                    continue
                
                if df_clean[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    # Fill categorical columns with mode
                    mode_value = df_clean[col].mode()
                    if not mode_value.empty:
                        df_clean[col].fillna(mode_value.iloc[0], inplace=True)
                    else:
                        df_clean[col].fillna('Unknown', inplace=True)
            
            elif strategy == 'drop':
                df_clean = df_clean.dropna()
                break
                
        return df_clean
        
    except Exception as e:
        print(f"Error handling missing values: {e}")
        return df

def generate_data_summary_report(df):
    """Generate a comprehensive data summary report"""
    try:
        report = {
            "basic_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
                "duplicate_rows": df.duplicated().sum()
            },
            "column_info": {},
            "missing_data": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict()
        }
        
        # Analyze each column
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "non_null_count": df[col].count(),
                "null_count": df[col].isnull().sum(),
                "unique_count": df[col].nunique()
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std()
                })
            
            elif pd.api.types.is_object_dtype(df[col]):
                top_values = df[col].value_counts().head(5).to_dict()
                col_info["top_values"] = top_values
            
            report["column_info"][col] = col_info
        
        return report
        
    except Exception as e:
        return {"error": f"Failed to generate report: {str(e)}"}
