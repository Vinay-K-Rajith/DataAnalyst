# AI Data Analyst

## Overview

This is a Streamlit-based AI-powered data analysis application that allows users to upload datasets and interact with their data through natural language queries. The system leverages Google's Gemini AI to analyze data and automatically generate appropriate visualizations based on user questions. Users can upload CSV, Excel, or JSON files and receive intelligent insights and visual representations of their data without needing to write code or understand complex data analysis techniques.

## User Preferences

Preferred communication style: Simple, everyday language.
UI Design: Enterprise-level professional interface with advanced styling, Inter font family, gradient headers, animated elements, and comprehensive data management tools.
Color Scheme: Professional blue theme (#4F7CFF primary, #6366F1 secondary) with enterprise-grade styling including hover effects, shadows, and responsive design.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Layout**: Wide layout with expandable sidebar for file upload and dataset information
- **State Management**: Session state management for chat history, current dataframe, and AI components
- **User Interface**: Chat-based interface allowing natural language queries about uploaded data

### Backend Architecture
- **Core Components**:
  - `DataAnalyzer`: Handles AI-powered data analysis using Google Gemini API
  - `VisualizationGenerator`: Creates interactive plots and charts using Plotly
  - `utils`: Provides data validation, optimization, and utility functions
- **Data Processing**: In-memory pandas dataframe processing with type optimization for memory efficiency
- **AI Integration**: Google Gemini API for natural language processing and data insights generation

### Data Storage Solutions
- **Primary Storage**: In-memory pandas dataframes stored in Streamlit session state
- **File Support**: CSV, Excel (.xlsx, .xls), and JSON file formats
- **Memory Optimization**: Automatic dataframe type optimization and chunking for large datasets
- **Data Safety**: Arrow serialization error handling for complex object columns
- **Display Optimization**: Safe dataframe display with complex object conversion to string representation
- **Validation**: Comprehensive data validation including missing value checks, duplicate detection, and memory usage monitoring

### Authentication and Authorization
- **API Authentication**: Google Gemini API key-based authentication
- **Security**: Environment variable-based API key management with fallback default key
- **Access Control**: No user authentication system - public access application

### Visualization Engine
- **Library**: Plotly for interactive visualizations with Kaleido for image export
- **Chart Types**: Comprehensive selection including bar charts, line charts, scatter plots, pie charts, histograms, box plots, and correlation heatmaps
- **Interactive Interface**: Dedicated visualization center in sidebar with step-by-step chart creation
- **Auto-Detection**: Intelligent visualization type selection based on query keywords and data characteristics
- **Professional Features**: Enterprise-grade styling, downloadable PNG exports, column selection, and color coding options
- **Customization**: Predefined blue color palette matching enterprise theme with responsive design

## External Dependencies

### AI Services
- **Google Gemini API**: Primary AI service for natural language processing and data analysis
- **API Integration**: Uses `google.genai` client library for AI-powered insights generation

### Data Processing Libraries
- **Pandas**: Core data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Plotly**: Interactive visualization library including Express, Graph Objects, and Figure Factory modules

### Web Framework
- **Streamlit**: Primary web application framework for UI and user interaction
- **Session Management**: Built-in Streamlit session state for maintaining application state

### File Processing
- **Native Support**: CSV and JSON parsing through pandas
- **Excel Support**: Excel file reading capabilities (.xlsx, .xls formats)
- **Memory Management**: Chunking and type optimization utilities for large dataset handling

### Development Dependencies
- **Python Standard Library**: JSON, IO, and OS modules for basic operations
- **Error Handling**: Comprehensive exception handling across all components