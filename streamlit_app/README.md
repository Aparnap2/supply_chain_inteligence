# Supply Chain Intelligence Dashboard

A comprehensive Streamlit-based dashboard for supply chain risk assessment and intelligence analysis.

## Features

### 🏠 Main Dashboard Interface
- **Executive Summary**: Key metrics and KPIs at a glance
- **Real-time Status Indicators**: System health and pipeline status
- **Supplier Overview**: Interactive table with risk indicators and filtering
- **Auto-refresh**: Configurable automatic data refresh

### 📈 Interactive Visualizations
- **Risk Heatmaps**: Correlation analysis using Plotly and Seaborn
- **Geographic Risk Maps**: Global risk distribution visualization
- **Time Series Charts**: Risk trend analysis over time
- **Industry Analysis**: Risk comparison across industries
- **ML Feature Importance**: Model interpretability charts

### 🔍 Detailed Analysis Views
- **Risk Deep Dive**: Comprehensive risk event analysis with drill-down
- **Supplier Assessment**: Individual supplier risk profiles
- **ML Model Dashboard**: Model performance metrics and validation
- **AI Agent Results**: Multi-agent analysis outputs with structured display

### 📤 Export & Reporting
- **Multiple Formats**: JSON, CSV, and PDF export options
- **Custom Reports**: Configurable report generation
- **Scheduled Reports**: Automated report delivery (demo)
- **Quick Export**: One-click data export from main dashboard

## Architecture

```
streamlit_app/
├── main.py                 # Main dashboard application
├── visualizations.py       # Interactive charts and plots
├── detailed_views.py       # Detailed analysis interfaces
├── export_reports.py       # Export and reporting functionality
├── utils.py                # Utility functions and data loading
├── config.py               # Configuration settings
└── README.md               # This file
```

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt):
  - streamlit
  - plotly
  - seaborn
  - pandas
  - numpy
  - pydantic

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the dashboard:
   ```bash
   # Option 1: Use the launcher script
   python run_dashboard.py
   
   # Option 2: Direct Streamlit command
   streamlit run streamlit_app/main.py
   ```

3. Open your browser to `http://localhost:8501`

### Testing
Run the test suite to verify installation:
```bash
python test_dashboard.py
```

## Usage

### Navigation
Use the sidebar to navigate between different views:
- **Dashboard Overview**: Main dashboard with summary metrics
- **Detailed Risk Analysis**: In-depth risk assessment with drill-down
- **ML Model Dashboard**: Machine learning model performance
- **AI Agent Results**: Multi-agent analysis results
- **Export & Reports**: Data export and report generation

### Data Upload
Upload supplier data via the sidebar:
- Supported formats: CSV, Excel, JSON
- Automatic data validation and quality scoring
- Integration with existing analysis pipeline

### Export Options
- **Quick Export**: JSON export from main dashboard header
- **Custom Reports**: Configurable report generation with section selection
- **Scheduled Reports**: Automated delivery (requires backend integration)
- **Multiple Formats**: JSON, CSV, PDF support

## Configuration

### Dashboard Settings
Modify `streamlit_app/config.py` to customize:
- Risk thresholds and color schemes
- Chart appearance and themes
- Export formats and options
- Performance settings

### Environment Variables
- `STREAMLIT_ENV`: Set to "production" for production settings
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENABLE_AUTH`: Enable authentication (requires implementation)

## Data Models

The dashboard works with the following core data models:
- **Supplier**: Company information and risk metrics
- **RiskEvent**: Identified risks with severity and impact
- **MLPrediction**: Machine learning model outputs
- **AnalysisResult**: Complete analysis with all components

## Customization

### Adding New Visualizations
1. Add chart creation methods to `SupplyChainVisualizations` class
2. Update the visualization tabs in `render_interactive_visualizations()`
3. Add any required data preparation methods

### Custom Export Formats
1. Extend `ExportReportManager` class with new export methods
2. Add format options to the export interface
3. Update the download handlers

### New Analysis Views
1. Add view methods to `DetailedAnalysisViews` class
2. Update the navigation options in the sidebar
3. Add any required data processing logic

## Performance Considerations

- **Caching**: Streamlit caching is used for data loading and processing
- **Lazy Loading**: Large datasets are loaded on-demand
- **Pagination**: Tables support pagination for large result sets
- **Async Processing**: Background processing for long-running operations

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Data Validation**: Check Pydantic model requirements
3. **Memory Issues**: Reduce dataset size or enable pagination
4. **Performance**: Check caching settings and data refresh intervals

### Debug Mode
Enable debug logging by setting `LOG_LEVEL=DEBUG` in environment variables.

### Support
For issues and questions:
- Check the test suite: `python test_dashboard.py`
- Review error logs in the Streamlit interface
- Verify data model compatibility

## Future Enhancements

- Real-time data streaming integration
- Advanced authentication and authorization
- Custom dashboard themes and branding
- Mobile-responsive design improvements
- Integration with external BI tools
- Advanced scheduling and notification systems

## License

This dashboard is part of the Supply Chain Intelligence Platform project.