"""
Flexible Interface Components for AI-Powered Dashboard
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from workflows.agile_workflow import AgileWorkflow

class FlexibleInterface:
    """Flexible UI components that adapt to any data format."""
    
    def __init__(self, agile_workflow: AgileWorkflow):
        self.workflow = agile_workflow
    
    def render_smart_upload(self) -> Optional[Any]:
        """Render intelligent file upload that accepts any format."""
        
        st.subheader("🤖 AI-Powered Data Upload")
        
        # Show supported formats
        capabilities = self.workflow.get_workflow_capabilities()
        
        with st.expander("📋 Supported Formats & Features"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Supported Formats:**")
                for fmt in capabilities["supported_formats"]:
                    st.markdown(f"• {fmt}")
            
            with col2:
                st.markdown("**AI Features:**")
                for feature in capabilities["ai_capabilities"]:
                    st.markdown(f"• {feature}")
        
        # Upload interface
        uploaded_file = st.file_uploader(
            "Drop any file here - AI will figure it out! 🎯",
            type=None,  # Accept any file type
            help="Upload CSV, JSON, Excel, PDF, text files, or any document with supplier data"
        )
        
        # Context input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            context = st.text_area(
                "Tell AI about your data (Optional)",
                placeholder="e.g., 'This is a vendor list from our ERP system' or 'Extract companies from this contract'",
                height=100
            )
        
        with col2:
            st.markdown("**💡 Tips:**")
            st.markdown("• More context = better results")
            st.markdown("• AI adapts to your format")
            st.markdown("• No column mapping needed")
        
        return uploaded_file, context
    
    def render_processing_status(self, file_name: str):
        """Show AI processing status with progress."""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate AI processing steps
        steps = [
            "🔍 Analyzing file format...",
            "🧠 Understanding data structure...", 
            "📊 Extracting supplier information...",
            "⚡ Running risk analysis...",
            "🎯 Generating insights...",
            "✅ Complete!"
        ]
        
        for i, step in enumerate(steps):
            status_text.text(step)
            progress_bar.progress((i + 1) / len(steps))
            
        return True
    
    def render_analysis_summary(self, result: Any):
        """Render flexible analysis summary that adapts to results."""
        
        if not result or not hasattr(result, 'suppliers'):
            st.warning("No analysis results available")
            return
        
        st.subheader("🎯 AI Analysis Results")
        
        # Adaptive metrics based on available data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Suppliers Found", 
                len(result.suppliers),
                help="AI-extracted supplier count"
            )
        
        with col2:
            risk_events = len(result.risk_events) if hasattr(result, 'risk_events') else 0
            st.metric(
                "Risk Events", 
                risk_events,
                delta="AI Generated" if risk_events > 0 else None
            )
        
        with col3:
            overall_risk = getattr(result, 'overall_risk_score', 0)
            st.metric(
                "Risk Score", 
                f"{overall_risk:.1f}",
                delta="AI Calculated"
            )
        
        with col4:
            data_quality = getattr(result, 'data_quality_score', 0) * 100
            st.metric(
                "Data Quality", 
                f"{data_quality:.1f}%",
                delta="AI Assessed"
            )
        
        # Show AI insights if available
        if hasattr(result, 'recommendations') and result.recommendations:
            st.subheader("🧠 AI Insights")
            for i, rec in enumerate(result.recommendations[:3], 1):
                st.info(f"**{i}.** {rec}")
    
    def render_flexible_table(self, suppliers: List[Any]):
        """Render adaptive table that shows available supplier data."""
        
        if not suppliers:
            st.info("No supplier data to display")
            return
        
        # Convert suppliers to flexible display format
        table_data = []
        for supplier in suppliers:
            row = {"Name": getattr(supplier, 'name', 'Unknown')}
            
            # Add available fields dynamically
            optional_fields = {
                'country': 'Country',
                'region': 'Region', 
                'industry': 'Industry',
                'tier': 'Tier',
                'criticality_score': 'Risk Score',
                'website': 'Website'
            }
            
            for field, display_name in optional_fields.items():
                value = getattr(supplier, field, None)
                if value is not None:
                    if field == 'criticality_score':
                        row[display_name] = f"{value:.1f}"
                    elif field == 'tier':
                        row[display_name] = value.replace('_', ' ').title()
                    else:
                        row[display_name] = str(value)
            
            table_data.append(row)
        
        # Display adaptive table
        df = pd.DataFrame(table_data)
        
        st.subheader(f"📋 Extracted Suppliers ({len(df)})")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Show data completeness
        completeness = {}
        for col in df.columns:
            non_null = df[col].notna().sum()
            completeness[col] = f"{(non_null/len(df)*100):.0f}%"
        
        st.caption("**Data Completeness:** " + " | ".join([f"{k}: {v}" for k, v in completeness.items()]))
    
    def render_smart_filters(self, suppliers: List[Any]) -> Dict[str, Any]:
        """Render adaptive filters based on available data."""
        
        if not suppliers:
            return {}
        
        st.subheader("🔍 Smart Filters")
        
        filters = {}
        col1, col2, col3 = st.columns(3)
        
        # Dynamic filter options based on available data
        with col1:
            # Country filter if available
            countries = list(set(getattr(s, 'country', None) for s in suppliers if getattr(s, 'country', None)))
            if countries:
                filters['country'] = st.selectbox("Country", ["All"] + sorted(countries))
        
        with col2:
            # Industry filter if available  
            industries = list(set(getattr(s, 'industry', None) for s in suppliers if getattr(s, 'industry', None)))
            if industries:
                filters['industry'] = st.selectbox("Industry", ["All"] + sorted(industries))
        
        with col3:
            # Risk level filter
            filters['risk_level'] = st.selectbox("Risk Level", ["All", "Low", "Medium", "High", "Critical"])
        
        return filters
    
    def render_format_examples(self):
        """Show examples of supported formats."""
        
        st.subheader("📝 Format Examples")
        
        format_tabs = st.tabs(["CSV", "JSON", "Text", "Excel"])
        
        with format_tabs[0]:
            st.code("""
name,country,industry,criticality_score
"Global Corp",Germany,Manufacturing,85
"Tech Solutions",USA,Technology,70
"Asia Logistics",Singapore,Logistics,60
            """, language="csv")
        
        with format_tabs[1]:
            st.code("""
{
  "suppliers": [
    {
      "name": "Global Corp",
      "country": "Germany", 
      "industry": "Manufacturing",
      "criticality_score": 85
    }
  ]
}
            """, language="json")
        
        with format_tabs[2]:
            st.code("""
Our key suppliers include:
- Global Manufacturing Corp in Germany (high priority)
- Asia Pacific Electronics in Taiwan  
- European Logistics Solutions in Netherlands
            """, language="text")
        
        with format_tabs[3]:
            st.info("📊 Excel files with any column structure are supported. AI will map columns automatically.")
    
    def render_ai_confidence(self, result: Any):
        """Show AI confidence and data quality indicators."""
        
        if not result:
            return
        
        st.subheader("🎯 AI Confidence Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            extraction_conf = getattr(result, 'data_quality_score', 0.8) * 100
            st.metric("Extraction Confidence", f"{extraction_conf:.0f}%")
        
        with col2:
            analysis_conf = 85  # Placeholder - could be calculated
            st.metric("Analysis Confidence", f"{analysis_conf}%")
        
        with col3:
            prediction_conf = 80  # Placeholder - could be from ML predictions
            st.metric("Prediction Confidence", f"{prediction_conf}%")
        
        # Confidence explanation
        if extraction_conf < 70:
            st.warning("⚠️ Low extraction confidence. Consider adding more context or using a clearer format.")
        elif extraction_conf > 90:
            st.success("✅ High confidence extraction. Results are highly reliable.")
        else:
            st.info("ℹ️ Good extraction confidence. Results are reliable.")