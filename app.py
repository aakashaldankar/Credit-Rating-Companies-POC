import gradio as gr
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
with open('credit_rating_model.pkl', 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
scaler = model_package['scaler']
label_encoder = model_package['label_encoder']
feature_names = model_package['feature_names']

# Define required columns
REQUIRED_COLUMNS = [
    'Sector',
    'Current Ratio',
    'Long-term Debt / Capital',
    'Debt/Equity Ratio',
    'Gross Margin',
    'Operating Margin',
    'EBIT Margin',
    'EBITDA Margin',
    'Pre-Tax Profit Margin',
    'Net Profit Margin',
    'Asset Turnover',
    'ROE - Return On Equity',
    'Return On Tangible Equity',
    'ROA - Return On Assets',
    'ROI - Return On Investment',
    'Operating Cash Flow Per Share',
    'Free Cash Flow Per Share'
]

RATING_DESCRIPTIONS = {
    'Excellent': 'AAA, AA+, AA, AA- - Highest credit quality with minimal risk',
    'Good': 'A+, A, A- - High credit quality with low default risk',
    'Moderate': 'BBB+, BBB, BBB- - Adequate credit quality, moderate credit risk',
    'Speculative': 'BB+, BB, BB- - Below investment grade, elevated risk',
    'Highly Speculative': 'B+, B, B- - Significant credit risk, vulnerable to default',
    'High Risk': 'CCC+, CCC, CCC-, CC, C, D - High default risk or in default'
}

RATING_COLORS = {
    'Excellent': '🟢',
    'Good': '🟢',
    'Moderate': '🟡',
    'Speculative': '🟠',
    'Highly Speculative': '🔴',
    'High Risk': '🔴'
}

def validate_csv(df):
    """Validate uploaded CSV has required columns."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    return True


def preprocess_data(df):
    """Preprocess data the same way as training."""
    df_processed = df[REQUIRED_COLUMNS].copy()
    df_encoded = pd.get_dummies(df_processed, columns=['Sector'], prefix='Sector', drop_first=False)


    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0


    df_encoded = df_encoded[feature_names]
    X_scaled = scaler.transform(df_encoded)


    return X_scaled


def predict_rating(file):
    """Main prediction function."""
    try:
        if file is None:
            return "⚠️ Please upload a CSV file to get started", None


        # Read file with explicit encoding and close it immediately
        with open(file.name, 'r', encoding='utf-8') as f:
            df = pd.read_csv(f)


        validate_csv(df)


        original_df = df.copy()
        X_scaled = preprocess_data(df)


        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)


        predicted_classes = label_encoder.inverse_transform(predictions)


        # Create results
        results = []
        for idx, (pred_class, probs) in enumerate(zip(predicted_classes, probabilities)):
            company_name = original_df.loc[idx, 'Corporation'] if 'Corporation' in original_df.columns else f'Company {idx+1}'
            sector = original_df.loc[idx, 'Sector'] if 'Sector' in original_df.columns else "N/A"
            confidence = probs[predictions[idx]] * 100


            results.append({
                'Status': RATING_COLORS[pred_class],
                'Company': company_name,
                'Sector': sector,
                'Predicted Rating': pred_class,
                'Confidence (%)': f'{confidence:.1f}%',
                'Description': RATING_DESCRIPTIONS[pred_class]
            })


        results_df = pd.DataFrame(results)


        # Create enhanced text summary for first row
        first_pred_class = predicted_classes[0]
        first_confidence = probabilities[0][predictions[0]] * 100
        status_emoji = RATING_COLORS[first_pred_class]


        summary = f"""
📊 CREDIT RATING PREDICTION RESULT


{status_emoji} RATING: {first_pred_class}
📊 CONFIDENCE: {first_confidence:.2f}%
🏢 SECTOR: {original_df.loc[0, 'Sector'] if 'Sector' in original_df.columns else 'N/A'}


📋 DESCRIPTION:
{RATING_DESCRIPTIONS[first_pred_class]}


─────────────────────────────────────────────────────────
CLASS PROBABILITY DISTRIBUTION:
"""
        for cls, prob in zip(label_encoder.classes_, probabilities[0]):
            bar_length = int(prob * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            summary += f"\n  {RATING_COLORS[cls]} {cls:20s} {bar} {prob*100:5.1f}%"


        summary += f"\n\n─────────────────────────────────────────────────────────"
        summary += f"\n✅ Successfully processed {len(df)} compan{'y' if len(df)==1 else 'ies'}!"


        return summary, results_df


    except Exception as e:
        error_msg = f"❌ ERROR: {str(e)}\n\n💡 Please ensure your CSV has all required columns:\n"
        error_msg += "\n".join([f"  • {col}" for col in REQUIRED_COLUMNS[:5]])
        error_msg += "\n  ... and more (see documentation)"
        return error_msg, None



# Custom CSS for enhanced styling with orange accents and ROUNDED button
custom_css = """
.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}


/* Header styling with orange gradient */
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #ff7a00 0%, #ff5722 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}


/* Upload box visuals */
.upload-section {
    border: 2px dashed #ff7a00;
    border-radius: 10px;
    padding: 2rem;
    background-color: #fff5f0;
    position: relative;
    z-index: 0;
}


/* MAKE SURE the file dropzone and its interactive children accept pointer events */
.upload-section, 
.upload-section * {
    pointer-events: auto !important;
}


/* Gradio internal dropzone selector(s) */
#file_upload .gr-file-dropzone,
#file_upload .gr-file,
#file_upload input[type="file"] {
    pointer-events: auto !important;
    z-index: 2;
}


/* If any overlay accidentally sits above the dropzone, lower it */
#file_upload::before,
#file_upload::after {
    display: none;
}


/* ALL BUTTON STYLING - Multiple selectors to ensure it works */
#predict_btn button,
#predict_btn .gr-button,
button#predict_btn,
.gr-button.primary {
    background: #ff7a00 !important;
    border-color: #ff7a00 !important;
    color: white !important;
    box-shadow: 0 6px 18px rgba(255, 122, 0, 0.18) !important;
    border-radius: 12px !important;  /* Smooth rounded corners */
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    -webkit-border-radius: 12px !important;
    -moz-border-radius: 12px !important;
}


/* Predict button hover/focus states */
#predict_btn button:hover,
#predict_btn button:focus,
#predict_btn .gr-button:hover,
#predict_btn .gr-button:focus {
    background: #ff8f29 !important;
    border-color: #ff8f29 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(255, 122, 0, 0.28) !important;
    border-radius: 12px !important;
    -webkit-border-radius: 12px !important;
    -moz-border-radius: 12px !important;
}


/* Target all buttons in the interface */
button, .gr-button {
    border-radius: 10px !important;
    -webkit-border-radius: 10px !important;
    -moz-border-radius: 10px !important;
}


/* Download button styling with rounded edges */
button[id*="download"],
.gr-button.secondary {
    border-radius: 10px !important;
    border-color: #ff7a00 !important;
    color: #ff7a00 !important;
    transition: all 0.3s ease !important;
    -webkit-border-radius: 10px !important;
    -moz-border-radius: 10px !important;
}


button[id*="download"]:hover,
.gr-button.secondary:hover {
    background: #fff5f0 !important;
    border-color: #ff8f29 !important;
    color: #ff8f29 !important;
    border-radius: 10px !important;
}


/* Orange accent for headers */
h1, h2, h3, h4 {
    color: #ff7a00 !important;
}


/* Labels with orange accent */
.gr-block label {
    color: #ff7a00 !important;
    font-weight: 600 !important;
}


/* Info box with orange border */
.info-box {
    background-color: #fff5f0;
    border-left: 4px solid #ff7a00;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}


/* Metric card */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(255, 122, 0, 0.1);
    margin: 0.5rem 0;
    border-top: 3px solid #ff7a00;
}


/* Accordion with orange styling */
.gr-accordion {
    border-radius: 12px !important;
    border-color: #ff7a00 !important;
}


/* Table header styling */
.gr-dataframe thead {
    background-color: #fff5f0 !important;
    color: #ff7a00 !important;
}


/* Links and interactive elements */
a {
    color: #ff7a00 !important;
    transition: color 0.3s ease !important;
}


a:hover {
    color: #ff5722 !important;
}


/* Tip box styling */
div[style*="background-color: #f0f7ff"] {
    background-color: #fff5f0 !important;
    border-left: 4px solid #ff7a00 !important;
}


div[style*="background-color: #f0f7ff"] strong {
    color: #ff7a00 !important;
}


div[style*="background-color: #f0f7ff"] p {
    color: #d95f00 !important;
}


/* Group containers */
.gr-group {
    border-radius: 12px !important;
    border-color: #ffe5d9 !important;
}


/* File upload button */
.gr-file-button {
    border-radius: 8px !important;
    -webkit-border-radius: 8px !important;
    -moz-border-radius: 8px !important;
}


footer {
    text-align: center;
    padding: 2rem 0;
    color: #ff7a00;
    font-weight: 500;
}
"""

# Create the enhanced Gradio interface with reordered sections
with gr.Blocks(
    title="📊 Corporate Credit Rating Classifier",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="red",
        neutral_hue="slate",
        spacing_size="lg",
        radius_size="lg",
        text_size="md",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
    css=custom_css
) as demo:


    # Header Section (keeps app title but primary interaction appears immediately below)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div style="text-align: center; padding: 1.5rem 0;">
                    <h1 style="font-size: 2.1rem; margin-bottom: 0.3rem; background: linear-gradient(135deg, #ff7a00 0%, #ff5722 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        📊 Corporate Credit Rating Classifier
                    </h1>
                    <p style="font-size: 1rem; color: #ff7a00; margin-top: 0; font-weight: 500;">
                        Credit Risk Assessment using Financial Metrics
                    </p>
                </div>
                """
            )


    gr.Markdown("---")


    # ---------------------------------------------------------------------
    # TOP: Main Prediction Interface - Upload & Predict (moved to top)
    # ---------------------------------------------------------------------
    with gr.Row(equal_height=True):
        # Left Column - Upload
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📤 Upload & Predict")


                # NOTE: we assign an elem_id to reliably target the dropzone in CSS
                file_input = gr.File(
                    label="Select CSV File",
                    file_types=['.csv'],
                    file_count="single",
                    elem_id="file_upload"
                )


                # Give predict button an elem_id to override color reliably
                predict_btn = gr.Button(
                    "🎯 Predict Credit Rating",
                    variant="primary",
                    size="lg",
                    scale=1,
                    elem_id="predict_btn"
                )


                gr.Markdown(
                    """
                    <div style="background-color: #fff5f0; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #ff7a00;">
                        <p style="margin: 0; color: #d95f00; font-size: 0.9rem;">
                            <strong style="color: #ff7a00;">💡 Tip:</strong> Ensure your CSV is properly formatted with all required columns for accurate predictions.
                        </p>
                    </div>
                    """
                )


        # Right Column - Results
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📊 Prediction Results")
                summary_output = gr.Textbox(
                    label="Summary",
                    lines=20,
                    max_lines=25,
                    show_label=False,
                    container=True,
                    placeholder="Results will appear here after prediction..."
                )


    # Detailed Results Table (keep near top so users see outcomes immediately)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Detailed Analysis")
            results_output = gr.Dataframe(
                label="Comprehensive Results",
                wrap=True,
                interactive=False,
                column_widths=["5%", "20%", "15%", "15%", "10%", "35%"]
            )


    gr.Markdown("---")


    # ---------------------------------------------------------------------
    # Download Example CSV - moved BELOW upload/predict as requested
    # ---------------------------------------------------------------------
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div style="background: linear-gradient(135deg, #fff5f015 0%, #ff772215 100%); padding: 1.2rem; border-radius: 10px; text-align: center; border: 2px solid #ffe5d9;">
                    <h3 style="margin-top: 0; color: #ff7a00;">💡 Need a Sample File?</h3>
                    <p style="color: #d95f00; margin-bottom: 0.7rem;">Download our example CSV to see the required format</p>
                </div>
                """
            )
            download_btn = gr.DownloadButton(
                label="📥 Download Example CSV",
                value="example_companies.csv",
                variant="secondary",
                size="lg",
                scale=1
            )


    gr.Markdown("---")


    # ---------------------------------------------------------------------
    # BOTTOM: Instructions Section (How to Use) - moved to the very bottom
    # ---------------------------------------------------------------------
    with gr.Accordion("📖 How to Use This Tool", open=False):
        gr.Markdown(
            """
            ### Getting Started
            
            1. **Prepare Your Data**: Ensure your CSV file contains all required financial metrics
            2. **Upload**: Click the upload button or drag & drop your CSV file
            3. **Analyze**: Click "Predict Credit Rating" to get instant results
            4. **Review**: Examine the detailed predictions and confidence scores
            
            ### Required Columns
            
            Your CSV must include the following columns:
            
            | Category | Metrics |
            |----------|---------|
            | **Sector** | Company industry sector |
            | **Liquidity** | Current Ratio |
            | **Leverage** | Long-term Debt/Capital, Debt/Equity Ratio |
            | **Profitability** | Gross Margin, Operating Margin, EBIT Margin, EBITDA Margin, Pre-Tax Profit Margin, Net Profit Margin |
            | **Efficiency** | Asset Turnover |
            | **Returns** | ROE, Return On Tangible Equity, ROA, ROI |
            | **Cash Flow** | Operating Cash Flow Per Share, Free Cash Flow Per Share |
            
            ### Credit Rating Categories
            
            | Rating | Description | Risk Level |
            |--------|-------------|------------|
            | 🟢 **Excellent** | AAA to AA- | Minimal Risk |
            | 🟢 **Good** | A+ to A- | Low Risk |
            | 🟡 **Moderate** | BBB+ to BBB- | Moderate Risk |
            | 🟠 **Speculative** | BB+ to BB- | Elevated Risk |
            | 🔴 **Highly Speculative** | B+ to B- | High Risk |
            | 🔴 **High Risk** | CCC+ to D | Very High Risk |
            """
        )


    gr.Markdown("---")


    # Footer (small, unobtrusive)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div style="text-align: center; padding: 1rem 0; color: #ff7a00; font-weight: 500;">
                    Built with ❤️ using Gradio • Deployed on Hugging Face Spaces
                </div>
                """
            )


    # Connect button to prediction function
    predict_btn.click(
        fn=predict_rating,
        inputs=file_input,
        outputs=[summary_output, results_output]
    )


# Launch configuration
if __name__ == "__main__":
    demo.launch()
