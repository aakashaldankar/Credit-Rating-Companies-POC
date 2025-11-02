# Corporate Credit Rating Classifier

A machine learning-powered web application that predicts corporate credit ratings based on financial metrics. This tool classifies companies into 6 credit rating categories (from "Excellent" to "High Risk") using a multinomial logistic regression model trained on 17 key financial indicators.

## Features

- **Intelligent Credit Assessment**: Predicts credit ratings across 6 risk categories using advanced ML algorithms
- **User-Friendly Web Interface**: Built with Gradio for easy CSV file uploads and instant predictions
- **Batch Processing**: Analyze multiple companies simultaneously from a single CSV file
- **Confidence Scoring**: Provides probability distributions across all rating categories
- **Visual Results**: Color-coded ratings with detailed breakdowns and probability bars
- **Example Templates**: Download sample CSV files to understand the required format

## Credit Rating Categories

| Rating | Description | Risk Level |
|--------|-------------|------------|
| 🟢 **Excellent** | AAA to AA- | Minimal Risk |
| 🟢 **Good** | A+ to A- | Low Risk |
| 🟡 **Moderate** | BBB+ to BBB- | Moderate Risk |
| 🟠 **Speculative** | BB+ to BB- | Elevated Risk |
| 🔴 **Highly Speculative** | B+ to B- | High Risk |
| 🔴 **High Risk** | CCC+ to D | Very High Risk |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Company Credit Rating POC"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv ccr_venv
   ```

3. **Activate the virtual environment**

   **Windows:**
   ```bash
   ccr_venv\Scripts\activate
   ```

   **Linux/Mac:**
   ```bash
   source ccr_venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

Launch the Gradio interface:

```bash
python app.py
```

The application will start a local web server (typically at `http://127.0.0.1:7860`). Open this URL in your browser to access the interface.

### Using the Application

1. **Prepare Your Data**: Ensure your CSV file contains all required columns (see below)
2. **Upload**: Click the upload button or drag & drop your CSV file
3. **Predict**: Click "🎯 Predict Credit Rating" to analyze
4. **Review**: Examine the detailed predictions, confidence scores, and risk assessments

### Required CSV Format

Your CSV file must include the following columns:

#### Categorical
- `Sector` - Company industry sector (e.g., Technology, Healthcare, Finance)

#### Financial Metrics (16 columns)
- `Current Ratio` - Liquidity measure
- `Long-term Debt / Capital` - Leverage ratio
- `Debt/Equity Ratio` - Financial leverage
- `Gross Margin` - Profitability metric
- `Operating Margin` - Operating efficiency
- `EBIT Margin` - Earnings before interest and taxes margin
- `EBITDA Margin` - Cash flow proxy
- `Pre-Tax Profit Margin` - Profitability before tax
- `Net Profit Margin` - Bottom-line profitability
- `Asset Turnover` - Asset efficiency
- `ROE - Return On Equity` - Shareholder return
- `Return On Tangible Equity` - Return on tangible assets
- `ROA - Return On Assets` - Asset profitability
- `ROI - Return On Investment` - Investment efficiency
- `Operating Cash Flow Per Share` - Cash generation
- `Free Cash Flow Per Share` - Available cash per share

#### Optional
- `Corporation` - Company name (used for display only)

**Example CSV Structure:**
```csv
Corporation,Sector,Current Ratio,Long-term Debt / Capital,Debt/Equity Ratio,...
Apple Inc.,Technology,1.5,0.35,1.2,...
Microsoft Corp.,Technology,2.1,0.28,0.9,...
```

> Download the example CSV file from the application interface to see the exact format.

## Training Your Own Model

If you want to retrain the model with your own data:

1. **Prepare training data**: Create a CSV file named `company_rating_data.csv` with:
   - All 17 required columns listed above
   - A `Rating` column with S&P-style ratings (AAA, AA+, AA, AA-, A+, etc.)
   - A `Rating Agency` column (optional)

2. **Run the training script**:
   ```bash
   python credit_rating_model.py
   ```

3. **Outputs generated**:
   - `credit_rating_model.pkl` - Trained model package
   - `model_report.txt` - Performance metrics and evaluation
   - `confusion_matrix.png` - Classification accuracy visualization
   - `roc_curves.png` - ROC curves for each rating category
   - `feature_importance.png` - Most influential financial metrics
   - `class_distribution.png` - Training data distribution
   - `precision_recall_curves.png` - Precision-recall analysis
   - `learning_curves.png` - Model learning progression

## Technology Stack

- **Machine Learning**: scikit-learn (Multinomial Logistic Regression)
- **Web Framework**: Gradio 5.36.2
- **Data Processing**: pandas 2.2.2, numpy 1.26.4
- **Visualization**: matplotlib, seaborn
- **Language**: Python 3.x

## Model Performance

The current model achieves:
- **Test Accuracy**: ~88-92% (depending on training data)
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Class Balancing**: Balanced class weights to handle rating distribution imbalance
- **Interpretability**: Feature importance analysis to understand key drivers

### Key Model Features

- **Algorithm**: Multinomial Logistic Regression with lbfgs solver
- **Features**: 17 input features (1 categorical + 16 numerical)
- **Preprocessing**: One-hot encoding for sectors, StandardScaler for normalization
- **Classes**: 6 grouped credit rating categories
- **Training**: 70/30 train-test split with stratification

## Project Structure

```
Company Credit Rating POC/
├── app.py                     # Gradio web application
├── credit_rating_model.py     # Model training pipeline
├── requirements.txt           # Python dependencies
├── credit_rating_model.pkl    # Trained model (generated)
├── example_companies.csv      # Sample data template
├── company_rating_data.csv    # Training dataset
├── ccr_venv/                  # Virtual environment (not in git)
├── README.md                  # This file
├── CLAUDE.md                  # AI assistant guidance
└── *.png                      # Generated visualizations
```

## How It Works

### Training Phase
1. Loads historical company financial data with known credit ratings
2. Groups 23 S&P-style ratings into 6 meaningful categories
3. Extracts and preprocesses 17 financial features
4. Trains a multinomial logistic regression model
5. Evaluates performance using multiple metrics (accuracy, ROC-AUC, precision-recall)
6. Saves the trained model and preprocessing pipeline

### Prediction Phase
1. User uploads a CSV file with company financial metrics
2. Application validates the file structure
3. Applies the same preprocessing as training (encoding, scaling)
4. Model predicts credit rating category with confidence scores
5. Results displayed with visual indicators and detailed analysis

## Limitations & Considerations

- **Data Quality**: Predictions are only as good as the input data quality
- **Training Data**: Model performance depends on the diversity and accuracy of training data
- **Sector Coverage**: Unknown sectors are handled but may affect accuracy
- **Temporal Factors**: Model doesn't account for macroeconomic conditions or market sentiment
- **Regulatory Use**: This is a POC tool, not suitable for regulatory compliance without validation

## Future Enhancements

Potential improvements for future versions:
- Support for time-series analysis and trend prediction
- Integration with real-time financial data APIs
- Deep learning models for improved accuracy
- Explainability features (SHAP values, LIME)
- Multi-agency rating reconciliation
- Industry-specific models
- API endpoint for programmatic access

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is a proof-of-concept for educational and research purposes.

## Acknowledgments

- Built with [Gradio](https://gradio.app/) for the web interface
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- Credit rating methodology inspired by S&P Global and Moody's rating systems

## Contact & Support

For questions, issues, or suggestions, please open an issue in the repository.

---

**Built with ❤️ for better credit risk assessment**
