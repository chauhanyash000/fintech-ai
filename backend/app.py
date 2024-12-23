from flask import Flask, render_template, request, send_file, jsonify
import os
import threading
from werkzeug.utils import secure_filename
import CTF
import KeyTakeaways
import AnalystsCoverage
import ECK
import EPR
import pricechanges
import researcharticles
from Combinedpdf import ReportBuilder  # New import for PDF generation
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Store the status of report generation and report builders
report_status = {}
report_builders = {}

def run_analysis_pipeline(ticker: str, company_name: str):
    """Run all analysis scripts in sequence"""
    try:
        api_key = os.getenv('OPENAI_API_KEY')  # Using environment variable instead of hardcoded key
        google_api_key = os.getenv('GOOGLE_API_KEY')
        custom_id = os.getenv('CUSTOM_ID')
        output_dir = f"analysis_pdfs_{ticker}"
        os.makedirs(output_dir, exist_ok=True)

        # Initialize ReportBuilder
        report_builders[ticker] = ReportBuilder(ticker, company_name)

        # Step 1: Run Company Technical and Fundamental Analysis
        report_status[ticker] = 'running_ctf'
        CTF.generate_stock_report(ticker, api_key, output_dir)

        # Step 2: Generate Key Takeaways
        report_status[ticker] = 'running_earnings'
        eck_pipeline = ECK.DocumentAnalysisPipeline(api_key)
        eck_pipeline.run_pipeline(ticker, company_name)
        

        # Step 3: Analyze Analyst Coverage
        report_status[ticker] = 'running_analyst_coverage'
        analyzer = AnalystsCoverage.AnalystCoverageAnalyzer()
        ratings_data = analyzer.search_analyst_ratings(ticker)
        if ratings_data:
            analyzer.generate_pdf_report(ratings_data, ticker, os.path.join(output_dir, f"{ticker}_AnalystCoverage.pdf"))

        # Step 4: Run Earnings Call Analysis
        report_status[ticker] = 'running_key_takeaways'
        KeyTakeaways.analyze_earnings(ticker, api_key)

        # Step 5: Run Press Release Analysis
        report_status[ticker] = 'running_press_release'
        epr_pipeline = EPR.DocumentAnalysisPipeline(api_key)
        epr_pipeline.run_pipeline(ticker, company_name)

        # Step 6: Generate Price Changes Analysis
        report_status[ticker] = 'running_price_changes'
        pricechanges.analyze_earnings(ticker, api_key)

        # Step 7: Generate Research Articles Analysis
        report_status[ticker] = 'running_research'
        research_assistant = researcharticles.FinancialResearchAssistant(
            openai_api_key=api_key,
            google_api_key=google_api_key,
            custom_id=custom_id
        )
        research_assistant.process_document(
            os.path.join(output_dir, f'{ticker}_earnings_analysis.pdf'),
            os.path.join(output_dir, f'{ticker}_research_articles.pdf'),
            ticker
        )

        # Final Step: Generate Professional Report using ReportBuilder
        report_status[ticker] = 'combining_reports'
        output_file = f"{ticker}_Investment_Analysis.pdf"
        success = report_builders[ticker].process_pdfs(output_dir, output_file, api_key)
        
        if success:
            report_status[ticker] = 'completed'
        else:
            report_status[ticker] = 'failed'

        # Cleanup temporary files
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                try:
                    os.remove(os.path.join(output_dir, file))
                except:
                    pass
            os.rmdir(output_dir)

    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")
        report_status[ticker] = 'failed'
    finally:
        # Cleanup the report builder instance
        if ticker in report_builders:
            del report_builders[ticker]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_report():
    ticker = request.form.get('ticker', '').upper()
    company_name = request.form.get('company_name', '')
    
    if not ticker or not company_name:
        return jsonify({'error': 'Please provide both ticker and company name'}), 400
    
    # Check if a report is already being generated for this ticker
    if ticker in report_status and report_status[ticker] not in ['completed', 'failed']:
        return jsonify({'error': 'Report generation already in progress'}), 409
    
    report_status[ticker] = 'starting'
    thread = threading.Thread(
        target=run_analysis_pipeline,
        args=(ticker, company_name)
    )
    thread.start()
    
    return jsonify({'status': 'starting', 'ticker': ticker})

@app.route('/status/<ticker>')
def check_status(ticker):
    ticker = ticker.upper()
    status = report_status.get(ticker, 'not_found')
    message = {
        'starting': 'Starting analysis pipeline...',
        'running_ctf': 'Analyzing technical and fundamental data...',
        'running_key_takeaways': 'Generating key takeaways...',
        'running_analyst_coverage': 'Analyzing analyst coverage...',
        'running_earnings': 'Processing earnings call...',
        'running_press_release': 'Analyzing press releases...',
        'running_price_changes': 'Analyzing price changes...',
        'running_research': 'Gathering research articles...',
        'combining_reports': 'Generating professional report...',
        'completed': 'Analysis complete',
        'failed': 'Analysis failed',
        'not_found': 'Analysis not found'
    }.get(status, 'Unknown status')
    
    return jsonify({'status': status, 'message': message})

@app.route('/download/<ticker>')
def download_report(ticker):
    ticker = ticker.upper()
    filename = f"{ticker}_Investment_Analysis.pdf"
    
    if os.path.exists(filename):
        return send_file(
            filename,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    else:
        return jsonify({'error': 'Report not found'}), 404

@app.route('/cleanup/<ticker>', methods=['POST'])
def cleanup_report(ticker):
    ticker = ticker.upper()
    filename = f"{ticker}_Investment_Analysis.pdf"
    
    try:
        if os.path.exists(filename):
            os.remove(filename)
        if ticker in report_status:
            del report_status[ticker]
        return jsonify({'message': 'Cleanup successful'})
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
