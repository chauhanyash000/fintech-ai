import requests
import logging
from datetime import datetime
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch
from bs4 import BeautifulSoup
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from dotenv import load_dotenv

# Previous helper functions remain the same
def sanitize_text(text: str) -> str:
    """
    Sanitize text by:
    1. Removing special characters
    2. Converting multiple spaces to single space
    3. Handling currency symbols
    4. Normalizing unicode characters
    5. Trimming whitespace
    """
    import re
    from unicodedata import normalize
    
    if not isinstance(text, str):
        return ""
        
    # Convert to string and normalize unicode characters
    text = normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII')
    
    # Handle currency and special characters
    text = text.replace('$', '')
    text = text.replace('€', '')
    text = text.replace('£', '')
    text = text.replace('%', '')
    
    # Remove any other special characters
    text = re.sub(r'[^\w\s.-]', '', text)
    
    # Convert multiple spaces to single space
    text = ' '.join(text.split())
    
    # Final trim
    return text.strip()

def parse_rating_action_target(text: str) -> tuple:
    """
    Split combined rating, action and price target text
    Returns tuple of (rating, action, price_target)
    """
    text = sanitize_text(text)
    parts = text.split()
    
    rating = ""
    action = ""
    price_target = ""
    
    if parts:
        rating = parts[0]
        
        for part in parts:
            if any(c.isdigit() for c in part):
                price_target = part
                break
                
        action_parts = []
        for part in parts[1:]:
            if not any(c.isdigit() for c in part):
                action_parts.append(part)
            else:
                break
        action = ' '.join(action_parts)
    
    return rating, action, price_target

class AnalystCoverageAnalyzer:
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.ticker = None
        self._setup_logging()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.gray
        ))
        
        self.styles.add(ParagraphStyle(
            name='Header',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=20
        ))

    def search_analyst_ratings(self, ticker: str) -> list:
        """Search for analyst ratings from StockAnalysis.com"""
        base_url = f"https://stockanalysis.com/stocks/{ticker}/ratings/"
        
        print("Search started...")
        session = requests.Session()
        retries = Retry(total=3,
                       backoff_factor=1,
                       status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = session.get(base_url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            ratings_table = soup.find('table', {'class': 'rating-table fade-out svelte-1c46ly0'})
            
            if not ratings_table:
                print("No ratings table found")
                return []
            
            ratings_data = []
            rows = ratings_table.find_all('tr')[1:10]  # Skip header row
            seen_entries = set()
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 8:
                    analyst = sanitize_text(cols[0].text)
                    rating, action, price_target = parse_rating_action_target(cols[2].text)
                    upside = sanitize_text(cols[6].text)
                    date = sanitize_text(cols[7].text)
                    
                    entry_id = f"{analyst}-{date}"
                    if entry_id in seen_entries:
                        continue
                    
                    seen_entries.add(entry_id)
                    
                    rating_data = {
                        'analyst': analyst,
                        'rating': rating,
                        'action': action,
                        'price_target': f"${price_target}" if price_target else "N/A",
                        'upside': f"{upside}%" if upside else "N/A",
                        'date': date
                    }
                    ratings_data.append(rating_data)
            
            return ratings_data
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return []

    def generate_pdf_report(self, ratings_data: list, ticker: str, output_path: str):
        """Generate a PDF report using ReportLab"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # Title
        title = Paragraph(f"Analyst Coverage Report - {ticker}", self.styles['Header'])
        story.append(title)
        
        # Add data source attribution
        attribution = Paragraph(
            "Data sourced from Yahoo Finance",
            self.styles['Disclaimer']
        )
        story.append(attribution)
        story.append(Spacer(1, 20))
        
        # Prepare table data
        headers = ['Analyst', 'Rating', 'Action', 'Price Target', 'Upside', 'Date']
        table_data = [headers]
        
        for rating in ratings_data:
            row = [
                rating['analyst'],
                rating['rating'],
                rating['action'],
                rating['price_target'],
                rating['upside'],
                rating['date']
            ]
            table_data.append(row)
        
        # Create table with styling
        table = Table(table_data, colWidths=[2*inch, inch, 1.2*inch, inch, inch, inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Add recommendations if available
        recommendations = self.generate_recommendations(ratings_data, ticker)
        if recommendations:
            story.append(Paragraph("Analysis & Recommendations", self.styles['SubHeader']))
            story.append(Paragraph(recommendations, self.styles['Normal']))
        
        # Add disclaimer
        story.append(Spacer(1, 30))
        disclaimer_text = (
            "Disclaimer: This report is for informational purposes only. "
        )
        story.append(Paragraph(disclaimer_text, self.styles['Disclaimer']))
        
        # Build the PDF
        try:
            doc.build(story)
        except Exception as e:
            self.logger.error(f"Error generating PDF: {e}")
            raise

    def generate_recommendations(self, ratings_data: list, ticker: str) -> str:
        """Generate recommendations using OpenAI"""
        try:
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')

            ratings_summary = "\n".join([
                f"- {rating['analyst']}: {rating['rating']} ({rating['action']}), "
                f"Target: {rating['price_target']}, Upside: {rating['upside']}"
                for rating in ratings_data
            ])
            
            prompt = f"""
            Analyze these recent analyst ratings for {ticker} and provide a concise, balanced recommendation paragraph:

            {ratings_summary}

            Focus on:
            1. Overall analyst sentiment
            2. Price target consensus
            3. Key actions (upgrades/downgrades)
            4. Potential upside/risks

            Format as a single, professional paragraph.
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst providing balanced, professional stock recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return "Unable to generate recommendations at this time."

def main():
    load_dotenv()
    try:
        ticker = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
        print(f"\nAnalyzing {ticker}...")
        
        analyzer = AnalystCoverageAnalyzer(debug=True)
        ratings_data = analyzer.search_analyst_ratings(ticker)
        
        if ratings_data:
            output_dir = f'analysis_pdfs_{ticker}'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{ticker}_AnalystCoverage.pdf')
            
            print("\nGenerating PDF report...")
            analyzer.generate_pdf_report(ratings_data, ticker, output_path)
            print(f"Report generated successfully: {output_path}")
        else:
            print("No analyst ratings found")
            
    except Exception as e:
        print(f"\nError generating report: {str(e)}")
        print("Please check your input and try again.")

if __name__ == "__main__":
    main()