import os
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from urllib.parse import urlparse, urljoin
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import hashlib
import calendar
import requests
import openai
from bs4 import BeautifulSoup
from fpdf import FPDF
from PyPDF2 import PdfReader, PdfWriter
import re
import yfinance
import pdfkit
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright
import random
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import sys
import unicodedata
from typing import List
import time
from datetime import datetime, timedelta
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from dotenv import load_dotenv

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListItem, ListFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping




class PDFCleanup:
    @staticmethod
    def remove_pages(input_path: str, output_path: str, start_remove: int, end_remove: int) -> None:
        """
        Remove specified pages from the beginning and end of a PDF file.
        
        Args:
            input_path: Path to input PDF file
            output_path: Path to save the modified PDF
            start_remove: Number of pages to remove from start (e.g., 13)
            end_remove: Number of pages to remove from end
        """
        try:
            # Create PDF reader object
            reader = PdfReader(input_path)
            writer = PdfWriter()
            
            # Calculate pages to keep
            total_pages = len(reader.pages)
            start_page = start_remove
            end_page = total_pages - end_remove
            
            # Validate page ranges
            if start_page >= end_page or start_page >= total_pages:
                raise ValueError(f"Invalid page range: start={start_page}, end={end_page}, total={total_pages}")
            
            # Add selected pages to writer
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])
            
            # Save the modified PDF
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
                
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    @staticmethod
    def process_final_pdf(pdf_path: str) -> None:
        """
        Process the final PDF by removing specified pages and replacing the original file.
        
        Args:
            pdf_path: Path to the PDF file to process
        """
        try:
            # Create temporary file path
            temp_path = pdf_path.replace('.pdf', '_temp.pdf')
            
            # Remove pages
            PDFCleanup.remove_pages(
                input_path=pdf_path,
                output_path=temp_path,
                start_remove=13,  # Remove first 13 pages
                end_remove=9      # Remove last page
            )
            
            # Replace original file with modified version
            os.replace(temp_path, pdf_path)
            
        except Exception as e:
            # Clean up temporary file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise Exception(f"Error in final PDF processing: {str(e)}")

def get_quarter_info(timestamp=None):
    """
    Determine quarter and year based on a timestamp.
    If no timestamp is provided, uses current date.
    
    Returns:
    - quarter_info: dict containing quarter number, year, and formatted string
    """
    # Use provided timestamp or current date
    if timestamp:
        if isinstance(timestamp, str):
            try:
                # Try to parse string timestamp
                date = datetime.strptime(timestamp, "%Y-%m-%d")
            except ValueError:
                try:
                    # Try alternative format
                    date = datetime.strptime(timestamp, "%Y%m%d")
                except ValueError:
                    raise ValueError("Invalid timestamp format. Use YYYY-MM-DD or YYYYMMDD")
        elif isinstance(timestamp, datetime):
            date = timestamp
        else:
            raise ValueError("Timestamp must be string or datetime object")
    else:
        date = datetime.now()

    # Get month and year
    month = date.month
    year = date.year

    # Determine quarter
    quarter = (month - 1) // 3 + 1

    # Get quarter start and end months
    quarter_start_month = 3 * (quarter - 1) + 1
    quarter_end_month = 3 * quarter

    # Get the last day of the end month
    _, last_day = calendar.monthrange(year, quarter_end_month)

    # Create quarter date ranges
    start_date = datetime(year, quarter_start_month, 1)
    end_date = datetime(year, quarter_end_month, last_day)

    quarter_info = {
        "quarter": quarter,
        "year": year,
        "quarter_str": f"{year} Q{quarter}",
        "fiscal_quarter": f"FY{year} Q{quarter} ",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "is_current": date == datetime.now().date()
    }

    return quarter_info

class TranscriptCrawler:
    def __init__(self):
        self._setup_logging()
        self._setup_session()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        self.logger.addHandler(console_handler)
        self.logger.info("Transcript Crawler initialized")

    def _setup_session(self):
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504, 429]
        )
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)



    def url_to_pdf(self, url: str, output_path: str) -> bool:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel='chrome',headless=False, args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-infobars',
                    '--start-maximized'
                ])  # Use visible browser
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
                    locale='en-US',
                    timezone_id='America/New_York',
                    geolocation={'latitude': 40.7128, 'longitude': -74.0060},
                    permissions=['geolocation'],
                    java_script_enabled=True,
                    has_touch=True,
                    color_scheme='light',
                    device_scale_factor=1,
                )

                # Add human-like behaviors
                page = context.new_page()
                page.set_default_navigation_timeout(60000)
                
                page.route('**/*', lambda route: route.continue_() if route.request.resource_type == 'document' else route.abort())
            
                # Close popups that might appear
                page.on("popup", lambda popup: popup.close())
                
                # Auto-dismiss dialogs
                page.on("dialog", lambda dialog: dialog.dismiss())
                
                # Click common popup close buttons
                async def close_popups():
                    selectors = [
                        'button[aria-label="close"]',
                        '.close-button',
                        '.popup-close',
                        'button.dismiss',
                        '[class*="close"]',
                        '[class*="modal"] button',
                        '#cookie-consent button'
                    ]
                    for selector in selectors:
                        try:
                            page.click(selector, timeout=1000)
                        except:
                            continue


                # Random mouse movements
                page.goto(url)
                for _ in range(3):
                    page.mouse.move(
                        random.randint(100, 800),
                        random.randint(100, 600)
                    )
                    time.sleep(random.uniform(0.5, 2))

                close_popups()

                # Scroll behavior
                page.evaluate("""
                    window.scrollTo({
                        top: document.body.scrollHeight,
                        behavior: 'smooth'
                    });
                """)
                time.sleep(3)


                page.pdf(path=output_path)
                browser.close()
                
                
                return True

        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            return False
               

    def process_url(self, url: str, ticker: str) -> Optional[str]:
        """Process URL and save as PDF"""
        try:
            url = url.strip()
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            # For PDF files, download directly
            if url.lower().endswith('.pdf'):
                response = self.session.get(url, stream=True, allow_redirects=True, timeout=30)
                response.raise_for_status()
                return self._download_pdf(url, response, ticker)
            
            # For webpages, convert to PDF
            timestamp = int(time.time())
            pdf_path = f'{ticker}_press_release.pdf'
            
            if self.url_to_pdf(url, pdf_path):
                return pdf_path
            return None
                
        except Exception as e:
            self.logger.error(f"Error processing URL {url}: {str(e)}")
            return None

    def _download_pdf(self, url: str, response: requests.Response, ticker: str) -> Optional[str]:
        """Download and save PDF file"""
        try:
            timestamp = int(time.time())
            filepath = f'{ticker}_press_release.pdf'
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            self.logger.info(f"Successfully downloaded PDF: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None

    def download_from_search_results(self, search_results: Dict, ticker: str) -> Optional[str]:
        """Download content from the first search result"""
        if 'items' not in search_results or not search_results['items']:
            self.logger.error("No search results found")
            return None
            
        url = search_results['items'][0]['link']
        self.logger.info(f"Processing first search result URL: {url}")
        return self.process_url(url, ticker)



class FinancialReportAnalyzer:
    def __init__(self, google_api_key: str, custom_search_id: str, debug: bool = True):
        self.api_key = google_api_key
        self.custom_search_id = custom_search_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.debug = debug
        self._setup_logging()
        

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def fetch_latest_press_release(self, ticker: str, company_name: str, quarter: str) -> Dict:
        """
        Fetch latest financial press release using Google Search API.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            year: Current year
        """
        # Modified search query to target press releases
        sites = ['reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'cnbc.com', 
                    'seekingalpha.com', 'fool.com', 'marketwatch.com']
        site_filter = ' OR '.join([f'site:{site}' for site in sites])
        
        search_query = f"{company_name} {ticker} latest quarterly financial results 10-Q or press release {quarter} {site_filter}"
        
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.custom_search_id,
                'q': search_query,
                'dateRestrict': 'm3',
                'num': 1
            }
            
            self.logger.debug(f"Making Google Search API call for query: {search_query}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            search_results = response.json()
            self._inspect_response(search_results)
            return search_results
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Google Search API call failed: {str(e)}")
            raise Exception(f"Google Search API call failed: {str(e)}")
    
    def _inspect_response(self, response: Dict):
        self.logger.info("\n=== GOOGLE SEARCH API RESPONSE INSPECTION ===")
        
        if 'items' in response and len(response['items']) > 0:
            first_result = response['items'][0]
            self.logger.info(f"\nFirst Result Title: {first_result.get('title', 'N/A')}")
            self.logger.info(f"URL: {first_result.get('link', 'N/A')}")
            self.logger.info(f"Snippet: {first_result.get('snippet', 'N/A')}")
        else:
            self.logger.info("No search results found")




class DocumentAnalysisPipeline:
    def __init__(self, openai_api_key: str, debug: bool = False):
        self.openai_api_key = openai_api_key
        self.debug = debug
        openai.api_key=openai_api_key
        self.analysis_results = {
            'financial': '',
            'business': '',
            'outlook': ''
        }
        self._request_timestamps: List[datetime] = []
        self._MAX_REQUESTS_PER_MINUTE= 50

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF file with UTF-8 encoding"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Normalize and clean the text
                        page_text = unicodedata.normalize('NFKD', page_text)
                        # Replace problematic characters
                        page_text = page_text.encode('utf-8', 'ignore').decode('utf-8')
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")

    def process_document(self, file_path: str):
        """Process document with step by step analysis"""
        try:
            # Extract text with UTF-8 encoding
            print("Extracting text from PDF...")
            self.file_content = self.extract_text_from_pdf(file_path)
            print(f"Extracted {len(self.file_content)} characters")

            # Run analyses
            print("Running financial analysis...")
            self.analysis_results['financial'] = self.analyze_content(
                self.get_financial_analysis_prompt(), 
                self.file_content
            )

            print("Running business update analysis...")
            self.analysis_results['business'] = self.analyze_content(
                self.get_business_update_prompt(), 
                self.file_content
            )

            print("Running outlook analysis...")
            self.analysis_results['outlook'] = self.analyze_content(
                self.get_outlook_prompt(), 
                self.file_content
            )

            return self.analysis_results
        
        except Exception as e:
            print(f"Processing error: {str(e)}")
            raise

    def run_pipeline(self, ticker: str, company_name: str) -> str:
        """Run the complete analysis pipeline"""
        try:
            
            # Get quarter information
            quarter_info = get_quarter_info()
            year = quarter_info['year']
            quarter= quarter_info['quarter_str']
            
            # Initialize components
            api_key = os.getenv('GOOGLE_API_KEY')
            custom_id= os.getenv('CUSTOM_ID')
            analyzer = FinancialReportAnalyzer(api_key, custom_id, debug=self.debug)
            crawler = TranscriptCrawler()
            
            # Fetch and download press release
            self.response = analyzer.fetch_latest_press_release(ticker, company_name, quarter)
            downloaded_file = crawler.download_from_search_results(self.response, ticker)
            
            if not downloaded_file:
                raise Exception("Failed to download press release")
            
            # Process document
            results = self.process_document(downloaded_file)
            
            # Generate final report
            output_dir=f'analysis_pdfs_{ticker}'
            output_path = os.path.join(output_dir, f"{ticker}_press_release_analysis.pdf")
            self.generate_report(results, output_path, self.response)
            
            return output_dir
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise

    def generate_report(self, results: Dict, output_path: str, article_url: str):
        """Generate PDF report from analysis results"""
        url = article_url['items'][0]['link'] if isinstance(article_url, dict) and 'items' in article_url and article_url['items'] else ""
        try:
            logging.info("Generating PDF report")
            pdf_report = PDFReport(article_url=url)
            pdf_report.add_analysis_to_pdf(results)
            pdf_report.output(output_path)
            logging.info(f"PDF report saved to: {output_path}")
            
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}", exc_info=True)
            raise

    def _check_rate_limit(self):
        """Check and enforce rate limits"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Remove timestamps older than 1 minute
        self._request_timestamps = [ts for ts in self._request_timestamps if ts > minute_ago]
        
        # If at rate limit, wait until we can make another request
        if len(self._request_timestamps) >= self._MAX_REQUESTS_PER_MINUTE:
            oldest_request = min(self._request_timestamps)
            sleep_time = (oldest_request + timedelta(minutes=1) - now).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Add current request timestamp
        self._request_timestamps.append(now)

    @retry(
        retry=retry_if_exception_type((openai.error.RateLimitError, openai.error.APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    def analyze_content(self, prompt: str, content: str) -> str:
        """
        Analyze content with a specific prompt, respecting rate limits
        
        Args:
            prompt: The analysis prompt template
            content: The content to analyze
        """
        try:
            # Check rate limit before making request
            self._check_rate_limit()
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in analyzing company press releases and financial statements. Make Sure to mention the Press Release Quarter and Year. Focus on key metrics, business updates, and future guidance."},
                    {"role": "user", "content": prompt.format(content=content)}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
            
        except openai.error.RateLimitError as e:
            self._check_rate_limit()  # Force wait on rate limit error
            raise e
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")
        
    # Updated prompts for press release analysis
    def get_financial_analysis_prompt(self) -> str:
        return """Analyze the financial metrics from this press release, provide nicely formatted info: [100 words]
        Make Headlines Bold.
        
        1. Revenue Analysis
        - Total revenue and growth rate
        - Segment/product revenue breakdown
        - Geographic revenue distribution
        
        2. Profitability Metrics
        - Gross margin and trends
        - Operating margin
        - Net income and EPS
        - Key profitability ratios
        
        3. Cash Flow & Balance Sheet
        - Operating cash flow
        - Capital expenditure
        - Cash position
        - Debt levels
        
        Document content:
        {content}"""
    
    def get_business_update_prompt(self) -> str:
        return """Analyze the business updates from this press release, provide nicely formatted info: [100 words]
        Make Headlines Bold.
        
        1. Key Business Highlights
        - Major achievements
        - New product launches
        - Strategic initiatives
        
        2. Operational Updates
        - Production metrics
        - Operational efficiency
        - Supply chain updates
        
        3. Market Position
        - Market share data
        - Competitive advantages
        - Industry trends
        
        Document content:
        {content}"""

    def get_outlook_prompt(self) -> str:
        return """Analyze the future outlook from this press release, provide nicely formatted info: [100 words]
        Make Headlines Bold.
        
        1. Financial Guidance
        - Revenue projections
        - Margin expectations
        - EPS guidance
        
        2. Business Outlook
        - Growth initiatives
        - Market expansion plans
        - Product roadmap
        
        3. Risk Factors
        - Market challenges
        - Operational risks
        - Economic factors
        
        Document content:
        {content}"""
    

class PDFReport:
    def __init__(self, article_url: str = ""):
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        self.elements = []
        self.article_url = article_url

    def _setup_styles(self):
        """Setup custom styles for the report"""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',  # Changed from MainTitle
            parent=self.styles['Heading1'],
            fontSize=12,
            spaceAfter=0.25*inch,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportSection',  # Changed from SectionTitle
            parent=self.styles['Heading2'],
            fontSize=11,
            spaceBefore=0.1*inch,
            spaceAfter=0.1*inch,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportText',  # Changed from BodyText
            parent=self.styles['Normal'],
            fontSize=9,
            spaceBefore=0.1*inch,
            leading=12
        ))

    def sanitize_text(self, text: str) -> str:
        """Sanitize text by removing or replacing problematic characters"""
        if text is None:
            return ""
        
        replacements = {
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201C': '"',  # Left double quotation mark
            '\u201D': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...', # Horizontal ellipsis
            '\u00A0': ' ',  # Non-breaking space
            '\u00AD': '-',  # Soft hyphen
            '\u2022': '*',  # Bullet point
            '\u00B0': 'degrees', # Degree sign
            '\u00AE': '(R)', # Registered trademark
            '\u2122': '(TM)', # Trademark
            '\u00A9': '(c)', # Copyright
            '\u20AC': 'EUR', # Euro sign
            '\u00A3': 'GBP', # Pound sign
            '\u00A5': 'YEN', # Yen sign
            '\u00BC': '1/4', # Quarter fraction
            '\u00BD': '1/2', # Half fraction
            '\u00BE': '3/4', # Three-quarters fraction
            '\n\n\n': '\n\n', # Remove excessive newlines
            '\t': ' '      # Replace tabs with spaces
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove any remaining non-ASCII characters
        text = text.encode('ascii', 'replace').decode('ascii')
        return text

    def _setup_styles(self):
        """Setup custom styles for the report"""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=0.3*inch,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#1a365d')  # Dark blue
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportSection',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceBefore=0.2*inch,
            spaceAfter=0.15*inch,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#2c5282'),  # Medium blue
            leading=16
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=0.05*inch,
            spaceAfter=0.1*inch,
            fontName='Helvetica',
            leading=14,  # Line spacing
            firstLineIndent=20  # Paragraph indent
        ))
        
        # Add style for bullet points
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=0.05*inch,
            spaceAfter=0.05*inch,
            fontName='Helvetica',
            leading=14,
            leftIndent=20,
            bulletIndent=10
        ))

    def _format_section_text(self, text: str) -> str:
        """Format text into proper paragraphs and bullet points"""
        lines = text.split('\n')
        formatted_lines = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_lines.append('')
            elif line.startswith(('•', '-', '*')):
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_lines.append(line)
            else:
                current_paragraph.append(line)
        
        if current_paragraph:
            formatted_lines.append(' '.join(current_paragraph))
        
        return '\n'.join(formatted_lines)

    def add_analysis_to_pdf(self, analysis_results: dict):
        """Create PDF report with press release analysis"""
        # Title
        self.elements.append(Paragraph('Financial Press Release Analysis', self.styles['ReportTitle']))
        self.elements.append(Spacer(1, 0.05*inch))
        
        # Financial Analysis Section
        self.elements.append(Paragraph('1. Financial Performance Analysis', self.styles['ReportSection']))
        text = self.sanitize_text(str(analysis_results.get('financial', '')))  # Convert to string
        formatted_text = self._format_section_text(text)
        
        for paragraph in formatted_text.split('\n'):
            if paragraph.strip():
                if paragraph.strip().startswith(('•', '-', '*')):
                    # Convert different bullet types to standard bullet
                    bullet_text = paragraph.strip().lstrip('•-* ')
                    self.elements.append(Paragraph(f'• {bullet_text}', self.styles['BulletPoint']))
                else:
                    self.elements.append(Paragraph(paragraph, self.styles['ReportText']))
        
        self.elements.append(Spacer(1, 0.05*inch))
        
        # Business Updates Section
        self.elements.append(Paragraph('2. Business Updates & Operations', self.styles['ReportSection']))
        text = self.sanitize_text(str(analysis_results.get('business', '')))  # Convert to string
        formatted_text = self._format_section_text(text)
        
        for paragraph in formatted_text.split('\n'):
            if paragraph.strip():
                if paragraph.strip().startswith(('•', '-', '*')):
                    bullet_text = paragraph.strip().lstrip('•-* ')
                    self.elements.append(Paragraph(f'• {bullet_text}', self.styles['BulletPoint']))
                else:
                    self.elements.append(Paragraph(paragraph, self.styles['ReportText']))
        
        self.elements.append(Spacer(1, 0.05*inch))
        
        # Future Outlook Section
        self.elements.append(Paragraph('3. Future Outlook & Guidance', self.styles['ReportSection']))
        text = self.sanitize_text(str(analysis_results.get('outlook', '')))  # Convert to string
        formatted_text = self._format_section_text(text)
        
        for paragraph in formatted_text.split('\n'):
            if paragraph.strip():
                if paragraph.strip().startswith(('•', '-', '*')):
                    bullet_text = paragraph.strip().lstrip('•-* ')
                    self.elements.append(Paragraph(f'• {bullet_text}', self.styles['BulletPoint']))
                else:
                    self.elements.append(Paragraph(paragraph, self.styles['ReportText']))


    def output(self, output_path: str):
        """Save the PDF to the specified path"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        def add_footer(canvas, doc):
            """Add page number and article URL to footer"""
            canvas.saveState()
            canvas.setFont('Helvetica', 8)
            
           
            # URL on left (truncated if too long)
            max_url_width = doc.pagesize[0] - 144  # Leave space for margins
            canvas.setFont('Helvetica', 8)
            url = self.article_url
            if canvas.stringWidth(url, 'Helvetica', 8) > max_url_width:
                while canvas.stringWidth(url + '...', 'Helvetica', 8) > max_url_width and len(url) > 0:
                    url = url[:-1]
                url = url + '...'
            
            canvas.drawString(72, 36, url)
            canvas.restoreState()

        # Build the document with footer on all pages
        doc.build(self.elements, onFirstPage=add_footer, onLaterPages=add_footer)


def main():

    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    ticker = input("Enter ticker symbol: ").upper()
    company_name = input("Enter company name: ")
    
    try:
        pipeline = DocumentAnalysisPipeline(openai_api_key, debug=True)
        output_dir = pipeline.run_pipeline(ticker, company_name)
        print(f"\nAnalysis complete. Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        logging.error("Pipeline execution failed", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()