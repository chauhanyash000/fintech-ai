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
import dotenv 


from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfgen import canvas



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

    def fetch_latest_transcript(self, ticker: str, company_name: str, quarter: str) -> Dict:
        """
        Fetch latest earnings call transcript using Google Search API.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            current_qtr: Current quarter (e.g. 'Q4 2024')
        
        Returns:
            Dict containing search results
        """
        search_query = f"{company_name} {ticker} latest earnings call transcript {quarter} site:Fool.com"
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.custom_search_id,
                'q': search_query,
                'dateRestrict':'m3',
                'num': 1  # Only need the first result
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
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse API response as JSON: {str(e)}")
            raise Exception("Failed to parse API response as JSON")

    def _inspect_response(self, response: Dict):
        self.logger.info("\n=== GOOGLE SEARCH API RESPONSE INSPECTION ===")
        
        if 'items' in response and len(response['items']) > 0:
            first_result = response['items'][0]
            self.logger.info(f"\nFirst Result Title: {first_result.get('title', 'N/A')}")
            self.logger.info(f"URL: {first_result.get('link', 'N/A')}")
            self.logger.info(f"Snippet: {first_result.get('snippet', 'N/A')}")
        else:
            self.logger.info("No search results found")

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
            pdf_path = f'{ticker}_earnings_call.pdf'
            
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
            filepath = f'{ticker}_earnings_call.pdf'
            
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


class PDFReport:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        self.transcript_url = None

    def _create_custom_styles(self):
        """Create custom paragraph styles with proper font handling"""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,
            textColor=colors.HexColor('#1B365D')
        ))
        
        self.styles.add(ParagraphStyle(
            name='MainSection',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=16,
            spaceBefore=16,
            textColor=colors.HexColor('#2F4F4F'),
            borderWidth=1,
            borderColor=colors.HexColor('#E8E8E8'),
            borderPadding=8,
            leading=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceBefore=12,
            spaceAfter=8,
            textColor=colors.HexColor('#1B365D'),
            leading=16
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontSize=10.5,
            spaceBefore=6,
            spaceAfter=10,
            leading=16,
            textColor=colors.HexColor('#333333')
        ))
        
        self.styles.add(ParagraphStyle(
            name='QAQuestion',
            parent=self.styles['Heading3'],
            fontSize=11,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor('#2F4F4F'),
            leading=16,
            leftIndent=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='QAAnswer',
            parent=self.styles['Normal'],
            fontSize=10.5,
            spaceBefore=4,
            spaceAfter=12,
            leading=16,
            leftIndent=20,
            textColor=colors.HexColor('#333333')
        ))

    def create_header_footer(self, canvas, doc):
        """Add header and footer to each page with transcript URL"""
        canvas.saveState()
        
        # Footer with page number and URL
        footer_y = doc.bottomMargin - 20
        
        
        # URL
        if self.transcript_url:
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.HexColor('#666666'))
            url_text = f'Source: {self.transcript_url}'
            url_width = canvas.stringWidth(url_text, 'Helvetica', 8)
            url_x = (doc.width - url_width) / 2
            canvas.drawString(url_x, footer_y - 15, url_text)
        
        canvas.restoreState()

    def sanitize_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '…': '...',
            '\u2022': '-',
            '\u00a0': ' ',
            '•': '-'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    def add_analysis_to_pdf(self, analysis_results: dict, output_path: str, transcript_url: str):
        """Create PDF report with enhanced formatting"""
        self.transcript_url = transcript_url
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        story = []
        
        # Process MECE analysis
        mece_sections = self.sanitize_text(analysis_results['mece']).split('\n')
        
        for section in mece_sections:
            if section.strip():
                if any(category in section.lower() for category in 
                      ['financial performance', 'strategic initiatives', 'market position', 
                       'future outlook', 'risk factors']):
                    story.append(Spacer(1, 10))
                    story.append(Paragraph(section, self.styles['SubSection']))
                else:
                    if not section.startswith('•'):
                        section = '• ' + section
                    story.append(Paragraph(section, self.styles['ReportBody']))

        story.append(PageBreak())

        # Pages 3-4: Q&A Analysis
        story.append(Paragraph('Q&A Discussion', self.styles['MainSection']))
        story.append(Spacer(1, 10))
        
        qa_content = self.sanitize_text(analysis_results['qa'])
        qa_sections = qa_content.split('\n')
        
        for section in qa_sections:
            if section.strip():
                if section.startswith(('Analyst:', 'Q:', 'Question:')):
                    story.append(Spacer(1, 8))
                    section = '→ ' + section.replace('Question:', 'Q:')
                    story.append(Paragraph(section, self.styles['QAQuestion']))
                elif section.startswith(('Executive:', 'A:', 'Answer:')):
                    section = '  ' + section.replace('Answer:', 'A:')
                    story.append(Paragraph(section, self.styles['QAAnswer']))
                    story.append(Spacer(1, 8))
                else:
                    story.append(Paragraph(section, self.styles['ReportBody']))

        doc.build(story, onFirstPage=self.create_header_footer, onLaterPages=self.create_header_footer)

class DocumentAnalysisPipeline:
    def __init__(self, openai_api_key: str, debug: bool = False):
        """Initialize the document analysis pipeline"""
        self.openai_api_key = openai_api_key
        self.debug = debug
        openai.api_key=openai_api_key
        self.setup_logging()
        # Initialize analysis_results dictionary
        self.analysis_results = {
            'mece': '',
            'topic': '',
            'qa': ''
        }
        self.file_content = ''
        self.summarized_content = ''


    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF file with robust Unicode handling"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Step 1: Normalize Unicode characters
                        page_text = unicodedata.normalize('NFKD', page_text)
                        
                        # Step 2: Handle special characters and encoding
                        # Replace problematic characters with their ASCII equivalents
                        replacements = {
                            '\u2019': "'",  # Smart single quote
                            '\u2018': "'",  # Smart single quote
                            '\u201c': '"',  # Smart double quote
                            '\u201d': '"',  # Smart double quote
                            '\u2013': '-',  # En dash
                            '\u2014': '--', # Em dash
                            '\u2026': '...', # Ellipsis
                            '\u00a0': ' ',  # Non-breaking space
                            '\u00ae': '(R)', # Registered trademark
                            '\u2122': '(TM)', # Trademark
                            '\u00a9': '(c)', # Copyright
                            '\u20ac': 'EUR', # Euro sign
                            '\u00a3': 'GBP', # Pound sign
                            '\u00a5': 'JPY', # Yen sign
                            '\u00b0': ' degrees', # Degree sign
                        }
                        
                        for old, new in replacements.items():
                            page_text = page_text.replace(old, new)
                        
                        # Step 3: Remove any remaining non-ASCII characters
                        page_text = ''.join(char for char in page_text if ord(char) < 128)
                        
                        # Step 4: Clean up whitespace
                        page_text = ' '.join(page_text.split())
                        
                        text += page_text + "\n"
                        
                # Final cleanup
                text = text.strip()
                
                # Verify the text is properly encoded
                text.encode('ascii', errors='ignore').decode('ascii')
                
                return text

        except Exception as e:
            logging.error(f"PDF extraction failed: {str(e)}")
            raise Exception(f"PDF extraction failed: {str(e)}")


        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")

    def summarize_content(self, text: str) -> str:
        """Summarize the content to 20% of original length"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial document summarizer. Create a concise summary focusing on key financial metrics, business updates, and important insights. Maintain all important numbers and data points."},
                    {"role": "user", "content": f"Summarize the following earnings call transcript to approximately 20% of its length, maintaining key financial data and insights:\n\n{text}"}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")

    def analyze_content(self, prompt: str, content: str) -> str:
        """Analyze content with a specific prompt"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in earnings call analysis. Provide detailed insights focusing on key metrics and business impact."},
                    {"role": "user", "content": prompt.format(content=content)}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")

    def process_document(self, file_path: str):
        """Process document with step by step analysis"""
        try:
            # Step 1: Extract text with UTF-8 encoding
            print("Extracting text from PDF...")
            self.file_content = self.extract_text_from_pdf(file_path)
            print(f"Extracted {len(self.file_content)} characters")

            # Step 2: Summarize content
            print("Summarizing content...")
            self.summarized_content = self.summarize_content(self.file_content)
            print("Content summarized to 20%")

            # Step 3: Run analyses using summarized content
            print("Running MECE analysis...")
            self.analysis_results['mece'] = self.analyze_content(
                self.get_mece_summary_prompt(), 
                self.summarized_content
            )

            print("Running topic analysis...")
            self.analysis_results['topic'] = self.analyze_content(
                self.get_topic_analysis_prompt(), 
                self.summarized_content
            )

            print("Running Q&A analysis...")
            # Use original content for Q&A to maintain full context
            self.analysis_results['qa'] = self.analyze_content(
                self.get_qa_extraction_prompt(), 
                self.file_content
            )

            return self.analysis_results
        
        except FileNotFoundError as e:
            print(f"File error: {str(e)}")
            raise
    

    # Your existing prompt methods remain the same
    def get_mece_summary_prompt(self) -> str:
        return """Analyze the document and provide a clear, comprehensive summary in these categories:
        1. Financial Performance & Metrics
        2. Strategic Initiatives & Business Updates 
        3. Market Position & Competitive Analysis
        4. Future Outlook & Guidance
        5. Risk Factors & Challenges

        For each category:
        - Provide 2-3 key points
        - Include specific numbers and data where relevant

        End with a brief "Overall Business Outlook" paragraph synthesizing the key themes.
        provide nicely formatted info that covers 1 page.

        Document content:
        {content}"""
    
    def get_topic_analysis_prompt(self) -> str:
        return """Analyze the Q&A section of this document:
        
        1. Identify top 5 most discussed topics across all questions
        2. Extract 5-10 key insights from the types of questions asked in each topics
        
        Format requirements:
       
        - Keep insights focused and specific
        - Keep separation of one line between each theme 
        
        Keep total response under 300 words, provide nicely formatted info 
        
        Document content:
        {content}"""

    def get_qa_extraction_prompt(self) -> str:
        return """Extract "All Q&A exchanges" from the document using this format:

        For each exchange:
        1. Analyst Name (Firm): Question: [summarize question text]
        2. Executive Name (Title): Answer: [summarize answer text in at max 3 lines]

        Requirements:
        - Provide all the questions and answers
        - Maintain original wording especially in the answer
        - Include speaker names and titles
        - Separate each Q&A exchange clearly

        
        Document content:
        {content}"""


        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('analysis_pipeline.log'),
                logging.StreamHandler()
            ]
        )

    def create_output_directory(self, ticker: str) -> str:
        """Create and return path to output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"analysis_pdfs_{ticker}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def download_transcripts(self, ticker: str, company_name: str, quarter: str) -> List[str]:
        """Download transcripts using the TranscriptCrawler"""
        try:
            logging.info(f"Fetching transcripts for {ticker} ({company_name})")
            api_key = os.getenv('GOOGLE_API_KEY')
            custom_id= os.getenv('CUSTOM_ID')
            analyzer = FinancialReportAnalyzer(api_key, custom_id, debug=self.debug)
            crawler = TranscriptCrawler()
            
            # Fetch latest transcript
            response = analyzer.fetch_latest_transcript(ticker, company_name, quarter)
            
            # Download transcripts
            downloaded_files = crawler.download_from_search_results(response, ticker)
            
            return downloaded_files
            
        except Exception as e:
            logging.error(f"Error downloading transcripts: {str(e)}", exc_info=True)
            raise

    def run_pipeline(self, ticker: str, company_name: str) -> str:
        """Run the complete analysis pipeline"""
        try:
            # Create output directory
            output_dir = self.create_output_directory(ticker)
            logging.info(f"Created output directory: {output_dir}")
            
            # Get quarter information
            quarter_info = get_quarter_info()
            year= quarter_info['year']
            quarter= quarter_info['quarter_str']
            
            # Download transcripts
            downloaded_files = self.download_transcripts(ticker, company_name, quarter)
            logging.info(f"Downloaded {len(downloaded_files)} transcript(s)")

            # Clean up PDF by removing specified pages
            logging.info("Processing final PDF - removing unnecessary pages")
            PDFCleanup.process_final_pdf(f'{ticker}_earnings_call.pdf')
            
            # Process each transcript
            results = self.process_document(f'{ticker}_earnings_call.pdf')
            
            api_key = os.getenv('GOOGLE_API_KEY')
            custom_id= os.getenv('CUSTOM_ID')
            analyzer = FinancialReportAnalyzer(api_key, custom_id, debug=self.debug)
            
            # Generate final report
            self.generate_report(results, output_dir, ticker, analyzer.fetch_latest_transcript(ticker, company_name, quarter))  # Using first transcript for now
            
            return output_dir
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise

    def generate_report(self, results: Dict, output_dir: str, ticker: str, transcript_url: str):
        try:
            logging.info("Generating PDF report")
            output_path = os.path.join(output_dir, f"{ticker}_earnings_analysis.pdf")
            
            transcript_url = transcript_url['items'][0]['link'] if isinstance(transcript_url, dict) and 'items' in transcript_url and transcript_url['items'] else ""
            pdf_report = PDFReport()
            pdf_report.add_analysis_to_pdf(results, output_path, transcript_url)
            
            logging.info(f"PDF report saved to: {output_path}")
            
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}", exc_info=True)
            raise
    
def main():
    # Get API keys
    dotenv.load_dotenv()
    openai_api_key =os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Get user input
    ticker = input("Enter ticker symbol: ").upper()
    company_name = input("Enter company name: ")
    
    # Initialize and run pipeline
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


