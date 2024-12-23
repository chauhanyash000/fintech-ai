import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv

class InvestorTakeawaysReport:
    def __init__(self, output_path):
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        self.styles = getSampleStyleSheet()
        self.elements = []
        
        # Create custom styles
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=10,
            alignment=1,  # Center alignment
            fontName='Times-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='TakeawayHeading',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=1,
            fontName='Times-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='TakeawayContent',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=5,
            leading=14,
            fontName='Times-Roman'
        ))

    def add_title(self, ticker):
        title = Paragraph(
            f'{ticker} - TOP 5 KEY Things to Know for Investors about {ticker}',
            self.styles['CustomHeading']
        )
        self.elements.append(title)

    def add_takeaway(self, number, headline, content):
        # Add takeaway heading
        heading = Paragraph(
            f"{number}. {headline.upper()}",
            self.styles['TakeawayHeading']
        )
        self.elements.append(heading)
        
        # Add takeaway content
        content_para = Paragraph(content, self.styles['TakeawayContent'])
        self.elements.append(content_para)
        
        # Add space after each takeaway
        self.elements.append(Spacer(1, 0.2 * inch))

    def save(self):
        self.doc.build(
            self.elements,
            onFirstPage=self._header_footer,
            onLaterPages=self._header_footer
        )

    def _header_footer(self, canvas, doc):
        # Header
        canvas.saveState()
        canvas.setFont('Times-Bold', 12)
        canvas.drawString(72, letter[1] - 50, 'Investor Key Takeaways Analysis')

def analyze_earnings(ticker: str, api_key: str) -> None:
    openai.api_key = api_key
    
    try:
        dir_path = f'analysis_pdfs_{ticker}'
        pdf_path = os.path.join(dir_path, f'{ticker}_earnings_analysis.pdf')
        output_path = os.path.join(dir_path, f'{ticker}_investor_takeaways.pdf')
        
        os.makedirs(dir_path, exist_ok=True)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Earnings analysis PDF not found for {ticker}")
            
        reader = PdfReader(pdf_path)
        content = ' '.join(page.extract_text() for page in reader.pages)
        
        prompt = f"""Based on the following earnings analysis, provide the top 5 most important takeaways for investors.
        Focus on key financial metrics, strategic initiatives, market position, and future outlook.
        
        For each takeaway:
        1. Provide a bold 2-3 word headline
        2. Follow with 1 paragraph of detailed information 
        3. Include specific numbers and metrics where relevant
        
        Format each takeaway as:
        [NUMBER]. [HEADLINE]: [Detailed information]
        
        Analysis Content:
        {content}
        
        Provide exactly 5 takeaways, numbered 1-5."""
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional financial analyst expert at analyzing earnings reports and making sound judgements."},
                {"role": "user", "content": prompt}
            ]
        )
        
        takeaways = response.choices[0].message.content.split('\n')
        
        report = InvestorTakeawaysReport(output_path)
        report.add_title(ticker)
        
        current_takeaway = []
        for line in takeaways:
            if not line.strip():
                continue
                
            if line[0].isdigit():
                if current_takeaway:
                    process_takeaway(report, current_takeaway)
                current_takeaway = [line]
            else:
                current_takeaway.append(line)
        
        if current_takeaway:
            process_takeaway(report, current_takeaway)
        
        report.save()
        print(f"Analysis complete. Report saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")

def process_takeaway(report: InvestorTakeawaysReport, takeaway_lines: list) -> None:
    number = int(takeaway_lines[0].split('.')[0])
    parts = takeaway_lines[0].split(':', 1)
    headline = parts[0].split('.', 1)[1].strip()
    content = parts[1].strip() if len(parts) > 1 else ' '.join(takeaway_lines[1:])
    report.add_takeaway(number, headline, content)

def main():
    load_dotenv()
    try:
        ticker = input("Enter ticker symbol: ").upper()
        api_key = os.getenv('OPENAI_API_KEY')
        analyze_earnings(ticker, api_key)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()