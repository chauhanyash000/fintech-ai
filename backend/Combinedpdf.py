import os
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
import openai
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
import fitz  # PyMuPDF
from datetime import datetime
import re
from io import BytesIO
from dotenv import load_dotenv
import shutil

COLORS = {
    'primary': colors.HexColor('#1A1A1A'),
    'secondary': colors.HexColor('#2F6690'),
    'accent': colors.HexColor('#CC0000'),
    'light_gray': colors.HexColor('#F2F2F2'),
    'medium_gray': colors.HexColor('#666666'),
    'border': colors.HexColor('#CCCCCC'),
}

class TextCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        replacements = {
            '\u2018': "'", '\u2019': "'",
            '\u201C': '"', '\u201D': '"',
            '\u2013': '–', '\u2014': '—',
            '\u2026': '...',
            '\u2022': '•',
            '\u00A9': '©',
            '\u00AE': '®',
            '\u2122': '™',
            '\u00B0': '°',
            '\u00A0': ' ',
            '\u20AC': '€',
            '\u00A3': '£',
            '\u00A5': '¥',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return ''.join(char if ord(char) < 128 else ' ' for char in text)

class ProfessionalStylesheet:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        self.styles.add(ParagraphStyle(
            name='CoverTitle',
            parent=self.styles['Title'],
            fontSize=32,
            textColor=COLORS['primary'],
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='CoverSubtitle',
            parent=self.styles['Title'],
            fontSize=20,
            textColor=COLORS['secondary'],
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=COLORS['secondary'],
            spaceBefore=20,
            spaceAfter=15,
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderColor=COLORS['border'],
            borderPadding=(10, 0, 10, 0)
        ))

        self.styles.add(ParagraphStyle(
            name='CustomBody',  # Changed from 'BodyText' to 'CustomBody'
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=COLORS['primary'],
            leading=14,
            alignment=TA_JUSTIFY,
            fontName='Times-Roman',
            spaceBefore=6,
            spaceAfter=6
        ))

        self.styles.add(ParagraphStyle(
            name='TOCEntry',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=COLORS['primary'],
            leading=16,
            leftIndent=20,
            fontName='Helvetica'
        ))

class PDFGenerator:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stylesheet = ProfessionalStylesheet()

    def create_header(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(COLORS['light_gray'])
        canvas.rect(0, doc.pagesize[1] - 40, doc.pagesize[0], 40, fill=True)
        
        canvas.setFillColor(COLORS['secondary'])
        canvas.setFont('Helvetica-Bold', 10)
        canvas.drawString(30, doc.pagesize[1] - 25, 
                         f'{self.ticker} - Investment Analysis Report')
        
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(COLORS['medium_gray'])
        date_text = datetime.now().strftime("%B %d, %Y")
        canvas.drawRightString(doc.pagesize[0] - 30, doc.pagesize[1] - 25, date_text)
        
        canvas.setStrokeColor(COLORS['accent'])
        canvas.setLineWidth(0.5)
        canvas.line(30, doc.pagesize[1] - 42, doc.pagesize[0] - 30, doc.pagesize[1] - 42)
        canvas.restoreState()

    def create_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(COLORS['border'])
        canvas.setLineWidth(0.5)
        canvas.line(30, 30, doc.pagesize[0] - 30, 30)
        
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(COLORS['medium_gray'])
        footer_text = f'Page {doc.page} | {self.ticker} Analysis'
        canvas.drawString(30, 15, footer_text)
        
        conf_text = 'Confidential - For Professional Investors Only'
        canvas.drawRightString(doc.pagesize[0] - 30, 15, conf_text)
        canvas.restoreState()

class ReportBuilder:
    def __init__(self, ticker: str, company_name: str):
        self.ticker = ticker.upper()
        self.company_name = company_name
        self.pdf_generator = PDFGenerator(ticker)
        self.text_cleaner = TextCleaner()

    def create_title_page(self, output_path: str) -> str:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=30 * mm,
            leftMargin=30 * mm,
            topMargin=30 * mm,
            bottomMargin=30 * mm
        )
        
        story = []
        story.extend([
            Spacer(1, 100),
            Paragraph(
                self.company_name.upper(),
                self.pdf_generator.stylesheet.styles['CoverTitle']
            ),
            Paragraph(
                f'NYSE: {self.ticker}',
                self.pdf_generator.stylesheet.styles['CoverSubtitle']
            ),
            Spacer(1, 40),
            Paragraph(
                'AI Investment Analysis Report',
                self.pdf_generator.stylesheet.styles['CoverSubtitle']
            ),
            Spacer(1, 20),
            Paragraph(
                datetime.now().strftime("%B %d, %Y"),
                self.pdf_generator.stylesheet.styles['CustomBody']  # Changed from 'BodyText' to 'CustomBody'
            )
        ])
        
        doc.build(story, onFirstPage=self._create_cover_footer)
        return output_path

    def _create_cover_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(COLORS['medium_gray'])
        canvas.setFont('Helvetica', 8)
        
        footer_text = [
            f"© {datetime.now().year}",
        ]
        
        y_position = 50
        for text in footer_text:
            canvas.drawCentredString(doc.pagesize[0]/2, y_position, text)
            y_position -= 12
        
        canvas.restoreState()

    def create_table_of_contents(self, output_path: str) -> str:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=30 * mm,
            leftMargin=30 * mm,
            topMargin=30 * mm,
            bottomMargin=30 * mm
        )
        
        story = []
        story.append(Paragraph(
            'Table of Contents',
            self.pdf_generator.stylesheet.styles['SectionHeader']
        ))
        story.append(Spacer(1, 20))
        
        sections = [
            "Executive Summary",
            "Company Analysis",
            "Investor Takeaways",
            "Press Release Analysis",
            "Price Changes",
            "Analysts Coverage",
            "Earnings Analysis",
            "Research Articles"
        ]
        
        toc_data = [[section, str(i+1)] for i, section in enumerate(sections, 1)]
        toc_table = Table(
            toc_data,
            colWidths=[400, 50],
            style=TableStyle([
                ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['primary']),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ])
        )
        
        story.append(toc_table)
        
        doc.build(
            story,
            onFirstPage=self.pdf_generator.create_header,
            onLaterPages=self.pdf_generator.create_header
        )
        return output_path

    def generate_summary(self, text: str, api_key: str) -> str:
        openai.api_key = api_key
        
        prompt = f"""Create a professional investment analysis "200 words" executive summary for {self.ticker} based on the following documents. 
        Structure the summary with these sections:

        1. Investment Highlights
        2. Risk Factors
        3. Market Position & Competition
        4. Growth Catalysts
        5. Analyst Consensus
        7. Investment Recommendation

        Focus on actionable insights and quantitative data where available:

        {self.text_cleaner.clean_text(text)}"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior investment analyst creating executive summaries. Provide in one page only concrete data, specific metrics, and actionable insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        
        return response.choices[0].message.content

    def process_pdfs(self, input_dir: str, output_path: str, api_key: str) -> bool:
        try:
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            # Try to import HRFlowable, use alternative if not available
            try:
                from reportlab.platypus import HRFlowable
                use_hr_flowable = True
            except ImportError:
                use_hr_flowable = False
                
            temp_dir = "temp_report_files"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create title and TOC pages
            title_path = os.path.join(temp_dir, "title.pdf")
            toc_path = os.path.join(temp_dir, "toc.pdf")
            
            self.create_title_page(title_path)
            self.create_table_of_contents(toc_path)
            
            # Define the expected order of PDF files
            expected_files = [
                f"{self.ticker}_analysis.pdf",
                f"{self.ticker}_investor_takeaways.pdf",
                f"{self.ticker}_press_release_analysis.pdf",
                f"{self.ticker}_pricechanges.pdf",
                f"{self.ticker}_AnalystCoverage.pdf",
                f"{self.ticker}_earnings_analysis.pdf",
                f"{self.ticker}_research_articles.pdf"
            ]
            
            # Define files to use for summary
            summary_source_files = [
                f"{self.ticker}_analysis.pdf",
                f"{self.ticker}_investor_takeaways.pdf",
                f"{self.ticker}_earnings_analysis.pdf"
            ]
            
            # Collect text for summary only from specified files
            summary_text = ""
            for filename in summary_source_files:
                file_path = os.path.join(input_dir, filename)
                if os.path.exists(file_path):
                    with fitz.open(file_path) as doc:
                        summary_text += "\n".join(page.get_text() for page in doc)
                else:
                    print(f"Warning: Summary source file missing: {filename}")
            
            # Generate summary
            summary = self.generate_summary(summary_text, api_key)
            summary_path = os.path.join(temp_dir, "executive_summary.pdf")
            
            # Define styles for executive summary
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='SummarySection',
                parent=styles['Heading2'],
                fontSize=12,
                spaceAfter=5,
                textColor=COLORS['primary'],
                fontName='Helvetica-Bold'
            ))
            styles.add(ParagraphStyle(
                name='SummaryText',
                parent=styles['BodyText'],
                fontSize=10,
                leading=14,
                spaceAfter=5,
                fontName='Helvetica'
            ))
            
            # Create PDF for executive summary with enhanced formatting
            doc = SimpleDocTemplate(
                summary_path,
                pagesize=A4,
                rightMargin=30 * mm,
                leftMargin=30 * mm,
                topMargin=30 * mm,
                bottomMargin=30 * mm
            )
            
            story = []
            
            # Add main title
            story.append(Paragraph(
                "Executive Summary",
                self.pdf_generator.stylesheet.styles['SectionHeader']
            ))
            story.append(Spacer(1, 5))
            
            # Add company info and date
            story.append(Paragraph(
                f"{self.company_name} ({self.ticker})",
                styles['SummarySection']
            ))
            story.append(Paragraph(
                f"Report Date: {datetime.now().strftime('%B %d, %Y')}",
                styles['SummaryText']
            ))
            story.append(Spacer(1, 10))
            
            # Process and format each section of the summary
            sections = summary.split('\n')
            current_section = None
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # Check if this is a section header (numbered sections from GPT response)
                if any(section.startswith(str(i) + '.') for i in range(1, 9)):
                    # Extract section title without the number
                    section_title = section.split('.', 1)[1].strip()
                    story.append(Paragraph(
                        section_title,
                        styles['SummarySection']
                    ))
                    current_section = None
                else:
                    # Add the content with proper formatting
                    if current_section is None:
                        current_section = []
                    
                    story.append(Paragraph(
                        section,
                        styles['SummaryText']
                    ))
            
            # Add a line at the end of the summary
            story.append(Spacer(1, 10))
            
            # Add horizontal line using HRFlowable if available, otherwise use alternative
            if use_hr_flowable:
                story.append(HRFlowable(
                    width="100%",
                    thickness=1,
                    color=COLORS['medium_gray'],
                    spaceBefore=10,
                    spaceAfter=10
                ))
            else:
                # Alternative: Create a thin table as a horizontal line
                hr_table = Table([['']], colWidths=[450], rowHeights=[1])
                hr_table.setStyle(TableStyle([
                    ('LINEABOVE', (0, 0), (-1, 0), 1, COLORS['medium_gray']),
                    ('TOPPADDING', (0, 0), (-1, 0), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
                ]))
                story.append(hr_table)
            
            doc.build(
                story,
                onFirstPage=self.pdf_generator.create_header,
                onLaterPages=self.pdf_generator.create_header
            )
            
            # Merge PDFs in specified order
            merger = PdfMerger()
            merger.append(title_path)
            merger.append(toc_path)
            merger.append(summary_path)
            
            # Add remaining PDFs in specified order
            missing_files = []
            for expected_file in expected_files:
                file_path = os.path.join(input_dir, expected_file)
                if os.path.exists(file_path):
                    merger.append(file_path)
                else:
                    missing_files.append(expected_file)
            
            merger.write(output_path)
            merger.close()
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            # Log missing files if any
            if missing_files:
                print("Warning: The following expected files were missing:")
                for missing_file in missing_files:
                    print(f"- {missing_file}")
            
            return True
            
        except Exception as e:
            print(f"Error processing PDFs: {str(e)}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False
    
def main():
    load_dotenv()
    
    ticker = input("Enter ticker symbol: ").upper()
    company_name = input("Enter company name: ")
    
    input_dir = f"analysis_pdfs_{ticker}"
    output_file = f"{ticker}_AI_Investment_Analysis.pdf"
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    builder = ReportBuilder(ticker, company_name)
    success = builder.process_pdfs(input_dir, output_file, api_key)
    
    if success:
        print(f"Successfully created professional investment report: {output_file}")
    else:
        print("Failed to create report")

if __name__ == "__main__":
    main()