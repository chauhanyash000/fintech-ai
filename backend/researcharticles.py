import os
from typing import List, Dict
import PyPDF2
import openai
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialResearchAssistant:
    def __init__(self, openai_api_key: str, google_api_key: str, custom_id: str):
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        self.custom_id = custom_id
        openai.api_key = openai_api_key
        self.search_service = build("customsearch", "v1", developerKey=self.google_api_key)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
            
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise

    def generate_search_queries(self, text: str) -> List[str]:
        """Generate search queries using OpenAI"""
        try:
            prompt = f"""
            You are a financial analyst. Based on this document text, first get the company name then generate 10 specific search queries 
            to find good research articles about that "company name only" and its financial performance.
            Focus on recent developments, financial metrics, and market position.

            Document text:
            {text[:4000]}

            Return atleast 10 search queries, one per line. Make them "specific to company" and focused on financial news.
            Do not include any other text or explanations.
            """

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )

            queries = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
            logger.info(f"Generated {len(queries)} search queries")
            logger.info(f"Queries: {queries}")
            return queries
        except Exception as e:
            logger.error(f"Error generating queries: {str(e)}")
            raise

    def search_articles(self, query: str, ticker: str) -> List[Dict]:
        """
        Search articles using Google Custom Search with date filtering
        Returns articles published in the last 6 months
        """
        try:
            # Define financial news sites
            sites = ['reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'cnbc.com', 
                    'seekingalpha.com', 'fool.com', 'marketwatch.com']
            site_filter = ' OR '.join([f'site:{site}' for site in sites])
            
            # Calculate date 6 months ago
            six_months_ago = datetime.now() - timedelta(days=180)
            date_str = six_months_ago.strftime('%Y-%m-%d')
            
            # Remove double quotes from query and construct full query
            clean_query = query.replace('"', '')
            full_query = f"{clean_query} {ticker}({site_filter})"
            logger.info(f"Executing search with query: {full_query}")

            results = self.search_service.cse().list(
                q=full_query,
                cx=self.custom_id,
                num=3,
                dateRestrict='m6'  # Restrict to last 6 months using dateRestrict parameter
            ).execute()

            articles = []
            if 'items' in results:
                # Process all results
                for item in results['items']:
                    article = {
                        'title': item['title'],
                        'link': item['link'],
                        'snippet': item['snippet'],
                        'published_date': item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', 'N/A')
                    }
                    articles.append(article)
                    logger.info(f"Found article: {article['title']} (Published: {article['published_date']})")
            else:
                logger.warning(f"No results found for query: {clean_query}")

            return articles
        except Exception as e:
            logger.error(f"Error searching articles: {str(e)}")
            raise

    def generate_article_content(self, articles: List[Dict]) -> List[Dict]:
        """Generate formatted content for each article using OpenAI"""
        try:
            formatted_articles = []
            
            for article in articles:
                prompt = f"""
                You are a financial analyst. Based on this article information, generate a detailed financial analysis 
                in exactly this format:

                Title: [Keep original title]
                Author: Financial Research Team
                URL: {article['link']}
                Summary: [Write a focused 3 lines summary of the key financial insights, numbers, and market implications]

                Article information:
                Title: {article['title']}
                Snippet: {article['snippet']}

                Focus on financial metrics, market analysis, and actionable insights in the summary.
                """

                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500
                )

                content = response.choices[0].message.content.strip()
                
                # Parse the generated content
                content_parts = content.split('\n')
                formatted_article = {
                    'title': content_parts[0].replace('Title:', '').strip(),
                    'author': content_parts[1].replace('Author:', '').strip(),
                    'url': article['link'],
                    'summary': '\n'.join(content_parts[3:]).replace('Summary:', '').strip()
                }
                
                formatted_articles.append(formatted_article)
                logger.info(f"Generated content for article: {formatted_article['title']}")
                
            return formatted_articles
        except Exception as e:
            logger.error(f"Error generating article content: {str(e)}")
            raise

    def save_to_pdf(self, articles: List[Dict], output_path: str):
        """Save formatted articles to PDF"""
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'ArticleTitle',
                parent=styles['Heading1'],
                fontSize=11,
                spaceAfter=2,
                textColor=colors.HexColor('#2C3E50')
            )
            
            body_style = ParagraphStyle(
                'ArticleBody',
                parent=styles['Normal'],
                fontSize=9,
                spaceAfter=2,
                leading=14
            )
            
            link_style = ParagraphStyle(
                'ArticleLink',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.blue,
                spaceAfter=2
            )
            
            story = []
            
            # Header
            story.append(Paragraph("Related Financial Research Articles", styles['Heading1']))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
            story.append(Spacer(1, 0.05 * inch))
            
            # Add articles
            for idx, article in enumerate(articles, 1):
                story.append(Paragraph(f"Article {idx}", styles['Heading2']))
                story.append(Paragraph(article['title'], title_style))
                story.append(Paragraph(
                    f'<link href="{article["url"]}">{article["url"]}</link>', 
                    link_style
                ))
                story.append(Paragraph(article['summary'], body_style))
                story.append(Spacer(1, 0.05 * inch))
            
            doc.build(story)
            logger.info(f"Successfully saved PDF to {output_path}")
        except Exception as e:
            logger.error(f"Error saving PDF: {str(e)}")
            raise

    def process_document(self, input_path: str, output_path: str, ticker:str):
        """Main processing pipeline"""
        try:
            logger.info(f"Starting document processing for {input_path}")
            
            # 1. Extract text from input PDF
            text = self.extract_text_from_pdf(input_path)
            if not text.strip():
                raise ValueError("No text extracted from PDF")
            
            # 2. Generate search queries
            queries = self.generate_search_queries(text)
            if not queries:
                raise ValueError("No search queries generated")
            
            # 3. Search for articles
            all_articles = []
            for query in queries:
                articles = self.search_articles(query, ticker)
                all_articles.extend(articles)

            del all_articles[0]
            
            if not all_articles:
                raise ValueError("No articles found from searches")
            
            # Remove duplicates based on URL
            unique_articles = list({article['link']: article for article in all_articles}.values())
            logger.info(f"Found {len(unique_articles)} unique articles")
            
            # 4. Generate formatted content
            formatted_articles = self.generate_article_content(unique_articles[:10])
            
            # 5. Save to PDF
            self.save_to_pdf(formatted_articles, output_path)
            
            logger.info("Document processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            raise

def main():
    try:
        # Get input parameters
        ticker = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
        logger.info(f"Processing research for ticker: {ticker}")

        load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')

        if not openai_api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")

        custom_id = os.getenv('CUSTOM_ID')

        if not openai_api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")

        google_api_key = os.getenv('GOOGLE_API_KEY')

        if not openai_api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        
        
        # Initialize assistant
        assistant = FinancialResearchAssistant(
            openai_api_key 
            ,google_api_key
            ,custom_id
        )
        # Set up paths
        dir_path = f'analysis_pdfs_{ticker}'
        os.makedirs(dir_path, exist_ok=True)
        
        input_path = os.path.join(dir_path, f'{ticker}_earnings_analysis.pdf')
        output_path = os.path.join(dir_path, f'{ticker}_research_articles.pdf')
        
        # Verify input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input PDF not found at: {input_path}")
        
        # Process document
        assistant.process_document(input_path, output_path, ticker)
        print(f"\nResearch articles have been saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    load_dotenv()
    main()
