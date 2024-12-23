import os
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import re
import openai
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER

# Define professional color scheme
COLORS = {
    'primary': colors.HexColor('#1A1A1A'),     # Dark gray for main text
    'secondary': colors.HexColor('#2F6690'),    # Professional blue for headings
    'accent': colors.HexColor('#CC0000'),      # Red for emphasis
    'light_gray': colors.HexColor('#F2F2F2'),  # Light gray for backgrounds
    'medium_gray': colors.HexColor('#666666'), # Medium gray for subtext
    'border': colors.HexColor('#CCCCCC'),      # Light gray for borders
}

def create_stylesheet():
    """Create professional document styles"""
    styles = getSampleStyleSheet()
    
    # Create custom styles with unique names
    styles.add(ParagraphStyle(
        name='CustomHeading1',  # Match the name used in analyze_earnings
        parent=styles['Heading1'],
        fontSize=16,
        textColor=COLORS['secondary'],
        spaceAfter=20,
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading2',  # Match the name used in analyze_earnings
        parent=styles['Heading2'],
        fontSize=14,
        textColor=COLORS['primary'],
        spaceAfter=15,
    ))
    
    styles.add(ParagraphStyle(
        name='CustomBody',  # Match the name used in analyze_earnings
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=COLORS['primary'],
    ))
    return styles

def header_footer(canvas, doc):
    """Add header and footer to each page"""
    canvas.saveState()

    
    # Footer
    canvas.setFont('Helvetica-Oblique', 8)
    canvas.setFillColor(COLORS['medium_gray'])
    
    # Source attribution
    canvas.drawString(30, 30, 'Data sourced from Yahoo Finance')
    
    # Timestamp and disclaimer
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    disclaimer = f'Generated on {timestamp} | For informational purposes only'
    canvas.drawCentredString(doc.pagesize[0]/2, 15, disclaimer)
    
    canvas.restoreState()

def get_earnings_data(ticker: str) -> Dict:
    """
    Fetches real earnings data from Yahoo Finance using income statement
    """
    try:
        # Get stock info
        stock = yf.Ticker(ticker)
        
        # Check if we can get the info
        info = stock.info
        if not info:
            raise ValueError(f"Could not retrieve information for {ticker}")
            
        company_name = str(info.get('longName', ticker))
        
        # Get quarterly income statement
        quarterly_stmt = stock.quarterly_income_stmt
        if quarterly_stmt is None or quarterly_stmt.empty:
            raise ValueError("No quarterly income statement data available")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=450)
        history = stock.history(start=start_date, end=end_date)
        
        if history.empty:
            raise ValueError("No historical price data available")
        
        # Make index timezone-naive
        history.index = history.index.tz_localize(None)
        
        earnings_data = []
        
        for date, data in quarterly_stmt.items():
            try:
                date_naive = pd.Timestamp(date).tz_localize(None)
                quarter = f"Q{pd.Timestamp(date).quarter} {pd.Timestamp(date).year}"
                date_str = date_naive.strftime('%Y-%m-%d')
                
                # Get EPS data safely
                try:
                    # First try to get net income
                    if 'Net Income' not in data:
                        continue
                    net_income = float(data.loc['Net Income'])
                    
                    # Try different ways to get shares outstanding
                    shares = None
                    for share_field in ['sharesOutstanding', 'impliedSharesOutstanding']:
                        shares = info.get(share_field)
                        if shares:
                            break
                    
                    # Calculate EPS
                    if shares and shares > 0:
                        eps_actual = net_income / shares
                    else:
                        # Try to get EPS directly from financials
                        financials = stock.quarterly_financials
                        if not financials.empty and 'Basic EPS' in financials.index:
                            eps_actual = float(financials.loc['Basic EPS', date])
                        else:
                            print(f"Warning: Could not calculate reliable EPS for {date_str}")
                            continue
                except Exception as eps_error:
                    print(f"Warning: EPS calculation failed for {date_str}: {str(eps_error)}")
                    continue
                
                # Get price data safely
                try:
                    date_idx = history.index.get_indexer([date_naive], method='nearest')[0]
                    if date_idx <= 0 or date_idx >= len(history) - 1:
                        continue
                        
                    price_data = {
                        'before': history.iloc[date_idx - 3]['Close'],
                        'date': history.iloc[date_idx]['Close'],
                        'after': history.iloc[date_idx + 3]['Close'],
                        'volume_before': history.iloc[date_idx - 3]['Volume'],
                        'volume_date': history.iloc[date_idx]['Volume'],
                        'volume_after': history.iloc[date_idx + 3]['Volume']
                    }
                    
                    # Verify all price data is valid
                    if any(not isinstance(v, (int, float)) or pd.isna(v) for v in price_data.values()):
                        continue
                        
                    earnings_data.append({
                        "quarter": quarter,
                        "date": date_str,
                        "eps": float(eps_actual),
                        "price_before": float(price_data['before']),
                        "price_date": float(price_data['date']),
                        "price_after": float(price_data['after']),
                        "volume_before": int(price_data['volume_before']),
                        "volume_date": int(price_data['volume_date']),
                        "volume_after": int(price_data['volume_after'])
                    })
                    
                except Exception as price_error:
                    print(f"Warning: Price data retrieval failed for {date_str}: {str(price_error)}")
                    continue
                    
            except Exception as entry_error:
                print(f"Warning: Failed to process earnings entry for {date}: {str(entry_error)}")
                continue
        
        # Sort and limit data
        earnings_data.sort(key=lambda x: x['date'], reverse=True)
        earnings_data = earnings_data[:6]
        
        if not earnings_data:
            raise ValueError("No valid earnings data could be processed")
            
        return {
            "company_name": company_name,
            "earnings_data": earnings_data
        }
            
    except Exception as e:
        raise Exception(f"Error getting data for {ticker}: {str(e)}")

# Helper function to safely get numeric values
def safe_float(value, default=None):
    """Safely convert a value to float, returning default if conversion fails"""
    try:
        result = float(value)
        return result if not pd.isna(result) else default
    except (TypeError, ValueError):
        return default

def create_earnings_table(data: Dict, styles) -> Table:
    """Create formatted earnings history table"""
    headers = ['Quarter', 'Date', 'EPS', 'Price Impact', 'Volume Impact']
    table_data = [headers]
    
    for entry in data['earnings_data']:
        price_change = ((entry['price_after'] - entry['price_before']) / entry['price_before']) * 100
        volume_change = ((entry['volume_date'] - entry['volume_before']) / entry['volume_before']) * 100
        
        row = [
            entry['quarter'],
            entry['date'],
            f"${entry['eps']:.2f}",
            f"${entry['price_before']:.2f} → ${entry['price_after']:.2f} ({price_change:+.1f}%)",
            f"{volume_change:+.1f}%"
        ]
        table_data.append(row)
    
    table = Table(table_data, colWidths=[0.8*inch, inch, 0.8*inch, 2*inch, inch])
    table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONT', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), COLORS['light_gray']),
        ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['primary']),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.25, COLORS['border']),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    
    return table

def analyze_earnings(ticker: str, api_key: str) -> None:
    try:
        openai.api_key = api_key
        
        dir_path = f'analysis_pdfs_{ticker}'
        os.makedirs(dir_path, exist_ok=True)
        output_path = os.path.join(dir_path, f'{ticker}_pricechanges.pdf')
        
        data = get_earnings_data(ticker)
        
        prompt = f"""Based on this earnings data for {ticker} ({data['company_name']}):
        {data['earnings_data']}
        
        Provide only 2 key investor takeaways. Provide nicely formatted info that covers 0.5 page only:
        1. One paragraph analysis with specific metrics
        2. Focus on trends, performance, and price reactions
        
        Format: [NUMBER]. [HEADLINE]: [Analysis]"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial analyst expert at analyzing earnings trends."},
                {"role": "user", "content": prompt}
            ]
        )
        
        analysis_text = response.choices[0].message.content
        
        styles = create_stylesheet()
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=30*mm,
            leftMargin=30*mm,
            topMargin=45*mm,
            bottomMargin=30*mm,
            title=f"{ticker} ({data['company_name']}) - Earnings Analysis"
        )
        
        story = []
        
        # Title with CustomHeading1
        story.append(Paragraph(
            f"{ticker} ({data['company_name']}) - Earnings Analysis",
            styles['CustomHeading1']
        ))
        story.append(Spacer(1, 12))
        
        # Recent Earnings History section with CustomHeading2
        story.append(Paragraph("Recent Earnings History", styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        story.append(create_earnings_table(data, styles))
        story.append(Spacer(1, 12))
        
        # Analysis section with CustomHeading2
        story.append(Paragraph("Price Changes Around Earnings Call", styles['CustomHeading2']))
        story.append(Spacer(1, 12))
        
        # Split analysis text into paragraphs with CustomBody
        for paragraph in analysis_text.split('\n\n'):
            if paragraph.strip():
                story.append(Paragraph(paragraph, styles['CustomBody']))
        
        doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
        print(f"Analysis saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")


def main():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    ticker = input("Enter stock ticker: ").upper()
    analyze_earnings(ticker, api_key)

if __name__ == "__main__":
    main()