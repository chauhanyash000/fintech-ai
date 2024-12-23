import os
import json
from typing import Dict, Any, List
from datetime import datetime
import logging
import pytz
import dotenv

# Third-party imports
import yfinance as yf
import pandas as pd
import numpy as np
import openai
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Frame, PageTemplate, BaseDocTemplate
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import registerFont, registerFontFamily
from reportlab.lib.fonts import addMapping

from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.colors import HexColor


def get_position(current, reference):
        """Determine if price is above or below a reference point"""
        if current > reference:
            return "above"
        elif current < reference:
            return "below"
        return "at"

def get_trend(current, previous):
        """Determine trend direction"""
        if current > previous:
            return "upward"
        elif current < previous:
            return "downward"
        return "sideways"

def calculate_vwap(data):
        """Calculate VWAP"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        return (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()

def find_support_resistance(data, window=20):
        """Find potential support and resistance levels using recent highs and lows"""
        highs = data['High'].nlargest(window).unique()[:3]
        lows = data['Low'].nsmallest(window).unique()[:3]
        return sorted(lows.tolist()), sorted(highs.tolist())


def get_fundamental_analysis(stock: yf.Ticker) -> Dict:
    """Get fundamental analysis data using income statement"""
    try:
        # Get income statement data
        income_stmt = stock.income_stmt
        info = stock.info
        
        if income_stmt.empty:
            raise ValueError("No income statement data available")
        
        # Convert any datetime index to strings to avoid datetime operations
        income_stmt.index = income_stmt.index.astype(str)
        if isinstance(income_stmt.columns, pd.DatetimeIndex):
            income_stmt.columns = income_stmt.columns.strftime('%Y-%m-%d')
        
        # Sort columns by date descending
        income_stmt = income_stmt.sort_index(axis=1, ascending=False)
        
        # Get quarterly data (last 4 quarters)
        quarterly_data = income_stmt.iloc[:, :4]
        
        # Process quarterly revenue data
        quarterly_revenue = []
        for date in quarterly_data.columns:
            try:
                revenue = quarterly_data.loc['Total Revenue', date] / 1e6 if 'Total Revenue' in quarterly_data.index else 0
                
                # Find year ago date by parsing and subtracting one year
                current_date = pd.to_datetime(date)
                year_ago_date = (current_date - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
                
                # Look for the closest matching date in the data
                matching_dates = [d for d in income_stmt.columns if year_ago_date[:7] in d]  # Match year-month
                year_ago_idx = matching_dates[0] if matching_dates else None
                
                year_ago_revenue = (income_stmt.loc['Total Revenue', year_ago_idx] / 1e6 
                                  if year_ago_idx and 'Total Revenue' in income_stmt.index 
                                  else revenue)
                
                change_pct = ((revenue - year_ago_revenue) / year_ago_revenue * 100) if year_ago_revenue else 0
                
                quarter_str = f"Q{pd.to_datetime(date).quarter} {pd.to_datetime(date).year}"
                quarterly_revenue.append({
                    "quarter": quarter_str,
                    "revenue": round(revenue, 2),
                    "change_pct": round(change_pct, 2)
                })
            except Exception as e:
                print(f"Error processing quarterly revenue for {date}: {e}")
                quarterly_revenue.append({
                    "quarter": "Unknown",
                    "revenue": 0,
                    "change_pct": 0
                })
        
        
        # Group data by year
        yearly_groups = {}
        for date in income_stmt.columns:
            year = pd.to_datetime(date).year
            if year not in yearly_groups:
                yearly_groups[year] = []
            yearly_groups[year].append(date)
        
        # Sort years in descending order and take last 3
        sorted_years = sorted(yearly_groups.keys(), reverse=True)[:3]
        
        # Process yearly revenue data
        yearly_revenue = []
        for year in sorted_years:
            try:
                year_dates = yearly_groups[year]
                revenue = sum(income_stmt.loc['Total Revenue', date] / 1e6 
                            for date in year_dates 
                            if 'Total Revenue' in income_stmt.index) if 'Total Revenue' in income_stmt.index else 0
                
                prev_year = year - 1
                prev_revenue = 0
                if prev_year in yearly_groups and 'Total Revenue' in income_stmt.index:
                    prev_dates = yearly_groups[prev_year]
                    prev_revenue = sum(income_stmt.loc['Total Revenue', date] / 1e6 for date in prev_dates)
                
                change_pct = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue else 0
                
                yearly_revenue.append({
                    "year": str(year),
                    "revenue": round(revenue, 2),
                    "change_pct": round(change_pct, 2)
                })
            except Exception as e:
                print(f"Error processing yearly revenue for {year}: {e}")
                yearly_revenue.append({
                    "year": str(year),
                    "revenue": 0,
                    "change_pct": 0
                })
        
        # Process quarterly profit/loss data
        quarterly_profit_loss = []
        for date in quarterly_data.columns:
            try:
                revenue = quarterly_data.loc['Total Revenue', date] / 1e6 if 'Total Revenue' in quarterly_data.index else 0
                net_income = quarterly_data.loc['Net Income', date] / 1e6 if 'Net Income' in quarterly_data.index else 0
                margin_pct = (net_income / revenue * 100) if revenue else 0
                
                quarter_str = f"Q{pd.to_datetime(date).quarter} {pd.to_datetime(date).year}"
                quarterly_profit_loss.append({
                    "quarter": quarter_str,
                    "profit_loss": round(net_income, 2),
                    "margin_pct": round(margin_pct, 2)
                })
            except Exception as e:
                print(f"Error processing quarterly profit/loss for {date}: {e}")
                quarterly_profit_loss.append({
                    "quarter": "Unknown",
                    "profit_loss": 0,
                    "margin_pct": 0
                })
        
        # Process yearly profit/loss data
        yearly_profit_loss = []
        for year in sorted_years:
            try:
                year_dates = yearly_groups[year]
                revenue = sum(income_stmt.loc['Total Revenue', date] / 1e6 
                            for date in year_dates 
                            if 'Total Revenue' in income_stmt.index) if 'Total Revenue' in income_stmt.index else 0
                net_income = sum(income_stmt.loc['Net Income', date] / 1e6 
                               for date in year_dates 
                               if 'Net Income' in income_stmt.index) if 'Net Income' in income_stmt.index else 0
                margin_pct = (net_income / revenue * 100) if revenue else 0
                
                yearly_profit_loss.append({
                    "year": str(year),
                    "profit_loss": round(net_income, 2),
                    "margin_pct": round(margin_pct, 2)
                })
            except Exception as e:
                print(f"Error processing yearly profit/loss for {year}: {e}")
                yearly_profit_loss.append({
                    "year": str(year),
                    "profit_loss": 0,
                    "margin_pct": 0
                })
        
        # Get market metrics with error handling
        market_metrics = {
            "pe_ratio": info.get('forwardPE', None),
            "market_cap": info.get('marketCap', 0) / 1e9,  # Convert to billions
            "dividend_yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            "ex_dividend_date": str(info.get('exDividendDate', '')) if info.get('exDividendDate') else None
        }
        
        # Calculate income metrics from most recent quarter with error handling
        try:
            latest_date = quarterly_data.columns[0]
            income_metrics = {
                "gross_profit_margin": round((quarterly_data.loc['Gross Profit', latest_date] / 
                                           quarterly_data.loc['Total Revenue', latest_date] * 100), 2)
                                           if all(x in quarterly_data.index for x in ['Gross Profit', 'Total Revenue']) else 0,
                "operating_margin": round((quarterly_data.loc['Operating Income', latest_date] / 
                                        quarterly_data.loc['Total Revenue', latest_date] * 100), 2)
                                        if all(x in quarterly_data.index for x in ['Operating Income', 'Total Revenue']) else 0,
                "ebitda_margin": round((quarterly_data.loc['EBITDA', latest_date] / 
                                      quarterly_data.loc['Total Revenue', latest_date] * 100), 2)
                                      if all(x in quarterly_data.index for x in ['EBITDA', 'Total Revenue']) else 0
            }
        except Exception as e:
            print(f"Error calculating income metrics: {e}")
            income_metrics = {
                "gross_profit_margin": 0,
                "operating_margin": 0,
                "ebitda_margin": 0
            }
        
        return {
            "quarterly_revenue": quarterly_revenue,
            "yearly_revenue": yearly_revenue,
            "quarterly_profit_loss": quarterly_profit_loss,
            "yearly_profit_loss": yearly_profit_loss,
            "market_metrics": market_metrics,
            "income_metrics": income_metrics
        }
        
    except Exception as e:
        logging.error(f"Error in fundamental analysis: {e}")
        return {
            "quarterly_revenue": [],
            "yearly_revenue": [],
            "quarterly_profit_loss": [],
            "yearly_profit_loss": [],
            "market_metrics": {
                "pe_ratio": None,
                "market_cap": 0,
                "dividend_yield": 0,
                "ex_dividend_date": None
            },
            "income_metrics": {
                "gross_profit_margin": 0,
                "operating_margin": 0,
                "ebitda_margin": 0
            }
        }
    
def fetch_analysis_summary(ticker: str, stock: yf.Ticker, openai_api_key: str) -> Dict[str, str]:
    technical_data = get_technical_analysis(ticker)
    fundamental_data = get_fundamental_analysis(stock)
    
    # Format support and resistance levels with error handling
    try:
        support_levels = [f"${float(level):.2f}" for level in technical_data['technical_summary']['key_levels']['support']]
        resistance_levels = [f"${float(level):.2f}" for level in technical_data['technical_summary']['key_levels']['resistance']]
        support_str = ', '.join(support_levels)
        resistance_str = ', '.join(resistance_levels)
    except (ValueError, TypeError):
        support_str = "N/A"
        resistance_str = "N/A"

    # Ensure all numeric values are properly converted to float
    prompt = f"""Provide a "one page" detailed summary in Bullet Points of {ticker} in exactly two sections: Fundamental Analysis (350 words) and Technical Analysis (350 words).

    Base your summary on these metrics:
    
    Technical Data:
    - Current Price: ${float(technical_data['price_data']['current_price']):.2f}
    - MA50: ${float(technical_data['moving_averages']['ma_50']['value']):.2f}
    - MA200: ${float(technical_data['moving_averages']['ma_200']['value']):.2f}
    - RSI: {float(technical_data['rsi']['value']):.2f}
    - MACD Trend: {technical_data['macd']['trend']}
    - Support Levels: {support_str}
    - Resistance Levels: {resistance_str}
    - VWAP
    - Bollinger Bands
    
    Fundamental Data:
    - Latest Quarterly Revenue: ${float(fundamental_data['quarterly_revenue'][0]['revenue'] if fundamental_data['quarterly_revenue'] else 0):.2f}M
    - YoY Revenue Growth: {float(fundamental_data['quarterly_revenue'][0]['change_pct'] if fundamental_data['quarterly_revenue'] else 0):.2f}%
    - Latest Quarterly Profit: ${float(fundamental_data['quarterly_profit_loss'][0]['profit_loss'] if fundamental_data['quarterly_profit_loss'] else 0):.2f}M
    - Profit Margin: {float(fundamental_data['quarterly_profit_loss'][0]['margin_pct'] if fundamental_data['quarterly_profit_loss'] else 0):.2f}%
    - Gross Profit Margin: {float(fundamental_data['income_metrics']['gross_profit_margin']):.2f}%
    - Operating Margin: {float(fundamental_data['income_metrics']['operating_margin']):.2f}%
    - EBITDA Margin: {float(fundamental_data['income_metrics']['ebitda_margin']):.2f}%
    - Market Cap: ${float(fundamental_data['market_metrics']['market_cap']):.2f}B
    - P/E Ratio: {fundamental_data['market_metrics']['pe_ratio'] if fundamental_data['market_metrics']['pe_ratio'] is not None else 'N/A'}
    """

    messages = [
        {
            "role": "system",
            "content": "You are a financial Trading Expert providing stock analysis summaries with quality insights."
        },
        {
            "role": "user",
            "content": prompt 
        }
    ]

    try:
        openai.api_key = openai_api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Updated to use correct model name
            messages=messages,
            temperature=0.5
        )
    
        if response.choices:
            content = response.choices[0].message.content
            # Split the content into two sections
            sections = content.split('Technical Analysis')
            fundamental = sections[0].replace('Fundamental Analysis', '').strip()
            technical = ('Technical Analysis' + sections[1]).strip() if len(sections) > 1 else ''
            
            return {
                'fundamental_analysis': fundamental,
                'technical_analysis': technical
            }
        else:
            return {
                'fundamental_analysis': 'Error generating fundamental analysis summary.',
                'technical_analysis': 'Error generating technical analysis summary.'
            }
            
    except Exception as e:
        print(f"Error in OpenAI API call: {str(e)}")
        return {
            'fundamental_analysis': 'Error generating fundamental analysis summary.',
            'technical_analysis': 'Error generating technical analysis summary.'
        }

def get_technical_analysis(symbol, period="6mo"):
    """Get comprehensive technical analysis for a given symbol"""
    # Fetch data
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    
    if len(data) == 0:
        raise ValueError(f"No data found for symbol {symbol}")

    # Ensure we have numeric data by converting to float
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['High'] = pd.to_numeric(data['High'], errors='coerce')
    data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

    # Current and previous values
    current_price = float(data['Close'].iloc[-1])
    previous_close = float(data['Close'].iloc[-2])

    # Calculate Moving Averages
    ma_50 = data['Close'].rolling(window=50).mean()
    ma_200 = data['Close'].rolling(window=200).mean()
    vwap = calculate_vwap(data)

    current_ma50 = float(ma_50.iloc[-1])
    current_ma200 = float(ma_200.iloc[-1])
    current_vwap = float(vwap.iloc[-1])

    # Calculate MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = float(rsi.iloc[-1])

    # Calculate Bollinger Bands
    middle_band = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    upper_band = middle_band + (std * 2)
    lower_band = middle_band - (std * 2)
    band_width = (upper_band - lower_band) / middle_band

    # Find support and resistance levels
    support_levels, resistance_levels = find_support_resistance(data)

    # Check for MACD crossover using numeric indices
    crossover_status = "no crossover"
    if len(macd_line) >= 2 and len(signal_line) >= 2:
        if (macd_line.iloc[-1] > signal_line.iloc[-1] and 
            macd_line.iloc[-2] <= signal_line.iloc[-2]):
            crossover_status = "potential bullish crossover"
        elif (macd_line.iloc[-1] < signal_line.iloc[-1] and 
              macd_line.iloc[-2] >= signal_line.iloc[-2]):
            crossover_status = "potential bearish crossover"

    # Compile technical analysis data
    technical_data = {
        "price_data": {
            "current_price": current_price,
            "previous_close": previous_close,
            "daily_change_pct": ((current_price - previous_close) / previous_close * 100) if previous_close else 0
        },
        "moving_averages": {
            "ma_50": {
                "value": current_ma50,
                "position": get_position(current_price, current_ma50),
                "trend": get_trend(float(ma_50.iloc[-1]), float(ma_50.iloc[-2]) if len(ma_50) > 1 else float(ma_50.iloc[-1]))
            },
            "ma_200": {
                "value": current_ma200,
                "position": get_position(current_price, current_ma200),
                "trend": get_trend(float(ma_200.iloc[-1]), float(ma_200.iloc[-2]) if len(ma_200) > 1 else float(ma_200.iloc[-1]))
            },
            "vwap": {
                "value": current_vwap,
                "position": get_position(current_price, current_vwap),
                "trend": get_trend(float(vwap.iloc[-1]), float(vwap.iloc[-2]) if len(vwap) > 1 else float(vwap.iloc[-1]))
            }
        },
        "macd": {
            "macd_line": float(macd_line.iloc[-1]),
            "signal_line": float(signal_line.iloc[-1]),
            "histogram": float(macd_histogram.iloc[-1]),
            "trend": "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "bearish",
            "crossover_status": crossover_status
        },
        "rsi": {
            "value": current_rsi,
            "condition": "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral",
            "trend": get_trend(float(rsi.iloc[-1]), float(rsi.iloc[-2]) if len(rsi) > 1 else float(rsi.iloc[-1]))
        },
        "bollinger_bands": {
            "upper_band": float(upper_band.iloc[-1]),
            "middle_band": float(middle_band.iloc[-1]),
            "lower_band": float(lower_band.iloc[-1]),
            "band_width": float(band_width.iloc[-1]),
            "position": "upper" if current_price > upper_band.iloc[-1] else "lower" if current_price < lower_band.iloc[-1] else "middle",
            "squeeze_status": "tight" if band_width.iloc[-1] < band_width.mean() else "normal"
        },
        "technical_summary": {
            "short_term_outlook": "bullish" if current_price > current_ma50 and current_rsi > 50 else "bearish",
            "medium_term_outlook": "bullish" if current_price > current_ma200 else "bearish",
            "key_levels": {
                "support": support_levels,
                "resistance": resistance_levels
            },
            "key_signals": [
                f"Price is {get_position(current_price, current_ma50)} 50-day MA",
                f"RSI indicates {current_rsi:.1f} ({current_rsi > 70 and 'overbought' or current_rsi < 30 and 'oversold' or 'neutral'})",
                f"MACD {macd_line.iloc[-1] > signal_line.iloc[-1] and 'shows bullish momentum' or 'shows bearish momentum'}"
            ]
        }
    }
    
    return technical_data

def get_fallback_data(info: dict, ticker: str) -> dict:
    """Provide fallback data when OpenAI API call fails"""
    ny_timezone = pytz.timezone('America/New_York')
    current_time = datetime.now(ny_timezone)
    
    return {
        'exchange': info.get('exchange', 'N/A'),
        'current_price': info.get('currentPrice', 0.0),
        'price_time': current_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'company_name': info.get('longName', ticker),
        'about': info.get('longBusinessSummary', 'No company description available.'),
        'management_team': [
            {
                'name': 'Information not available',
            } for _ in range(3)
        ],
        'products_services': [
            {
                'category': 'Information not available',
                'description': 'N/A',
            } for _ in range(3)
        ]
    }

def fetch_company_basics(ticker: str, stock: yf.Ticker, openai_api_key: str) -> dict:
    """Fetch basic company information using yfinance and OpenAI API"""
    info = stock.info
    ny_timezone = pytz.timezone('America/New_York')
    current_time = datetime.now(ny_timezone)

    messages = [
        {
            "role": "system",
            "content": "You are a financial analyst. Provide detailed company information in a structured JSON format."
        },
        {
            "role": "user",
            "content": f"""
            Please analyze {ticker} ({info.get('longName', '')}) and provide information in this exact JSON format:
            {{
                "company_description": "Brief 300-word description of the company's business model and market position",
                "management_team": [
                    {{
                        "name": "Executive Full Name",
                        "position": "Current Position"
                    }},
                    {{
                        "name": "Executive Full Name",
                        "position": "Current Position"
                    }},
                    {{
                        "name": "Executive Full Name",
                        "position": "Current Position"
                    }}
                ],
                "products_services": [
                    {{
                        "category": "Main Product/Service Category",
                        "description": "50 word description of the category"
                    }},
                    {{
                        "category": "Secondary Product/Service Category",
                        "description": "50 word escription of the category"

                    }},
                    {{
                        "category": "Tertiary Product/Service Category",
                        "description": "50 word description of the category"

                    }}
                ]
            }}
            
            Ensure the response is valid JSON and maintain exact field names as shown.
            """
        }
    ]

    try:
        openai.api_key = openai_api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Changed from gpt-4o-mini to a valid model
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        if not response.choices:
            print("No response choices available")
            return get_fallback_data(info, ticker)
            
        try:
            content = response.choices[0].message.content
            # Clean the response string - remove any potential markdown formatting
            content = content.replace('```json', '').replace('```', '').strip()
            openai_data = json.loads(content)
            
            return {
                'exchange': info.get('exchange', 'N/A'),
                'current_price': info.get('currentPrice', 0),
                'price_time': current_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'company_name': info.get('longName', ticker),
                'about': openai_data.get('company_description', 'No description available'),
                'management_team': openai_data.get('management_team', [])[:3],
                'products_services': openai_data.get('products_services', [])[:3]
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Received content: {content}")
            return get_fallback_data(info, ticker)
            
    except Exception as e:
        print(f"Error in OpenAI API call: {str(e)}")
        return get_fallback_data(info, ticker)
    

class StockReportTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        self.page_width, self.page_height = letter
        
        # Define frames for main content
        content_frame = Frame(
            self.leftMargin, 
            self.bottomMargin + 20,
            self.width, 
            self.height - 20,
            id='content'
        )
        
        # Define frame for footer
        footer_frame = Frame(
            self.leftMargin, 
            self.bottomMargin,
            self.width,
            20,
            id='footer'
        )
        
        # Create page template
        template = PageTemplate(
            'normal',
            [content_frame, footer_frame],
            onPage=self.add_footer
        )
        self.addPageTemplates([template])
    
    def add_footer(self, canvas, doc):
        canvas.saveState()
        # Use standard Helvetica instead of Helvetica-Italic
        canvas.setFont('Helvetica', 8)

        canvas.drawString(50, 30, "Data Source: Yahoo Finance")
        
        canvas.restoreState()


def add_company_basics_section(story: List, basics: Dict):
    """Add company basics section using ReportLab"""
    styles = getSampleStyleSheet()
    
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=8
    )
    
    subsection_title = ParagraphStyle(
        'SubsectionTitle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=4
    )
    
    normal_text = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4
    )
    
    # Section Title
    story.append(Paragraph('1. Company Basics', section_title))
    story.append(Spacer(1, 0.1*inch))
    
    # Trading Information
    story.append(Paragraph('Trading Information', subsection_title))
    trading_data = [
        [f"Exchange: {basics['exchange']}", f"Price: ${basics['current_price']:.2f}", f"As of: {basics['price_time']}"]
    ]
    trading_table = Table(trading_data, colWidths=[2*inch, 2*inch, 3*inch])
    trading_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(trading_table)
    story.append(Spacer(1, 0.1*inch))
    
    # Company Description
    story.append(Paragraph('About the Company', subsection_title))
    story.append(Paragraph(basics['about'], normal_text))
    story.append(Spacer(1, 0.1*inch))
    
    # Management Team
    story.append(Paragraph('Management Team', subsection_title))
    for executive in basics['management_team']:
        story.append(Paragraph(
            f"<b>{executive['name']} - {executive['position']}</b>",
            normal_text
        ))
    story.append(Spacer(1, 0.1*inch))
    
    # Products/Services
    story.append(Paragraph('Products and Services', subsection_title))
    for category in basics['products_services']:
        story.append(Paragraph(f"<b>{category['category']}</b>", normal_text))
        story.append(Paragraph(category['description'], normal_text))

    
    story.append(PageBreak())

def add_analysis_summary_section(story: List, summary: Dict[str, str]):
    """Add analysis summary section using ReportLab with enhanced formatting"""
    styles = getSampleStyleSheet()
    
    # Main section title style
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=0.3*inch,
        spaceBefore=0.2*inch,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#1a365d'),  # Dark blue
        leading=16
    )
    
    # Subsection title style
    subsection_title = ParagraphStyle(
        'SubsectionTitle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=0.15*inch,
        spaceBefore=0.1*inch,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#2c5282'),  # Medium blue
        leading=14
    )
    
    # Body text style
    body_text = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=0.05*inch,
        spaceAfter=0.1*inch,
        fontName='Helvetica',
        leading=14,
        alignment=TA_JUSTIFY,
        firstLineIndent=20
    )
    
    # Bullet point style
    bullet_style = ParagraphStyle(
        'BulletPoint',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=0.05*inch,
        spaceAfter=0.05*inch,
        fontName='Helvetica',
        leading=14,
        leftIndent=20,
        bulletIndent=10
    )
    
    def sanitize_text(text: str) -> str:
        """Clean and sanitize text for PDF rendering"""
        if not isinstance(text, str):
            text = str(text)
            
        replacements = {
            '\u2018': "'",  # Left single quotation
            '\u2019': "'",  # Right single quotation
            '\u201C': '"',  # Left double quotation
            '\u201D': '"',  # Right double quotation
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2022': '*',  # Bullet
            '\u00A0': ' ',  # Non-breaking space
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&apos;'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
    
    def format_analysis_text(text: str) -> List[str]:
        """Format analysis text into properly structured paragraphs and bullet points"""
        if not text:
            return []
            
        text = sanitize_text(text)
        formatted_paragraphs = []
        current_paragraph = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                if current_paragraph:
                    formatted_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            elif line.startswith(('•', '-', '*')):
                if current_paragraph:
                    formatted_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                bullet_text = line.lstrip('•-* ')
                formatted_paragraphs.append(f'• {bullet_text}')
            elif line.startswith('**') and line.endswith('**'):
                if current_paragraph:
                    formatted_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_paragraphs.append(f'<b>{line[2:-2]}</b>')
            else:
                current_paragraph.append(line)
        
        if current_paragraph:
            formatted_paragraphs.append(' '.join(current_paragraph))
        
        return formatted_paragraphs
    
    # Add main section title
    story.append(Paragraph('2. Analysis Summary', section_title))
    story.append(Spacer(1, 0.1*inch))
    
    try:
        # Add Fundamental Analysis section
        story.append(Paragraph('Fundamental Analysis', subsection_title))
        fundamental_paragraphs = format_analysis_text(summary.get('fundamental_analysis', ''))
        
        for paragraph in fundamental_paragraphs:
            if paragraph.startswith('•'):
                story.append(Paragraph(paragraph, bullet_style))
            else:
                story.append(Paragraph(paragraph, body_text))
        
        story.append(Spacer(1, 0.1*inch))
        
        # Add Technical Analysis section
        story.append(Paragraph('Technical Analysis', subsection_title))
        technical_paragraphs = format_analysis_text(summary.get('technical_analysis', ''))
        
        for paragraph in technical_paragraphs:
            if paragraph.startswith('•'):
                story.append(Paragraph(paragraph, bullet_style))
            else:
                story.append(Paragraph(paragraph, body_text))
        
    except Exception as e:
        # Add error message to PDF if text processing fails
        error_style = ParagraphStyle(
            'ErrorText',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.red
        )
        story.append(Paragraph(f"Error processing text: {str(e)}", error_style))
    
    # Add final spacing
    story.append(Spacer(1, 0.3*inch))

def generate_stock_report(ticker: str, openai_api_key: str, output_dir: str) -> str:
    """Generate a complete stock analysis report using ReportLab"""
    try:
        # Setup
        output_file = f"{ticker}_analysis.pdf"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        
        class NumberedCanvas(canvas.Canvas):
            def __init__(self, *args, **kwargs):
                canvas.Canvas.__init__(self, *args, **kwargs)
                self._saved_page_states = []

            def showPage(self):
                self._saved_page_states.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                num_pages = len(self._saved_page_states)
                for state in self._saved_page_states:
                    self.__dict__.update(state)
                    self.draw_page_number(num_pages)
                    canvas.Canvas.showPage(self)
                canvas.Canvas.save(self)

            def draw_page_number(self, page_count):
                page = self._pageNumber
                self.setFont("Helvetica", 8)
                self.drawRightString(self._pagesize[0] - 50, 30, 
                                   f"Page {page}")
                if page == 2:
                    self.drawString(50, 30, "Data Source: Yahoo Finance")
        
        # Initialize document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Initialize stock
        stock = yf.Ticker(ticker)
        
        # Create story (content) list
        story = []
        
        # Add title
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=20,
            alignment=1
        )
        story.append(Paragraph(f"Stock Analysis Report: {ticker}", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Fetch and add company basics
        basics = fetch_company_basics(ticker, stock, openai_api_key)
        add_company_basics_section(story, basics)
        
        # Fetch and add analysis summary
        summary = fetch_analysis_summary(ticker, stock, openai_api_key)
        add_analysis_summary_section(story, summary)
        
        # Build PDF
        doc.build(story, canvasmaker=NumberedCanvas)
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Failed to generate report for {ticker}: {str(e)}")

    
def main():
    dotenv.load_dotenv()
    print("API Key loaded:", os.getenv('OPENAI_API_KEY') is not None)
    # Get API key from environment variable
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    try:
        ticker = input("Enter ticker symbol: ").upper()
        output_dir = f"analysis_pdfs_{ticker}"
        
        output_path = generate_stock_report(
            ticker=ticker,
            openai_api_key=OPENAI_API_KEY,
            output_dir=output_dir
        )
        
        print(f"\nStock analysis report generated successfully!")
        print(f"Report location: {output_path}")
        
    except Exception as e:
        print(f"\nError: Failed to generate report")
        print(f"Details: {str(e)}")

if __name__ == "__main__":
    main()