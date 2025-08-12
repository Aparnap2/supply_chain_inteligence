"""
API Data Collector for External Data Sources

This module implements data collection from external APIs including
World Bank economic indicators, weather data, and trade information.

Requirements covered: 1.3, 1.5
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from models.supplier import Supplier


logger = logging.getLogger(__name__)


class APIDataCollector:
    """
    Collects data from external APIs for supply chain risk assessment.
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize the API data collector.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.timeout = timeout
        self.max_retries = max_retries
        
        # API endpoints
        self.world_bank_base_url = "https://api.worldbank.org/v2"
        self.openweather_base_url = "https://api.openweathermap.org/data/2.5"
        
        # Country code mapping for APIs
        self.country_codes = {
            'United States': 'US',
            'China': 'CN',
            'Germany': 'DE',
            'Japan': 'JP',
            'United Kingdom': 'GB',
            'France': 'FR',
            'India': 'IN',
            'Brazil': 'BR',
            'Canada': 'CA',
            'Russia': 'RU',
            'South Korea': 'KR',
            'Australia': 'AU',
            'Mexico': 'MX',
            'Indonesia': 'ID',
            'Netherlands': 'NL',
            'Saudi Arabia': 'SA',
            'Turkey': 'TR',
            'Taiwan': 'TW',
            'Belgium': 'BE',
            'Argentina': 'AR',
            'Ireland': 'IE',
            'Israel': 'IL',
            'Thailand': 'TH',
            'Nigeria': 'NG',
            'Egypt': 'EG',
            'South Africa': 'ZA',
            'Philippines': 'PH',
            'Bangladesh': 'BD',
            'Vietnam': 'VN',
            'Chile': 'CL',
            'Finland': 'FI',
            'Malaysia': 'MY',
            'Singapore': 'SG',
            'New Zealand': 'NZ',
            'Norway': 'NO',
            'United Arab Emirates': 'AE',
            'Ukraine': 'UA',
            'Morocco': 'MA',
            'Kenya': 'KE',
            'Ethiopia': 'ET',
            'Ghana': 'GH',
            'Tanzania': 'TZ',
            'Uganda': 'UG',
            'Zambia': 'ZM',
            'Zimbabwe': 'ZW'
        }
    
    async def _make_request(self, session: aiohttp.ClientSession, 
                          url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make HTTP request with retry logic.
        
        Args:
            session: aiohttp session
            url: Request URL
            params: Query parameters
            
        Returns:
            Response data or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, params=params, timeout=self.timeout) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Request failed with status {response.status}: {url}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1}): {url}")
            except Exception as e:
                logger.warning(f"Request error (attempt {attempt + 1}): {url} - {str(e)}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"All retry attempts failed for: {url}")
        return None
    
    async def collect_economic_indicators(self, suppliers: List[Supplier]) -> Dict[str, Any]:
        """
        Collect economic indicators from World Bank API.
        
        Args:
            suppliers: List of suppliers to collect data for
            
        Returns:
            Dictionary with economic indicators by country
        """
        logger.info(f"Collecting economic indicators for {len(suppliers)} suppliers")
        
        # Get unique countries
        countries = list(set(supplier.country for supplier in suppliers))
        economic_data = {}
        
        async with aiohttp.ClientSession() as session:
            for country in countries:
                country_code = self.country_codes.get(country)
                if not country_code:
                    logger.warning(f"No country code mapping for: {country}")
                    continue
                
                try:
                    # Collect GDP data
                    gdp_url = f"{self.world_bank_base_url}/country/{country_code}/indicator/NY.GDP.MKTP.CD"
                    gdp_params = {
                        'format': 'json',
                        'date': f"{datetime.now().year - 2}:{datetime.now().year - 1}",
                        'per_page': 10
                    }
                    
                    gdp_data = await self._make_request(session, gdp_url, gdp_params)
                    
                    # Collect inflation data
                    inflation_url = f"{self.world_bank_base_url}/country/{country_code}/indicator/FP.CPI.TOTL.ZG"
                    inflation_data = await self._make_request(session, inflation_url, gdp_params)
                    
                    # Collect unemployment data
                    unemployment_url = f"{self.world_bank_base_url}/country/{country_code}/indicator/SL.UEM.TOTL.ZS"
                    unemployment_data = await self._make_request(session, unemployment_url, gdp_params)
                    
                    # Process and store data
                    country_indicators = {
                        'country': country,
                        'country_code': country_code,
                        'gdp': self._extract_latest_value(gdp_data),
                        'inflation': self._extract_latest_value(inflation_data),
                        'unemployment': self._extract_latest_value(unemployment_data),
                        'collected_at': datetime.now().isoformat()
                    }
                    
                    economic_data[country] = country_indicators
                    logger.debug(f"Collected economic data for {country}")
                    
                except Exception as e:
                    logger.error(f"Failed to collect economic data for {country}: {str(e)}")
                    economic_data[country] = {
                        'country': country,
                        'error': str(e),
                        'collected_at': datetime.now().isoformat()
                    }
        
        logger.info(f"Economic data collection completed for {len(economic_data)} countries")
        return economic_data
    
    async def collect_weather_risk_data(self, suppliers: List[Supplier]) -> Dict[str, Any]:
        """
        Collect weather risk data for supplier locations.
        
        Args:
            suppliers: List of suppliers to collect data for
            
        Returns:
            Dictionary with weather risk data by country
        """
        logger.info(f"Collecting weather risk data for {len(suppliers)} suppliers")
        
        # Note: This is a simplified implementation
        # In production, you would use actual weather APIs with API keys
        
        countries = list(set(supplier.country for supplier in suppliers))
        weather_data = {}
        
        # Simulate weather risk data collection
        for country in countries:
            try:
                # In a real implementation, you would make actual API calls
                # For now, we'll create mock data based on known risk patterns
                risk_score = self._calculate_weather_risk_score(country)
                
                weather_data[country] = {
                    'country': country,
                    'weather_risk_score': risk_score,
                    'risk_factors': self._get_weather_risk_factors(country),
                    'seasonal_patterns': self._get_seasonal_patterns(country),
                    'collected_at': datetime.now().isoformat(),
                    'data_source': 'simulated'  # Would be actual API in production
                }
                
                logger.debug(f"Collected weather risk data for {country}")
                
            except Exception as e:
                logger.error(f"Failed to collect weather data for {country}: {str(e)}")
                weather_data[country] = {
                    'country': country,
                    'error': str(e),
                    'collected_at': datetime.now().isoformat()
                }
        
        logger.info(f"Weather risk data collection completed for {len(weather_data)} countries")
        return weather_data
    
    async def collect_trade_data(self, suppliers: List[Supplier]) -> Dict[str, Any]:
        """
        Collect trade flow data for supplier countries.
        
        Args:
            suppliers: List of suppliers to collect data for
            
        Returns:
            Dictionary with trade data by country
        """
        logger.info(f"Collecting trade data for {len(suppliers)} suppliers")
        
        countries = list(set(supplier.country for supplier in suppliers))
        trade_data = {}
        
        # Simulate trade data collection
        # In production, you would use actual trade APIs
        
        for country in countries:
            try:
                trade_data[country] = {
                    'country': country,
                    'trade_volume_usd': self._estimate_trade_volume(country),
                    'main_trading_partners': self._get_main_trading_partners(country),
                    'trade_balance': self._estimate_trade_balance(country),
                    'export_diversification': self._calculate_export_diversification(country),
                    'trade_risk_score': self._calculate_trade_risk_score(country),
                    'collected_at': datetime.now().isoformat(),
                    'data_source': 'simulated'  # Would be actual API in production
                }
                
                logger.debug(f"Collected trade data for {country}")
                
            except Exception as e:
                logger.error(f"Failed to collect trade data for {country}: {str(e)}")
                trade_data[country] = {
                    'country': country,
                    'error': str(e),
                    'collected_at': datetime.now().isoformat()
                }
        
        logger.info(f"Trade data collection completed for {len(trade_data)} countries")
        return trade_data
    
    def _extract_latest_value(self, api_response: Optional[Dict]) -> Optional[float]:
        """Extract the latest value from World Bank API response."""
        if not api_response or len(api_response) < 2:
            return None
        
        data_points = api_response[1]  # World Bank API returns [metadata, data]
        if not data_points:
            return None
        
        # Find the most recent non-null value
        for point in data_points:
            if point and point.get('value') is not None:
                return float(point['value'])
        
        return None
    
    def _calculate_weather_risk_score(self, country: str) -> float:
        """Calculate weather risk score based on country characteristics."""
        # Simplified risk scoring based on known climate patterns
        high_risk_countries = [
            'Philippines', 'Bangladesh', 'Myanmar', 'Vietnam', 'Thailand',
            'India', 'Pakistan', 'Indonesia', 'Japan', 'South Korea'
        ]
        
        medium_risk_countries = [
            'United States', 'China', 'Brazil', 'Australia', 'Mexico',
            'Turkey', 'Iran', 'Egypt', 'South Africa', 'Argentina'
        ]
        
        if country in high_risk_countries:
            return 0.8
        elif country in medium_risk_countries:
            return 0.5
        else:
            return 0.3
    
    def _get_weather_risk_factors(self, country: str) -> List[str]:
        """Get weather risk factors for a country."""
        risk_factors_map = {
            'Philippines': ['typhoons', 'flooding', 'landslides'],
            'Bangladesh': ['flooding', 'cyclones', 'monsoons'],
            'Japan': ['earthquakes', 'tsunamis', 'typhoons'],
            'United States': ['hurricanes', 'tornadoes', 'wildfires'],
            'Australia': ['bushfires', 'droughts', 'cyclones'],
            'India': ['monsoons', 'flooding', 'heat waves'],
            'China': ['flooding', 'typhoons', 'droughts']
        }
        
        return risk_factors_map.get(country, ['seasonal variations'])
    
    def _get_seasonal_patterns(self, country: str) -> Dict[str, str]:
        """Get seasonal risk patterns for a country."""
        return {
            'high_risk_season': 'varies by region',
            'low_risk_season': 'varies by region',
            'peak_risk_months': 'varies by region'
        }
    
    def _estimate_trade_volume(self, country: str) -> float:
        """Estimate trade volume for a country (in billions USD)."""
        # Simplified estimates based on economic size
        large_economies = {
            'United States': 4000.0,
            'China': 4500.0,
            'Germany': 3000.0,
            'Japan': 1400.0,
            'United Kingdom': 1200.0,
            'France': 1100.0
        }
        
        medium_economies = {
            'India': 800.0,
            'Brazil': 500.0,
            'Canada': 900.0,
            'Russia': 600.0,
            'South Korea': 1100.0,
            'Australia': 500.0
        }
        
        if country in large_economies:
            return large_economies[country]
        elif country in medium_economies:
            return medium_economies[country]
        else:
            return 100.0  # Default for smaller economies
    
    def _get_main_trading_partners(self, country: str) -> List[str]:
        """Get main trading partners for a country."""
        # Simplified trading partner data
        partners_map = {
            'United States': ['China', 'Canada', 'Mexico', 'Japan', 'Germany'],
            'China': ['United States', 'Japan', 'South Korea', 'Germany', 'Australia'],
            'Germany': ['United States', 'China', 'France', 'Netherlands', 'United Kingdom'],
            'Japan': ['China', 'United States', 'South Korea', 'Taiwan', 'Germany']
        }
        
        return partners_map.get(country, ['United States', 'China', 'Germany'])
    
    def _estimate_trade_balance(self, country: str) -> float:
        """Estimate trade balance for a country (in billions USD)."""
        # Simplified trade balance estimates
        surplus_countries = {
            'Germany': 250.0,
            'China': 400.0,
            'Japan': 50.0,
            'South Korea': 80.0,
            'Netherlands': 70.0
        }
        
        deficit_countries = {
            'United States': -800.0,
            'United Kingdom': -150.0,
            'India': -100.0,
            'Turkey': -50.0
        }
        
        if country in surplus_countries:
            return surplus_countries[country]
        elif country in deficit_countries:
            return deficit_countries[country]
        else:
            return 0.0  # Balanced trade
    
    def _calculate_export_diversification(self, country: str) -> float:
        """Calculate export diversification score (0-1, higher is more diversified)."""
        # Simplified diversification scores
        highly_diversified = [
            'Germany', 'United States', 'Japan', 'South Korea', 
            'France', 'United Kingdom', 'Italy'
        ]
        
        moderately_diversified = [
            'China', 'India', 'Brazil', 'Mexico', 'Turkey', 
            'Thailand', 'Malaysia', 'Indonesia'
        ]
        
        if country in highly_diversified:
            return 0.8
        elif country in moderately_diversified:
            return 0.6
        else:
            return 0.4  # Less diversified economies
    
    def _calculate_trade_risk_score(self, country: str) -> float:
        """Calculate overall trade risk score (0-1, higher is riskier)."""
        # Factors: political stability, trade disputes, sanctions, etc.
        high_risk_countries = [
            'Russia', 'Iran', 'North Korea', 'Venezuela', 'Myanmar'
        ]
        
        medium_risk_countries = [
            'Turkey', 'Argentina', 'Pakistan', 'Nigeria', 'Egypt'
        ]
        
        if country in high_risk_countries:
            return 0.9
        elif country in medium_risk_countries:
            return 0.6
        else:
            return 0.3  # Lower risk countries