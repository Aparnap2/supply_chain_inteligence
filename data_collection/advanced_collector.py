"""
Advanced data collector using Crawl4AI for intelligent web scraping.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid

from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    RegexExtractionStrategy,
    JsonCssExtractionStrategy
)
from crawl4ai import LLMConfig
from crawl4ai.chunking_strategy import RegexChunking
from crawl4ai.content_filter_strategy import PruningContentFilter
from pydantic import ValidationError

from models.supplier import Supplier
from models.scraped_data import ScrapedData


logger = logging.getLogger(__name__)


class AdvancedDataCollector:
    """
    Advanced data collector with Crawl4AI integration for intelligent web scraping.
    Implements multiple extraction strategies with fallback mechanisms.
    """
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        """
        Initialize the advanced data collector.
        
        Args:
            api_key: OpenAI API key for LLM extraction
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.crawler = None
        
        # Risk-related keywords for content analysis
        self.risk_keywords = [
            'bankruptcy', 'financial crisis', 'supply chain disruption',
            'regulatory changes', 'sanctions', 'trade war', 'shortage',
            'delay', 'quality issues', 'recall', 'investigation',
            'lawsuit', 'cyber attack', 'data breach', 'strike',
            'natural disaster', 'pandemic', 'lockdown', 'closure'
        ]
        
        # Sentiment indicators
        self.negative_indicators = [
            'decline', 'decrease', 'loss', 'problem', 'issue', 'concern',
            'risk', 'threat', 'challenge', 'difficulty', 'failure'
        ]
        
        self.positive_indicators = [
            'growth', 'increase', 'success', 'expansion', 'improvement',
            'opportunity', 'achievement', 'progress', 'innovation'
        ]
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.crawler = AsyncWebCrawler(verbose=True)
        await self.crawler.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
    
    def _create_llm_extraction_strategy(self) -> LLMExtractionStrategy:
        """Create LLM extraction strategy for intelligent content parsing."""
        schema = {
            "type": "object",
            "properties": {
                "company_info": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "industry": {"type": "string"},
                        "location": {"type": "string"}
                    }
                },
                "financial_indicators": {
                    "type": "object",
                    "properties": {
                        "revenue": {"type": "string"},
                        "profit": {"type": "string"},
                        "financial_health": {"type": "string"}
                    }
                },
                "risk_factors": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "news_sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative"]
                },
                "key_events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "event": {"type": "string"},
                            "date": {"type": "string"},
                            "impact": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        llm_config = LLMConfig(
            provider="openai/gpt-4o-mini",
            api_token=self.api_key
        ) if self.api_key else None
        
        return LLMExtractionStrategy(
            llm_config=llm_config,
            schema=schema,
            extraction_type="schema",
            instruction=(
                "Extract comprehensive information about the company including "
                "financial indicators, risk factors, recent news sentiment, and key events. "
                "Focus on supply chain related information, operational status, "
                "and any potential risk indicators."
            )
        )
    
    def _create_regex_extraction_strategy(self) -> RegexExtractionStrategy:
        """Create regex extraction strategy for structured data extraction."""
        patterns = {
            "financial_data": r"(?:revenue|sales|profit|loss)[\s:]+\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
            "employee_count": r"(?:employees?|staff|workforce)[\s:]+(\d{1,6})",
            "locations": r"(?:located|based|headquarters?)[\s:]+in\s+([A-Za-z\s,]+)",
            "contact_info": r"(?:phone|tel|contact)[\s:]+(\+?[\d\s\-\(\)]+)",
            "email": r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "risk_mentions": f"({'|'.join(self.risk_keywords)})",
        }
        
        return RegexExtractionStrategy(patterns=patterns)
    
    def _create_css_extraction_strategy(self) -> JsonCssExtractionStrategy:
        """Create CSS selector extraction strategy for structured content."""
        schema = {
            "name": "company_data",
            "baseSelector": "body",
            "fields": [
                {
                    "name": "title",
                    "selector": "title, h1",
                    "type": "text"
                },
                {
                    "name": "description",
                    "selector": "meta[name='description']",
                    "type": "attribute",
                    "attribute": "content"
                },
                {
                    "name": "news_headlines",
                    "selector": "h1, h2, h3, .headline, .news-title",
                    "type": "text",
                    "multiple": True
                },
                {
                    "name": "content_paragraphs",
                    "selector": "p, .content, .article-body",
                    "type": "text",
                    "multiple": True
                },
                {
                    "name": "contact_info",
                    "selector": ".contact, .address, .phone, .email",
                    "type": "text",
                    "multiple": True
                }
            ]
        }
        
        return JsonCssExtractionStrategy(schema)
    
    def _calculate_relevance_score(self, content: str, supplier: Supplier) -> float:
        """Calculate relevance score based on content analysis."""
        content_lower = content.lower()
        score = 0.0
        
        # Check for supplier name mentions
        if supplier.name.lower() in content_lower:
            score += 0.3
        
        # Check for industry-related terms
        if supplier.industry.lower() in content_lower:
            score += 0.2
        
        # Check for location mentions
        if supplier.country.lower() in content_lower:
            score += 0.1
        
        # Check for risk-related keywords
        risk_mentions = sum(1 for keyword in self.risk_keywords if keyword in content_lower)
        score += min(risk_mentions * 0.1, 0.4)
        
        return min(score, 1.0)
    
    def _calculate_quality_score(self, content: str, extraction_method: str) -> float:
        """Calculate data quality score based on content characteristics."""
        if not content or len(content.strip()) < 50:
            return 0.1
        
        score = 0.5  # Base score
        
        # Length-based scoring
        if len(content) > 500:
            score += 0.2
        elif len(content) > 200:
            score += 0.1
        
        # Structure-based scoring
        if any(char in content for char in ['.', '!', '?']):
            score += 0.1
        
        # Method-based scoring
        if extraction_method == "LLMExtractionStrategy":
            score += 0.2
        elif extraction_method == "JsonCssExtractionStrategy":
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_sentiment_score(self, content: str) -> float:
        """Calculate sentiment score based on keyword analysis."""
        content_lower = content.lower()
        
        positive_count = sum(1 for word in self.positive_indicators if word in content_lower)
        negative_count = sum(1 for word in self.negative_indicators if word in content_lower)
        
        if positive_count == 0 and negative_count == 0:
            return 0.0  # Neutral
        
        total_count = positive_count + negative_count
        sentiment = (positive_count - negative_count) / total_count
        
        return max(-1.0, min(1.0, sentiment))
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract relevant keywords from content."""
        content_lower = content.lower()
        keywords = []
        
        # Extract risk-related keywords
        for keyword in self.risk_keywords:
            if keyword in content_lower:
                keywords.append(keyword)
        
        # Extract industry-specific terms (simple approach)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content)
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            if word_lower not in ['this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'said']:
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # Get top frequent words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords.extend([word for word, _ in top_words])
        
        return list(set(keywords))
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content (simple approach)."""
        # Simple regex-based entity extraction
        entities = []
        
        # Company names (capitalized words)
        company_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Ltd|LLC|Company|Group)\b'
        companies = re.findall(company_pattern, content)
        entities.extend(companies)
        
        # Locations (capitalized words before common location indicators)
        location_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?=\s+(?:city|state|country|region))'
        locations = re.findall(location_pattern, content)
        entities.extend(locations)
        
        return list(set(entities))
    
    def _extract_risk_indicators(self, content: str) -> List[str]:
        """Extract risk indicators from content."""
        content_lower = content.lower()
        indicators = []
        
        for keyword in self.risk_keywords:
            if keyword in content_lower:
                indicators.append(keyword)
        
        return indicators
    
    async def _scrape_with_strategy(self, url: str, strategy, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Scrape URL with a specific extraction strategy."""
        try:
            result = await self.crawler.arun(
                url=url,
                extraction_strategy=strategy,
                chunking_strategy=RegexChunking(),
                content_filter=PruningContentFilter(threshold=0.48),
                bypass_cache=True
            )
            
            if result.success:
                # Try extracted content first, then fall back to markdown/html
                content = None
                if result.extracted_content:
                    content = result.extracted_content
                elif result.markdown:
                    content = result.markdown
                elif result.html:
                    content = result.html
                
                if content:
                    if strategy_name == "LLMExtractionStrategy":
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            return {"raw_content": content}
                    else:
                        return {"raw_content": content, "html": result.html}
            
            return None
            
        except Exception as e:
            logger.warning(f"Strategy {strategy_name} failed for {url}: {str(e)}")
            return None
    
    async def scrape_supplier_website(self, supplier: Supplier) -> Optional[ScrapedData]:
        """
        Scrape a single supplier website with multiple extraction strategies.
        
        Args:
            supplier: Supplier object with website information
            
        Returns:
            ScrapedData object or None if scraping fails
        """
        if not supplier.website:
            logger.warning(f"No website provided for supplier {supplier.id}")
            return None
        
        url = supplier.website
        strategies = [
            (self._create_llm_extraction_strategy(), "LLMExtractionStrategy"),
            (self._create_css_extraction_strategy(), "JsonCssExtractionStrategy"),
            (self._create_regex_extraction_strategy(), "RegexExtractionStrategy")
        ]
        
        for attempt in range(self.max_retries):
            for strategy, strategy_name in strategies:
                try:
                    logger.info(f"Attempting {strategy_name} for {url} (attempt {attempt + 1})")
                    
                    result = await self._scrape_with_strategy(url, strategy, strategy_name)
                    
                    if result:
                        # Extract content for analysis
                        if "raw_content" in result:
                            content = result["raw_content"]
                        else:
                            content = json.dumps(result, indent=2)
                        
                        # Ensure content is not empty
                        if not content or len(content.strip()) == 0:
                            content = f"No content extracted using {strategy_name}"
                        
                        # Create ScrapedData object
                        scraped_data = ScrapedData(
                            data_id=f"DATA_{uuid.uuid4().hex[:8].upper()}",
                            supplier_id=supplier.id,
                            source_url=url,
                            content=content,
                            extraction_method=strategy_name,
                            relevance_score=self._calculate_relevance_score(content, supplier),
                            quality_score=self._calculate_quality_score(content, strategy_name),
                            sentiment_score=self._calculate_sentiment_score(content),
                            keywords=self._extract_keywords(content),
                            entities=self._extract_entities(content),
                            risk_indicators=self._extract_risk_indicators(content),
                            metadata={
                                "extraction_attempt": attempt + 1,
                                "strategy_used": strategy_name,
                                "content_length": len(content),
                                "url_domain": url.split('/')[2] if '/' in url else url
                            },
                            processing_status="processed"
                        )
                        
                        logger.info(f"Successfully scraped {url} using {strategy_name}")
                        return scraped_data
                        
                except Exception as e:
                    logger.error(f"Error with {strategy_name} for {url}: {str(e)}")
                    continue
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # All strategies failed
        logger.error(f"All extraction strategies failed for {url}")
        return ScrapedData(
            data_id=f"DATA_{uuid.uuid4().hex[:8].upper()}",
            supplier_id=supplier.id,
            source_url=url,
            content="Failed to extract content",
            extraction_method="failed",
            relevance_score=0.0,
            quality_score=0.0,
            processing_status="failed",
            error_message=f"All extraction strategies failed after {self.max_retries} attempts"
        )
    
    async def scrape_supplier_websites(self, suppliers: List[Supplier]) -> List[ScrapedData]:
        """
        Scrape multiple supplier websites concurrently.
        
        Args:
            suppliers: List of Supplier objects
            
        Returns:
            List of ScrapedData objects
        """
        logger.info(f"Starting to scrape {len(suppliers)} supplier websites")
        
        # Filter suppliers with websites
        suppliers_with_websites = [s for s in suppliers if s.website]
        logger.info(f"Found {len(suppliers_with_websites)} suppliers with websites")
        
        if not suppliers_with_websites:
            return []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def scrape_with_semaphore(supplier):
            async with semaphore:
                return await self.scrape_supplier_website(supplier)
        
        # Execute scraping tasks concurrently
        tasks = [scrape_with_semaphore(supplier) for supplier in suppliers_with_websites]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        scraped_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception scraping supplier {suppliers_with_websites[i].id}: {result}")
            elif result is not None:
                scraped_data.append(result)
        
        logger.info(f"Successfully scraped {len(scraped_data)} websites")
        return scraped_data
    
    def get_scraping_statistics(self, scraped_data: List[ScrapedData]) -> Dict[str, Any]:
        """
        Generate statistics about scraping results.
        
        Args:
            scraped_data: List of ScrapedData objects
            
        Returns:
            Dictionary with scraping statistics
        """
        if not scraped_data:
            return {"total": 0, "successful": 0, "failed": 0}
        
        successful = [d for d in scraped_data if d.processing_status == "processed"]
        failed = [d for d in scraped_data if d.processing_status == "failed"]
        
        avg_quality = sum(d.quality_score for d in successful) / len(successful) if successful else 0
        avg_relevance = sum(d.relevance_score for d in successful) / len(successful) if successful else 0
        
        extraction_methods = {}
        for data in successful:
            method = data.extraction_method
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
        
        return {
            "total": len(scraped_data),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(scraped_data) if scraped_data else 0,
            "average_quality_score": avg_quality,
            "average_relevance_score": avg_relevance,
            "extraction_methods_used": extraction_methods,
            "total_risk_indicators": sum(len(d.risk_indicators) for d in successful),
            "suppliers_with_risk_signals": len([d for d in successful if d.has_risk_signals()])
        }


# Convenience function for easy usage
async def collect_supplier_data(suppliers: List[Supplier], api_key: Optional[str] = None) -> List[ScrapedData]:
    """
    Convenience function to collect data from supplier websites.
    
    Args:
        suppliers: List of Supplier objects
        api_key: OpenAI API key for LLM extraction
        
    Returns:
        List of ScrapedData objects
    """
    async with AdvancedDataCollector(api_key=api_key) as collector:
        return await collector.scrape_supplier_websites(suppliers)