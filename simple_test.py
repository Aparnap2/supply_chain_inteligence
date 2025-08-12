"""
Simple test to debug crawl4ai extraction.
"""

import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import RegexExtractionStrategy


async def test_simple_extraction():
    """Test simple extraction to see what we get."""
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url="https://httpbin.org/html",
            extraction_strategy=RegexExtractionStrategy(patterns={"title": r"<title>(.*?)</title>"})
        )
        
        print(f"Success: {result.success}")
        print(f"HTML length: {len(result.html) if result.html else 0}")
        print(f"Markdown length: {len(result.markdown) if result.markdown else 0}")
        print(f"Extracted content: {result.extracted_content}")
        print(f"Links: {len(result.links) if result.links else 0}")
        
        if result.html:
            print(f"HTML preview: {result.html[:200]}...")


if __name__ == "__main__":
    asyncio.run(test_simple_extraction())