"""
Web Read Tool for the Digital Clone AI Assistant.

Extracts and processes web page content with denoising and chunking.
"""

import requests
import logging
from typing import Dict, Any, List
from urllib.parse import urlparse
import re
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)

TOOL_META = {
    "name": "web_read",
    "description": "获取网页内容，提取正文，去除噪声，返回结构化的文本片段",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要读取的网页URL"
            },
            "max_length": {
                "type": "integer",
                "description": "最大返回字符数",
                "default": 4000,
                "minimum": 500,
                "maximum": 10000
            }
        },
        "required": ["url"],
        "additionalProperties": False
    },
    "strict": True,
    "executor_default": "chat",
    "complexity": "simple",
    "arg_hint": "url为完整的网页地址；max_length控制返回内容长度(500-10000字符)",
    "caller_snippet": "用于读取网页详细内容，获取完整文章或特定页面的信息。结合web_search使用。"
}

def _is_valid_url(url: str) -> bool:
    """Validate URL format."""
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except:
        return False

def _extract_main_content(html: str, url: str) -> str:
    """
    Extract main content from HTML with noise removal.

    Uses BeautifulSoup to find main content areas and remove noise.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Remove unwanted elements
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
        tag.decompose()

    # Try to find main content containers
    content_selectors = [
        'main',
        '[role="main"]',
        '.content',
        '.article',
        '.post',
        '.entry',
        '#content',
        '#main',
        '.main-content'
    ]

    main_content = None
    for selector in content_selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break

    # Fallback to body if no main content found
    if not main_content:
        main_content = soup.body or soup

    # Extract text and clean it
    text = main_content.get_text(separator='\n', strip=True)

    # Clean up whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n\n'.join(lines)

    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text

def _chunk_text(text: str, max_chunk_length: int = 1000) -> List[str]:
    """Split text into manageable chunks."""
    if len(text) <= max_chunk_length:
        return [text]

    chunks = []
    paragraphs = text.split('\n\n')

    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_length:
            current_chunk += para + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + '\n\n'

            # If single paragraph is too long, split it
            if len(current_chunk) > max_chunk_length:
                words = current_chunk.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_chunk_length:
                        current_chunk += word + ' '
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word + ' '

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def run(url: str, max_length: int = 4000) -> Dict[str, Any]:
    """
    Execute web read tool.

    Args:
        url: URL to read
        max_length: Maximum character length to return

    Returns:
        Dictionary with ok/value/error fields
    """
    try:
        if not url or not url.strip():
            return {"ok": False, "error": "URL不能为空"}

        url = url.strip()
        if not _is_valid_url(url):
            return {"ok": False, "error": "无效的URL格式"}

        logger.info(f"Reading web page: {url}")

        # Fetch webpage
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        if 'text/html' not in response.headers.get('content-type', '').lower():
            return {"ok": False, "error": "URL不是有效的HTML页面"}

        # Extract content
        html_content = response.text
        main_text = _extract_main_content(html_content, url)

        if not main_text or len(main_text.strip()) < 100:
            return {"ok": False, "error": "无法提取页面主要内容"}

        # Chunk and limit length
        chunks = _chunk_text(main_text)
        combined_text = '\n\n---\n\n'.join(chunks)

        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length-3] + "..."

        result = {
            "url": url,
            "title": "",  # Could extract from HTML title
            "content": combined_text,
            "chunks": len(chunks),
            "total_length": len(combined_text)
        }

        # Try to get page title
        soup = BeautifulSoup(html_content, 'html.parser')
        title_tag = soup.find('title')
        if title_tag:
            result["title"] = title_tag.get_text().strip()

        logger.info(f"Web read extracted {len(combined_text)} characters in {len(chunks)} chunks")
        return {"ok": True, "value": result}

    except requests.exceptions.Timeout:
        error_msg = "网页读取超时"
        logger.error(f"Web read timeout for {url}")
        return {"ok": False, "error": error_msg}

    except requests.exceptions.RequestException as e:
        error_msg = f"网络请求失败: {str(e)}"
        logger.error(f"Web read request error for {url}: {e}")
        return {"ok": False, "error": error_msg}

    except Exception as e:
        error_msg = f"网页读取失败: {str(e)}"
        logger.error(f"Web read error for {url}: {e}")
        return {"ok": False, "error": error_msg}
