"""
Web Search Tool for the Digital Clone AI Assistant.

Provides web search functionality using SerpAPI or fallback implementation.
"""

import os
import requests
import logging
from typing import Dict, Any, List
import json

logger = logging.getLogger(__name__)

TOOL_META = {
    "name": "web_search",
    "description": "执行网络搜索，返回相关网页的标题、摘要和链接",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询字符串"
            },
            "k": {
                "type": "integer",
                "description": "返回结果数量",
                "default": 5,
                "minimum": 1,
                "maximum": 10
            }
        },
        "required": ["query"],
        "additionalProperties": False
    },
    "strict": True,
    "executor_default": "chat",
    "complexity": "simple",
    "arg_hint": "query为搜索关键词；k控制返回结果数量(1-10)",
    "caller_snippet": "用于查找最新信息、调研数据或相关资源。优先使用此工具获取外部知识，避免主观推测。"
}

def _search_with_serpapi(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search using SerpAPI if available."""
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise ValueError("SerpAPI key not configured")

    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key,
        "num": k
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        if "organic_results" in data:
            for item in data["organic_results"][:k]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })

        return results

    except Exception as e:
        logger.warning(f"SerpAPI search failed: {e}")
        raise

def _search_with_duckduckgo(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Fallback search using DuckDuckGo (no API key required)."""
    try:
        # Use DuckDuckGo instant answers API
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []

        # Add instant answer if available
        if data.get("Answer"):
            results.append({
                "title": "Instant Answer",
                "link": data.get("AnswerURL", ""),
                "snippet": data.get("Answer", "")
            })

        # Add abstract if available
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "Abstract"),
                "link": data.get("AbstractURL", ""),
                "snippet": data.get("Abstract", "")
            })

        # Add related topics
        for topic in data.get("RelatedTopics", [])[:k-len(results)]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("FirstURL", "").split('/')[-1] if topic.get("FirstURL") else "Related Topic",
                    "link": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", "")
                })

        return results[:k]

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        raise

def run(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Execute web search tool.

    Args:
        query: Search query string
        k: Number of results to return

    Returns:
        Dictionary with ok/value/error fields
    """
    try:
        if not query or not query.strip():
            return {"ok": False, "error": "搜索查询不能为空"}

        query = query.strip()
        logger.info(f"Performing web search for: {query}")

        # Try SerpAPI first, fallback to DuckDuckGo
        try:
            results = _search_with_serpapi(query, k)
        except Exception:
            logger.info("Falling back to DuckDuckGo search")
            results = _search_with_duckduckgo(query, k)

        if not results:
            return {"ok": False, "error": "未找到搜索结果"}

        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", "")
            })

        logger.info(f"Web search returned {len(formatted_results)} results")
        return {"ok": True, "value": formatted_results}

    except Exception as e:
        error_msg = f"网络搜索失败: {str(e)}"
        logger.error(f"Web search error for '{query}': {e}")
        return {"ok": False, "error": error_msg}
