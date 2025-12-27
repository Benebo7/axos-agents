"""Tools for fetching and analyzing crypto news"""
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt.tool_node import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import List, Dict, Union, Optional, Annotated
import time
import json
import httpx
import os

from tools.stream_utils import (
    get_stream_writer_safe,
    stream_custom_event,
    stream_progress_event,
)

# CryptoNews API configuration
CRYPTONEWS_API_KEY = os.getenv("CRYPTONEWS_API_KEY") or os.getenv("VITE_CRYPTONEWS_API_KEY")
CRYPTONEWS_BASE_URL = "https://cryptonews-api.com/api/v1"


@tool
def generate_news_brief(
    limit: int = 20,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """
    Generates a complete AI Brief with news data, sentiment analysis, and summary.
    This is the MAIN tool for the 'brief' mode - it does everything in one call.

    Args:
        limit: Number of news items to fetch (default: 20)

    Returns:
        Command with complete brief data including news, sentiment, and summary
    """
    print(f"📰 generate_news_brief started (limit: {limit})")

    writer = get_stream_writer_safe()
    
    # Step 1: Fetch news
    stream_progress_event(
        writer,
        stage="fetch_news",
        status="running",
        message="Fetching latest crypto news",
        order=1,
        total_steps=3,
    )

    news = []
    try:
        with httpx.Client(timeout=30.0) as client:
            params = {
                "token": CRYPTONEWS_API_KEY,
                "items": limit,
                "page": 1,
                "sortby": "date",
                "section": "general",
            }
            response = client.get(f"{CRYPTONEWS_BASE_URL}/category", params=params)
            response.raise_for_status()
            data = response.json()
            news = data.get("data", [])
            print(f"   ✅ Fetched {len(news)} news items")
    except Exception as e:
        print(f"   ❌ Error fetching news: {e}")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error fetching news: {str(e)}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # Send progress event
    stream_custom_event(
        writer,
        event_type="news_fetched",
        data={
            "count": len(news),
            "message": f"Fetched {len(news)} news articles",
        },
    )

    # Step 2: Analyze sentiment
    stream_progress_event(
        writer,
        stage="analyze_sentiment",
        status="running",
        message="Analyzing news sentiment",
        order=2,
        total_steps=3,
    )

    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for item in news:
        sentiment = item.get("sentiment", "Neutral")
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
        else:
            sentiment_counts["Neutral"] += 1

    total = len(news) or 1
    positive_pct = (sentiment_counts["Positive"] / total) * 100
    negative_pct = (sentiment_counts["Negative"] / total) * 100

    if positive_pct >= 60:
        overall = "bullish"
        confidence = int(positive_pct)
    elif negative_pct >= 60:
        overall = "bearish"
        confidence = int(negative_pct)
    else:
        overall = "neutral"
        confidence = int(100 - abs(positive_pct - negative_pct))

    sentiment_analysis = {
        "total_analyzed": len(news),
        "positive": sentiment_counts["Positive"],
        "negative": sentiment_counts["Negative"],
        "neutral": sentiment_counts["Neutral"],
        "positive_percentage": round(positive_pct, 1),
        "negative_percentage": round(negative_pct, 1),
        "overall_sentiment": overall,
        "confidence": confidence,
        "timestamp": int(time.time()),
    }
    print(f"   ✅ Sentiment: {overall} ({confidence}%)")

    # Send sentiment event
    stream_custom_event(
        writer,
        event_type="sentiment_analyzed",
        data=sentiment_analysis,
    )

    # Step 3: Generate summary
    stream_progress_event(
        writer,
        stage="generate_summary",
        status="running",
        message="Generating summary",
        order=3,
        total_steps=3,
    )

    # Get top headlines
    headlines = [n.get("title", "") for n in news[:10]]
    sources = list(set(n.get("source_name", "") for n in news[:10] if n.get("source_name")))
    tickers = []
    for n in news:
        tickers.extend(n.get("tickers", []))
    top_tickers = list(set(tickers))[:10]

    news_summary = {
        "headlines": headlines,
        "sources": sources,
        "top_tickers": top_tickers,
        "total_news": len(news),
        "timestamp": int(time.time()),
    }

    # Send summary event
    stream_custom_event(
        writer,
        event_type="summary_generated",
        data=news_summary,
    )

    stream_progress_event(
        writer,
        stage="complete",
        status="done",
        message="Brief generation complete",
        order=3,
        total_steps=3,
    )

    # Format news for state
    formatted_news = [
        {
            "id": n.get("news_url", str(i)),
            "title": n.get("title", ""),
            "description": n.get("text", "")[:300],
            "source": n.get("source_name", ""),
            "url": n.get("news_url", ""),
            "pubDate": n.get("date", ""),
            "tickers": n.get("tickers", []),
            "sentiment": n.get("sentiment", "Neutral"),
            "imageUrl": n.get("image_url", ""),
        }
        for i, n in enumerate(news)
    ]

    result = {
        "news": formatted_news,
        "sentiment_analysis": sentiment_analysis,
        "news_summary": news_summary,
        "summary_complete": True,
    }

    return Command(
        update={
            **result,
            "messages": [
                ToolMessage(
                    content=json.dumps({
                        "status": "success",
                        "sentiment": overall,
                        "confidence": confidence,
                        "total_news": len(news),
                        "positive": sentiment_counts["Positive"],
                        "negative": sentiment_counts["Negative"],
                        "neutral": sentiment_counts["Neutral"],
                        "top_headlines": headlines[:5],
                        "top_tickers": top_tickers[:5],
                    }, ensure_ascii=False),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def fetch_crypto_news(
    limit: int = 20,
    tickers: Optional[List[str]] = None,
    sort: str = "latest",
    search: Optional[str] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """
    Fetches crypto news from CryptoNews API.

    Args:
        limit: Number of news items to fetch (default: 20, max: 50)
        tickers: Filter by specific tickers (e.g., ["BTC", "ETH"])
        sort: Sort by "latest" or "trending"
        search: Search query to filter news

    Returns:
        Command that updates state with fetched news
    """
    print(f"📰 fetch_crypto_news started")
    print(f"   limit: {limit}, tickers: {tickers}, sort: {sort}, search: {search}")

    writer = get_stream_writer_safe()
    stream_progress_event(
        writer,
        stage="fetch_news",
        status="running",
        message="Fetching latest crypto news",
        order=1,
        total_steps=3,
        details={"limit": limit, "tickers": tickers},
    )

    if not CRYPTONEWS_API_KEY:
        error_msg = "CryptoNews API key not configured"
        print(f"❌ {error_msg}")
        stream_progress_event(
            writer,
            stage="fetch_news",
            status="error",
            message=error_msg,
            order=1,
            total_steps=3,
        )
        return Command(
            update={
                "news": [],
                "messages": [ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id
                )]
            }
        )

    try:
        # Build API URL
        params = {
            "token": CRYPTONEWS_API_KEY,
            "items": min(limit, 50),
            "page": 1,
            "sortby": "rank" if sort == "trending" else "date",
        }

        if tickers:
            params["tickers"] = ",".join(tickers)
        else:
            params["section"] = "general"

        if search:
            params["search"] = search

        url = f"{CRYPTONEWS_BASE_URL}/category"
        response = httpx.get(url, params=params, timeout=15.0)
        response.raise_for_status()
        data = response.json()

        if not data or not data.get("data"):
            print("⚠️ No news data returned from API")
            return Command(
                update={
                    "news": [],
                    "messages": [ToolMessage(
                        content="No news available at the moment.",
                        tool_call_id=tool_call_id
                    )]
                }
            )

        # Parse news items
        news_items = []
        for item in data["data"][:limit]:
            news_item = {
                "id": item.get("news_url", f"news_{int(time.time() * 1000)}"),
                "title": item.get("title", "").strip(),
                "description": (item.get("text", "")[:200] + "...") if item.get("text") else "",
                "content": item.get("text", ""),
                "source": item.get("source_name", "Unknown"),
                "url": item.get("news_url", "#"),
                "pubDate": item.get("date", ""),
                "tickers": item.get("tickers", []),
                "sentiment": item.get("sentiment", "Neutral"),
                "imageUrl": item.get("image_url"),
            }
            news_items.append(news_item)

            # Stream each news item
            stream_custom_event(writer, "news_item", news_item)

        print(f"✅ Fetched {len(news_items)} news items")

        stream_progress_event(
            writer,
            stage="fetch_news",
            status="completed",
            message=f"Fetched {len(news_items)} news articles",
            order=1,
            total_steps=3,
            details={"count": len(news_items)},
        )

        return Command(
            update={
                "news": news_items,
                "messages": [ToolMessage(
                    content=f"✅ Fetched {len(news_items)} crypto news articles.",
                    tool_call_id=tool_call_id
                )]
            }
        )

    except httpx.HTTPStatusError as e:
        error_msg = f"API error: {e.response.status_code}"
        print(f"❌ {error_msg}")
        stream_progress_event(
            writer,
            stage="fetch_news",
            status="error",
            message=error_msg,
            order=1,
            total_steps=3,
        )
        return Command(
            update={
                "news": [],
                "messages": [ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id
                )]
            }
        )
    except Exception as e:
        error_msg = f"Error fetching news: {str(e)}"
        print(f"❌ {error_msg}")
        return Command(
            update={
                "news": [],
                "messages": [ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id
                )]
            }
        )


@tool
def analyze_news_sentiment(
    news_json: Optional[Union[str, List[Dict]]] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None
) -> Command:
    """
    Analyzes sentiment distribution across news articles.

    IMPORTANT: You MUST pass the news data from fetch_crypto_news result.
    Example: analyze_news_sentiment(news_json=<result from fetch_crypto_news>)

    Args:
        news_json: JSON string or list of news items (REQUIRED - pass data from fetch_crypto_news)

    Returns:
        Command that updates state with sentiment analysis
    """
    print("📊 analyze_news_sentiment started")
    print(f"   news_json type: {type(news_json)}")
    print(f"   state available: {state is not None}")
    if state:
        print(f"   state keys: {list(state.keys()) if isinstance(state, dict) else 'not a dict'}")
        if "news" in state:
            print(f"   state['news'] length: {len(state.get('news', []))}")

    writer = get_stream_writer_safe()
    stream_progress_event(
        writer,
        stage="analyze_sentiment",
        status="running",
        message="Analyzing news sentiment distribution",
        order=2,
        total_steps=3,
    )

    # Parse news data - try multiple sources
    news = []
    
    if isinstance(news_json, str):
        try:
            news = json.loads(news_json)
            print(f"   ✅ Parsed from JSON string: {len(news)} items")
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parse error: {e}")
    elif isinstance(news_json, list):
        news = news_json
        print(f"   ✅ Received list directly: {len(news)} items")
    elif isinstance(news_json, dict):
        # Sometimes the data comes wrapped
        news = [news_json]
        print(f"   ✅ Received single dict, wrapped: 1 item")
    
    # Fallback to state if no data provided
    if not news and state:
        if isinstance(state, dict) and "news" in state:
            news = state["news"]
            print(f"   ✅ Read {len(news)} news items from state")
        else:
            print(f"   ⚠️ State doesn't contain 'news' key")
    
    if not news:
        print("   ❌ No news data available from any source")

    if not news:
        return Command(
            update={
                "sentiment_analysis": {"error": "No news to analyze"},
                "messages": [ToolMessage(
                    content="No news available for sentiment analysis.",
                    tool_call_id=tool_call_id
                )]
            }
        )

    # Count sentiments
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for item in news:
        sentiment = item.get("sentiment", "Neutral")
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
        else:
            sentiment_counts["Neutral"] += 1

    total = len(news)
    
    # Calculate overall sentiment
    positive_ratio = sentiment_counts["Positive"] / total if total > 0 else 0
    negative_ratio = sentiment_counts["Negative"] / total if total > 0 else 0
    
    if positive_ratio > 0.5:
        overall = "bullish"
    elif negative_ratio > 0.5:
        overall = "bearish"
    else:
        overall = "neutral"

    # Calculate confidence based on sentiment clarity
    max_ratio = max(positive_ratio, negative_ratio, sentiment_counts["Neutral"] / total if total > 0 else 0)
    confidence = min(95, int(max_ratio * 100 + 20))

    sentiment_analysis = {
        "total_analyzed": total,
        "positive": sentiment_counts["Positive"],
        "negative": sentiment_counts["Negative"],
        "neutral": sentiment_counts["Neutral"],
        "positive_percentage": round(positive_ratio * 100, 1),
        "negative_percentage": round(negative_ratio * 100, 1),
        "overall_sentiment": overall,
        "confidence": confidence,
        "timestamp": time.time(),
    }

    print(f"✅ Sentiment analysis complete: {overall} ({confidence}% confidence)")

    # Stream sentiment analysis
    stream_custom_event(writer, "sentiment_analysis", sentiment_analysis)

    stream_progress_event(
        writer,
        stage="analyze_sentiment",
        status="completed",
        message=f"Market sentiment: {overall.upper()} ({confidence}%)",
        order=2,
        total_steps=3,
        details=sentiment_analysis,
    )

    return Command(
        update={
            "sentiment_analysis": sentiment_analysis,
            "messages": [ToolMessage(
                content=f"✅ Sentiment analysis complete. Overall: {overall.upper()} with {confidence}% confidence. Positive: {sentiment_counts['Positive']}, Negative: {sentiment_counts['Negative']}, Neutral: {sentiment_counts['Neutral']}.",
                tool_call_id=tool_call_id
            )]
        }
    )


@tool
def generate_news_summary(
    news_json: Optional[Union[str, List[Dict]]] = None,
    max_news: int = 10,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None
) -> Command:
    """
    Generates a concise summary of the top news stories.
    
    IMPORTANT: You MUST pass the news data from fetch_crypto_news result.
    Example: generate_news_summary(news_json=<result from fetch_crypto_news>)

    Args:
        news_json: JSON string or list of news items (REQUIRED - pass data from fetch_crypto_news)
        max_news: Maximum number of news items to include (default: 10)

    Returns:
        Command that updates state with news summary data
    """
    print(f"📝 generate_news_summary started (max: {max_news})")
    print(f"   news_json type: {type(news_json)}")
    print(f"   state available: {state is not None}")
    if state:
        print(f"   state keys: {list(state.keys()) if isinstance(state, dict) else 'not a dict'}")
        if "news" in state:
            print(f"   state['news'] length: {len(state.get('news', []))}")

    writer = get_stream_writer_safe()
    stream_progress_event(
        writer,
        stage="generate_summary",
        status="running",
        message="Preparing news summary",
        order=3,
        total_steps=3,
    )

    # Parse news data - try multiple sources
    news = []
    
    if isinstance(news_json, str):
        try:
            news = json.loads(news_json)
            print(f"   ✅ Parsed from JSON string: {len(news)} items")
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parse error: {e}")
    elif isinstance(news_json, list):
        news = news_json
        print(f"   ✅ Received list directly: {len(news)} items")
    elif isinstance(news_json, dict):
        news = [news_json]
        print(f"   ✅ Received single dict, wrapped: 1 item")
    
    # Fallback to state if no data provided
    if not news and state:
        if isinstance(state, dict) and "news" in state:
            news = state["news"]
            print(f"   ✅ Read {len(news)} news items from state")
        else:
            print(f"   ⚠️ State doesn't contain 'news' key")
    
    if not news:
        print("   ❌ No news data available from any source")

    if not news:
        return Command(
            update={
                "news_summary": {"error": "No news to summarize"},
                "messages": [ToolMessage(
                    content="No news available for summary generation.",
                    tool_call_id=tool_call_id
                )]
            }
        )

    # Get top news items
    top_news = news[:max_news]

    # Extract key information
    headlines = [item.get("title", "") for item in top_news]
    sources = list(set(item.get("source", "Unknown") for item in top_news))
    tickers_mentioned = list(set(
        ticker for item in top_news 
        for ticker in (item.get("tickers") or [])
    ))[:10]  # Limit to top 10 tickers

    # Count sentiment for top news
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for item in top_news:
        sentiment = item.get("sentiment", "Neutral")
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1

    # Build summary structure
    news_summary = {
        "headlines": headlines,
        "sources": sources[:5],  # Top 5 sources
        "tickers_mentioned": tickers_mentioned,
        "sentiment_breakdown": sentiment_counts,
        "total_news": len(news),
        "summarized_count": len(top_news),
        "generated_at": time.time(),
    }

    # Create a brief text summary for the LLM to enhance
    brief_lines = []
    if sentiment_counts["Positive"] > sentiment_counts["Negative"]:
        brief_lines.append("O mercado demonstra sentimento otimista.")
    elif sentiment_counts["Negative"] > sentiment_counts["Positive"]:
        brief_lines.append("O mercado demonstra cautela no momento.")
    else:
        brief_lines.append("O mercado apresenta sinais mistos.")

    brief_lines.append(f"Foram analisadas {len(top_news)} notícias de {len(sources)} fontes.")
    
    if tickers_mentioned:
        brief_lines.append(f"Destaques: {', '.join(tickers_mentioned[:5])}.")

    news_summary["brief"] = " ".join(brief_lines)

    print(f"✅ Summary generated for {len(top_news)} news items")

    # Stream summary
    stream_custom_event(writer, "news_summary", news_summary)

    stream_progress_event(
        writer,
        stage="generate_summary",
        status="completed",
        message="News summary generated",
        order=3,
        total_steps=3,
        details={"count": len(top_news)},
    )

    # Signal completion
    stream_custom_event(writer, "complete", {
        "status": "success",
        "timestamp": time.time(),
    })

    return Command(
        update={
            "news_summary": news_summary,
            "summary_complete": True,
            "messages": [ToolMessage(
                content=f"✅ News summary generated from {len(top_news)} articles. {news_summary['brief']}",
                tool_call_id=tool_call_id
            )]
        }
    )


@tool
def search_news_by_topic(
    topic: str,
    limit: int = 10,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """
    Searches news by a specific topic or keyword.

    Args:
        topic: The topic or keyword to search for
        limit: Maximum number of results (default: 10)

    Returns:
        Command that updates state with search results
    """
    print(f"🔍 search_news_by_topic: '{topic}' (limit: {limit})")

    writer = get_stream_writer_safe()
    stream_progress_event(
        writer,
        stage="search_news",
        status="running",
        message=f"Searching news for: {topic}",
        order=1,
        total_steps=1,
        details={"topic": topic},
    )

    if not CRYPTONEWS_API_KEY:
        error_msg = "CryptoNews API key not configured"
        return Command(
            update={
                "search_results": [],
                "messages": [ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id
                )]
            }
        )

    try:
        params = {
            "token": CRYPTONEWS_API_KEY,
            "items": min(limit, 30),
            "page": 1,
            "search": topic,
            "sortby": "date",
        }

        response = httpx.get(f"{CRYPTONEWS_BASE_URL}/category", params=params, timeout=15.0)
        response.raise_for_status()
        data = response.json()

        if not data or not data.get("data"):
            return Command(
                update={
                    "search_results": [],
                    "messages": [ToolMessage(
                        content=f"No news found for '{topic}'.",
                        tool_call_id=tool_call_id
                    )]
                }
            )

        results = []
        for item in data["data"][:limit]:
            result = {
                "id": item.get("news_url", f"news_{int(time.time() * 1000)}"),
                "title": item.get("title", "").strip(),
                "description": (item.get("text", "")[:200] + "...") if item.get("text") else "",
                "source": item.get("source_name", "Unknown"),
                "url": item.get("news_url", "#"),
                "pubDate": item.get("date", ""),
                "tickers": item.get("tickers", []),
                "sentiment": item.get("sentiment", "Neutral"),
            }
            results.append(result)

        print(f"✅ Found {len(results)} news items for '{topic}'")

        stream_progress_event(
            writer,
            stage="search_news",
            status="completed",
            message=f"Found {len(results)} results for '{topic}'",
            order=1,
            total_steps=1,
            details={"count": len(results)},
        )

        return Command(
            update={
                "search_results": results,
                "messages": [ToolMessage(
                    content=f"✅ Found {len(results)} news articles about '{topic}'.",
                    tool_call_id=tool_call_id
                )]
            }
        )

    except Exception as e:
        error_msg = f"Error searching news: {str(e)}"
        print(f"❌ {error_msg}")
        return Command(
            update={
                "search_results": [],
                "messages": [ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id
                )]
            }
        )


def get_news_tools():
    """Returns list of news tools"""
    return [
        generate_news_brief,  # Main tool for brief mode - does everything in one call
        fetch_crypto_news,
        analyze_news_sentiment,
        generate_news_summary,
        search_news_by_topic,
    ]

