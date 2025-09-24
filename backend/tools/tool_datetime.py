"""
DateTime tool for the Digital Clone AI Assistant.

Provides current date and time information in various formats.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

TOOL_META = {
    "name": "datetime",
    "description": "获取当前日期和时间信息，支持本地时间和UTC时间",
    "parameters": {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["date", "time", "datetime", "iso", "timestamp"],
                "description": "输出格式：date(仅日期)、time(仅时间)、datetime(完整日期时间)、iso(ISO格式)、timestamp(Unix时间戳)",
                "default": "datetime"
            },
            "tz": {
                "type": "string",
                "enum": ["local", "utc"],
                "description": "时区选择：local(本地时区)、utc(UTC时区)",
                "default": "local"
            }
        },
        "required": [],
        "additionalProperties": False
    }
}


def run(format: str = "datetime", tz: str = "local") -> Dict[str, Any]:
    """
    Execute datetime tool.

    Args:
        format: Output format ("date", "time", "datetime", "iso", "timestamp")
        tz: Timezone ("local" or "utc")

    Returns:
        Dictionary with ok/value/error fields
    """
    try:
        logger.info(f"Getting datetime with format={format}, timezone={tz}")

        # Get current time
        if tz == "utc":
            now = datetime.now(timezone.utc)
        else:  # local
            now = datetime.now()

        # Format the output
        if format == "date":
            result = now.strftime("%Y-%m-%d")
        elif format == "time":
            result = now.strftime("%H:%M:%S")
        elif format == "datetime":
            if tz == "utc":
                result = now.strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                result = now.strftime("%Y-%m-%d %H:%M:%S")
        elif format == "iso":
            result = now.isoformat()
        elif format == "timestamp":
            # Unix timestamp (seconds since epoch)
            result = now.timestamp()
        else:
            return {"ok": False, "error": f"不支持的格式: {format}"}

        logger.info(f"Datetime result: {result}")
        return {"ok": True, "value": result}

    except Exception as e:
        error_msg = f"获取时间失败: {str(e)}"
        logger.error(f"Datetime tool error: {e}")
        return {"ok": False, "error": error_msg}
