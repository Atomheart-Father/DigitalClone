"""
Conversation Logger for the Digital Clone AI Assistant.

Logs conversation turns to JSONL files for analysis and future RAG training.
Handles sanitization to avoid logging sensitive information.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Updated imports for new structure
try:
    import sys
    import os
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from core.config import config
    from schemas.messages import Message, RouteDecision
except ImportError:
    # Fallback imports for backward compatibility
    try:
        from backend.config import config
        from backend.message_types import Message, RouteDecision
    except ImportError:
        # Final fallback - assume we're in the old structure
        from config import config
        from message_types import Message, RouteDecision

logger = logging.getLogger(__name__)


class ConversationLogger:
    """Logger for conversation transcripts."""

    def __init__(self):
        self.log_dir = Path(config.LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_date = None
        self.current_file = None

    def _get_log_file(self, date: Optional[str] = None) -> Path:
        """Get the log file path for the specified date."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        return config.get_log_file_path(date)

    def _sanitize_message(self, message: Message) -> Dict[str, Any]:
        """Sanitize a message for logging (remove sensitive data)."""
        sanitized = {
            "role": message.role.value,
            "content_length": len(message.content),
            "has_tool_call": message.tool_call is not None,
            "has_tool_result": message.tool_result is not None,
            "timestamp": datetime.now().isoformat()
        }

        # Include content preview (first 100 chars) for analysis
        if message.content:
            preview_length = min(100, len(message.content))
            sanitized["content_preview"] = message.content[:preview_length]

        # Include tool call info (without sensitive parameters)
        if message.tool_call:
            sanitized["tool_call"] = {
                "name": message.tool_call.name,
                "arguments_count": len(message.tool_call.arguments)
            }

        # Include tool result info
        if message.tool_result:
            sanitized["tool_result"] = {
                "name": message.tool_result.name,
                "content_length": len(message.tool_result.content)
            }

        # Include metadata if present
        if message.metadata:
            sanitized["metadata"] = message.metadata

        return sanitized

    def log_turn(
        self,
        route_decision: RouteDecision,
        messages: List[Message],
        tool_calls_count: int,
        ask_cycles_used: int
    ):
        """Log a complete conversation turn."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "conversation_turn",
                "route_decision": {
                    "engine": route_decision.engine,
                    "reason": route_decision.reason,
                    "confidence": route_decision.confidence
                },
                "statistics": {
                    "total_messages": len(messages),
                    "tool_calls_count": tool_calls_count,
                    "ask_cycles_used": ask_cycles_used
                },
                "messages": [self._sanitize_message(msg) for msg in messages]
            }

            self._write_log_entry(log_entry)
            logger.debug(f"Logged conversation turn with {len(messages)} messages")

        except Exception as e:
            logger.error(f"Failed to log conversation turn: {e}")

    def log_continuation(
        self,
        clarification: str,
        messages: List[Message],
        tool_calls_count: int
    ):
        """Log a conversation continuation after user clarification."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "conversation_continuation",
                "clarification_length": len(clarification),
                "statistics": {
                    "total_messages": len(messages),
                    "tool_calls_count": tool_calls_count
                },
                "messages": [self._sanitize_message(msg) for msg in messages]
            }

            self._write_log_entry(log_entry)
            logger.debug("Logged conversation continuation")

        except Exception as e:
            logger.error(f"Failed to log conversation continuation: {e}")

    def log_tool_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        execution_time: float
    ):
        """Log individual tool execution."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "tool_execution",
                "tool_name": tool_name,
                "parameters_sanitized": self._sanitize_parameters(parameters),
                "result": {
                    "ok": result.get("ok"),
                    "has_value": "value" in result,
                    "has_error": "error" in result,
                    "error_length": len(result.get("error", ""))
                },
                "execution_time": execution_time
            }

            self._write_log_entry(log_entry)
            logger.debug(f"Logged tool execution: {tool_name}")

        except Exception as e:
            logger.error(f"Failed to log tool execution: {e}")

    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log an error that occurred during processing."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "error",
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {}
            }

            self._write_log_entry(log_entry)
            logger.warning(f"Logged error: {error_type}")

        except Exception as e:
            logger.error(f"Failed to log error: {e}")

    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize tool parameters for logging."""
        sanitized = {}
        sensitive_keys = {'api_key', 'password', 'token', 'secret', 'key'}

        for key, value in parameters.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, str) and len(value) > 100:
                # Truncate long string values
                sanitized[key] = value[:100] + "..."
            else:
                sanitized[key] = value

        return sanitized

    def _write_log_entry(self, entry: Dict[str, Any]):
        """Write a log entry to the appropriate JSONL file."""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self._get_log_file(today)

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        except Exception as e:
            logger.error(f"Failed to write log entry to {log_file}: {e}")

    def get_recent_logs(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent log entries for analysis."""
        logs = []
        today = datetime.now()

        for i in range(days):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            log_file = self._get_log_file(date)

            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                logs.append(json.loads(line))
                except Exception as e:
                    logger.warning(f"Failed to read log file {log_file}: {e}")

        return logs

    def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get usage statistics from recent logs."""
        logs = self.get_recent_logs(days)

        stats = {
            "total_turns": 0,
            "total_continuations": 0,
            "total_tool_calls": 0,
            "total_errors": 0,
            "engine_usage": {"chat": 0, "reasoner": 0},
            "tool_usage": {},
            "avg_messages_per_turn": 0.0
        }

        total_messages = 0

        for entry in logs:
            entry_type = entry.get("type")

            if entry_type == "conversation_turn":
                stats["total_turns"] += 1
                stats["total_tool_calls"] += entry.get("statistics", {}).get("tool_calls_count", 0)

                # Engine usage
                engine = entry.get("route_decision", {}).get("engine")
                if engine in stats["engine_usage"]:
                    stats["engine_usage"][engine] += 1

                # Messages count
                messages_count = entry.get("statistics", {}).get("total_messages", 0)
                total_messages += messages_count

            elif entry_type == "conversation_continuation":
                stats["total_continuations"] += 1
                stats["total_tool_calls"] += entry.get("statistics", {}).get("tool_calls_count", 0)

            elif entry_type == "tool_execution":
                tool_name = entry.get("tool_name")
                if tool_name:
                    stats["tool_usage"][tool_name] = stats["tool_usage"].get(tool_name, 0) + 1

            elif entry_type == "error":
                stats["total_errors"] += 1

        # Calculate averages
        if stats["total_turns"] > 0:
            stats["avg_messages_per_turn"] = total_messages / stats["total_turns"]

        return stats

    def close(self):
        """Close the logger (no-op for file-based logging)."""
        pass


# Global logger instance
conversation_logger = ConversationLogger()
