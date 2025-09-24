"""
Ask User Policy Module for the Digital Clone AI Assistant.

Contains logic for determining when to ask users for clarification.
"""

def needs_user_clarification(content: str) -> bool:
    """
    Determine if the response content indicates a need for user clarification.

    Enhanced version that avoids false positives from normal greetings.

    Args:
        content: The response content to analyze

    Returns:
        True if clarification is needed, False otherwise
    """
    # First, exclude normal conversational responses that shouldn't trigger clarification
    exclusion_indicators = [
        "有什么我可以帮助你的吗",
        "很高兴为你服务",
        "有什么可以帮到你",
        "我可以帮您做什么",
        "有什么需要帮助的",
        "how can i help you",
        "what can i help you with",
        "is there anything i can help"
    ]

    content_lower = content.lower()
    for exclusion in exclusion_indicators:
        if exclusion in content_lower:
            return False  # This is just a normal greeting, not a clarification request

    # Now check for actual clarification needs
    clarification_indicators = [
        "请告诉我",
        "您能提供",
        "需要更多信息",
        "请问您",
        "我想了解",
        "能否告诉我",
        "需要澄清",
        "请补充",
        "为了更好地帮助您",
        "tell me more about",
        "could you provide",
        "need more information",
        "please clarify",
        "I need to know",
        "can you tell me more"
    ]

    return any(indicator in content_lower for indicator in clarification_indicators)


def get_clarification_question(content: str) -> str:
    """
    Extract or generate a clarification question from response content.

    Args:
        content: The response content containing clarification needs

    Returns:
        A clear question to ask the user
    """
    # Simple extraction - look for question marks
    sentences = content.split('。')
    for sentence in sentences:
        sentence = sentence.strip()
        if '？' in sentence or '?' in sentence:
            # Clean up the sentence
            question = sentence.replace('请', '').replace('您', '').replace('你', '').strip()
            if len(question) > 10:  # Ensure it's a substantial question
                return question

    # Fallback question
    return "为了更好地帮助您，我需要更多信息。请问您能提供更多细节吗？"
