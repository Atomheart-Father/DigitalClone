"""
Ask User Policy Module for the Digital Clone AI Assistant.

Contains logic for determining when to ask users for clarification.
"""

def needs_user_clarification(content: str) -> bool:
    """
    Determine if the response content indicates a need for user clarification.

    This is a simplified version - in practice, you might use more sophisticated
    detection based on the model's output patterns.

    Args:
        content: The response content to analyze

    Returns:
        True if clarification is needed, False otherwise
    """
    clarification_indicators = [
        "请告诉我",
        "您能提供",
        "需要更多信息",
        "请问",
        "我想了解",
        "能否告诉我",
        "需要澄清",
        "请补充",
        "tell me",
        "could you provide",
        "need more information",
        "please clarify",
        "I need to know",
        "can you tell me"
    ]

    content_lower = content.lower()
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
