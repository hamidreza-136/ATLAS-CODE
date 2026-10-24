"""Object extraction prompt (Appendix A, Fig. 5)."""

OBJECT_PROMPT = (
    "{expression} \n"
    "What is the main object (or objects) referred to in the given expression or question? \n"
    "Focus on the **primary subject or agent** involved in the described action or behavior. "
    "Respond with a single word (e.g., 'cat', 'person', 'dog') that best describes the target object(s). "
)


def build_prompt(expression: str) -> str:
    """Lower-case the expression, end it with punctuation, and wrap it in the prompt."""
    expression = expression.lower()
    if not expression.endswith(("?", ".")):
        expression += "."
    return OBJECT_PROMPT.format(expression=expression)
