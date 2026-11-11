"""The default 7-item survey: four 1-5 ratings and three open-text questions."""

from __future__ import annotations

from .schemas import Question


def default_questions() -> list[Question]:
    return [
        Question("appeal", "Overall, how appealing is this product concept to you? (1 = not at all, 5 = extremely)", "rating"),
        Question("clarity", "How clear is it what this product is and how it would taste? (1 = very unclear, 5 = very clear)", "rating"),
        Question("uniqueness", "How different is this concept from drinks you already know? (1 = not at all, 5 = extremely)", "rating"),
        Question("purchase_intent", "How likely would you be to buy this if it were available? (1 = definitely not, 5 = definitely)", "rating"),
        Question("first_reaction", "In one or two sentences, what is your honest first reaction to this concept?", "text"),
        Question("concerns", "What, if anything, concerns or puts you off about this concept?", "text"),
        Question("improvement", "What one change would make you more likely to try it?", "text"),
    ]
