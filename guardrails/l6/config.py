from guardrails import Guard
from guardrails.hub import RestrictToTopic
guard = Guard()

guard.name = 'topic_guard'
guard.use(
    RestrictToTopic(
        valid_topics=["pizza", "food", "restaurant", "order", "menu"],
        invalid_topics=["politics", "automobiles"],
        on_fail="exception"
    )
)