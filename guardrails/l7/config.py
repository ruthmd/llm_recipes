from guardrails import Guard
from guardrails.hub import DetectPII


guard = Guard(
    name='pii_guard',
)

guard.use(
    DetectPII(
        pii_entities=['PERSON', 'PHONE_NUMBER'],
        on_fail="refrain"
    ), 
    on="messages"
).use(
    DetectPII(
        pii_entities=['PERSON', 'PHONE_NUMBER'],
        on_fail="fix"
    ), 
    on="output"
)