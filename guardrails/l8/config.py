from guardrails import Guard
from guardrails.hub import CompetitorCheck
guard = Guard()
guard.name = 'competitor_check'

guard.use(CompetitorCheck(competitors=['Pizza by Alfredo'], on_fail="exception"))