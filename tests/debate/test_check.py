from agentic_crypto_trading_system.debate.consensus import (

    UnanimousConsensus, 
    Position, 
    DebateArgument,
    MajorityConsensus,
    WeightedConsensus
)

# def test_check():

#     strategy = UnanimousConsensus()

#     arguments = [
#         DebateArgument(
#         agent_role="...",
#         position=Position.BUY,
#         confidence=0.85,
#         reasoning="..."
#     ),
#     DebateArgument(
#             agent_role="sentiment",
#             position=Position.BUY,
#             confidence=0.90,
#             reasoning="Positive news"
#         )
#     ]

#     check = strategy.check(arguments)
#     print(check)

#     assert check[0] is True

# def test_majority_consensus():
#     framework = MajorityConsensus()
#     arguments = [
#         DebateArgument(
#             agent_role="...",
#             position=Position.BUY,
#             confidence=0.85,
#             reasoning="..."
#         )
#     ]
#     result = framework.check(arguments)
#     print (result)

def test_weighted_consensus():
    framework = WeightedConsensus()

    arguments = [
        DebateArgument(
            agent_role="sentiment",
            position=Position.BUY,
            confidence=0.90,
            reasoning="Positive news"
        )
    ]
    results = framework.check(arguments)
    print (results)
    # assert results[1] == Position.BUY
    assert results[2] > 0.5
