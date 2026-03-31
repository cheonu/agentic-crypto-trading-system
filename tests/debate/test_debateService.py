from agentic_crypto_trading_system.debate.debate_service import (

    DebateService
)

from agentic_crypto_trading_system.debate.consensus import (

    UnanimousConsensus, 
    Position, 
    DebateArgument,
    MajorityConsensus,
    DebateTranscript,
    WeightedConsensus,
    ConsensusMode,
    ConsensusStrategy,
    DebateArgument,
    DebateRound,
    DebateStatus,
)
from datetime import datetime
import uuid

# def test_debate_services():
#     framework = DebateService()

#     result = framework.initiate_debate("BTC/USD", ["technical_analyst"])
#     print (result)

def mock_generator(role, task, round_num, previous_round): 
        DebateArgument(
            agent_role=role,
            position=Position.BUY,
            round_number=1,
            previous_round=None
        )

# def test_conduct_round():
#     framework = DebateService()
#     transcript = framework.initiate_debate(
#         "BTC/USD", 
#         ["technical_analyst"])

#     argument_generators = {
#         "technical_analyst" : mock_generator
#     }

#     result = framework.conduct_round(transcript,"BTC/USD trading", argument_generators)
#     print(result)
#     print ("Agents:", transcript.participating_agents)
#     print("Generators:", argument_generators.keys())

def test_check_consensus():
    framework = DebateService()

    transcript = framework.initiate_debate(
        "BTC/USD", 
        ["technical_analyst"])

    argument_generators = {
        "technical_analyst" : mock_generator
    }

    debate_round = framework.conduct_round(transcript,"BTC/USD trading", argument_generators)
    result = framework.check_consensus(debate_round)
    print(result)


  
    