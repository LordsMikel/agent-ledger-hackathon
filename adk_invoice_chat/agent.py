"""Expose the Firestore-backed invoice chat as the Google ADK root agent.

Author: Miguel Medina Cantos
"""

from config.settings import Settings
from src.main import build_chat_agent_system


root_agent = build_chat_agent_system(Settings.from_env())
