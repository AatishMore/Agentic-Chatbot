

from graph import app


class Agent:
    def __init__(self):
        self.app = app

    def run(self, query: str, history=None) -> dict:
        state = {
            "messages":     [],
            "query":        query,
            "final_answer": "",
            "tool_used":    "",
            "history":      history or [],
        }
        result = self.app.invoke(state)
        return result



_agent = Agent()


def run_agent(query: str, history=None) -> dict:
    return _agent.run(query, history)
