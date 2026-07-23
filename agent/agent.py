import os
import argparse
from pathlib import Path
from backends import get_backend
from actions import ActionRunner

base_dir = Path(__file__).resolve().parent
temp_dir = base_dir / "../temp"
temp_dir.mkdir(parents=True, exist_ok=True)
context_file = os.path.join(temp_dir, "context.txt")
_env = os.environ.copy()
_env["TORCH_CPP_LOG_LEVEL"] = "ERROR"


class Agent:
    def __init__(self, max_tokens: int, max_steps: int = 20):
        self.backend = get_backend()
        self.actions = ActionRunner()
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.end_pattern = "[[END]]"
        self.memory = []
        self.react_note = f"""IMPORTANT:
You are running in a ReAct loop: Thought -> Action -> Observation -> Thought.
When you need to use a tool, output exactly one action and then stop immediately.
Never output a second Thought or Action before the system returns an Observation.
Never write Observation yourself; Observation is written only by the system after an action runs.
Use this format:
Thought: explain the next step briefly.
Action: one available action name
Action Input:
```text
arguments for the action
```

After the system returns an Observation, use it to decide the next step.
When you have the final answer, use:
Thought: brief summary of what you learned.
Final Answer: the answer for the user.

You may also return {self.end_pattern} after the final answer to end the conversation.
If an Observation shows an action failed, reason about why it failed and try a corrected action.
"""

    def initialize(self, text: str, context: str = "") -> None:
        prompt = f"""You are an autonomous ReAct agent.

Available actions:
{self.actions.desc()}

Context:
{context}

User task:
{text}
"""
        self.memory = [prompt]

    def step(self) -> str:
        prompt = "\n\n".join(self.memory)
        resp = ""
        stream = self.backend.stream_response(
            prompt + "\n\n" + self.react_note, self.max_tokens
        )
        try:
            for chunk in stream:
                chunk_start = len(resp)
                resp += chunk
                action_end = self.actions.first_complete_action_end(resp)
                if action_end is None:
                    print(chunk, end="", flush=True)
                    continue

                print(resp[chunk_start:action_end], end="", flush=True)
                resp = resp[:action_end]
                break
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

        resp = self.actions.trim_to_first_action(resp)
        self.memory.append(resp)
        return resp

    def maybe_take_action(self):
        res = self.actions(self.memory[-1])
        if res is not None:
            self.observe(res)
            return res
        else:
            return None

    def observe(self, text: str) -> None:
        observation = f"\n\nObservation: {text}\n\n"
        print(observation, flush=True)
        self.memory.append(observation)

    def is_end(self) -> bool:
        last_message = self.memory[-1]
        if self.end_pattern in last_message:
            return True
        return (
            "final answer:" in last_message.lower()
            and not self.actions.parse_action(last_message)
        )

    def run(self, text: str):
        self.initialize(text)
        state = "model"
        steps = 0

        while state != "done":
            if state == "model":
                if steps >= self.max_steps:
                    state = "max_steps"
                else:
                    steps += 1
                    self.step()
                    state = "done" if self.is_end() else "action"

            elif state == "action":
                state = "model" if self.maybe_take_action() is not None else "repair"

            elif state == "repair":
                self.observe(
                    "No valid action was found. Continue with either a valid "
                    "ReAct action or a Final Answer."
                )
                state = "model"

            elif state == "max_steps":
                self.observe(
                    f"Reached max_steps={self.max_steps}. Stop and provide the best final "
                    "answer from the collected observations."
                )
                self.step()
                state = "done"

        print("", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AToy: An agent create shit")
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--input", type=str, default="None")
    args = parser.parse_args()
    agent = Agent(max_tokens=args.max_tokens, max_steps=args.max_steps)
    with open(args.input.strip(), "r") as f:
        text = f.read()
    agent.run(text)
