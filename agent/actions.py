import re
import os
import inspect
import traceback
import subprocess
import shutil
from dataclasses import dataclass


def python_code(code: str) -> str:
    """
    Directly write a Python script and use _result_ to represent the string result to be returned.
    """
    try:
        scope = {}
        exec(code, scope)
        if "_result_" not in scope:
            return "[python_code completed without setting _result_]"
        return _format_text_result(str(scope["_result_"]))
    except Exception:
        return traceback.format_exc()


def run_cmd(cmd: str) -> str:
    """
    Directly write a shell script. Return the command status, stdout, and stderr.
    """
    try:
        kwargs, shell_error = _shell_kwargs_for(cmd)
        if shell_error:
            return shell_error

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            **kwargs,
        )
        return _format_command_result(result.returncode, result.stdout, result.stderr)
    except Exception:
        return traceback.format_exc()


def _shell_kwargs_for(cmd: str):
    bash_path = shutil.which("bash")
    if os.name != "nt":
        return ({"executable": bash_path} if bash_path else {}), None

    if not _uses_posix_shell_syntax(cmd):
        return {}, None

    if bash_path and _bash_works(bash_path):
        return {"executable": bash_path}, None

    return {}, "\n".join(
        [
            "command not executed",
            "reason: POSIX shell syntax was detected, but no usable bash was found on Windows.",
            "hint: use python_code for Python snippets, or rewrite the command for Windows shell.",
        ]
    )


def _uses_posix_shell_syntax(cmd: str) -> bool:
    return bool(re.search(r"<<\s*['\"]?\w+|/dev/null|\|\|\s*true", cmd))


def _bash_works(bash_path: str) -> bool:
    try:
        result = subprocess.run(
            [bash_path, "-lc", "true"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _format_text_result(text: str) -> str:
    return text if text.strip() else "[action completed with empty output]"


def _format_stream(name: str, text: str) -> str:
    text = text.replace("\x00", "").rstrip()
    return f"{name}:\n{text if text else '[empty]'}"


def _format_command_result(returncode: int, stdout: str, stderr: str) -> str:
    status = "succeeded" if returncode == 0 else "failed"
    return "\n".join(
        [
            f"command {status}",
            f"returncode: {returncode}",
            _format_stream("stdout", stdout),
            _format_stream("stderr", stderr),
        ]
    )


GLOBAL_ACTIONS = [
    python_code,
    run_cmd,
]


@dataclass
class ActionCall:
    name: str
    args: str
    warning: str = ""


class ActionRunner:
    def __init__(self):
        self.action_pattern = re.compile(r"(?im)^\s*Action\s*:\s*([A-Za-z_]\w*)\s*$")
        self.action_input_pattern = re.compile(r"(?im)^\s*Action\s*Input\s*:\s*")
        global GLOBAL_ACTIONS
        self.actions = {}
        for function in GLOBAL_ACTIONS:
            name = str(function.__name__)
            desc = str(inspect.getsource(function))
            self.actions[name] = {"name": name, "desc": desc, "func": function}

    def __call__(self, text: str):
        action_call = self.parse_action(text)
        if action_call:
            action_name = action_call.name
            action_args = action_call.args
            if self.actions.get(action_name, None):
                output = str(self.actions[action_name]["func"](action_args))
                output = _format_text_result(output)
                if action_call.warning:
                    return f"{action_call.warning}\n{output}"
                return output
            else:
                return f"Invalid action name: {action_name}"
        return None

    def trim_to_first_action(self, text: str) -> str:
        text = self._strip_leading_observation(text)
        action_match = self.action_pattern.search(text)
        if not action_match:
            return text

        tail = text[action_match.end() :]
        action_input_match = self.action_input_pattern.search(tail)
        if not action_input_match:
            return text[: action_match.end()].rstrip()

        action_input_start = action_match.end() + action_input_match.end()
        action_input = text[action_input_start:]
        fenced_match = re.match(
            r"(?ims)\s*```[^\n]*\n.*?^\s*```\s*",
            action_input,
        )
        if fenced_match:
            return text[: action_input_start + fenced_match.end()].rstrip()

        next_block = re.search(
            r"(?im)^\s*(?:Thought|Action|Observation|Final Answer)\s*:",
            action_input,
        )
        if next_block:
            return text[: action_input_start + next_block.start()].rstrip()
        return text.rstrip()

    def _strip_leading_observation(self, text: str) -> str:
        return re.sub(
            r"(?is)^\s*Observation\s*:\s*(?=(?:Thought|Action|Final Answer)\s*:)",
            "",
            text,
            count=1,
        )

    def parse_action(self, text: str):
        action_matches = list(self.action_pattern.finditer(text))
        if not action_matches:
            return None

        action_match = action_matches[0]
        tail = text[action_match.end() :]
        action_input_match = self.action_input_pattern.search(tail)
        if not action_input_match:
            return ActionCall(
                name=action_match.group(1).strip(),
                args="",
                warning="[warning: Action Input was missing; executed with empty input]",
            )

        action_input = tail[action_input_match.end() :]
        warning = ""
        if len(action_matches) > 1:
            warning = (
                f"[warning: response contained {len(action_matches)} Action blocks; "
                "executed only the first one]"
            )

        return ActionCall(
            name=action_match.group(1).strip(),
            args=self._extract_action_input(action_input),
            warning=warning,
        )

    def _extract_action_input(self, text: str) -> str:
        text = text.lstrip()
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            for idx, line in enumerate(lines[1:], start=1):
                if line.strip() == "```":
                    return "\n".join(lines[1:idx]).rstrip()

        action_input = re.split(
            r"(?im)^\s*(?:Thought|Action|Observation|Final Answer)\s*:",
            text,
            maxsplit=1,
        )[0]
        return action_input.strip()

    def desc(self) -> str:
        shell_note = (
            "On Windows, run_cmd uses the default Windows shell. Do not use POSIX "
            "heredocs such as `python - <<'PY'`; use python_code for Python snippets "
            "or write Windows-compatible commands."
            if os.name == "nt"
            else "On POSIX systems, run_cmd uses bash when it is available."
        )
        text_head = f"""Use the ReAct format when an action is required:
Thought: explain what you need to do next.
Action: $ACTION
Action Input:
```text
$ARGS
```

Output exactly one Action block, then stop and wait for Observation.
{shell_note}

Below are the available $ACTION options along with their descriptions and code:\n\n"""
        text_actions = []
        for key, value in self.actions.items():
            text_actions.append(f"$ACTION={key}\n{value['desc']}")
        text_actions = "\n".join(text_actions)
        return text_head + text_actions


if __name__ == "__main__":
    action_runner = ActionRunner()
    print(action_runner.desc())
