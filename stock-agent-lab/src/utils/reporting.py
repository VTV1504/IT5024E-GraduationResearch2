from __future__ import annotations

import os
from datetime import datetime


def make_report_path(agent_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{agent_name}_report_{timestamp}.txt"
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    return os.path.join(reports_dir, filename)


def write_report(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)
