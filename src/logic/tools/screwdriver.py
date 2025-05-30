from dataclasses import dataclass
from logic.tools.tool import Tool


@dataclass
class Screwdriver(Tool):
    picked_up_fastener_name: str | None = None
    picked_up_fastener: bool
