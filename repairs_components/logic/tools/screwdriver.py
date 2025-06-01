from dataclasses import dataclass
from repairs_components.logic.tools.tool import Tool


@dataclass
class Screwdriver(Tool):
    picked_up_fastener_name: str | None = None
    picked_up_fastener: bool = False
