"""Reader backends. Both return a flowmind.schema.FlowGraph (spec §7.1)."""

from flowmind.reader.mermaid_reader import mermaid_to_graph

__all__ = ["mermaid_to_graph"]
