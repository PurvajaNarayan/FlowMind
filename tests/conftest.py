import json
from pathlib import Path

import pytest

from flowmind.reader.mermaid_reader import mermaid_to_graph

FIXTURE = Path(__file__).parent / "fixtures" / "sample.json"


@pytest.fixture
def dataset():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def sample(dataset):
    return dataset["code00453"]


@pytest.fixture
def graph(sample):
    return mermaid_to_graph(sample["mermaid"])
