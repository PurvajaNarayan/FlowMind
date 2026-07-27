from flowmind.data import iter_qa


def test_iter_qa_flattens(dataset):
    items = list(iter_qa(dataset))
    assert len(items) == 3
    q4 = next(i for i in items if i.question_id == "4")
    assert q4.qa_type == "topological"
    assert q4.answers == ["7"]
    assert q4.code is not None  # code subset


def test_multi_reference_answers(dataset):
    items = list(iter_qa(dataset))
    q1 = next(i for i in items if i.question_id == "1")
    assert len(q1.answers) == 3  # A1/A2/A3 all present (spec §6)
