import pytest

from evaluation.run_eval import load_csv


def test_csv_question_set(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text(
        "question;expected;cards\n"
        "My refund hasn't arrived yet;answer;Q5\n"
        "I have a problem with billing;clarify;Q4 Q5 Q6\n"
        "\n"
        "What is the weather today?;decline;\n"
        "Can I connect Trello?;answer|decline;Q7\n",
        encoding="utf-8-sig",  # Notepad may add a BOM
    )
    items = load_csv(path)
    assert [i["id"] for i in items] == ["V01", "V02", "V03", "V04"]
    assert items[1]["expected_cards"] == ["Q4", "Q5", "Q6"]
    assert items[2]["expected_cards"] == []
    assert items[3]["expected_action"] == ["answer", "decline"]


def test_csv_with_a_wrong_action_is_rejected(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text("question;expected;cards\nHello;reply;\n", encoding="utf-8")
    with pytest.raises(ValueError, match="answer, clarify or decline"):
        load_csv(path)
