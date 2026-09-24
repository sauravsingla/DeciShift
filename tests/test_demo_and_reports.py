import json

from decishift.demo import run_demo
from decishift.reports import render_json, render_markdown, render_terminal


def test_demo_is_reproducible():
    _, one = run_demo(500)
    _, two = run_demo(500)
    assert one.records.equals(two.records)
    assert one.attribution.equals(two.attribution)
    assert one.interactions.equals(two.interactions)


def test_report_formats_are_machine_and_human_readable():
    _, result = run_demo(300)
    terminal = render_terminal(result)
    markdown = render_markdown(result)
    payload = json.loads(render_json(result))
    assert "Decision shift rate" in terminal
    assert "# DeciShift comparison" in markdown
    assert payload["summary"]["total_records"] == 300
    assert "scientific_note" in payload
