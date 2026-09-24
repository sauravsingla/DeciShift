import json
from decishift.demo import run_demo
from decishift.reports import render_html, render_json, render_markdown, render_terminal


def test_demo_is_reproducible():
    _,one=run_demo(500); _,two=run_demo(500); assert one.records.equals(two.records); assert one.attribution.equals(two.attribution); assert one.interactions.equals(two.interactions)


def test_report_formats_are_machine_and_human_readable():
    _,result=run_demo(300); terminal=render_terminal(result); markdown=render_markdown(result); payload=json.loads(render_json(result)); html=render_html(result); assert "Decision shift rate" in terminal; assert "# DeciShift comparison" in markdown; assert payload["summary"]["total_records"]==300; assert "scientific_note" in payload; assert "<!doctype html>" in html.lower(); assert "cdn" not in html.lower()


def test_json_report_sanitizes_nonfinite_metadata():
    _, result = run_demo(20)
    result.metadata["nonfinite_test"] = float("inf")
    text = render_json(result)
    assert "Infinity" not in text
    assert json.loads(text)["summary"]["nonfinite_test"] is None
