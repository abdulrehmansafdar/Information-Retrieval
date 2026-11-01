"""Smoke-test the Streamlit app through AppTest: both sections render.

    cd week07
    ..\\.venv\\Scripts\\python.exe apptest_smoke.py

The app is driven the way a user drives it: the section bar, the query box, the model
buttons, the settings, and the example buttons, which must fill in the query.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent / "app.py"
PAGES = ("Search", "Documents")

passed = True


def fail(message: str) -> None:
    global passed
    print(f"FAIL {message}")
    passed = False


def report(name: str, at: AppTest) -> bool:
    if at.exception:
        fail(name)
        for error in at.exception:
            print(f"   {error.value}")
        return False
    print(f"ok   {name}: {len(at.metric)} metric(s), {len(at.markdown)} block(s), "
          f"{len(at.error)} error(s)")
    return True


def page_text(at: AppTest) -> str:
    return "\n".join(block.value for block in at.markdown)


# 1. initial render, and the section bar has exactly the two sections
at = AppTest.from_file(str(APP), default_timeout=240).run()
report("initial render (Search)", at)
options = tuple(at.segmented_control(key="page").options)
if len(options) != len(PAGES):
    fail(f"expected {len(PAGES)} sections, found {options}")

# 2. an example button fills in the query, and the angle graphs appear
example = [b for b in at.button if b.label == "Try: wheezing"]
if not example:
    fail("no example button for 'wheezing'")
else:
    at = example[0].click().run()
    text = page_text(at)
    if at.text_input(key="query").value != "wheezing":
        fail("the example button did not fill in the query")
    elif "no shared word" not in text:
        fail("LSI on 'wheezing' showed no article matched on meaning")
    elif "Numbers are the ranks" not in text:
        fail("the graph of all the results is missing")
    elif "cos θ = 0.874" not in text:
        fail("the graph of one result is missing COPD's cosine")
    elif "Cosine similarity" not in text:
        fail("the cosine working is missing")
    else:
        report("example button -> results, graphs and working", at)

# every occurrence the counts use must be on the page, in every section
at.text_input(key="query").set_value("mosquito bite").run()
report("a query whose words are in two sections", at)
marked = page_text(at)
if "<mark>mosquito</mark>" not in marked or "<mark>mosquitoes</mark>" not in marked:
    fail("the word in Dengue's Treatment section is not marked")
at.text_input(key="query").set_value("wheezing").run()

# 3. the same query under GVSM reaches only the article with the word
at.segmented_control(key="model").set_value("gvsm").run()
report("the same query under GVSM", at)
if "no shared word" in page_text(at):
    fail("GVSM claimed an article with no shared word")
if "cos θ = 0.212" not in page_text(at):
    fail("GVSM's graph is missing Asthma's cosine")

# 4. every weighting, and a change of k
at.segmented_control(key="model").set_value("lsi").run()
for weighting in ("count", "binary", "tfidf"):
    at.radio(key="weighting").set_value(weighting).run()
    report(f"LSI with {weighting} weighting", at)
for k in (2, 30, 15):
    at.slider(key="k").set_value(k).run()
    report(f"LSI with k = {k}", at)

# 5. a query with many results, whose markers crowd together
at.text_input(key="query").set_value("chest pain and blood vessels").run()
report("a query with many results", at)

# 6. queries that cannot be answered show a message, not a crash
for bad in ("the of and", "zzzqqq"):
    at.text_input(key="query").set_value(bad).run()
    if not at.error:
        fail(f"{bad!r} showed no error")
    else:
        report(f"{bad!r} is shown as an error", at)

# 7. the documents
at.segmented_control(key="page").set_value("Documents").run()
report("documents section", at)
if len(at.expander) != 30:
    fail(f"expected 30 articles, found {len(at.expander)}")

print()
if passed:
    print("ALL PASSED")
    raise SystemExit(0)
print("FAILED")
raise SystemExit(1)
