"""Smoke-test the Streamlit app through AppTest: every section renders.

    cd week06
    ..\\.venv\\Scripts\\python.exe apptest_smoke.py

The app is driven the way a user drives it: the section bar, the query box, the
mode buttons, and the example buttons on Start here, which must move the user to
Search with the query filled in.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent / "app.py"
PAGES = ("Start here", "Search", "Compare", "Membership", "Set view", "Articles")

passed = True


def report(name: str, at: AppTest) -> bool:
    global passed
    if at.exception:
        print(f"FAIL {name}")
        for error in at.exception:
            print(f"   {error.value}")
        passed = False
        return False
    print(f"ok   {name}: {len(at.metric)} metric(s), {len(at.markdown)} block(s), "
          f"{len(at.error)} error(s)")
    return True


def goto(at: AppTest, page: str) -> AppTest:
    at.segmented_control(key="page").set_value(page)
    return at.run()


# 1. initial render, and the section bar has every page
at = AppTest.from_file(str(APP), default_timeout=240).run()
report("initial render (Start here)", at)
options = tuple(at.segmented_control(key="page").options)
if options != PAGES:
    print(f"FAIL expected sections {PAGES}, found {options}")
    passed = False

# 2. an example button moves to Search with the query filled in
example = [b for b in at.button if b.label.startswith("Try: hypertension AND NOT")]
if not example:
    print("FAIL no example button for 'hypertension AND NOT diabetes'")
    passed = False
else:
    at = example[0].click().run()
    if at.segmented_control(key="page").value != "Search":
        print("FAIL the example button did not reach Search")
        passed = False
    elif at.text_input(key="query").value != "hypertension AND NOT diabetes":
        print("FAIL the example button did not fill in the query")
        passed = False
    else:
        report("example button -> Search with the query filled in", at)

# 3. a typo in fuzzy mode, then the same typo in Boolean mode
at.text_input(key="query").set_value("hypertenion").run()
report("typo 'hypertenion' in fuzzy mode", at)
at.segmented_control(key="mode").set_value("boolean").run()
report("the same query in Boolean mode", at)
at.segmented_control(key="mode").set_value("extended").run()
report("the same query in extended mode", at)

# 4. a query the parser refuses shows an error, not a crash
at.text_input(key="query").set_value("hypertension AND").run()
if not at.error:
    print("FAIL a dangling operator showed no error")
    passed = False
else:
    report("a dangling operator is shown as an error", at)

# 5. every other section renders
at = goto(at, "Compare")
report("compare section", at)
at = goto(at, "Membership")
at.text_input(key="mem_word").set_value("hypertenion").run()
report("membership of a typo", at)
at = goto(at, "Set view")
report("set view section", at)
at = goto(at, "Articles")
report("articles section", at)

print()
if passed:
    print("ALL PASSED")
    raise SystemExit(0)
print("FAILED")
raise SystemExit(1)
