"""Smoke-test the Streamlit app through AppTest: every section renders.

    cd week03
    ..\\.venv\\Scripts\\python.exe apptest_smoke.py

Week 3 uses a page selector rather than st.tabs, because a tab cannot be changed
from code and the Home "browse" button has to move the user to Browse. So this
drives the selector between sections the way a user would, and checks that the
per-row preview buttons open a document inline.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent / "app.py"

passed = True


def report(name: str, at: AppTest) -> bool:
    global passed
    if at.exception:
        print(f"FAIL {name}")
        for error in at.exception:
            print(f"   {error.value}")
        passed = False
        return False
    print(f"ok   {name}: {len(at.dataframe)} dataframe(s), "
          f"{len(at.metric)} metric(s), {len(at.success)} success, "
          f"{len(at.error)} error")
    return True


def goto(at: AppTest, page: str) -> AppTest:
    at.segmented_control(key="page").set_value(page)
    return at.run()


def click(at: AppTest, key: str) -> AppTest | None:
    buttons = [b for b in at.button if b.key == key]
    if not buttons:
        return None
    return buttons[0].click().run()


def page_of(at: AppTest) -> str:
    return at.segmented_control(key="page").value


# 1. initial render
at = AppTest.from_file(str(APP), default_timeout=240).run()
report("initial render", at)
sections = at.segmented_control(key="page")
if len(sections.options) != 7:
    print(f"FAIL expected 7 sections, found {len(sections.options)}")
    passed = False

# 2. Home -> browse really moves to Browse (the button used to do nothing,
#    because a tab cannot be switched from code)
at = click(at, "home_root/ir") or at
if page_of(at) != "Browse":
    print(f"FAIL home browse did not reach Browse, page is {page_of(at)!r}")
    passed = False
else:
    report("home browse -> Browse", at)

# 3. Search, then an inline preview from a result row
at = goto(at, "Search")
at.text_input(key="search_box").set_value("proximity window").run()
report("search 'proximity window'", at)
previewed = click(at, "search_preview_p04")
if previewed is None:
    print("FAIL no preview button on the p04 row")
    passed = False
else:
    report("inline preview under a result row", previewed)
    if not previewed.expander:
        print("FAIL the inline preview rendered no sections")
        passed = False
    anchors = sum("?doc=" in m.value for m in previewed.markdown)
    if anchors == 0:
        print("FAIL the preview has no inline hypertext links")
        passed = False
    else:
        print(f"ok   inline hypertext links: {anchors} markdown block(s) "
              f"carry ?doc= anchors")

# 4. Browse, then an inline preview from a document row
at = goto(at, "Browse")
at.session_state["node"] = "root/ir"
at.run()
report("browse into root/ir", at)
previewed = click(at, "browse_preview_p17")
if previewed is None:
    print("FAIL no preview button on the p17 row")
    passed = False
else:
    report("inline preview under a browse row", previewed)

# 5. a search that matches nothing is a message, not a crash
at = goto(at, "Search")
at.text_input(key="search_box").set_value("zzzznotaword").run()
report("search with no matches", at)

# 6. a filter narrows the collection without erroring
at.session_state["f_domain"] = "Information Retrieval"
at.session_state["f_year"] = "2021"
at.run()
report("filter by domain and year", at)

# 7. the hypertext section finds a path
at = goto(at, "Hypertext")
at.selectbox(key="path_a").set_value("p01").run()
at.selectbox(key="path_b").set_value("p18").run()
found = click(at, "path_btn")
if found is None:
    print("FAIL no path button")
    passed = False
else:
    report("find a navigation path", found)

# 8. the Document section still renders the full page
at = goto(at, "Document")
report("document section", at)

print()
if passed:
    print("ALL PASSED")
    raise SystemExit(0)
print("FAILED")
raise SystemExit(1)
