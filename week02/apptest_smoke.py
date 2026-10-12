"""Smoke-test the Streamlit app through AppTest: every tab, every mode."""

from streamlit.testing.v1 import AppTest


def report(name, at):
    if at.exception:
        print(f"FAIL {name}")
        for e in at.exception:
            print(f"   {e.value}")
        return False
    print(f"ok   {name}: {len(at.dataframe)} df, "
          f"{len(at.success)} success, {len(at.error)} error")
    return True


passed = True

# 1. initial render
at = AppTest.from_file("app.py", default_timeout=240).run()
passed &= report("initial render", at)

# 2. ranked search
at.text_input(key="q_bm25").set_value("proximity window phrase query").run()
passed &= report("bm25 search", at)

# 3. bim
at = AppTest.from_file("app.py", default_timeout=240).run()
at.radio[1].set_value("bim").run()
at.text_input(key="q_bim").set_value("term weighting saturation").run()
passed &= report("bim search", at)

# 4. boolean exact
at = AppTest.from_file("app.py", default_timeout=240).run()
at.radio[1].set_value("boolean").run()
at.text_input(key="q_boolean").set_value("boolean retrieval").run()
passed &= report("boolean", at)

# 5. field restricted + cross-field AND
at = AppTest.from_file("app.py", default_timeout=240).run()
at.radio[1].set_value("fields").run()
at.text_input(key="q_fields").set_value("title:search AND results:precision").run()
passed &= report("fields cross-field", at)

# 6. bad field name -> readable error, no exception
at = AppTest.from_file("app.py", default_timeout=240).run()
at.radio[1].set_value("fields").run()
at.text_input(key="q_fields").set_value("keywords:search").run()
if at.exception:
    print("FAIL bad field raised")
    for e in at.exception:
        print(f"   {e.value}")
    passed = False
else:
    print(f"ok   bad field: {len(at.error)} error message(s) shown cleanly")

# 7. bm25f
at = AppTest.from_file("app.py", default_timeout=240).run()
at.radio[1].set_value("bm25f").run()
at.text_input(key="q_bm25f").set_value("structured field search").run()
passed &= report("bm25f", at)

# 8. overlapped / non-overlapped through the search tab
for mode in ("overlapped", "non-overlapped"):
    at = AppTest.from_file("app.py", default_timeout=240).run()
    at.radio[1].set_value(mode).run()
    at.text_input(key=f"q_{mode}").set_value("document index positions").run()
    passed &= report(mode, at)

# 9. k1 = 0 turns BM25 into BIM - the slider should say so.
# Streamlit renders every tab on every run, so there is no tab switching here:
# the tab contents are all exercised by the plain run() above.
at = AppTest.from_file("app.py", default_timeout=240).run()
labels = [s.label for s in at.slider]
if "k1 — tf saturation" not in labels:
    print(f"FAIL k1 slider missing, sliders are {labels}")
    passed = False
else:
    at.slider[labels.index("k1 — tf saturation")].set_value(0.0).run()
    messages = [s.value for s in at.info] + [e.value for e in at.error]
    hit = any("Binary Independence Model" in m for m in messages)
    print(f"ok   k1=0 explains it is BIM: {hit}")
    passed &= hit

# 10. Fields & Lists tab
at = AppTest.from_file("app.py", default_timeout=240).run()
at.text_input(key="q_fields_compare").set_value("title:search").run()
at.text_input(key="q_list_compare").set_value("document index positions").run()
passed &= report("Fields & Lists tab", at)

# 11. Proximity tab
at = AppTest.from_file("app.py", default_timeout=240).run()
at.text_input(key="q_prox_tab").set_value("section with(2) retrieval").run()
passed &= report("Proximity tab", at)

# 12. Proximity tab with a syntax error
at = AppTest.from_file("app.py", default_timeout=240).run()
at.text_input(key="q_prox_tab").set_value("paragraph methodology").run()
if at.exception:
    print("FAIL proximity syntax error raised")
    for e in at.exception:
        print(f"   {e.value}")
    passed = False
else:
    print(f"ok   proximity syntax error: {len(at.error)} message(s)")

# 13. Grouped field query. Brackets used to be ignored, so this used to render
# a result set for a different question with nothing on screen to say so.
at = AppTest.from_file("app.py", default_timeout=240).run()
at.radio[1].set_value("fields").run()
at.text_input(key="q_fields").set_value(
    "(title:search OR abstract:rank) AND abstract:result").run()
passed &= report("fields with brackets", at)

# 14. Every malformed field query must show a message, never raise.
at = AppTest.from_file("app.py", default_timeout=240).run()
at.radio[1].set_value("fields").run()
shown = 0
for bad in ("keywords:search", "title:search NOT results:x", "(title:a OR title:b",
            ":search", "title:", "title:search AND"):
    at.text_input(key="q_fields").set_value(bad).run()
    if at.exception:
        print(f"FAIL malformed field query raised: {bad!r}")
        for e in at.exception:
            print(f"   {e.value}")
        passed = False
    elif at.error:
        shown += 1
print(f"ok   {shown}/6 malformed field queries explained on screen, no crashes")
passed &= shown == 6

# 15-17. Evaluation, Corpus and Pipeline tabs all render on every run
at = AppTest.from_file("app.py", default_timeout=240).run()
texts = " ".join(t.value for t in at.markdown) + " ".join(
    c.value for c in at.caption) + " ".join(h.value for h in at.subheader) \
    + " ".join(h.value for h in at.header)
for label in ("Effectiveness", "Corpus and structure", "Pipeline walkthrough"):
    found = label in texts
    print(f"ok   tab {label!r} present: {found}")
    passed &= found
passed &= report("all tabs rendered", at)

print()
print("ALL PASSED" if passed else "FAILURES ABOVE")