"""Step 3b: the navigation history.

The back/forward stack is the one piece of per-user state in the module, so it
gets its own file. The behaviour that matters is the truncation on `visit`: after
going back, moving somewhere new discards the redone future, which is what makes
`forward` mean anything.
"""

from __future__ import annotations

from browsing import NavigationState


class TestVisit:
    def test_visiting_appends_and_advances(self, state):
        state.visit("root")
        state.visit("root/ir")
        assert state.history == ["root", "root/ir"]
        assert state.current == "root/ir"
        assert state.position == 1

    def test_visiting_the_same_node_twice_is_a_no_op(self, state):
        state.visit("root")
        state.visit("root")
        assert state.history == ["root"]

    def test_a_new_node_after_going_back_truncates_the_future(self, state):
        state.visit("root")
        state.visit("root/ir")
        state.visit("doc/p04")
        state.back()
        state.visit("root/ml")
        assert state.history == ["root", "root/ir", "root/ml"]
        assert not state.can_go_forward()

    def test_the_history_is_bounded(self):
        state = NavigationState(limit=3)
        for node in ("a", "b", "c", "d"):
            state.visit(node)
        assert state.history == ["b", "c", "d"]
        assert state.current == "d"


class TestBackForward:
    def test_back_and_forward_walk_the_stack(self, state):
        for node in ("root", "root/ir", "doc/p04"):
            state.visit(node)
        assert state.back() == "root/ir"
        assert state.back() == "root"
        assert state.forward() == "root/ir"
        assert state.forward() == "doc/p04"

    def test_back_at_the_start_is_a_no_op(self, state):
        state.visit("root")
        assert state.back() == "root"
        assert not state.can_go_back()

    def test_forward_at_the_end_is_a_no_op(self, state):
        state.visit("root")
        state.visit("root/ir")
        assert state.forward() == "root/ir"
        assert not state.can_go_forward()

    def test_the_flags_track_the_position(self, state):
        assert not state.can_go_back() and not state.can_go_forward()
        state.visit("a")
        assert not state.can_go_back() and not state.can_go_forward()
        state.visit("b")
        assert state.can_go_back() and not state.can_go_forward()
        state.back()
        assert not state.can_go_back() and state.can_go_forward()


class TestRecent:
    def test_recent_is_most_recent_first_and_unique(self, state):
        for node in ("root", "root/ir", "root", "doc/p04"):
            state.visit(node)
        assert state.recent() == ["doc/p04", "root", "root/ir"]

    def test_recent_respects_n(self, state):
        for node in ("a", "b", "c", "d"):
            state.visit(node)
        assert state.recent(2) == ["d", "c"]

    def test_recent_is_empty_when_nothing_was_visited(self, state):
        assert state.recent() == []

    def test_recent_ignores_the_redone_future(self, state):
        state.visit("a")
        state.visit("b")
        state.back()
        assert state.recent() == ["a"]


class TestClear:
    def test_clear_resets_everything(self, state):
        state.visit("root")
        state.visit("root/ir")
        state.clear()
        assert state.history == []
        assert state.position == -1
        assert state.current == ""

    def test_breadcrumb_nodes_returns_the_past(self, state):
        for node in ("root", "root/ir", "root/ir/ranking"):
            state.visit(node)
        assert state.breadcrumb_nodes() == \
            ["root", "root/ir", "root/ir/ranking"]
