"""The query language: precedence, brackets, BUT NOT, the minus sign, and errors."""

from __future__ import annotations

import pytest

from query_parser import And, Not, Or, QueryError, Term, parse_query, terms_in, to_text


def word(node):
    assert isinstance(node, Term), node
    return node.word


class TestShapes:
    def test_one_word_is_one_term(self):
        assert word(parse_query("hypertension")) == "hypertension"

    def test_and_joins_two_words(self):
        tree = parse_query("hypertension AND treatment")
        assert isinstance(tree, And)
        assert [word(c) for c in tree.children] == ["hypertension", "treatment"]

    def test_or_joins_two_words(self):
        tree = parse_query("asthma OR COPD")
        assert isinstance(tree, Or)

    def test_not_negates_one_word(self):
        tree = parse_query("NOT diabetes")
        assert isinstance(tree, Not) and word(tree.child) == "diabetes"

    def test_a_minus_sign_is_the_same_as_not(self):
        assert parse_query("-fever") == parse_query("NOT fever")

    def test_but_not_is_and_with_a_negated_part(self):
        tree = parse_query("asthma BUT NOT smoking")
        assert isinstance(tree, And)
        assert isinstance(tree.children[1], Not)

    def test_a_chain_of_and_is_one_node_not_nested_pairs(self):
        tree = parse_query("heart AND failure AND treatment")
        assert isinstance(tree, And) and len(tree.children) == 3

    def test_a_hyphenated_word_keeps_its_parts(self):
        tree = parse_query("iron-deficiency")
        assert tree.stems == ("iron", "defici")


class TestPrecedence:
    def test_and_binds_tighter_than_or(self):
        tree = parse_query("asthma OR COPD AND smoking")
        assert isinstance(tree, Or)
        assert isinstance(tree.children[1], And)

    def test_not_binds_tighter_than_and(self):
        tree = parse_query("NOT asthma AND COPD")
        assert isinstance(tree, And)
        assert isinstance(tree.children[0], Not)

    def test_brackets_override_precedence(self):
        tree = parse_query("(asthma OR COPD) AND smoking")
        assert isinstance(tree, And)
        assert isinstance(tree.children[0], Or)

    def test_the_text_shows_the_brackets_the_parser_chose(self):
        assert to_text(parse_query("asthma OR COPD AND smoking")) == \
            "asthma OR (COPD AND smoking)"


class TestImplicitOperator:
    def test_two_words_mean_and_by_default(self):
        assert isinstance(parse_query("heart failure"), And)

    def test_two_words_can_mean_or(self):
        assert isinstance(parse_query("heart failure", implicit="OR"), Or)

    def test_implicit_or_returns_a_wider_query_than_implicit_and(self):
        narrow = parse_query("heart failure treatment", implicit="AND")
        wide = parse_query("heart failure treatment", implicit="OR")
        assert isinstance(narrow, And) and isinstance(wide, Or)

    def test_an_unknown_implicit_operator_is_refused(self):
        with pytest.raises(ValueError):
            parse_query("heart failure", implicit="XOR")


class TestWordsAndCase:
    def test_lower_case_operators_are_ordinary_words(self):
        with pytest.raises(QueryError, match="'and' has no searchable word"):
            parse_query("asthma and copd")

    def test_a_stop_word_is_refused_with_a_reason(self):
        with pytest.raises(QueryError, match="stop word"):
            parse_query("the")

    def test_terms_are_listed_in_the_order_typed(self):
        tree = parse_query("copd OR asthma AND NOT smoking")
        assert [t.word for t in terms_in(tree)] == ["copd", "asthma", "smoking"]


class TestErrors:
    @pytest.mark.parametrize("query, message", [
        ("", "empty"),
        ("   ", "empty"),
        ("hypertension AND", "ends with an operator"),
        ("(asthma", "never closed"),
        ("asthma)", "not expected here"),
        ("BUT NOT smoking", "needs a word or a bracket"),
        ("asthma BUT smoking", "BUT must be followed by NOT"),
        ("()", "needs a word or a bracket"),
    ])
    def test_a_bad_query_says_what_is_wrong(self, query, message):
        with pytest.raises(QueryError, match=message):
            parse_query(query)

    def test_an_error_points_at_its_character(self):
        with pytest.raises(QueryError) as caught:
            parse_query("asthma AND ) COPD")
        assert caught.value.position == 11

    def test_the_error_message_gives_a_one_based_position(self):
        with pytest.raises(QueryError, match=r"at character 12"):
            parse_query("asthma AND ) COPD")


class TestRoundTrip:
    @pytest.mark.parametrize("query", [
        "hypertension",
        "asthma OR COPD",
        "hypertension AND NOT diabetes",
        "(asthma OR COPD) AND smoking",
        "heart failure BUT NOT (asthma OR -fever)",
    ])
    def test_the_text_of_a_tree_parses_back_to_the_same_tree(self, query):
        tree = parse_query(query)
        assert parse_query(to_text(tree)) == tree
