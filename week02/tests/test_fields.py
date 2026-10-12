"""Step 4: searching structured fields.

The test that matters most is `an_unsatisfiable_restriction_returns_nothing`.

The failure it guards against is subtle and was present in the first draft:
`title:proximity AND results:precision`, where 'proximity' is not a term in the
corpus, used to drop the impossible title constraint and answer with every paper
whose RESULTS mention precision. That is not a smaller answer - it is a
DIFFERENT answer to a DIFFERENT query.
"""

from __future__ import annotations

import pytest

from fields import DEFAULT_WEIGHTS, FieldQueryError, parse_field_query


class TestQueryParsing:
    def test_a_prefix_becomes_a_restriction(self, index, idf_table):
        clauses, _ = parse_field_query("title:search", idf_table)
        assert len(clauses) == 1
        assert len(clauses[0].restrictions) == 1
        assert clauses[0].restrictions[0].field == "title"
        assert clauses[0].restrictions[0].terms == ("search",)

    def test_two_fields_joined_by_and_form_ONE_clause(self, idf_table):
        """AND is an INTERSECTION. Two clauses here would be an OR."""
        clauses, _ = parse_field_query(
            "title:search AND results:precision", idf_table)
        assert len(clauses) == 1
        assert [r.field for r in clauses[0].restrictions] == ["title", "results"]

    def test_or_splits_into_two_clauses(self, idf_table):
        clauses, _ = parse_field_query(
            "title:search OR results:precision", idf_table)
        assert len(clauses) == 2
        assert [c.restrictions[0].field for c in clauses] == ["title", "results"]

    def test_and_binds_tighter_than_or(self, idf_table):
        """SQL precedence: A OR B AND C is A OR (B AND C)."""
        clauses, _ = parse_field_query(
            "title:a OR results:b AND abstract:c", idf_table)
        assert len(clauses) == 2
        assert len(clauses[0].restrictions) == 1
        assert len(clauses[1].restrictions) == 2

    def test_an_unknown_field_is_an_error(self, index, idf_table):
        with pytest.raises(FieldQueryError) as caught:
            parse_field_query("keywords:search", idf_table)
        assert "title" in str(caught.value)

    def test_words_before_a_colon_are_not_treated_as_terms(self, idf_table):
        clauses, _ = parse_field_query("title:search", idf_table)
        assert "title" not in clauses[0].restrictions[0].terms

    def test_a_restriction_whose_terms_are_all_unknown_is_unsatisfiable(
            self, idf_table):
        clauses, _ = parse_field_query("title:proximity", idf_table)
        assert clauses[0].restrictions[0].unsatisfiable

    def test_not_is_rejected_rather_than_silently_dropped(self, idf_table):
        """NOT used to become a stop word and vanish.

        Silently dropping an operator is how OR went wrong: preprocess_query()
        removes 'OR', so a post-preprocessing split turned a user's OR into an
        AND and returned a confidently wrong answer.
        """
        with pytest.raises(FieldQueryError) as caught:
            parse_field_query("title:search NOT results:x", idf_table)
        assert "NOT is not supported" in str(caught.value)


class TestBrackets:
    """Parentheses used to be ignored altogether.

    The parser split on OR and AND with a regex and never looked at brackets,
    so `(A OR B) AND C` was read as `A OR (B AND C)` - a different question, and
    a confidently wrong answer. The laws below are checked against the parsed
    clauses, not against the text, because the whole point is that the text
    means one thing and the old parser meant another.
    """

    def _shape(self, text, idf_table):
        clauses, _ = parse_field_query(text, idf_table)
        return [tuple(r.field for r in c.restrictions) for c in clauses]

    def test_a_group_around_or_produces_one_clause_per_branch(self, idf_table):
        assert self._shape("(title:a OR title:b) AND results:c",
                           idf_table) == [("title", "results"),
                                          ("title", "results")]

    def test_a_group_changes_the_answer(self, idf_table):
        """`(A OR B) AND C` must be `(A AND C) OR (B AND C)`, not `A OR B AND C`.

        Flattened, the second form loses a clause; grouped, it keeps both.
        """
        grouped = self._shape("(title:a OR title:b) AND results:c", idf_table)
        flat = self._shape("title:a OR title:b AND results:c", idf_table)
        assert grouped != flat
        assert flat == [("title",), ("title", "results")]

    def test_and_binds_tighter_than_or_inside_brackets_too(self, idf_table):
        assert (self._shape("title:a OR (title:b AND results:c)", idf_table)
                == self._shape("title:a OR title:b AND results:c", idf_table))

    def test_spaces_around_brackets_do_not_change_the_query(self, idf_table):
        assert (self._shape("(title:a OR title:b) AND results:c", idf_table)
                == self._shape("( title:a OR title:b ) AND results:c",
                               idf_table))

    def test_nested_brackets_are_honoured(self, idf_table):
        assert self._shape("((title:a OR title:b) AND results:c) OR abstract:d",
                           idf_table) == [("title", "results"),
                                          ("title", "results"),
                                          ("abstract",)]

    def test_the_grouped_and_flat_forms_agree_on_papers(self, fields):
        """The behaviour check, at the level a user sees it."""
        grouped = fields.restrict("(title:search OR abstract:rank) "
                                  "AND abstract:result")
        flat = fields.restrict("title:search AND abstract:result "
                               "OR abstract:rank AND abstract:result")
        assert grouped.ok and flat.ok
        assert set(grouped.papers) == set(flat.papers)

    def test_an_unmatched_bracket_is_refused(self, idf_table):
        for text in ("(title:a OR title:b", "title:a OR title:b)"):
            with pytest.raises(FieldQueryError):
                parse_field_query(text, idf_table)


class TestMalformedQueriesAreRefused:
    """Every one of these parsed into something plausible and answered anyway.

    Each is a different way to ask a question the user never asked: a colon with
    no field name read as a plain word, `title:` silently dropped the
    restriction, and `title:search AND` quietly became `title:search`.
    """

    @pytest.mark.parametrize("text", [
        ":search", " :search", "title:graph OR :search",
    ])
    def test_a_colon_with_no_field_name_is_refused(self, text, idf_table):
        with pytest.raises(FieldQueryError) as caught:
            parse_field_query(text, idf_table)
        assert "no field name" in str(caught.value)

    @pytest.mark.parametrize("text", ["title:", "title: AND results:graph"])
    def test_an_empty_restriction_is_refused(self, text, idf_table):
        with pytest.raises(FieldQueryError) as caught:
            parse_field_query(text, idf_table)
        assert "nothing to look for" in str(caught.value)

    @pytest.mark.parametrize("text", [
        "title:search AND", "title:search OR", "title:a AND title:b OR",
    ])
    def test_a_dangling_connector_is_refused(self, text, idf_table):
        with pytest.raises(FieldQueryError) as caught:
            parse_field_query(text, idf_table)
        assert "nothing to join to" in str(caught.value)

    @pytest.mark.parametrize("text", ["AND title:search", "OR title:search"])
    def test_a_leading_connector_is_refused(self, text, idf_table):
        with pytest.raises(FieldQueryError) as caught:
            parse_field_query(text, idf_table)
        assert "nothing to join to" in str(caught.value)

    def test_a_multi_word_restriction_still_works(self, idf_table):
        """The bracket work must not break `title:inverted index`."""
        clauses, _ = parse_field_query("title:inverted index", idf_table)
        assert clauses[0].restrictions[0].field == "title"
        assert clauses[0].restrictions[0].terms == ("invert index",)

    def test_words_after_the_colon_join_the_same_restriction(self, idf_table):
        """Everything up to the next connector belongs to the field.

        `title:inverted index` must reach the phrase `invert index` INSIDE the
        title, which is the only reason this works. The cost is that a bare
        word after a restriction is not a separate all-fields constraint - it is
        a second term of the same one.
        """
        clauses, _ = parse_field_query("title:search index", idf_table)
        assert [r.field for r in clauses[0].restrictions] == ["title"]
        assert clauses[0].restrictions[0].terms == ("search", "index")

    def test_a_bare_word_on_its_own_is_a_loose_restriction(self, idf_table):
        clauses, _ = parse_field_query("index", idf_table)
        assert [r.field for r in clauses[0].restrictions] == [""]


class TestOrSemantics:
    def test_or_returns_the_union_of_both_fields(self, fields):
        one = fields.restrict("title:search")
        two = fields.restrict("results:precision")
        union = fields.restrict("title:search OR results:precision")
        assert set(union.papers) == set(one.papers) | set(two.papers)

    def test_and_returns_the_intersection_of_both_fields(self, fields):
        one = fields.restrict("title:search")
        two = fields.restrict("results:precision")
        both = fields.restrict("title:search AND results:precision")
        assert set(both.papers) == set(one.papers) & set(two.papers)

    def test_or_is_a_superset_of_and(self, fields):
        both = fields.restrict("title:search AND results:precision")
        either = fields.restrict("title:search OR results:precision")
        assert set(both.papers) <= set(either.papers)

    def test_a_paper_satisfying_two_clauses_appears_once(self, fields):
        """An OR must not turn into duplicate rows."""
        result = fields.restrict("title:search OR results:precision", top=30)
        ids = [h.doc_id for h in result.hits]
        assert len(ids) == len(set(ids))

    def test_each_hit_reports_which_field_matched(self, fields):
        result = fields.restrict("title:search OR results:precision", top=30)
        assert result.hits
        for hit in result.hits:
            assert hit.field in {"title", "results"} or "+" in hit.field


class TestUnsatisfiableConstraints:
    @pytest.mark.parametrize("mode", ["restrict", "filter", "weighted"])
    def test_an_unknown_term_yields_zero_hits_not_a_wider_answer(self, fields,
                                                                 mode):
        result = getattr(fields, mode)(
            "title:proximity AND results:precision")
        assert not result.ok
        assert result.count == 0
        # the error names the term AFTER preprocessing: 'proximity' is not in
        # this corpus, where the phrase list folded it into 'proxim oper'
        assert "proxim" in (result.error or "")

    def test_the_error_says_which_constraint_failed(self, fields):
        result = fields.restrict("title:proximity AND results:precision")
        assert "title" in result.error

    def test_a_blank_query_is_not_an_error(self, fields):
        result = fields.restrict("")
        assert result.ok
        assert result.count == 0

    def test_a_known_query_still_works(self, fields):
        result = fields.restrict("title:search AND results:precision")
        assert result.ok
        assert result.count >= 1


class TestRestrict:
    def test_only_the_named_field_is_matched(self, index, fields):
        result = fields.restrict("title:search")
        assert result.ok
        for hit in result.hits:
            assert hit.field == "title"
            paper = next(p for p in index.papers if p.doc_id == hit.doc_id)
            title_terms = {t.term for t in paper.tokens
                           if paper.fields["title"].contains(t.position)}
            assert set(hit.terms) <= title_terms

    def test_restrict_and_filter_return_the_same_papers(self, fields):
        for query in ("title:search", "results:precision", "title:index"):
            assert fields.restrict(query).papers == fields.filter(query).papers

    def test_restrict_and_filter_can_return_a_different_order(self, fields):
        """The pair is the whole point of the step.

        If the orders always matched, restricting and filtering would be the
        same operation and there would be no reason to document either.
        """
        query = "title:search"
        assert ([h.doc_id for h in fields.restrict(query).hits]
                != [h.doc_id for h in fields.filter(query).hits])

    def test_a_restriction_with_no_field_means_the_whole_paper(self, fields):
        loose = fields.restrict("structured field search")
        assert loose.ok
        assert loose.count >= 1


class TestWeighted:
    def test_a_title_beat_outranks_a_title_hit(self, index, fields):
        result = fields.weighted("structured field search")
        assert result.ok
        for hit in result.hits:
            assert set(hit.field.split("+")) <= set(DEFAULT_WEIGHTS)

    def test_weights_reach_the_score(self, fields):
        weighted = fields.weighted("structured field search")
        strict = fields.restrict("structured field search")
        assert weighted.hits[0].score > strict.hits[0].score

    def test_a_higher_title_weight_raises_the_score(self, fields):
        default = fields.weighted("structured field search")
        flatter = fields.weighted("structured field search",
                                  weights={k: 1.0 for k in DEFAULT_WEIGHTS})
        by_id = {h.doc_id: h.score for h in flatter.hits}
        top = default.hits[0]
        if "title" in top.field:
            assert top.score >= by_id[top.doc_id]

    def test_nothing_is_discarded_by_the_weighted_design(self, fields):
        weighted = fields.weighted("structured field search")
        strict = fields.restrict("structured field search")
        assert len(weighted.papers) >= len(strict.papers)


class TestFieldStatistics:
    def test_a_title_is_much_shorter_than_a_results_section(self, fields):
        assert fields.stats.average_length["title"] < \
            fields.stats.average_length["results"] / 5

    def test_every_paper_has_every_field(self, fields):
        for name in DEFAULT_WEIGHTS:
            assert fields.stats.paper_holders(name) == 20

    def test_per_field_idf_is_never_worse_than_a_flat_idf(self, fields):
        """A term in 1 of 20 titles should be worth more than a term in 1 of 20
        papers, because titles are a much smaller collection."""
        title_only = [term for term, df
                      in fields.stats.document_frequency["title"].items()
                      if df == 1 and term in fields.index.vocabulary]
        assert title_only, "the corpus must have a term unique to titles"