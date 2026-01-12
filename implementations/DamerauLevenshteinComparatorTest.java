package de.pseudonymisierung.mainzelliste.matcher;

import de.pseudonymisierung.mainzelliste.PlainTextField;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DamerauLevenshteinComparatorTest {

    private static final double EPS = 1e-12;

    private static PlainTextField field(String value) {
        PlainTextField f = new PlainTextField();
        f.setValue(value);
        return f;
    }

    @Test
    void Similarity_is_zero_when_a_field_is_missing() {
        DamerauLevenshteinComparator cmp = new DamerauLevenshteinComparator();

        assertEquals(0.0, cmp.compareBackend(null, field("abc")), EPS);
        assertEquals(0.0, cmp.compareBackend(field("abc"), null), EPS);
        //assertEquals(0.0, cmp.compareBackend(null, null), EPS);
    }

    @Test
    void Similarity_is_zero_when_a_field_value_is_missing() {
        DamerauLevenshteinComparator cmp = new DamerauLevenshteinComparator();

        assertEquals(0.0, cmp.compareBackend(field(null), field("abc")), EPS);
        assertEquals(0.0, cmp.compareBackend(field("abc"), field(null)), EPS);
        assertEquals(0.0, cmp.compareBackend(field(null), field(null)), EPS);
    }

    @Test
    void Similarity_is_one_for_the_same_string_instance() {
        DamerauLevenshteinComparator cmp = new DamerauLevenshteinComparator();

        String same = "abc";
        assertEquals(1.0, cmp.compareBackend(field(same), field(same)), EPS);
    }

    @Test
    void Similarity_is_one_for_two_equal_strings() {
        DamerauLevenshteinComparator cmp = new DamerauLevenshteinComparator();

        String left = "abc";
        String right = "abc";
        assertNotSame(left, right);
        assertEquals(1.0, cmp.compareBackend(field(left), field(right)), EPS);
    }

    @Test
    void Similarity_is_zero_when_one_string_is_empty() {
        DamerauLevenshteinComparator cmp = new DamerauLevenshteinComparator();

        assertEquals(0.0, cmp.compareBackend(field(""), field("a")), EPS);
        assertEquals(0.0, cmp.compareBackend(field("a"), field("")), EPS);
    }

    @Test
    void One_character_substitution_reduces_similarity_by_one_over_the_maximum_length() {
        DamerauLevenshteinComparator cmp = new DamerauLevenshteinComparator();

        // distance("test","tent") = 1, maxLen=4 => similarity = 1 - 1/4 = 0.75
        assertEquals(0.75, cmp.compareBackend(field("test"), field("tent")), EPS);
    }

    @Test
    void Comparison_works_when_the_second_string_is_longer() {
        DamerauLevenshteinComparator cmp = new DamerauLevenshteinComparator();

        // distance("ab","abcd") = 2 insertions, maxLen=4 => 0.5
        assertEquals(0.5, cmp.compareBackend(field("ab"), field("abcd")), EPS);
    }

    @Test
    void Adjacent_character_swap_counts_as_one_edit() {
        DamerauLevenshteinComparator cmp = new DamerauLevenshteinComparator();

        // "CA" <-> "AC" ist eine Transposition => Distanz=1, maxLen=2 => 0.5
        assertEquals(0.5, cmp.compareBackend(field("CA"), field("AC")), EPS);
    }

    @Test
    void Similarity_is_not_negative() {
        DamerauLevenshteinComparator cmp = new DamerauLevenshteinComparator();

        // distance("a","b") = 1, maxLen=1 => similarity = 0 => method returns 0
        assertEquals(0.0, cmp.compareBackend(field("a"), field("b")), EPS);

        // OSA-Variante (Transpositionen): distance("CA","ABC") = 3, maxLen=3 => similarity 0.
        assertEquals(0.0, cmp.compareBackend(field("CA"), field("ABC")), EPS);
    }

    @Test
    void Multiple_comparisons_do_not_affect_each_other() {
        DamerauLevenshteinComparator cmp = new DamerauLevenshteinComparator();

        double s1 = cmp.compareBackend(field("test"), field("tent"));
        double s2 = cmp.compareBackend(field("CA"), field("AC"));

        assertEquals(0.75, s1, EPS);
        assertEquals(0.5, s2, EPS);
    }

    @Test
    void Similarity_is_always_between_zero_and_one() {
        DamerauLevenshteinComparator cmp = new DamerauLevenshteinComparator();

        double v = cmp.compareBackend(field("kitten"), field("sitting"));
        assertTrue(v >= 0.0 && v <= 1.0, "similarity out of range: " + v);
    }
}
