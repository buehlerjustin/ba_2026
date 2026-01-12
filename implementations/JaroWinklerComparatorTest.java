package de.pseudonymisierung.mainzelliste.matcher;

import de.pseudonymisierung.mainzelliste.PlainTextField;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.*;

class JaroWinklerComparatorTest {

    //um unterschiedlichen Rundungen vorzubeugen --> kleines Delta
    private static final double DELTA = 1e-12;

    //Helfer-Methode
    private static PlainTextField field(String value) {
        PlainTextField f = new PlainTextField();
        f.setValue(value);
        return f;
    }

    //forciert Overflow des stamp Counters, sodass "if (stamp == 0)" ausgeführt wird
    private static void forceStampToHitZeroNextCall() {
        try {
            Field tlField = JaroWinklerComparator.class.getDeclaredField("THREAD_LOCAL");
            tlField.setAccessible(true);
            ThreadLocal<?> tl = (ThreadLocal<?>) tlField.get(null);

            Object buffers = tl.get();
            Field stampField = buffers.getClass().getDeclaredField("stamp");
            stampField.setAccessible(true);
            stampField.setInt(buffers, -1); // ++stamp -> 0
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("Failed to force stamp wrap", e);
        }
    }

    // Tests für eventuelle Null-Felder
    @Test
    void Similarity_is_zero_when_a_field_is_missing() {
        JaroWinklerComparator cmp = new JaroWinklerComparator();

        assertEquals(0.0, cmp.compareBackend(null, field("abc")), DELTA);
        assertEquals(0.0, cmp.compareBackend(field("abc"), null), DELTA);
    }

    @Test
    void Similarity_is_zero_when_a_field_value_is_missing() {
        JaroWinklerComparator cmp = new JaroWinklerComparator();

        assertEquals(0.0, cmp.compareBackend(field(null), field("abc")), DELTA);
        assertEquals(0.0, cmp.compareBackend(field("abc"), field(null)), DELTA);
        assertEquals(0.0, cmp.compareBackend(field(null), field(null)), DELTA);
    }

    @Test
    void Similarity_is_one_for_the_same_string_instance() {
        JaroWinklerComparator cmp = new JaroWinklerComparator();

        String same = "abc";
        assertEquals(1.0, cmp.compareBackend(field(same), field(same)), DELTA);
    }

    @Test
    void Similarity_is_one_for_two_equal_strings() {
        JaroWinklerComparator cmp = new JaroWinklerComparator();

        String left = "abc";
        String right = "abc";
        assertNotSame(left, right);
        assertEquals(1.0, cmp.compareBackend(field(left), field(right)), DELTA);
    }

    @Test
    void Similarity_is_zero_when_one_string_is_empty() {
        JaroWinklerComparator cmp = new JaroWinklerComparator();

        assertEquals(0.0, cmp.compareBackend(field(""), field("a")), DELTA);
        assertEquals(0.0, cmp.compareBackend(field("a"), field("")), DELTA);
    }

    @Test
    void Similarity_is_zero_when_there_are_no_character_matches() {
        JaroWinklerComparator cmp = new JaroWinklerComparator();

        // Für Länge 2, match-Distanz ist 0, nur Matches mit selbem Index erlaubt
        assertEquals(0.0, cmp.compareBackend(field("ab"), field("ba")), DELTA);
    }

    @Test
    void Similarity_matches_reference_values_for_standard_examples() {
        JaroWinklerComparator cmp = new JaroWinklerComparator();

        // Standard example values (scaling=0.1, maxPrefix=4, threshold=0.7)
        assertEquals(0.9611111111111111, cmp.compareBackend(field("MARTHA"), field("MARHTA")), 1e-12);
        assertEquals(0.8133333333333332, cmp.compareBackend(field("DIXON"), field("DICKSONX")), 1e-12);
        assertEquals(0.8962962962962964, cmp.compareBackend(field("JELLYFISH"), field("SMELLYFISH")), 1e-12);
    }

    @Test
    void Winkler_boost_is_not_applied_when_base_similarity_is_below_the_threshold() {
        JaroWinklerComparator cmp = new JaroWinklerComparator();

        // 1 matching Character ('a'), also ist Jaro niedrig (< 0.7) und Winkler-Boost wird nicht ausgeführt
        // Jaro = (1/5 + 1/5 + 1)/3 = 0.46666...
        assertEquals(0.4666666666666667, cmp.compareBackend(field("a0000"), field("a9999")), 1e-12);
    }

    @Test
    void Winkler_boost_is_not_applied_when_strings_have_no_common_prefix() {
        JaroWinklerComparator cmp = new JaroWinklerComparator();

        // Jaro >= 0.7, aber Preifx-Länge ist 0
        assertEquals(0.7777777777777777, cmp.compareBackend(field("abc"), field("xbc")), 1e-12);
    }

    @Test
    void Winkler_boost_is_disabled_when_scaling_factor_is_zero() {
        // Mit scalingFactor = 0, Winkler-Boost sollte nicht ausgeführt werden.
        JaroWinklerComparator cmp = new JaroWinklerComparator(null, null, 0.0, 4, 0.70);

        // Nur Jaro für MARTHA/MARHTA ist 0.9444444444444445
        assertEquals(0.9444444444444445, cmp.compareBackend(field("MARTHA"), field("MARHTA")), 1e-12);
    }

    @Test
    void Similarity_is_capped_at_one() {
        // Wähle Parameter die Winkler-Boost über 1.0 ergeben würden
        JaroWinklerComparator cmp = new JaroWinklerComparator(null, null, 1.0, 10, 0.0);

        assertEquals(1.0, cmp.compareBackend(field("test"), field("test1")), DELTA);
    }

    @Test
    void Comparison_works_even_when_the_internal_stamp_counter_wraps_around() {
        forceStampToHitZeroNextCall();

        JaroWinklerComparator cmp = new JaroWinklerComparator();

        // Irgendein nicht-trivialer Vergleic, der stamp-Inkrementierung auslöst
        assertEquals(0.7777777777777777, cmp.compareBackend(field("abc"), field("xbc")), 1e-12);
    }

    @Test
    void Multiple_comparisons_do_not_affect_each_other() {
        JaroWinklerComparator cmp = new JaroWinklerComparator();

        double first = cmp.compareBackend(field("MARTHA"), field("MARHTA"));
        double second = cmp.compareBackend(field("DIXON"), field("DICKSONX"));

        assertEquals(0.9611111111111111, first, 1e-12);
        assertEquals(0.8133333333333332, second, 1e-12);
    }
}
