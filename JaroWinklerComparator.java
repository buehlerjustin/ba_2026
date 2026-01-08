package de.pseudonymisierung.mainzelliste.matcher;

import de.pseudonymisierung.mainzelliste.PlainTextField;

public final class JaroWinklerComparator extends FieldComparator<PlainTextField> {

    private final double scalingFactor;     // Standard ist 0.10
    private final int maxPrefixLength;      // 1-4 möglich
    private final double winklerThreshold;  // häufig ist 0.70 --> in Konfig auslagern ?

    private static final class Buffers {
        int stamp = 1;
        int[] m1 = new int[0];
        int[] m2 = new int[0];
        char[] ms1 = new char[0];
        char[] ms2 = new char[0];
    }

    private static final ThreadLocal<Buffers> THREAD_LOCAL = ThreadLocal.withInitial(Buffers::new);

    public JaroWinklerComparator() {
        this(null, null, 0.10, 4, 0.70);
    }

    public JaroWinklerComparator(String fieldLeft, String fieldRight) {
        this(fieldLeft, fieldRight, 0.10, 4, 0.70);
    }

    public JaroWinklerComparator(String fieldLeft, String fieldRight,
                                     double scalingFactor, int maxPrefixLength, double winklerThreshold) {
        super(fieldLeft, fieldRight);
        this.scalingFactor = scalingFactor;
        this.maxPrefixLength = Math.max(0, maxPrefixLength);
        this.winklerThreshold = winklerThreshold;
    }

    @Override
    public double compareBackend(PlainTextField fieldLeft, PlainTextField fieldRight) {
        if (fieldLeft == null || fieldRight == null) return 0.0;

        final String stringLeft = fieldLeft.getValue();
        final String stringRight = fieldRight.getValue();

        if (stringLeft == null || stringRight == null) return 0.0;
        if (stringLeft == stringRight || stringLeft.equals(stringRight)) return 1.0;

        final int len1 = stringLeft.length();
        final int len2 = stringRight.length();
        if (len1 == 0 || len2 == 0) return 0.0;

        final Buffers buffers = THREAD_LOCAL.get();

        int stamp = ++buffers.stamp;
        if (stamp == 0) {
            buffers.stamp = stamp = 1;
        }

        if (buffers.m1.length < len1) buffers.m1 = new int[len1];
        if (buffers.m2.length < len2) buffers.m2 = new int[len2];

        final int matchDistance = Math.max(0, (Math.max(len1, len2) / 2) - 1);

        int matches = 0;

        for (int i = 0; i < len1; i++) {
            final char c = stringLeft.charAt(i);
            final int start = Math.max(0, i - matchDistance);
            final int end = Math.min(i + matchDistance + 1, len2);

            for (int j = start; j < end; j++) {
                if (buffers.m2[j] != stamp && c == stringRight.charAt(j)) {
                    buffers.m1[i] = stamp;
                    buffers.m2[j] = stamp;
                    matches++;
                    break;
                }
            }
        }

        if (matches == 0) return 0.0;

        if (buffers.ms1.length < matches) buffers.ms1 = new char[Math.max(matches, buffers.ms1.length * 2 + 1)];
        if (buffers.ms2.length < matches) buffers.ms2 = new char[Math.max(matches, buffers.ms2.length * 2 + 1)];

        int k = 0;
        for (int i = 0; i < len1; i++) {
            if (buffers.m1[i] == stamp) buffers.ms1[k++] = stringLeft.charAt(i);
        }
        k = 0;
        for (int j = 0; j < len2; j++) {
            if (buffers.m2[j] == stamp) buffers.ms2[k++] = stringRight.charAt(j);
        }

        int transpositions = 0;
        for (int i = 0; i < matches; i++) {
            if (buffers.ms1[i] != buffers.ms2[i]) transpositions++;
        }
        final double t = transpositions / 2.0;

        final double m = matches;
        final double jaro = (m / len1 + m / len2 + (m - t) / m) / 3.0;

        double jw = jaro;
        if (jaro >= winklerThreshold && scalingFactor > 0.0 && maxPrefixLength > 0) {
            final int max = Math.min(Math.min(maxPrefixLength, len1), len2);
            int prefix = 0;
            while (prefix < max && stringLeft.charAt(prefix) == stringRight.charAt(prefix)) prefix++;
            if (prefix > 0) {
                jw = jaro + prefix * scalingFactor * (1.0 - jaro);
            }
        }

        return Math.min(jw, 1.0);
    }

}
