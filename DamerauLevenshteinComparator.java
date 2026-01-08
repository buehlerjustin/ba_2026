package de.pseudonymisierung.mainzelliste.matcher;

import de.pseudonymisierung.mainzelliste.PlainTextField;

public final class DamerauLevenshteinComparator extends FieldComparator<PlainTextField> {

    private static final class Buffers {
        int stamp = 1;

        int[] m0 = new int[0];
        int[] m1 = new int[0];
        int[] m2 = new int[0];

        char[] ms1 = new char[0];
        char[] ms2 = new char[0];
    }

    private static final ThreadLocal<Buffers> THREAD_LOCAL = ThreadLocal.withInitial(Buffers::new);

    public DamerauLevenshteinComparator() {
        this(null, null);
    }

    public DamerauLevenshteinComparator(String fieldLeft, String fieldRight) {
        super(fieldLeft, fieldRight);
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

        final int maxLen = Math.max(len1, len2);

        String s = stringLeft;
        String t = stringRight;
        int n = len1;
        int m = len2;

        if (m > n) {
            final String tmpStr = s; s = t; t = tmpStr;
            final int tmpLen = n; n = m; m = tmpLen;
        }

        final Buffers buffers = THREAD_LOCAL.get();


        if (buffers.ms1.length < n) buffers.ms1 = new char[Math.max(n, buffers.ms1.length * 2 + 1)];
        if (buffers.ms2.length < m) buffers.ms2 = new char[Math.max(m, buffers.ms2.length * 2 + 1)];

        s.getChars(0, n, buffers.ms1, 0);
        t.getChars(0, m, buffers.ms2, 0);

        final int rowSize = m + 1;
        if (buffers.m0.length < rowSize) buffers.m0 = new int[Math.max(rowSize, buffers.m0.length * 2 + 1)];
        if (buffers.m1.length < rowSize) buffers.m1 = new int[Math.max(rowSize, buffers.m1.length * 2 + 1)];
        if (buffers.m2.length < rowSize) buffers.m2 = new int[Math.max(rowSize, buffers.m2.length * 2 + 1)];

        int[] prevPrev = buffers.m0; // row i-2
        int[] prev = buffers.m1;     // row i-1
        int[] curr = buffers.m2;     // row i

        for (int j = 0; j <= m; j++) {
            prev[j] = j;
            prevPrev[j] = j;
        }

        for (int i = 1; i <= n; i++) {
            curr[0] = i;

            final char s_i_1 = buffers.ms1[i - 1];

            for (int j = 1; j <= m; j++) {
                final char t_j_1 = buffers.ms2[j - 1];

                final int cost = (s_i_1 == t_j_1) ? 0 : 1;

                int best = prev[j] + 1;             // Deletion
                final int ins = curr[j - 1] + 1;    // Insertion
                if (ins < best) best = ins;

                final int sub = prev[j - 1] + cost; // Substitution
                if (sub < best) best = sub;

                // Adjacent Transposition (Optimal String Alignment)
                if (i > 1 && j > 1
                        && s_i_1 == buffers.ms2[j - 2]
                        && buffers.ms1[i - 2] == t_j_1) {
                    final int tr = prevPrev[j - 2] + 1;
                    if (tr < best) best = tr;
                }

                curr[j] = best;
            }

            final int[] tmp = prevPrev;
            prevPrev = prev;
            prev = curr;
            curr = tmp;
        }

        buffers.m0 = prevPrev;
        buffers.m1 = prev;
        buffers.m2 = curr;

        final int distance = prev[m];
        final double similarity = 1.0 - (distance / (double) maxLen);

        return (similarity <= 0.0) ? 0.0 : Math.min(similarity, 1.0);
    }
}
