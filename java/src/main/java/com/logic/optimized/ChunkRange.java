package com.logic.optimized;

/**
 * Represents the row boundaries of a data chunk.
 */
public record ChunkRange(int startInclusive, int endExclusive) {}
