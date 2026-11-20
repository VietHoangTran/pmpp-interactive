// Lab 16.1: the scoring constants of the Smith-Waterman recurrence. Read-only.
//
//   H[r][q] = max(0, H[r-1][q-1] + s, H[r][q-1] + SW_GAP, H[r-1][q] + SW_GAP)
//   s = SW_MATCH if a[r-1] == b[q-1], else SW_MISMATCH
//
// The values are the book's example (p. 382: +3 for a match, -3 for a mismatch, a gap
// penalty of 2); the book does not say why it chose them, and nothing in this lab
// depends on them. The book's kernel ADDS its penalty constants (Fig. 16.9:
// `w + DELETION, n + INSERTION`), so the gap is stored negative here. The book names
// two penalties (insertion, deletion) with the same value; one constant covers both.
//
// main.cpp (the CPU reference) and the .cu files include this header. The Metal
// kernels are compiled at run time and cannot include it, so the driver passes the
// same three values to the Metal compiler as preprocessor definitions.
#pragma once

#define SW_MATCH 3
#define SW_MISMATCH (-3)
#define SW_GAP (-2)
