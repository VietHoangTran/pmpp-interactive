// Lab 8.1: the seven weights of the 3D 7-point stencil. Read-only for you.
//
// Every kernel in this lab (yours in work/, the provided ones in ../shared/) takes
// them by value. The pairing follows book Figs. 8.6 and 8.8 (pp. 188, 191):
//
//   out[i][j][k] = c0 * in[i  ][j  ][k  ]
//                + c1 * in[i  ][j  ][k-1]  +  c2 * in[i  ][j  ][k+1]     x neighbours
//                + c3 * in[i  ][j-1][k  ]  +  c4 * in[i  ][j+1][k  ]     y neighbours
//                + c5 * in[i-1][j  ][k  ]  +  c6 * in[i+1][j  ][k  ]     z neighbours
//
// with i the z index, j the y index and k the x index of an N x N x N grid stored
// row-major, x fastest: element (i, j, k) is at i*N*N + j*N + k.
//
// Figs. 8.10 and 8.12 (pp. 194, 197) print c3 with the y+1 neighbour and c4 with y-1,
// the other way round. The test uses seven different values, so that pairing fails.
//
// kernel.metal declares the same struct (a Metal source compiled at run time cannot
// include this file); the two must keep the same seven floats in the same order.
#pragma once

struct StencilCoeffs {
    float c0, c1, c2, c3, c4, c5, c6;
};
