
# flamp - Faster linear algebra with multiple precision [![Build Status](https://github.com/c-f-h/flamp/actions/workflows/python-package.yml/badge.svg)](https://github.com/c-f-h/flamp/actions/workflows/python-package.yml)

`flamp` contains ports of many real and complex linear algebra routines from
the [`mpmath`](https://mpmath.org/) package, but using numpy object arrays
containing `gmpy2` multiprecision floating point numbers instead of the
`mpmath` floating point numbers. The resulting linear algebra routines are
typically by a factor of 10x-15x faster than those in `mpmath`.

`flamp` is based on `mpmath` by Fredrik Johansson and the matrix algorithms
by Timo Hartmann contained therein. `flamp` is BSD-licensed.
