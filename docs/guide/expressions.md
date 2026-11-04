# Expressions

Many keys take a string that Mallard evaluates at each point (and, where the
key says so, each time): initial and boundary states, sources, sponges,
regions and probe curves. They are written in kexpr (Kokkos expressions),
Mallard's expression language, compiled once when the input is read and
evaluated by the same code on CPUs and GPUs.

```toml
[initialize]
type = "analytical"
u = ["0.0", "0.0"]
p = "1.0"
# A density bump: a local variable, then the value
rho = "var r2 := (x - 0.5)^2 + (y - 0.5)^2; 1 + 0.2 * exp(-r2 / 0.01)"
```

kexpr follows the grammar, operator semantics and numerics of
[exprtk](https://www.partow.net/programming/exprtk/) (Arash Partow, MIT License), to
the last bit of its literal conversion and integer powers, without its
strings and vectors, and with case-sensitive names. Every key accepts the
same language and reports the same errors, with the expression and the
position of the problem, and `mallard check` reports them before a run. Every
section also evaluates it with the same code, on the device where the solver
runs: the values on GPUs are those on CPUs up to the last bits of library
functions such as `sin` and `exp`.

**Variables.** `x`, `y`, `z` in meters (`z` is 0 in 2D) and, where the key
says so, the time `t` in seconds; sources also read the local and
domain-averaged state ([Sources](sources.md)), initial states the columns of
tabulated data ([Initialization](initialization.md#tabulated-data-and-point-clouds))
and probe curves their parameter `s`. Names are case sensitive: `T` is not `t`.

**Operators and functions.**

| What | Syntax |
|---|---|
| Arithmetic | `+ - * / % ^`; `^` binds from the right and above unary minus (`-x^2` is `-(x^2)`); `2x`, `2(x + 1)` and `(x)(y)` multiply |
| Comparisons | `< <= > >= == !=` (also `=` and `<>`), giving 1 or 0 |
| Logic | `and`, `or`, `xor`, `nand`, `nor`, `xnor`, `&`, `\|`, `not(x)`; any value other than 0 is true |
| Conditionals | `c ? a : b`, `if (c, a, b)`, and the statements below |
| Functions | `abs`, `sgn`, `floor`, `ceil`, `round`, `roundn(x, n)`, `trunc`, `frac`, `min`, `max`, `sum`, `mul`, `avg` (any number of arguments), `clamp(lo, x, hi)`, `inrange(lo, x, hi)`, `sqrt`, `root(x, n)`, `pow(x, y)`, `exp`, `expm1`, `log`, `log10`, `log2`, `log1p`, `logn(x, n)`, `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `asin`, `acos`, `atan`, `atan2(y, x)`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `sinc`, `hypot(x, y)`, `erf`, `erfc`, `ncdf`, `deg2rad`, `rad2deg` |
| Constants | `pi`, `epsilon` (1e-10), `inf`, `true`, `false` |

**Statements.** Statements are separated by `;`; the value of an expression
is that of its last statement. Long expressions fit on several lines with
TOML's `"""` and a trailing `\` on each line.

| Statement | Meaning |
|---|---|
| `var r := sqrt(x^2 + y^2);` | A local variable (0 without `:=`), until the end of its `{ }` block. `x`, `y`, `z`, `t` themselves cannot be assigned |
| `r := 2 r;` `r += 1;` (`-=`, `*=`, `/=`, `%=`) | Assignment; its value is the new value |
| `if (c) a; else b;` | `a` if `c` is true, else `b`; without `else`, NaN when `c` is false. `{ ... }` holds several statements; `else if (d) ...` chains |
| `switch { case c1 : a; case c2 : b; default : d; }` | The value of the first true case, else `d` (NaN without `default`) |
| `while (c) { ... }` | Runs the body while `c` is true |
| `for (var i := 0; i < n; i += 1) { ... }` | Initializer, condition, step (which may be empty), body |
| `repeat ...; until (c)` | Runs the statements, then again until `c` is true |
| `break;` `break[v];` `continue;` | Leave the loop (giving it the value `v`, or NaN), or go on with its next iteration |
| `~{a; b; c}` | Several statements where one value is expected |

A loop's value is that of its body's last complete run (0 if it never ran).
An evaluation stops with an error after 100000 loop iterations in all, so a
loop that never ends cannot hang a run, on the CPU or in a GPU kernel.

```toml
# Piecewise, with if-else
rho = """\
    if (x < 0.4) 1.0; \
    else if (x < 0.6) 1.0 - 4.375 * (x - 0.4); \
    else 0.125;"""

# A square wave from 20 terms of its Fourier series
p = """\
    var s := 0; \
    for (var k := 1; k <= 39; k += 2) { s += sin(2 * pi * k * x) / k }; \
    1 + 0.1 * 4 / pi * s"""

# Layers by height
T = "switch { case y < 0.1 : 300; case y < 0.2 : 600; default : 900; }"
```

```toml
# The root of s^3 + s = x by Newton's method
T = """\
    var s := x; var ds := 1; \
    repeat \
        ds := (s^3 + s - x) / (3 * s^2 + 1); \
        s -= ds; \
    until (abs(ds) < 1e-12); \
    300 + 10 * s"""
```

Not supported, with an error that says so: strings (`'abc'`), vectors
(`var v[3]`), `return`, and functions or variables defined in other
expressions.

