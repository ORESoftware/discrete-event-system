'use strict';

// RUST MIGRATION:
// - Target: src/des/general/expr.rs
// - Expr is a direct Rust enum with variants Num, Var, Bin, Neg, and Func.
//   The node interfaces become enum payload structs only if variant fields grow.
// - Parser and lexer helpers can remain private module functions. Parse/evaluate
//   errors should return `Result<Expr, ExprError>` and `Result<f64, ExprError>`
//   instead of throwing.
// - FUNC_IMPL becomes a match on FuncName. FuncName itself should be a Rust enum
//   with FromStr/Display impls for parse and stringify.
// - toFunction is JavaScript-specific closure packaging; in Rust prefer an
//   Evaluator struct that owns Expr plus argument ordering, or expose evaluate
//   directly over a HashMap/BTreeMap environment.
// - numericalDerivative/numericalGradient may stay free functions. If used in a
//   DES graph, wrap them later as PureTransform implementors.

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/expr.rs  (module des::general::expr)
// 1:1 file move. AST `type Expr = ... | ...` -> `enum Expr`; variant constructors
// (num/v/add/…) -> `Expr::Num(..)` style; algorithm classes (ExprParser,
// ExprEvaluator, ExprCompiler, ExprPrinter, ExprDifferentiator, ExprSimplifier,
// Numerical*Derivative/Gradient) -> struct + impl Transform; `switch (e.kind)`
// -> `match e`. ExprCompiler returns a closure -> return `impl Fn(&[f64]) -> f64`.
// @deprecated free-fn shims -> drop in Rust.
// =============================================================================

// =============================================================================
// Tiny symbolic expression engine.
//
// Capabilities
//   - Parse human strings: "x^2 * sin(x) + exp(-x)"      (ExprParser)
//   - Construct AST programmatically: mul(num(2), v('x'))
//   - Numerically evaluate over an environment             (ExprEvaluator)
//   - Symbolically differentiate w.r.t. any variable       (ExprDifferentiator)
//   - Algebraic simplification (constant folding, x*0=0, …) (ExprSimplifier)
//   - Pretty-print back to a string                         (ExprPrinter)
//   - Convert to JS function: toFunction(ast, ['x'])
//
// MIGRATION SHAPE
//   `Expr` is a DISCRIMINATED UNION (`kind: …`) — the idiomatic TypeScript
//   stand-in for a Rust `enum`, and every algorithm pattern-matches via
//   `switch (e.kind)`, mapping to a Rust `match`. Each algorithm is a class
//   (a `PureTransform` where it has a clean single-input shape) so the unit of
//   behaviour maps to a Rust `struct + impl`. The variant CONSTRUCTORS
//   (`num`, `v`, `add`, …) stay as small factories — they map to Rust
//   `Expr::Num(..)` style variant construction. Thin `@deprecated` function
//   shims preserve the historical free-function API.
//
// Supported function nodes:
//   sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, log, sqrt, abs
// =============================================================================

import {PureTransform} from '../shared/transform';

// -----------------------------------------------------------------------------
// AST  (discriminated union ⇒ Rust enum)
// -----------------------------------------------------------------------------

export type Expr =
  | NumNode
  | VarNode
  | BinNode
  | UnaryNeg
  | FuncNode;

export interface NumNode    { kind: 'num';  value: number; }
export interface VarNode    { kind: 'var';  name: string; }
export interface BinNode    { kind: 'bin';  op: '+' | '-' | '*' | '/' | '^'; left: Expr; right: Expr; }
export interface UnaryNeg   { kind: 'neg';  arg: Expr; }
export interface FuncNode   { kind: 'func'; name: FuncName; arg: Expr; }

export type FuncName =
  | 'sin' | 'cos' | 'tan'
  | 'asin' | 'acos' | 'atan'
  | 'sinh' | 'cosh' | 'tanh'
  | 'exp' | 'log' | 'sqrt' | 'abs';

const FUNCS: ReadonlySet<string> = new Set([
  'sin','cos','tan','asin','acos','atan','sinh','cosh','tanh',
  'exp','log','ln','sqrt','abs',
]);

// -----------------------------------------------------------------------------
// Variant constructors (programmatic API ⇒ Rust enum-variant construction).
// -----------------------------------------------------------------------------

export const num = (v: number): NumNode => ({kind: 'num', value: v});
export const v   = (name: string): VarNode => ({kind: 'var', name});
export const add = (a: Expr, b: Expr): BinNode => ({kind: 'bin', op: '+', left: a, right: b});
export const sub = (a: Expr, b: Expr): BinNode => ({kind: 'bin', op: '-', left: a, right: b});
export const mul = (a: Expr, b: Expr): BinNode => ({kind: 'bin', op: '*', left: a, right: b});
export const div = (a: Expr, b: Expr): BinNode => ({kind: 'bin', op: '/', left: a, right: b});
export const pow = (a: Expr, b: Expr): BinNode => ({kind: 'bin', op: '^', left: a, right: b});
export const neg = (a: Expr): UnaryNeg => ({kind: 'neg', arg: a});
export const fn  = (name: FuncName, a: Expr): FuncNode => ({kind: 'func', name, arg: a});

export const ZERO = num(0);
export const ONE  = num(1);

/** Evaluate a built-in unary math function by name (Rust `match` on FuncName). */
export class MathFn {
  static apply(name: FuncName, x: number): number {
    switch (name) {
      case 'sin':  return Math.sin(x);
      case 'cos':  return Math.cos(x);
      case 'tan':  return Math.tan(x);
      case 'asin': return Math.asin(x);
      case 'acos': return Math.acos(x);
      case 'atan': return Math.atan(x);
      case 'sinh': return Math.sinh(x);
      case 'cosh': return Math.cosh(x);
      case 'tanh': return Math.tanh(x);
      case 'exp':  return Math.exp(x);
      case 'log':  return Math.log(x);
      case 'sqrt': return Math.sqrt(x);
      case 'abs':  return Math.abs(x);
    }
  }
}

// -----------------------------------------------------------------------------
// Parser: precedence-climbing recursive descent.
//   expression → term ('+' term | '-' term)*
//   term       → factor ('*' factor | '/' factor)*
//   factor     → unary ('^' factor)?       (right-assoc)
//   unary      → '-' unary | atom
//   atom       → number | variable | function '(' expression ')' | '(' expression ')'
// -----------------------------------------------------------------------------

interface Token {
  kind: 'num' | 'op' | 'lparen' | 'rparen' | 'ident' | 'comma' | 'eof';
  text: string;
}

/**
 * Stateful recursive-descent parser. Owns the scan position (`i`) and the
 * current token as instance fields, so the lexer/parser scratch state lives on
 * `this` rather than in free functions. Maps to a Rust struct + impl.
 */
export class ExprParser extends PureTransform<string, Expr> {
  private s = '';
  private i = 0;
  private cur: Token = {kind: 'eof', text: ''};

  transform(src: string): Expr {
    this.s = src;
    this.i = 0;
    this.cur = this.nextToken();
    const e = this.parseExpression();
    if (this.cur.kind !== 'eof') {
      console.warn(`[expr.parse] trailing token "${this.cur.text}" after a complete expression in ${JSON.stringify(src)} — likely a missing operator or extra characters.`);
      throw new Error(`unexpected trailing token "${this.cur.text}"`);
    }
    return e;
  }

  private skipWS(): void {
    while (this.i < this.s.length && /\s/.test(this.s[this.i])) this.i++;
  }

  private nextToken(): Token {
    this.skipWS();
    if (this.i >= this.s.length) return {kind: 'eof', text: ''};
    const c = this.s[this.i];
    if (c === '(') { this.i++; return {kind: 'lparen', text: '('}; }
    if (c === ')') { this.i++; return {kind: 'rparen', text: ')'}; }
    if (c === ',') { this.i++; return {kind: 'comma',  text: ','}; }
    if ('+-*/^'.includes(c)) { this.i++; return {kind: 'op', text: c}; }
    if (/[0-9.]/.test(c)) {
      let j = this.i;
      while (j < this.s.length && /[0-9.]/.test(this.s[j])) j++;
      if (j < this.s.length && (this.s[j] === 'e' || this.s[j] === 'E')) {
        j++;
        if (j < this.s.length && (this.s[j] === '+' || this.s[j] === '-')) j++;
        while (j < this.s.length && /[0-9]/.test(this.s[j])) j++;
      }
      const text = this.s.slice(this.i, j);
      this.i = j;
      return {kind: 'num', text};
    }
    if (/[a-zA-Z_]/.test(c)) {
      let j = this.i;
      while (j < this.s.length && /[a-zA-Z_0-9]/.test(this.s[j])) j++;
      const text = this.s.slice(this.i, j);
      this.i = j;
      return {kind: 'ident', text};
    }
    console.warn(`[expr.lex] unexpected character '${c}' at position ${this.i} in ${JSON.stringify(this.s)}.`);
    throw new Error(`unexpected char '${c}' at position ${this.i}`);
  }

  private advance(): Token { const t = this.cur; this.cur = this.nextToken(); return t; }

  private expect(kind: Token['kind'], text?: string): Token {
    if (this.cur.kind !== kind || (text !== undefined && this.cur.text !== text)) {
      console.warn(`[expr.parse] syntax error: expected ${kind}${text ? ' "' + text + '"' : ''} but got ${this.cur.kind} "${this.cur.text}" at position ${this.i}.`);
      throw new Error(`expected ${kind}${text ? ' "' + text + '"' : ''}, got ${this.cur.kind} "${this.cur.text}"`);
    }
    return this.advance();
  }

  private parseExpression(): Expr {
    let left = this.parseTerm();
    while (this.cur.kind === 'op' && (this.cur.text === '+' || this.cur.text === '-')) {
      const op = this.advance().text as '+' | '-';
      const right = this.parseTerm();
      left = {kind: 'bin', op, left, right};
    }
    return left;
  }

  private parseTerm(): Expr {
    let left = this.parseFactor();
    while (this.cur.kind === 'op' && (this.cur.text === '*' || this.cur.text === '/')) {
      const op = this.advance().text as '*' | '/';
      const right = this.parseFactor();
      left = {kind: 'bin', op, left, right};
    }
    return left;
  }

  private parseFactor(): Expr {
    const left = this.parseUnary();
    if (this.cur.kind === 'op' && this.cur.text === '^') {
      this.advance();
      const right = this.parseFactor();   // right-associative
      return {kind: 'bin', op: '^', left, right};
    }
    return left;
  }

  private parseUnary(): Expr {
    if (this.cur.kind === 'op' && this.cur.text === '-') {
      this.advance();
      return {kind: 'neg', arg: this.parseUnary()};
    }
    if (this.cur.kind === 'op' && this.cur.text === '+') {
      this.advance();
      return this.parseUnary();
    }
    return this.parseAtom();
  }

  private parseAtom(): Expr {
    if (this.cur.kind === 'num') {
      const t = this.advance();
      return {kind: 'num', value: parseFloat(t.text)};
    }
    if (this.cur.kind === 'lparen') {
      this.advance();
      const e = this.parseExpression();
      this.expect('rparen');
      return e;
    }
    if (this.cur.kind === 'ident') {
      const t = this.advance();
      if ((this.cur as Token).kind === 'lparen' && FUNCS.has(t.text)) {
        this.advance();
        const arg = this.parseExpression();
        this.expect('rparen');
        const fname: FuncName = (t.text === 'ln' ? 'log' : t.text) as FuncName;
        return {kind: 'func', name: fname, arg};
      }
      return {kind: 'var', name: t.text};
    }
    console.warn(`[expr.parse] unexpected token ${this.cur.kind} "${this.cur.text}" while parsing an atom (position ${this.i}).`);
    throw new Error(`unexpected token ${this.cur.kind} "${this.cur.text}"`);
  }
}

// -----------------------------------------------------------------------------
// Numerical evaluation.
// -----------------------------------------------------------------------------

export type Env = Record<string, number>;

export interface EvalInput {
  expr: Expr;
  env: Env;
}

/** Evaluate an AST over a variable environment. */
export class ExprEvaluator extends PureTransform<EvalInput, number> {
  transform(input: EvalInput): number {
    return this.eval(input.expr, input.env);
  }

  eval(e: Expr, env: Env): number {
    switch (e.kind) {
      case 'num': return e.value;
      case 'var': {
        const value = env[e.name];
        if (value === undefined) {
          console.warn(`[expr.evaluate] undefined variable '${e.name}' (env has: ${Object.keys(env).join(', ') || '(empty)'}) — variable not bound in the evaluation environment.`);
          throw new Error(`undefined variable '${e.name}'`);
        }
        return value;
      }
      case 'neg': return -this.eval(e.arg, env);
      case 'func': return MathFn.apply(e.name, this.eval(e.arg, env));
      case 'bin': {
        const a = this.eval(e.left, env);
        const b = this.eval(e.right, env);
        switch (e.op) {
          case '+': return a + b;
          case '-': return a - b;
          case '*': return a * b;
          case '/': return a / b;
          case '^': return Math.pow(a, b);
        }
      }
    }
  }
}

/**
 * Compile an expression to a JS function over a fixed argument list. Faster
 * than re-walking the AST repeatedly; the generated function does NOT use eval
 * — it walks the AST under closure capture.
 */
export class ExprCompiler extends PureTransform<{expr: Expr; args: string[]}, (...vals: number[]) => number> {
  private readonly evaluator = new ExprEvaluator();

  transform(input: {expr: Expr; args: string[]}): (...vals: number[]) => number {
    const {expr, args} = input;
    const evaluator = this.evaluator;
    return (...vals: number[]) => {
      const env: Env = {};
      for (let i = 0; i < args.length; i++) env[args[i]] = vals[i];
      return evaluator.eval(expr, env);
    };
  }
}

// -----------------------------------------------------------------------------
// Pretty-printer.
// -----------------------------------------------------------------------------

const PREC: Record<string, number> = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3, 'neg': 4};

/** Render an AST back to an infix string with minimal parenthesization. */
export class ExprPrinter extends PureTransform<Expr, string> {
  transform(e: Expr): string {
    return this.print(e, 0);
  }

  print(e: Expr, parentPrec: number): string {
    switch (e.kind) {
      case 'num': {
        const value = e.value;
        if (value < 0) return `(${value})`;
        return Number.isInteger(value) ? value.toFixed(0) : value.toString();
      }
      case 'var': return e.name;
      case 'neg': {
        const inner = this.print(e.arg, PREC['neg']);
        return PREC['neg'] < parentPrec ? `(-${inner})` : `-${inner}`;
      }
      case 'func': return `${e.name}(${this.print(e.arg, 0)})`;
      case 'bin': {
        const p = PREC[e.op];
        const l = this.print(e.left, p);
        const r = this.print(e.right, e.op === '^' ? p - 1 : p + 1);
        const s = `${l} ${e.op} ${r}`;
        return p < parentPrec ? `(${s})` : s;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Symbolic differentiation.
// -----------------------------------------------------------------------------

/**
 * d/dvar of an expression. Returns a new AST, simplified (constant folding +
 * identity rules) so derivatives stay readable.
 */
export class ExprDifferentiator extends PureTransform<{expr: Expr; varName: string}, Expr> {
  private readonly simplifier = new ExprSimplifier();

  transform(input: {expr: Expr; varName: string}): Expr {
    return this.simplifier.transform(this.diffRaw(input.expr, input.varName));
  }

  private diffRaw(e: Expr, x: string): Expr {
    switch (e.kind) {
      case 'num': return ZERO;
      case 'var': return e.name === x ? ONE : ZERO;
      case 'neg': return neg(this.diffRaw(e.arg, x));
      case 'bin': {
        const u = e.left, w = e.right;
        const du = this.diffRaw(u, x), dw = this.diffRaw(w, x);
        switch (e.op) {
          case '+': return add(du, dw);
          case '-': return sub(du, dw);
          case '*': return add(mul(du, w), mul(u, dw));   // (uv)' = u'v + uv'
          case '/': return div(sub(mul(du, w), mul(u, dw)), pow(w, num(2)));   // (u/v)' = (u'v − uv')/v²
          case '^': {
            // d/dx [u^v]: if v is a constant, easy:  v · u^(v-1) · u'
            // general:  u^v · (v' · ln u + v · u'/u)
            if (w.kind === 'num') {
              return mul(mul(num(w.value), pow(u, num(w.value - 1))), du);
            }
            return mul(pow(u, w), add(mul(dw, fn('log', u)), mul(w, div(du, u))));
          }
        }
      }
      case 'func': {
        const u = e.arg;
        const du = this.diffRaw(u, x);
        switch (e.name) {
          case 'sin':  return mul(fn('cos', u), du);
          case 'cos':  return mul(neg(fn('sin', u)), du);
          case 'tan':  return mul(div(ONE, pow(fn('cos', u), num(2))), du);
          case 'asin': return mul(div(ONE, fn('sqrt', sub(ONE, pow(u, num(2))))), du);
          case 'acos': return mul(neg(div(ONE, fn('sqrt', sub(ONE, pow(u, num(2)))))), du);
          case 'atan': return mul(div(ONE, add(ONE, pow(u, num(2)))), du);
          case 'sinh': return mul(fn('cosh', u), du);
          case 'cosh': return mul(fn('sinh', u), du);
          case 'tanh': return mul(sub(ONE, pow(fn('tanh', u), num(2))), du);
          case 'exp':  return mul(fn('exp', u), du);
          case 'log':  return mul(div(ONE, u), du);
          case 'sqrt': return mul(div(ONE, mul(num(2), fn('sqrt', u))), du);
          case 'abs':  return mul(div(u, fn('abs', u)), du);   // weak: undefined at 0
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Simplification: one bottom-up pass with constant folding and identity rules.
// -----------------------------------------------------------------------------

export class ExprSimplifier extends PureTransform<Expr, Expr> {
  transform(e: Expr): Expr {
    switch (e.kind) {
      case 'num': case 'var': return e;
      case 'neg': {
        const a = this.transform(e.arg);
        if (a.kind === 'num') return num(-a.value);
        if (a.kind === 'neg') return a.arg;            // --x = x
        return neg(a);
      }
      case 'func': {
        const a = this.transform(e.arg);
        if (a.kind === 'num') {
          return num(MathFn.apply(e.name, a.value));
        }
        return fn(e.name, a);
      }
      case 'bin': {
        const a = this.transform(e.left);
        const b = this.transform(e.right);
        // Constant folding.
        if (a.kind === 'num' && b.kind === 'num') {
          switch (e.op) {
            case '+': return num(a.value + b.value);
            case '-': return num(a.value - b.value);
            case '*': return num(a.value * b.value);
            case '/': if (b.value !== 0) return num(a.value / b.value); break;
            case '^': return num(Math.pow(a.value, b.value));
          }
        }
        // Identity rules.
        switch (e.op) {
          case '+':
            if (a.kind === 'num' && a.value === 0) return b;
            if (b.kind === 'num' && b.value === 0) return a;
            break;
          case '-':
            if (b.kind === 'num' && b.value === 0) return a;
            if (a.kind === 'num' && a.value === 0) return neg(b);
            break;
          case '*':
            if (a.kind === 'num' && a.value === 0) return ZERO;
            if (b.kind === 'num' && b.value === 0) return ZERO;
            if (a.kind === 'num' && a.value === 1) return b;
            if (b.kind === 'num' && b.value === 1) return a;
            if (a.kind === 'num' && a.value === -1) return this.transform(neg(b));
            if (b.kind === 'num' && b.value === -1) return this.transform(neg(a));
            break;
          case '/':
            if (b.kind === 'num' && b.value === 1) return a;
            if (a.kind === 'num' && a.value === 0) return ZERO;
            break;
          case '^':
            if (b.kind === 'num' && b.value === 0) return ONE;
            if (b.kind === 'num' && b.value === 1) return a;
            if (a.kind === 'num' && a.value === 1) return ONE;
            if (a.kind === 'num' && a.value === 0 && b.kind === 'num' && b.value > 0) return ZERO;
            break;
        }
        return {kind: 'bin', op: e.op, left: a, right: b};
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Numerical derivatives for black-box JS functions.
// -----------------------------------------------------------------------------

/** Central-difference derivative; O(h²). Choose h ≈ 1e-6 for f64 problems. */
export class NumericalDerivative extends PureTransform<{f: (x: number) => number; x: number}, number> {
  constructor(private readonly h = 1e-6) { super(); }

  transform(input: {f: (x: number) => number; x: number}): number {
    const {f, x} = input;
    return (f(x + this.h) - f(x - this.h)) / (2 * this.h);
  }
}

/** Richardson-extrapolated derivative; O(h⁴). */
export class RichardsonDerivative extends PureTransform<{f: (x: number) => number; x: number}, number> {
  constructor(private readonly h = 1e-3) { super(); }

  transform(input: {f: (x: number) => number; x: number}): number {
    const {f, x} = input;
    const h = this.h;
    const d1 = (f(x + h) - f(x - h)) / (2 * h);
    const d2 = (f(x + h / 2) - f(x - h / 2)) / h;
    return (4 * d2 - d1) / 3;
  }
}

/** Central-difference gradient for f: Rⁿ → R. */
export class NumericalGradient extends PureTransform<{f: (x: number[]) => number; x: number[]}, number[]> {
  constructor(private readonly h = 1e-6) { super(); }

  transform(input: {f: (x: number[]) => number; x: number[]}): number[] {
    const {f, x} = input;
    const h = this.h;
    const n = x.length;
    const grad = new Array<number>(n);
    const xc = x.slice();
    for (let i = 0; i < n; i++) {
      const orig = xc[i];
      xc[i] = orig + h; const fp = f(xc);
      xc[i] = orig - h; const fm = f(xc);
      xc[i] = orig;
      grad[i] = (fp - fm) / (2 * h);
    }
    return grad;
  }
}

// -----------------------------------------------------------------------------
// Backward-compatible function shims (kept thin; prefer the classes above).
// -----------------------------------------------------------------------------

/** @deprecated Use `new ExprParser().transform(src)`. */
export function parse(src: string): Expr {
  return new ExprParser().transform(src);
}

/** @deprecated Use `new ExprEvaluator().transform({expr, env})`. */
export function evaluate(e: Expr, env: Env): number {
  return new ExprEvaluator().eval(e, env);
}

/** @deprecated Use `new ExprCompiler().transform({expr, args})`. */
export function toFunction(e: Expr, args: string[]): (...vals: number[]) => number {
  return new ExprCompiler().transform({expr: e, args});
}

/** @deprecated Use `new ExprPrinter().transform(e)`. */
export function stringify(e: Expr, parentPrec = 0): string {
  return new ExprPrinter().print(e, parentPrec);
}

/** @deprecated Use `new ExprDifferentiator().transform({expr, varName})`. */
export function diff(e: Expr, varName: string): Expr {
  return new ExprDifferentiator().transform({expr: e, varName});
}

/** @deprecated Use `new ExprSimplifier().transform(e)`. */
export function simplify(e: Expr): Expr {
  return new ExprSimplifier().transform(e);
}

/** @deprecated Use `new NumericalDerivative(h).transform({f, x})`. */
export function numericalDerivative(f: (x: number) => number, x: number, h = 1e-6): number {
  return new NumericalDerivative(h).transform({f, x});
}

/** @deprecated Use `new RichardsonDerivative(h).transform({f, x})`. */
export function richardsonDerivative(f: (x: number) => number, x: number, h = 1e-3): number {
  return new RichardsonDerivative(h).transform({f, x});
}

/** @deprecated Use `new NumericalGradient(h).transform({f, x})`. */
export function numericalGradient(f: (x: number[]) => number, x: number[], h = 1e-6): number[] {
  return new NumericalGradient(h).transform({f, x});
}
