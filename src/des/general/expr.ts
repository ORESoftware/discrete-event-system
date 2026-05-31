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
// Tiny symbolic expression engine.
//
// Capabilities
//   - Parse human strings: "x^2 * sin(x) + exp(-x)"
//   - Construct AST programmatically: mul(num(2), v('x'))
//   - Numerically evaluate over an environment: eval(ast, {x: 2})
//   - Symbolically differentiate w.r.t. any variable: diff(ast, 'x')
//   - Algebraic simplification (constant folding, x*0=0, x*1=x, …)
//   - Pretty-print back to a string: stringify(ast)
//   - Convert to JS function: toFunction(ast, ['x']) → (x) => number
//
// Supported nodes:
//   NumNode      a literal number
//   VarNode      a named variable
//   BinNode      Add | Sub | Mul | Div | Pow
//   UnaryNeg     unary minus
//   FuncNode     sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
//                exp, log (natural), sqrt, abs
//
// Differentiation rules cover all of the above. Simplification is
// conservative (one pass of constant folding + identity rules) to keep
// derivatives readable; it does not attempt full canonicalisation.
//
// This module is the "math language" used by main-calculus.ts and any
// solver that wants symbolic input (gradients, Jacobians). Every solver
// also accepts plain JS functions, so the expression layer is optional.
// =============================================================================

// -----------------------------------------------------------------------------
// AST
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
// Construction helpers (programmatic API).
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

// -----------------------------------------------------------------------------
// Parser: precedence-climbing recursive descent.
//   expression → term ('+' term | '-' term)*
//   term       → factor ('*' factor | '/' factor)*
//   factor     → unary ('^' factor)?       (right-assoc)
//   unary      → '-' unary | atom
//   atom       → number | variable | function '(' expression ')' | '(' expression ')'
// -----------------------------------------------------------------------------

interface Lexer {
  s: string;
  i: number;
}

function peekChar(L: Lexer): string { return L.i < L.s.length ? L.s[L.i] : ''; }
function skipWS(L: Lexer): void {
  while (L.i < L.s.length && /\s/.test(L.s[L.i])) L.i++;
}

interface Token {
  kind: 'num' | 'op' | 'lparen' | 'rparen' | 'ident' | 'comma' | 'eof';
  text: string;
}

function nextToken(L: Lexer): Token {
  skipWS(L);
  if (L.i >= L.s.length) return {kind: 'eof', text: ''};
  const c = L.s[L.i];
  if (c === '(') { L.i++; return {kind: 'lparen', text: '('}; }
  if (c === ')') { L.i++; return {kind: 'rparen', text: ')'}; }
  if (c === ',') { L.i++; return {kind: 'comma',  text: ','}; }
  if ('+-*/^'.includes(c)) { L.i++; return {kind: 'op', text: c}; }
  if (/[0-9.]/.test(c)) {
    let j = L.i;
    while (j < L.s.length && /[0-9.]/.test(L.s[j])) j++;
    if (j < L.s.length && (L.s[j] === 'e' || L.s[j] === 'E')) {
      j++;
      if (j < L.s.length && (L.s[j] === '+' || L.s[j] === '-')) j++;
      while (j < L.s.length && /[0-9]/.test(L.s[j])) j++;
    }
    const text = L.s.slice(L.i, j);
    L.i = j;
    return {kind: 'num', text};
  }
  if (/[a-zA-Z_]/.test(c)) {
    let j = L.i;
    while (j < L.s.length && /[a-zA-Z_0-9]/.test(L.s[j])) j++;
    const text = L.s.slice(L.i, j);
    L.i = j;
    return {kind: 'ident', text};
  }
  console.warn(`[expr.lex] unexpected character '${c}' at position ${L.i} in ${JSON.stringify(L.s)}.`);
  throw new Error(`unexpected char '${c}' at position ${L.i}`);
}

interface Parser {
  L: Lexer;
  cur: Token;
}

function advance(P: Parser): Token { const t = P.cur; P.cur = nextToken(P.L); return t; }
function expect(P: Parser, kind: Token['kind'], text?: string): Token {
  if (P.cur.kind !== kind || (text !== undefined && P.cur.text !== text)) {
    console.warn(`[expr.parse] syntax error: expected ${kind}${text ? ' "' + text + '"' : ''} but got ${P.cur.kind} "${P.cur.text}" at position ${P.L.i}.`);
    throw new Error(`expected ${kind}${text ? ' "' + text + '"' : ''}, got ${P.cur.kind} "${P.cur.text}"`);
  }
  return advance(P);
}

function parseExpression(P: Parser): Expr {
  let left = parseTerm(P);
  while (P.cur.kind === 'op' && (P.cur.text === '+' || P.cur.text === '-')) {
    const op = advance(P).text as '+' | '-';
    const right = parseTerm(P);
    left = {kind: 'bin', op, left, right};
  }
  return left;
}
function parseTerm(P: Parser): Expr {
  let left = parseFactor(P);
  while (P.cur.kind === 'op' && (P.cur.text === '*' || P.cur.text === '/')) {
    const op = advance(P).text as '*' | '/';
    const right = parseFactor(P);
    left = {kind: 'bin', op, left, right};
  }
  return left;
}
function parseFactor(P: Parser): Expr {
  const left = parseUnary(P);
  if (P.cur.kind === 'op' && P.cur.text === '^') {
    advance(P);
    const right = parseFactor(P);   // right-associative
    return {kind: 'bin', op: '^', left, right};
  }
  return left;
}
function parseUnary(P: Parser): Expr {
  if (P.cur.kind === 'op' && P.cur.text === '-') {
    advance(P);
    return {kind: 'neg', arg: parseUnary(P)};
  }
  if (P.cur.kind === 'op' && P.cur.text === '+') {
    advance(P);
    return parseUnary(P);
  }
  return parseAtom(P);
}
function parseAtom(P: Parser): Expr {
  if (P.cur.kind === 'num') {
    const t = advance(P);
    return {kind: 'num', value: parseFloat(t.text)};
  }
  if (P.cur.kind === 'lparen') {
    advance(P);
    const e = parseExpression(P);
    expect(P, 'rparen');
    return e;
  }
  if (P.cur.kind === 'ident') {
    const t = advance(P);
    if ((P.cur as Token).kind === 'lparen' && FUNCS.has(t.text)) {
      advance(P);
      const arg = parseExpression(P);
      expect(P, 'rparen');
      const fname: FuncName = (t.text === 'ln' ? 'log' : t.text) as FuncName;
      return {kind: 'func', name: fname, arg};
    }
    return {kind: 'var', name: t.text};
  }
  console.warn(`[expr.parse] unexpected token ${P.cur.kind} "${P.cur.text}" while parsing an atom (position ${P.L.i}).`);
  throw new Error(`unexpected token ${P.cur.kind} "${P.cur.text}"`);
}

export function parse(src: string): Expr {
  const L: Lexer = {s: src, i: 0};
  const P: Parser = {L, cur: nextToken(L)};
  const e = parseExpression(P);
  if (P.cur.kind !== 'eof') {
    console.warn(`[expr.parse] trailing token "${P.cur.text}" after a complete expression in ${JSON.stringify(src)} — likely a missing operator or extra characters.`);
    throw new Error(`unexpected trailing token "${P.cur.text}"`);
  }
  return e;
}

// -----------------------------------------------------------------------------
// Numerical evaluation.
// -----------------------------------------------------------------------------

export type Env = Record<string, number>;

const FUNC_IMPL: Record<FuncName, (x: number) => number> = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan,
  asin: Math.asin, acos: Math.acos, atan: Math.atan,
  sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
  exp: Math.exp, log: Math.log, sqrt: Math.sqrt, abs: Math.abs,
};

export function evaluate(e: Expr, env: Env): number {
  switch (e.kind) {
    case 'num': return e.value;
    case 'var': {
      const v = env[e.name];
      if (v === undefined) {
        console.warn(`[expr.evaluate] undefined variable '${e.name}' (env has: ${Object.keys(env).join(', ') || '(empty)'}) — variable not bound in the evaluation environment.`);
        throw new Error(`undefined variable '${e.name}'`);
      }
      return v;
    }
    case 'neg': return -evaluate(e.arg, env);
    case 'func': return FUNC_IMPL[e.name](evaluate(e.arg, env));
    case 'bin': {
      const a = evaluate(e.left, env);
      const b = evaluate(e.right, env);
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

/**
 * Compile an expression to a JS function over a fixed argument list.
 * Faster than re-walking the AST repeatedly. Generated function does
 * NOT use eval — it walks the AST under closure capture.
 */
export function toFunction(e: Expr, args: string[]): (...vals: number[]) => number {
  return (...vals: number[]) => {
    const env: Env = {};
    for (let i = 0; i < args.length; i++) env[args[i]] = vals[i];
    return evaluate(e, env);
  };
}

// -----------------------------------------------------------------------------
// Pretty-printer.
// -----------------------------------------------------------------------------

const PREC: Record<string, number> = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3, 'neg': 4};

export function stringify(e: Expr, parentPrec = 0): string {
  switch (e.kind) {
    case 'num': {
      const v = e.value;
      if (v < 0) return `(${v})`;
      return Number.isInteger(v) ? v.toFixed(0) : v.toString();
    }
    case 'var': return e.name;
    case 'neg': {
      const inner = stringify(e.arg, PREC['neg']);
      return PREC['neg'] < parentPrec ? `(-${inner})` : `-${inner}`;
    }
    case 'func': return `${e.name}(${stringify(e.arg, 0)})`;
    case 'bin': {
      const p = PREC[e.op];
      const l = stringify(e.left, p);
      const r = stringify(e.right, e.op === '^' ? p - 1 : p + 1);
      const s = `${l} ${e.op} ${r}`;
      return p < parentPrec ? `(${s})` : s;
    }
  }
}

// -----------------------------------------------------------------------------
// Symbolic differentiation.
// -----------------------------------------------------------------------------

/**
 * d/dvar of expression e. Returns a new AST. The result is simplified
 * (constant folding + identity rules) so derivatives stay readable.
 */
export function diff(e: Expr, varName: string): Expr {
  return simplify(diffRaw(e, varName));
}

function diffRaw(e: Expr, x: string): Expr {
  switch (e.kind) {
    case 'num': return ZERO;
    case 'var': return e.name === x ? ONE : ZERO;
    case 'neg': return neg(diffRaw(e.arg, x));
    case 'bin': {
      const u = e.left, v = e.right;
      const du = diffRaw(u, x), dv = diffRaw(v, x);
      switch (e.op) {
        case '+': return add(du, dv);
        case '-': return sub(du, dv);
        case '*': return add(mul(du, v), mul(u, dv));   // (uv)' = u'v + uv'
        case '/': return div(sub(mul(du, v), mul(u, dv)), pow(v, num(2)));   // (u/v)' = (u'v − uv')/v²
        case '^': {
          // d/dx [u^v]: if v is a constant, easy:  v · u^(v-1) · u'
          // general:  u^v · (v' · ln u + v · u'/u)
          if (v.kind === 'num') {
            return mul(mul(num(v.value), pow(u, num(v.value - 1))), du);
          }
          return mul(pow(u, v), add(mul(dv, fn('log', u)), mul(v, div(du, u))));
        }
      }
    }
    case 'func': {
      const u = e.arg;
      const du = diffRaw(u, x);
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

// -----------------------------------------------------------------------------
// Simplification: one bottom-up pass with constant folding and identity rules.
// -----------------------------------------------------------------------------

export function simplify(e: Expr): Expr {
  switch (e.kind) {
    case 'num': case 'var': return e;
    case 'neg': {
      const a = simplify(e.arg);
      if (a.kind === 'num') return num(-a.value);
      if (a.kind === 'neg') return a.arg;            // --x = x
      return neg(a);
    }
    case 'func': {
      const a = simplify(e.arg);
      if (a.kind === 'num') {
        return num(FUNC_IMPL[e.name](a.value));
      }
      return fn(e.name, a);
    }
    case 'bin': {
      const a = simplify(e.left);
      const b = simplify(e.right);
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
          if (a.kind === 'num' && a.value === -1) return simplify(neg(b));
          if (b.kind === 'num' && b.value === -1) return simplify(neg(a));
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

// -----------------------------------------------------------------------------
// Convenience: numerical derivative (central difference). Useful for
// black-box JS functions where the analytical derivative isn't
// available. Order O(h²); choose h ≈ 1e-6 for f64 problems.
// -----------------------------------------------------------------------------

export function numericalDerivative(
  f: (x: number) => number,
  x: number,
  h: number = 1e-6,
): number {
  return (f(x + h) - f(x - h)) / (2 * h);
}

/** Richardson-extrapolated derivative; O(h⁴). */
export function richardsonDerivative(
  f: (x: number) => number,
  x: number,
  h: number = 1e-3,
): number {
  const d1 = (f(x + h) - f(x - h)) / (2 * h);
  const d2 = (f(x + h / 2) - f(x - h / 2)) / h;
  return (4 * d2 - d1) / 3;
}

/** Central-difference gradient for f: R^n → R. */
export function numericalGradient(
  f: (x: number[]) => number,
  x: number[],
  h: number = 1e-6,
): number[] {
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
