// =============================================================================
// RUST MIGRATION  —  target: src/des/test/ts_test.rs  (or a doc-test / examples/)
// 1:1 file move. A tiny TypeScript inheritance/override demo (no assertions),
// not a real test — keep it only as an illustrative example or drop it.
//
// Test harness → Rust:
//   none — there is no PASS/FAIL harness here; nothing to map to #[test].
//
// Conversion notes (file-specific):
//   - `abstract class HasMethod -> SuperClass -> SubClass` is an inheritance
//     chain: in Rust use `trait HasMethod { fn implement_me(&self) -> T }` with
//     structs implementing it; there is no method-override-via-extends.
//   - `T = any` / `: any` return -> pick a concrete type or a generic; avoid Any.
// =============================================================================

abstract class HasMethod<T =any> {
  abstract implementMe():T
}

class SuperClass extends HasMethod {

  implementMe() {
    return {
      foo: 'bar'
    }
  }
}

class SubClass extends SuperClass {

  implementMe() : any{
    return {
      north: 'star'
    }
  }
}