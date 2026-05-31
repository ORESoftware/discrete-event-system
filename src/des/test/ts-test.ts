// RUST MIGRATION: Port file-for-file to `tests/ts_test.rs` only for behavior that still matters after TypeScript-only type experiments become Rust traits/generics.
// Test-port notes: convert compile/runtime expectations into `#[test]` functions returning `Result<()>`; replace manual checks with `assert!` and `assert_eq!`; prefer trait-bound examples over structural typing assumptions.



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
