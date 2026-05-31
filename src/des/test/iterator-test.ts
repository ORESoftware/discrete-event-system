// RUST MIGRATION: Port file-for-file to `tests/iterator_test.rs` if this external queue smoke check remains part of the Rust migration.
// Test-port notes: convert iterator expectations into `#[test]` functions returning `Result<()>`; replace console/manual checks with `assert!` and `assert_eq!`; keep any generated queue contents deterministic.

import {LinkedQueue} from "@oresoftware/linked-queue";


const q = new LinkedQueue();


q.enqueue({foo:1});
q.enqueue({foo:2});
q.enqueue({foo:3});
q.enqueue({foo:4});


for(const v of q.iterator()){
  console.log(v);
}

for(const v of q.iterator()){
  console.log(v);
  break;
}
