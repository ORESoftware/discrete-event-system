// =============================================================================
// RUST MIGRATION  —  target: examples/linked_queue_iterator.rs   (example binary)
// 1:1 file move. A LinkedQueue iterator smoke/demo that only PRINTS items, so
// it fits an `examples/` binary; promote to `#[cfg(test)] mod tests` (asserting
// the yielded order) if it should guard behaviour.
//
// Test harness → Rust:
//   no PASS/FAIL harness — console.log dumps -> assert_eq! on collected items
//   if promoted to a #[test].
//
// Conversion notes (file-specific):
//   - LinkedQueue (@oresoftware/linked-queue) -> std VecDeque<T>; q.iterator()
//     -> .iter(); the early `break` just exercises partial iteration.
// =============================================================================

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