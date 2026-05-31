// RUST MIGRATION: target module src/des/general/do_audit.rs.
// RUST MIGRATION: doAudit is a small diagnostic command and can be a free function or src/bin/do_audit.rs wrapper around the registry.
// RUST MIGRATION: Replace thrown makeError paths with Result<(), Error>; avoid any-typed entity values by using the Rust entity trait hierarchy.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/do-audit.rs  (module des::general::do_audit)
// 1:1 file move. Debug invariant check: total registered-entity size must stay constant.
//
// Declarations → Rust:
//   const doAudit = () => {...}   -> fn do_audit(reg: &Registry) (or method on the registry)
//
// Conversion notes (file-specific):
//   - Module-level mutable state `first`/`previousTotal` -> not idiomatic globals;
//     hold them in a struct (auditor) or `OnceCell`/thread-local, not file statics.
//   - Pulls the global `reg` singleton (see entity-registration.rs) — pass it in by ref.
//   - `throw makeError(...)` on mismatch is an invariant violation -> `panic!`.
// =============================================================================

import {reg} from "./entity-registration";
import {makeError} from "./general";


let first = true;
let previousTotal = 0;

export const doAudit = () => {

  let total = 0;

  for (const v of reg.getAllProcessors()) {
    // console.log(v.doAudit());
    total += v.doAudit().totalSize;
  }

  for (const v of reg.getAllSinks()) {
    // console.log(v.doAudit());
    total += v.doAudit().totalSize;
  }

  for (const v of reg.getAllDecisionNodes()) {
    // console.log(v.doAudit());
    total += v.doAudit().totalSize;
  }

  if (!first) {
    if (previousTotal !== total) {
      throw makeError('totals are not equal', previousTotal, total);
    }
  }

  previousTotal = total;
  first = false;

};

