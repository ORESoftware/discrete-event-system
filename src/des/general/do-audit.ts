// RUST MIGRATION: target module src/des/general/do_audit.rs.
// RUST MIGRATION: doAudit is a small diagnostic command and can be a free function or src/bin/do_audit.rs wrapper around the registry.
// RUST MIGRATION: Replace thrown makeError paths with Result<(), Error>; avoid any-typed entity values by using the Rust entity trait hierarchy.
'use strict';

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

