'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/entity-registration.rs  (module des::general::entity_registration)
// 1:1 file move. Global registry of all sources/sinks/processors/decision nodes.
//
// Declarations → Rust:
//   const reg = { ...methods }   -> struct Registry { sets... } + impl (register*/getAll*)
//
// Conversion notes (file-specific):
//   - `reg` is a process-wide MUTABLE SINGLETON over `Set<EntityX>`; in Rust prefer
//     an explicit owned `Registry` passed around, or `OnceCell<Mutex<Registry>>` if a
//     true global is required (avoid hidden ambient state).
//   - The `Set<...>` collections -> `HashSet`/`Vec` of `Rc<RefCell<dyn ...>>` trait objects
//     (entities need identity/`Hash+Eq`, e.g. by id).
//   - `<any, any>` generics on entities -> concrete trait objects (`dyn EntitySource` etc.).
// =============================================================================

import {EntityProcessor} from "../entity-processing/processing";
import {EntitySource} from "../entity-source/source";
import {EntitySink} from "../entity-sink/sink";
import {ProbabilityDecisionEntity} from "../entity-decision/probability-decision";


const vals = {
  allProcessors: new Set<EntityProcessor<any, any>>(),
  allSources: new Set<EntitySource<any, any>>(),
  allSinks: new Set<EntitySink<any, any>>(),
  allDecision: new Set<ProbabilityDecisionEntity<any, any>>
};


export const reg = {

  getAllDecisionNodes() {
    return vals.allDecision;
  },

  getAllSources() {
    return vals.allSources;
  },

  getAllSinks() {
    return vals.allSinks;
  },

  getAllProcessors() {
    return vals.allProcessors;
  },

  registerSource(v: EntitySource<any, any>) {
    return vals.allSources.add(v);
  },

  registerSink(v: EntitySink<any, any>) {
    return vals.allSinks.add(v);
  },

  registerProcessor(v: EntityProcessor<any, any>) {
    return vals.allProcessors.add(v);
  },

  registerDecision(v: ProbabilityDecisionEntity<any, any>) {
    return vals.allDecision.add(v);
  }

}

