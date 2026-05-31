'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/smart_movable.rs  (module des::general::des_base::smart_movable)
// 1:1 file move. SmartMovable — a token that is ALSO a run-loop participant
// (it moves through the graph yet advances itself each tick).
//
// Declarations → Rust:
//   abstract class SmartMovable (implements Token, IterativeDESParticipant)
//     -> trait SmartMovable: Token + DESRunLoopEntity
//        (provided activate/deactivate/isActive/hasWork over an `active: bool` field;
//         required `run_time_step`)
//
// Conversion notes (file-specific):
//   - Multi-interface implements (Token + IterativeDESParticipant) -> trait bounds /
//     supertraits; write explicit impls (no structural satisfaction).
//   - Only `runTimeStep` is abstract -> the one required trait method; the rest are
//     provided defaults backed by the `active` flag.
// =============================================================================

import {Token} from './station';
import {IterativeDESParticipant} from './runner';
import {ValidationCheck} from './validation';

export abstract class SmartMovable implements Token, IterativeDESParticipant {
  protected active = false;

  constructor(readonly id: string) {}

  activate(): void {
    this.active = true;
  }

  deactivate(): void {
    this.active = false;
  }

  isActive(): boolean {
    return this.active;
  }

  assertPreconditions(): void {}

  hasWork(): boolean {
    return this.active;
  }

  onFinalize(): void {}

  numValidators(): number {
    return 0;
  }

  runValidation(): ValidationCheck[] {
    return [];
  }

  abstract runTimeStep(): void;
}
