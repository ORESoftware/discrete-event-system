'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/smart_movable.rs
// - Keep file-for-file. SmartMovable becomes a trait plus shared state struct
//   carrying id/token identity and run-loop hooks.
// - Token and IterativeDESParticipant implementations map to trait impls; any
//   subclass-specific mutable behavior should live in concrete structs.
// - ValidationCheck lists should use Vec<ValidationCheck> from validation.rs.
// - No free helpers here; pure behavior lifted into the DES graph should use
//   PureTransform/PureTransformEntity. Convert validation failures to Result.

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
