'use strict';

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
