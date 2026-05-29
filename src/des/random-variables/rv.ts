import * as math from "mathjs";
import {BigNumber, number} from "mathjs";
import {bgn, getReasonableU, getReasonableUNative, makeError} from "../general/general";
import {Serializable} from "../abstract/abstract";

export abstract class RandomVariable extends Serializable<any> {

  abstract getRate(): math.BigNumber;

  abstract getNextEventQuantity(timeStep: math.BigNumber): number;

  abstract getNextEvents(timeStep: math.BigNumber): Generator<number>;
  abstract getSerializableData(): any

}

type RT = ReturnType<PoissonRandomVariable['getSerializableData']>

export class BernoulliRandomVariable extends RandomVariable {

  getRate() {
    return bgn(0.3);
  }

  getNextEventQuantity(timeStep: math.BigNumber): number {
    return math.random() > 0.5 ? 1 : 0;
  }

  getSerializableData() {
    return {
      lambda: 5
    };
  }

  *getNextEvents(timeStep: math.BigNumber): Generator<number> {
    return undefined;
  }
}

export class PoissonRandomVariable extends RandomVariable {

  getRate() {
    return bgn(0.3);
  }

  getNextEventQuantity(timeStep: math.BigNumber): number {
    return math.random() > 0.5 ? 1 : 0;
  }

  getSerializableData() {
    return {
      lambda: 5
    };
  }

  *getNextEvents(timeStep: math.BigNumber): Generator<number> {
    return undefined;
  }
}

export class ExponentialRandomVariable extends RandomVariable {

  lambda = bgn(-1);
  preComputedRHS: math.BigNumber | null = null;
  maxVal: math.BigNumber | null = null;

  getRate() {
    return bgn(0.3);
  }

  constructor({lambda, timeStep}: { lambda: math.BigNumber, timeStep: math.BigNumber }) {
    super();

    // TODO: set a max value for the random x, so it's not infinity

    this.lambda = lambda;
    if (math.smallerEq(lambda, bgn(0))) {
      console.warn(`[ExponentialRandomVariable] invalid lambda=${lambda.toString()} (must be > 0); rate parameter cannot be zero or negative.`);
      throw new Error('lambda must be larger than 0.')
    }
    this.preComputedRHS = math.exp(
      <math.BigNumber>math.multiply(
        bgn(-1),
        math.multiply(
          timeStep,
          lambda,
          // this.getAdjustedExpectedVal(timeStep),
          // lambda
          // lambda,
          // this.getAdjustedExpectedVal(timeStep),
          // timeStep
        )
      )
    );
  }


  getAdjustedExpectedVal(timeStep: math.BigNumber) {
    return math.divide(this.lambda, timeStep);
  }

  getNextEventQuantity(timeStep: math.BigNumber): number {

    const rhs = this.preComputedRHS as math.BigNumber;

    let uProduct = bgn(1);
    let q = -1;

    while (math.larger(uProduct, rhs)) {
      q = q + 1;
      const u = getReasonableU();
      uProduct = math.multiply(u, uProduct) as math.BigNumber;
    }

    // console.log({rhs, q, uProduct});
    return q;
  }

  *getNextEvents(timeStep: math.BigNumber): Generator<number> {
    const rhs = this.preComputedRHS as math.BigNumber;

    let uProduct = bgn(1);
    let q = -1;

    while (math.larger(uProduct, rhs)) {
      q = q + 1;
      const u = getReasonableU();
      uProduct = math.multiply(u, uProduct) as math.BigNumber;
    }

    // console.log({rhs, q, uProduct});
    return q;
  }


  getNextEventCount(timeStep: math.BigNumber): number {

    let sum = bgn(0);
    let q = -1;

    while (math.smaller(sum, timeStep)) {
      q = q + 1;
      const u = getReasonableU();
      const diff = math.subtract(bgn(1), u);
      const t = math.multiply(bgn(-1), math.log(diff));
      sum = math.add(sum, t) as math.BigNumber;
    }

    return q;
  }

  getSerializableData(): any {
    return {
      lambda: Number(this.lambda)
    }
  }


}

export class ExponentialRandomVariable3 extends RandomVariable {

  nextEvent = bgn(0);
  lambda = bgn(-1)
  first = true;
  timeStep = bgn(-1);
  nextU = bgn(Math.random())
  precomputedRHS: math.BigNumber = bgn(-1);

  constructor(v: { lambda: math.BigNumber, timeStep: math.BigNumber }) {
    super();
    this.timeStep = v.timeStep;
    this.lambda = v.lambda;
    this.precomputedRHS = math.exp(   //  U(0,1) = e^(-lambda*t)
      math.multiply(bgn(-1),
        math.multiply(this.lambda, this.timeStep)
      ) as math.BigNumber
    ) as math.BigNumber;
    if (math.smaller(this.lambda, bgn(0.00000001))) {
      throw new Error('Width of uniform distribution needs to be greater than 0.00000001')
    }
  }

  getRate() {
    return bgn(0.3);
  }

  getNextVal(): math.BigNumber {
    const u = getReasonableU();
    const v = math.multiply(
      math.divide(bgn(-1), bgn(this.lambda)),
      math.log(math.subtract(bgn(1), u))
    ) as math.BigNumber;
    return math.add(v, this.nextEvent) as math.BigNumber;
  }

  getNextEventQuantity(timeStep: math.BigNumber): number {

    if (this.first) {
      this.first = false;
      this.nextU = bgn(Math.random());
    } else {
      this.nextEvent = math.subtract(this.nextEvent, timeStep);
    }

    // console.log({nextevent: this.nextEvent})

    // this.nextEvent = math.max(
    //   bgn(0),  // it might be wrong to take the max of this
    //   math.subtract(this.nextEvent, timeStep)
    // );


    let q = 0;

    while (math.smaller(this.nextEvent, timeStep)) {
      q = q + 1;
      this.nextEvent = this.getNextVal();
    }

    return q;

  }

  getSerializableData() {
    return {
      lambda: Number(this.lambda)
    }
  }

  *getNextEvents(timeStep: math.BigNumber): Generator<number> {
    return undefined;
  }
}


export class ExponentialRandomVariable2 extends RandomVariable {

  nextEvent = bgn(0);
  lambda = bgn(-1)
  first = true;
  timeStep = bgn(-1);

  constructor(v: { lambda: math.BigNumber, timeStep: math.BigNumber }) {
    super();
    this.timeStep = v.timeStep;
    this.lambda = v.lambda;
    if (math.smaller(this.lambda, bgn(0.00000001))) {
      throw new Error('Width of uniform distribution needs to be greater than 0.00000001')
    }
  }

  getRate() {
    return bgn(0.3);
  }

  getNextVal(): math.BigNumber {
    const u = getReasonableU();
    const v = math.multiply(
      math.divide(bgn(-1), bgn(this.lambda)),
      math.log(math.subtract(bgn(1), u))
    ) as math.BigNumber;
    return math.add(v, this.nextEvent) as math.BigNumber;
  }

  getNextEventQuantity(timeStep: math.BigNumber): number {

    if (this.first) {
      this.first = false;
      this.nextEvent = this.getNextVal();
    } else {
      this.nextEvent = math.subtract(this.nextEvent, timeStep);
    }

    // console.log({nextevent: this.nextEvent})

    // this.nextEvent = math.max(
    //   bgn(0),  // it might be wrong to take the max of this
    //   math.subtract(this.nextEvent, timeStep)
    // );


    let q = 0;

    while (math.smaller(this.nextEvent, timeStep)) {
      q = q + 1;
      this.nextEvent = this.getNextVal();
    }

    return q;

  }

  *getNextEvents(timeStep: math.BigNumber): Generator<number> {
    return undefined;
  }

  getSerializableData() {
    return {
      lambda: Number(this.lambda)
    }
  }
}

export class UniformRandomVariable extends RandomVariable {

  nextEvent = -1;
  aVal = NaN;
  bVal = NaN;
  width = NaN;
  first = true;

  constructor({aVal, bVal}: { aVal: math.BigNumber, bVal: math.BigNumber }) {
    super();
    this.aVal = Number(aVal);
    this.bVal = Number(bVal);

    if (isNaN(this.aVal)) {
      console.warn(`[UniformRandomVariable] lower bound aVal parsed to NaN (raw=${String(aVal)}).`);
      throw makeError('this.aVal is not a number.')
    }
    if (isNaN(this.bVal)) {
      console.warn(`[UniformRandomVariable] upper bound bVal parsed to NaN (raw=${String(bVal)}).`);
      throw makeError('this.bVal is not a number.')
    }

    this.width = Number(bVal) - Number(aVal);
    if (math.smaller(this.width, 0.00000001)) {
      console.warn(`[UniformRandomVariable] degenerate interval: width=${this.width} from [${this.aVal}, ${this.bVal}] (need width > 1e-8).`);
      throw new Error('Width of uniform distribution needs to be greater than 0.00000001')
    }

  }

  getRate() {
    return bgn(0.3);
  }

  getNextVal(): number {
    const u = getReasonableUNative();
    const v = this.aVal + (u * this.width);
    return v + this.nextEvent;
  }

  getNextEventQuantity(timeStep: math.BigNumber): number {

    // return 1;

    if (this.first) {
      this.first = false;
      this.nextEvent = this.getNextVal();
    } else {
      this.nextEvent = this.nextEvent - Number(timeStep);
    }

    // console.log('next event:', this.nextEvent);

    // this.nextEvent = math.max(
    //   bgn(0),  // it might be wrong to take the max of this
    //   math.subtract(this.nextEvent, timeStep)
    // );

    let q = 0;

    while (this.nextEvent < Number(timeStep)) {
      q = q + 1;
      this.nextEvent = this.getNextVal();
      // console.log({nextEvent:this.nextEvent})
    }

    // console.log({q});

    return q;

  }

  * getNextEvents(timeStep: math.BigNumber): Generator<number> {

    // return 1;

    if (this.first) {
      this.first = false;
      this.nextEvent = this.getNextVal();
    } else {
      this.nextEvent = this.nextEvent - Number(timeStep);
    }

    // console.log('next event:', this.nextEvent);

    // this.nextEvent = math.max(
    //   bgn(0),  // it might be wrong to take the max of this
    //   math.subtract(this.nextEvent, timeStep)
    // );

    let q = 0;

    while (this.nextEvent < Number(timeStep)) {
      q = q + 1;
      yield this.nextEvent;
      this.nextEvent = this.getNextVal();
      // console.log({nextEvent:this.nextEvent})
    }

    // console.log({q});

  }

  getSerializableData(): any {
    return this;
  }


}


export class UniformRandomVariable2 extends RandomVariable {

  nextEvent = bgn(0);
  aVal: math.BigNumber | null = null;
  bVal: math.BigNumber | null = null;
  width: math.BigNumber | null = null;
  first = true;

  constructor({aVal, bVal}: { aVal: math.BigNumber, bVal: math.BigNumber }) {
    super();
    this.aVal = aVal;
    this.bVal = bVal;
    this.width = bgn(math.subtract(bVal, aVal));
    if (math.smaller(this.width, bgn(0.00000001))) {
      throw new Error('Width of uniform distribution needs to be greater than 0.00000001')
    }
  }

  getRate() {
    return bgn(0.3);
  }

  getNextVal(): math.BigNumber {
    const u = getReasonableU();
    const v = math.add(this.aVal as math.BigNumber, math.multiply(u, this.width as math.BigNumber));
    return math.add(v, this.nextEvent) as math.BigNumber;
  }

  getNextEventQuantity(timeStep: math.BigNumber): number {

    return 1;

    if (this.first) {
      this.first = false;
      this.nextEvent = this.getNextVal();
    } else {
      this.nextEvent = math.subtract(this.nextEvent, timeStep);
    }

    // this.nextEvent = math.max(
    //   bgn(0),  // it might be wrong to take the max of this
    //   math.subtract(this.nextEvent, timeStep)
    // );

    let q = 0;

    while (math.smaller(this.nextEvent, timeStep)) {
      q = q + 1;
      this.nextEvent = this.getNextVal();
      // console.log({nextEvent:this.nextEvent})
    }

    return q;

  }

  *getNextEvents(timeStep: math.BigNumber): Generator<number> {
    return undefined;
  }

  getSerializableData() {
    return {
      a: this.aVal,
      b: this.bVal
    }
  }
}