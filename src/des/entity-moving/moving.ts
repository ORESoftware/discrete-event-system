'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-moving/moving.rs  (module des::entity_moving::moving)
// 1:1 file move. The "tokens" that flow through the network (customers/jobs).
//
// Declarations → Rust:
//   abstract class AbstractMovingEntity<E,V> -> trait MovingEntity (default fns)
//                                               + base struct holding the shared
//                                               timing/visit fields
//   abstract class BasicMovingEntity<V>      -> trait/struct extending the above
//   interface ProcessingTimeValue            -> struct ProcessingTimeValue (#[derive(Clone)])
//   class ProcessableMovingEntity<V>         -> struct + impl
//   class BasicQuantityMovingEntity          -> struct { value: f64 } + impl
//
// Conversion notes (file-specific):
//   - INHERITANCE chain Entity -> AbstractMovingEntity -> BasicMovingEntity ->
//     ProcessableMovingEntity. Flatten to a shared field-bag struct + trait
//     default methods; do NOT mirror `extends`.
//   - `static nextMovingId` auto-increment -> `AtomicU64` (not a struct field).
//   - `math.BigNumber` (timeInSystem, totalWaitTime, totalInProcessTime, …) ->
//     one decimal crate or `f64`, chosen engine-wide; `math.add/subtract` -> ops.
//   - `Date.now()` in init()/doFinish() -> inject `Clock` (shared/capabilities).
//   - `uuid.v4().slice(-10)` -> `uuid::Uuid::new_v4()` then take last 10 chars.
//   - getters/setters startTime/endTime -> `fn start_time(&self)` / `set_start_time`.
//   - `stationsVisited: Map<string,{count}>` -> `HashMap<String, VisitCount>`.
//   - `processingTimeByStation: LinkedQueue<ProcessingTimeValue,string>` is a
//     KEYED queue -> `IndexMap`/`HashMap<String,_>` + ordering; std has none.
//   - `<[any,any]>...get(stationId)` tuple casts -> typed `Option<(K, &mut V)>`.
//   - `(this as any)['value']` dynamic access in getValue -> use a concrete field.
//   - `value = <unknown>null as number` placeholder -> `Option<f64>` / required field.
//   - `throw makeError(..)` / `throw new Error('not yet implemented')` ->
//     `Result` (recoverable) or `panic!`/`unimplemented!` (bug/stub).
//   - `getSerializableData(): any` -> a `#[derive(Serialize)]` DTO struct.
// =============================================================================

import * as math from "mathjs";
import {number} from "mathjs";
import * as uuid from "uuid";
import {bgn, HasComputedProperties, makeError} from "../general/general";
import {Entity, StationaryEntity, TimeStepOpts} from "../abstract/abstract";
import {EntityGraphData} from "../abstract/interfaces";
import {IsVoid, LinkedQueue} from "@oresoftware/linked-queue";
import * as des from "../general/time-accrued";

export abstract class AbstractMovingEntity<E, V = any>
  extends Entity<E> {

  static nextMovingId = 0;
  stationsVisitedCount = 0;
  totalWaitTime = bgn(0);
  totalInProcessTime = bgn(0);

  movingId: number = AbstractMovingEntity.nextMovingId++;
  movingUuid: string;

  private _startTime: number = -1
  private _endTime: number = -1;

  currentContainer: StationaryEntity<E> | null = null;
  timeInSystem: math.BigNumber = bgn(0);
  hasExitedSystem: boolean = false;
  realTimeInSystem = -1;
  outQueueWaitTime = bgn(0);
  stationsVisited = new Map<string, { count: number }>();

  constructor(id: string) {
    super(id);
    // if(!id){
    //   throw new Error('etagage');
    // }
    // console.log({id});
    this.movingUuid = id;
  }

  abstract getValue(): V;

  init() {
    this.startTime = Date.now();
    return this;
  }

  abstract runTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts): void;


  bumpTimeInSystem(stepSize: math.BigNumber): this {
    this.timeInSystem = math.add(this.timeInSystem, stepSize);
    return this;
  }

  doTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts) {
    throw new Error('probably a misfire');
    this.timeInSystem = math.add(this.timeInSystem, stepSize);
    return this.runTimeStep(stepSize, opts);
  }

  get startTime(): number {
    return this._startTime;
  }

  set startTime(value: number) {
    this._startTime = value;
  }

  get endTime(): number {
    return this._endTime;
  }

  set endTime(value: number) {
    this._endTime = value;
  }

  doFinish() {
    // exit the system
    this.endTime = Date.now();
    this.realTimeInSystem = this.endTime - this.startTime;
    this.hasExitedSystem = true;
    return this.runFinish();
  }

  addVisitedStation(name: string) {
    if (!this.stationsVisited.has(name)) {
      this.stationsVisited.set(name, {count: 0})
    }
    (this.stationsVisited.get(name) as { count: number }).count++
  }

  abstract runFinish(): void;

}


export abstract class BasicMovingEntity<V = any> extends AbstractMovingEntity<V> {

  constructor() {
    super(uuid.v4().slice(-10));
  }

  doValidationBeforeRun(): boolean {
    return true;
  }

  getGraphData(): EntityGraphData {
    return Object.assign(
      this.getWithComputedProperties(),
      this.getSerializableData()
    );
  }

  getSerializableData(): any {
    return {
      totalInProcessTime: Number(this.totalInProcessTime.toFixed(5)),
      stationsVisitedCount: this.stationsVisitedCount
    };
  }

  runFinish(): void {
    // throw new Error('not yet implemented.')
  }

  runTimeStep(stepSize: math.BigNumber): void {
    throw new Error('not yet implemented.')
  }

  getWithComputedProperties() {
    return {
      timeInSystem: Number(this.timeInSystem.toFixed(5)),
      hasExitedSystem: this.hasExitedSystem
    }
  }

  getValue() {
    return {
      id: this.id,
      value: (this as any)['value']
    }
  }

  doValidation(): void {
  }

}

export interface ProcessingTimeValue {
  stationId: string,
  timeInInputQueue: number,
  timeInProcessQueue: number,
  timeInOutputQueue: number,
  startTimeInInputQueue: number,
  startTimeInProcessQueue: number,
  startTimeInOutputQueue: number
}

export class ProcessableMovingEntity<V> extends BasicMovingEntity<V> {

  processingTimeByStation = new LinkedQueue<ProcessingTimeValue, string>();

  constructor() {
    super();
  }

  startNewStation(stationId: string) {

    this.processingTimeByStation.remove(stationId);

    // if(this.processingTimeByStation.size > 15){
    // TODO: don't delete here because other stations may be writing to this
    //   this.processingTimeByStation.dequeue();
    // }

    const currentTime = des.getTimeAccrued();

    this.processingTimeByStation.enqueue(stationId, {
      timeInInputQueue: -1,
      timeInProcessQueue: -1,
      timeInOutputQueue: -1,
      startTimeInInputQueue: currentTime,
      startTimeInProcessQueue: -1,
      startTimeInOutputQueue: -1
    });

  }

  setStartTimeInProcessQueue(stationId: string) {
    const [k, z] = <[any, any]>this.processingTimeByStation.get(stationId);

    if (IsVoid.check(k)) {
      throw makeError('missing value:', stationId);
    }

    if (z.startTimeInProcessQueue !== -1) {
      throw makeError('value should be 0.');
    }

    const currentTime = des.getTimeAccrued();
    z.startTimeInProcessQueue = currentTime;
  }

  setStartTimeInOutputQueue(stationId: string) {
    const [k, z] = <[any, any]>this.processingTimeByStation.get(stationId);

    if (IsVoid.check(k)) {
      throw makeError('missing value:', stationId);
    }

    if (z.startTimeInOutputQueue !== -1) {
      throw makeError('value should be 0.');
    }

    const currentTime = des.getTimeAccrued();
    z.startTimeInOutputQueue = currentTime;
  }

  setTimeInInputQueue(stationId: string): math.BigNumber {
    const [k, z] = <[any, any]>this.processingTimeByStation.get(stationId);

    if (IsVoid.check(k)) {
      throw makeError('missing value:', stationId);
    }

    if (z.timeInInputQueue !== -1) {
      throw makeError('value should be 0.');
    }

    const currentTime = des.getTimeAccrued();
    return z.timeInInputQueue = math.subtract(currentTime, z.startTimeInInputQueue);
  }

  setTimeInProcessingQueue(stationId: string): math.BigNumber {

    const [k, z] = <[any, any]>this.processingTimeByStation.get(stationId);

    if (IsVoid.check(k)) {
      throw makeError('missing value:', stationId);
    }

    if (z.timeInProcessQueue !== -1) {
      throw makeError('value should be 0.');
    }

    const currentTime = des.getTimeAccrued();
    return z.timeInProcessQueue = math.subtract(currentTime, z.startTimeInProcessQueue);
  }


  setTimeInOutputQueue(stationId: string): math.BigNumber {
    const [k, z] = <[any, any]>this.processingTimeByStation.get(stationId);

    if (IsVoid.check(k)) {
      throw makeError('missing value:', stationId);
    }

    if (z.timeInOutputQueue !== 0) {
      throw makeError('value should be 0.');
    }

    const currentTime = des.getTimeAccrued();
    return z.timeInOutputQueue = math.subtract(currentTime, z.startTimeInOutputQueue);
  }


  bumpTotalWaitTime(inMillis: math.BigNumber): this {
    this.totalWaitTime = math.add(this.totalWaitTime, inMillis);
    return this;
  }

  bumpOutQueueWaitTime(inMillis: math.BigNumber): this {
    this.outQueueWaitTime = math.add(this.outQueueWaitTime, inMillis);
    return this;
  }

  bumpTotalProcessingTime(inMillis: math.BigNumber): this {
    this.totalInProcessTime = math.add(this.totalInProcessTime, inMillis);
    return this;
  }


}

export class BasicQuantityMovingEntity extends AbstractMovingEntity<any> {

  value = <unknown>null as number;

  constructor(q: number) {
    super(uuid.v4().slice(-10));
    this.value = q;
  }

  doValidationBeforeRun(): boolean {
    return true;
  }

  getGraphData(): EntityGraphData {
    return Object.assign(
      this.getWithComputedProperties(),
      this.getSerializableData()
    );
  }

  getSerializableData(): any {
    return {
      totalInProcessTime: Number(this.totalInProcessTime.toFixed(5)),
      stationsVisitedCount: this.stationsVisitedCount
    };
  }

  runFinish(): void {
    // throw new Error('not yet implemented.')
  }

  runTimeStep(stepSize: math.BigNumber): void {
    throw new Error('not yet implemented.')
  }

  getWithComputedProperties() {
    return {
      timeInSystem: Number(this.timeInSystem.toFixed(5)),
      hasExitedSystem: this.hasExitedSystem
    }
  }

  getValue() {
    return {
      q: this.value
    }
  }

  doValidation(): void {
  }

}