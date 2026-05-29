'use strict';

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

