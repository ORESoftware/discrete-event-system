'use strict';

import {Entity} from "../abstract/abstract";
import * as math from "mathjs";
import {EntityGraphData, HasManyInputConnections, HasManyOutputConnections} from "../abstract/interfaces";

export interface ConnectionOpts {
    travelTime: number;
    isBidirectional: false;
}