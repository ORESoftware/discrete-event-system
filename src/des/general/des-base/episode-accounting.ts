'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/episode_accounting.rs  (module des::general::des_base::episode_accounting)
// 1:1 file move. Reward/length bookkeeping for RL episodes (scalar + vector).
//
// Declarations → Rust:
//   interface EpisodeSummary / VectorEpisodeSummary -> struct (#[derive(Clone, Copy/Debug)])
//   class EpisodeAccounting        -> struct EpisodeAccounting { histories: Vec<f64>, .. }
//   class VectorEpisodeAccounting  -> struct (dimension: usize; currentRewards: Vec<f64>)
//
// Conversion notes (file-specific):
//   - Pure scalar bookkeeping; methods mutate self -> `&mut self`. No I/O, no RNG.
//   - `new Array(dim).fill(0)` -> `vec![0.0; dim]`; `slice()` copies -> `.clone()`.
//   - `throw new Error` on dimension mismatch -> `Result`/`panic!` (programmer error).
//   - `rewardHistory: number[][]` -> `Vec<Vec<f64>>`.
// =============================================================================

export interface EpisodeSummary {
  reward: number;
  length: number;
}

export interface VectorEpisodeSummary {
  rewards: number[];
  length: number;
}

export class EpisodeAccounting {
  readonly rewardHistory: number[] = [];
  readonly lengthHistory: number[] = [];
  currentReward = 0;
  currentLength = 0;
  totalSteps = 0;

  recordStep(reward: number): void {
    this.totalSteps += 1;
    this.currentReward += reward;
    this.currentLength += 1;
  }

  finishEpisode(): EpisodeSummary {
    const summary = {reward: this.currentReward, length: this.currentLength};
    this.rewardHistory.push(summary.reward);
    this.lengthHistory.push(summary.length);
    this.resetCurrent();
    return summary;
  }

  resetCurrent(): void {
    this.currentReward = 0;
    this.currentLength = 0;
  }
}

export class VectorEpisodeAccounting {
  readonly rewardHistory: number[][] = [];
  readonly lengthHistory: number[] = [];
  readonly currentRewards: number[];
  totalSteps = 0;

  constructor(readonly dimension: number) {
    this.currentRewards = new Array(dimension).fill(0);
  }

  recordStep(rewards: readonly number[]): void {
    if (rewards.length !== this.dimension) {
      throw new Error(`expected ${this.dimension} rewards, got ${rewards.length}`);
    }
    this.totalSteps += 1;
    for (let i = 0; i < this.dimension; i++) this.currentRewards[i] += rewards[i];
  }

  finishEpisode(length: number): VectorEpisodeSummary {
    const rewards = this.currentRewards.slice();
    this.rewardHistory.push(rewards);
    this.lengthHistory.push(length);
    this.resetCurrent();
    return {rewards, length};
  }

  resetCurrent(): void {
    this.currentRewards.fill(0);
  }

  setCurrentRewards(rewards: readonly number[]): void {
    if (rewards.length !== this.dimension) {
      throw new Error(`expected ${this.dimension} rewards, got ${rewards.length}`);
    }
    for (let i = 0; i < this.dimension; i++) this.currentRewards[i] = rewards[i];
  }
}
