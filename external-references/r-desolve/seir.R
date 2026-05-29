#!/usr/bin/env Rscript
## R reference implementation of the SEIR-with-hospitalization model using
## the `deSolve` package. Three independent verifications:
##   1. Closed-form steady state via solve(-A, b).
##   2. Forward-Euler difference equation (manual loop).
##   3. ODE via deSolve::lsoda (Livermore stiff/non-stiff adaptive solver).
##
## How to run (from courses/hdm-fall-2022/des):
##
##   Rscript external-references/r-desolve/seir.R \
##           --out out/external/r-desolve/standard.json
##
## The default `Rscript` command on $PATH is used. Override with
##   RSCRIPT_BIN=Rscript-4.3 ./external-references/run-all.sh
##
## Requires:
##   install.packages(c('deSolve', 'jsonlite'))

suppressPackageStartupMessages({
  library(deSolve)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out_path <- NULL
dt <- 0.05
i <- 1
while (i <= length(args)) {
  if (args[i] == "--out")  { out_path <- args[i + 1]; i <- i + 2; next }
  if (args[i] == "--dt")   { dt <- as.numeric(args[i + 1]); i <- i + 2; next }
  if (args[i] == "--seed") { i <- i + 2; next }
  i <- i + 1
}
if (is.null(out_path)) stop("--out is required")

## --- Configuration (mirrors DEFAULT_CONFIG in src/des/runners/types.ts) ----
horizon_days <- 1200
phase1_days  <- 800
source_cap   <- 500
arr_iat      <- c(0.7, 1.3)

mu <- list(
  arrival = mean(arr_iat),
  S  = mean(c(0.20, 0.40)),  E  = mean(c(0.20, 0.40)),
  IP = mean(c(0.20, 0.40)),  IA = mean(c(0.20, 0.40)),
  IS = mean(c(0.20, 0.40)),  IH = mean(c(0.20, 0.40)),
  R  = mean(c(1.50, 2.50)),  D  = mean(c(0.10, 0.30))
)
p_a <- 0.40
p_h <- 0.20
p_d <- 0.12
lambda <- 1 / mu$arrival

A <- matrix(0, 7, 7)
A[1, 1] <- -1 / mu$S;          A[1, 7] <-  1 / mu$R           # S
A[2, 1] <-  1 / mu$S;          A[2, 2] <- -1 / mu$E           # E
A[3, 2] <-  1 / mu$E;          A[3, 3] <- -1 / mu$IP          # I-P
A[4, 3] <-  p_a       / mu$IP; A[4, 4] <- -1 / mu$IA          # I-A
A[5, 3] <- (1 - p_a)  / mu$IP; A[5, 5] <- -1 / mu$IS          # I-S
A[6, 5] <-  p_h       / mu$IS; A[6, 6] <- -1 / mu$IH          # I-H
A[7, 4] <-  1 / mu$IA
A[7, 5] <- (1 - p_h)  / mu$IS
A[7, 6] <- (1 - p_d)  / mu$IH
A[7, 7] <- -1 / mu$R                                          # R

b_const <- c(lambda, 0, 0, 0, 0, 0, 0)

## --- 1. Closed-form steady state (open system) ----------------------------
N_star <- as.vector(solve(-A, b_const))

## --- 2. Forward-Euler difference equation ---------------------------------
n_steps  <- round(horizon_days / dt)
N <- rep(0, 7); C <- 0; deaths <- 0
pop_sums <- rep(0, 7); peak <- rep(0, 7)
diverged <- FALSE
for (step_i in 1:n_steps) {
  pop_sums <- pop_sums + N * dt
  t_now <- (step_i - 1) * dt
  src <- if (C < source_cap && t_now < phase1_days) lambda else 0
  b_t <- c(src, 0, 0, 0, 0, 0, 0)
  dN <- as.vector(A %*% N + b_t)
  deaths <- deaths + N[6] * p_d / mu$IH * dt
  N <- N + dt * dN
  C <- C + dt * src
  peak <- pmax(peak, N)
  if (any(!is.finite(N))) { diverged <- TRUE; break }
}
diff_final     <- N
diff_time_avg  <- pop_sums / horizon_days
diff_peak      <- peak

## --- 3. ODE via deSolve::lsoda --------------------------------------------
##
## Adaptive solvers struggle with the discontinuous source-cutoff. We
## therefore integrate in two SMOOTH phases:
##   phase 1  [0,    t_off] : src = lambda
##   phase 2  [t_off, T]    : src = 0
## with t_off = min(phase1Days, sourceCap / lambda).
make_rhs <- function(src_value) {
  function(t, y, parms) {
    N <- y[1:7]
    dN <- as.vector(A %*% N + c(src_value, 0, 0, 0, 0, 0, 0))
    dC <- src_value
    dD <- N[6] * p_d / mu$IH
    list(c(dN, dC, dD))
  }
}

t_off <- min(phase1_days, source_cap / lambda)
t_off <- min(t_off, horizon_days)
y0 <- rep(0, 9)
t0 <- proc.time()

if (t_off > 0) {
  t_eval_1 <- seq(0, t_off, by = 1)
  if (tail(t_eval_1, 1) != t_off) t_eval_1 <- c(t_eval_1, t_off)
  out1 <- lsoda(y0, t_eval_1, make_rhs(lambda), NULL, rtol = 1e-8, atol = 1e-10)
  y_mid <- as.numeric(out1[nrow(out1), 2:10])
} else {
  out1 <- NULL
  y_mid <- y0
}
if (t_off < horizon_days) {
  t_eval_2 <- seq(t_off, horizon_days, by = 1)
  if (head(t_eval_2, 1) != t_off) t_eval_2 <- c(t_off, t_eval_2)
  out2 <- lsoda(y_mid, t_eval_2, make_rhs(0), NULL, rtol = 1e-8, atol = 1e-10)
} else {
  out2 <- NULL
}
elapsed_ms <- as.numeric((proc.time() - t0)["elapsed"]) * 1000

if (!is.null(out1) && !is.null(out2)) {
  out <- rbind(out1, out2[-1, , drop = FALSE])
} else if (!is.null(out1)) {
  out <- out1
} else {
  out <- out2
}

t_eval      <- out[, 1]
ode_pops    <- out[, 2:8]
ode_final   <- as.numeric(out[nrow(out), 2:8])
trap        <- function(t, y) sum(diff(t) * (head(y, -1) + tail(y, -1)) / 2)
ode_time_avg <- sapply(1:7, function(j) trap(t_eval, ode_pops[, j])) / horizon_days
ode_peak    <- apply(ode_pops, 2, max)
ode_C       <- as.numeric(out[nrow(out), 9])
ode_D       <- as.numeric(out[nrow(out), 10])

## --- Build JSON in our schema ---------------------------------------------
labels  <- c("S", "E", "I-P", "I-A", "I-S", "I-H", "R")
mk_dict <- function(vals) setNames(as.list(as.numeric(vals)), labels)

splits <- list(
  `__source__` = list(S = 1),
  S            = list(E = 1),
  E            = list(`I-P` = 1),
  `I-P`        = list(`I-A` = p_a, `I-S` = 1 - p_a),
  `I-A`        = list(R = 1),
  `I-S`        = list(R = 1 - p_h, `I-H` = p_h),
  `I-H`        = list(R = 1 - p_d, D = p_d),
  R            = list(S = 1),
  D            = list(`main-sink` = 1)
)

result <- list(
  kernel             = "r-desolve",
  seed               = 0,
  totals             = list(created = ode_C, absorbed = ode_D),
  finalPopulations   = mk_dict(ode_final),
  transitionCounts   = splits,
  splitProbs         = splits,
  timeAvgPopulations = mk_dict(ode_time_avg),
  peakPopulations    = mk_dict(ode_peak),
  elapsedMs          = elapsed_ms,
  `_extras` = list(
    closedFormSteadyState = mk_dict(N_star),
    differenceEquation = list(
      dt                 = dt,
      diverged           = diverged,
      finalPopulations   = mk_dict(diff_final),
      timeAvgPopulations = mk_dict(diff_time_avg),
      peakPopulations    = mk_dict(diff_peak),
      totals             = list(created = C, absorbed = deaths)
    )
  )
)

parent_dir <- dirname(out_path)
if (!dir.exists(parent_dir)) dir.create(parent_dir, recursive = TRUE)
write_json(result, out_path, pretty = TRUE, auto_unbox = TRUE, digits = 10)

cat(sprintf("r-desolve      -> %s  (%.1f ms lsoda)\n", out_path, elapsed_ms))
if (diverged) {
  cat(sprintf("  diff-eq DIVERGED at dt=%g (expected if dt > 2*min(mu_c) = %g)\n",
              dt, 2 * min(unlist(mu)[-1])))
}
