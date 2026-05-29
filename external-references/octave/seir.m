#!/usr/bin/env octave
%% Octave reference implementation of the SEIR-with-hospitalization model.
%%
%% Three independent verifications:
%%   1. Closed-form steady state via mldivide ('\'). The open-system fixed
%%      point is N* = -A \ b. Should match the analytical f_S = lambda/q
%%      derivation in MATH.md.
%%   2. Forward-Euler difference equation in matrix form. Independent
%%      reimplementation of difference-runner.ts.
%%   3. ODE via lsode (Octave's stiff/non-stiff adaptive solver).
%%
%% How to run (from courses/hdm-fall-2022/des):
%%
%%   octave --no-gui --quiet external-references/octave/seir.m -- \
%%          --out out/external/octave/standard.json
%%
%% The default 'octave' command on $PATH is used. Override with
%%   OCTAVE_BIN=octave-cli ./external-references/run-all.sh
%%
%% Output JSON matches the RunResult schema in src/des/runners/types.ts so
%% it drops straight into validate-with-externals.ts as a column.

%% Parse CLI args:  -- --out PATH --dt N
args = argv();
out_path = '';
dt = 0.05;
i = 1;
while i <= length(args)
  switch args{i}
    case '--out'
      out_path = args{i+1}; i = i + 2;
    case '--dt'
      dt = str2double(args{i+1}); i = i + 2;
    otherwise
      i = i + 1;
  endswitch
endwhile

if isempty(out_path)
  fprintf(stderr, "usage: octave seir.m -- --out PATH [--dt N]\n");
  exit(2);
endif

%% --- Configuration (mirrors DEFAULT_CONFIG in src/des/runners/types.ts) ----
horizonDays = 1200;
phase1Days  = 800;
sourceCap   = 500;
arr_iat     = [0.7, 1.3];
res = struct(...
  'S',   [0.20, 0.40], 'E',   [0.20, 0.40], 'I_P', [0.20, 0.40], ...
  'I_A', [0.20, 0.40], 'I_S', [0.20, 0.40], 'I_H', [0.20, 0.40], ...
  'R',   [1.50, 2.50], 'D',   [0.10, 0.30]);
p_a = 0.40;
p_h = 0.20;
p_d = 0.12;

mu_S  = mean(res.S);  mu_E  = mean(res.E);
mu_IP = mean(res.I_P); mu_IA = mean(res.I_A);
mu_IS = mean(res.I_S); mu_IH = mean(res.I_H);
mu_R  = mean(res.R);   mu_D  = mean(res.D);
mu_arr = mean(arr_iat);
lambda = 1 / mu_arr;

%% --- Build the 7x7 transition matrix A and source vector b ----------------
A = zeros(7, 7);
A(1, 1) = -1/mu_S; A(1, 7) =  1/mu_R;            % S
A(2, 1) =  1/mu_S; A(2, 2) = -1/mu_E;            % E
A(3, 2) =  1/mu_E; A(3, 3) = -1/mu_IP;           % I-P
A(4, 3) =  p_a       /mu_IP; A(4, 4) = -1/mu_IA; % I-A
A(5, 3) = (1-p_a)    /mu_IP; A(5, 5) = -1/mu_IS; % I-S
A(6, 5) =  p_h       /mu_IS; A(6, 6) = -1/mu_IH; % I-H
A(7, 4) =  1/mu_IA;
A(7, 5) = (1-p_h)/mu_IS;
A(7, 6) = (1-p_d)/mu_IH;
A(7, 7) = -1/mu_R;                               % R

b_const = [lambda; 0; 0; 0; 0; 0; 0];

%% --- 1. Closed-form steady state (open system) ----------------------------
N_star = -A \ b_const;

%% --- 2. Forward-Euler difference equation ---------------------------------
n_steps = round(horizonDays / dt);
N = zeros(7, 1);
C = 0.0;
deaths = 0.0;
pop_sums = zeros(7, 1);
peak     = zeros(7, 1);
diverged = false;
for i = 1:n_steps
  pop_sums = pop_sums + N * dt;
  t_now = (i - 1) * dt;
  if (C < sourceCap) && (t_now < phase1Days)
    src = lambda;
  else
    src = 0;
  endif
  b_t = [src; 0; 0; 0; 0; 0; 0];
  dN = A * N + b_t;
  deaths = deaths + N(6) * p_d / mu_IH * dt;
  N = N + dt * dN;
  C = C + dt * src;
  peak = max(peak, N);
  if any(!isfinite(N))
    diverged = true;
    break;
  endif
endfor
diff_final    = N;
diff_time_avg = pop_sums / horizonDays;
diff_peak     = peak;
diff_C        = C;
diff_deaths   = deaths;

%% --- 3. ODE via lsode (split across the source-cutoff for smoothness) ----
%% Adaptive solvers struggle with the discontinuity at the source-cutoff
%% time t_off = min(phase1Days, sourceCap / lambda). Integrate in two
%% smooth phases instead and concatenate.
function dy = odefun(y, t, A, mu_IH, p_d, src_value)
  N_ = y(1:7);
  b_  = [src_value; 0; 0; 0; 0; 0; 0];
  dN_ = A * N_ + b_;
  dC_ = src_value;
  dD_ = N_(6) * p_d / mu_IH;
  dy  = [dN_; dC_; dD_];
endfunction

t_off = min(phase1Days, sourceCap / lambda);
t_off = min(t_off, horizonDays);

lsode_options('relative tolerance', 1e-8);
lsode_options('absolute tolerance', 1e-10);
y0 = zeros(9, 1);
ode_t0 = tic();

if t_off > 0
  t_eval_1 = (0:1:t_off)';
  if t_eval_1(end) != t_off; t_eval_1 = [t_eval_1; t_off]; endif
  y_out_1 = lsode(@(y, t) odefun(y, t, A, mu_IH, p_d, lambda), y0, t_eval_1);
  y_mid = y_out_1(end, :)';
else
  t_eval_1 = []; y_out_1 = []; y_mid = y0;
endif

if t_off < horizonDays
  t_eval_2 = (t_off:1:horizonDays)';
  if t_eval_2(1) != t_off; t_eval_2 = [t_off; t_eval_2]; endif
  y_out_2 = lsode(@(y, t) odefun(y, t, A, mu_IH, p_d, 0), y_mid, t_eval_2);
else
  t_eval_2 = []; y_out_2 = [];
endif
ode_elapsed_ms = toc(ode_t0) * 1000;

if !isempty(y_out_1) && !isempty(y_out_2)
  t_eval = [t_eval_1; t_eval_2(2:end)];
  y_out  = [y_out_1; y_out_2(2:end, :)];
elseif !isempty(y_out_1)
  t_eval = t_eval_1; y_out = y_out_1;
else
  t_eval = t_eval_2; y_out = y_out_2;
endif

ode_pops      = y_out(:, 1:7);
ode_final     = y_out(end, 1:7)';
ode_time_avg  = trapz(t_eval, ode_pops) / horizonDays;
ode_peak      = max(ode_pops)';
ode_C         = y_out(end, 8);
ode_D         = y_out(end, 9);

%% --- Build JSON manually (Octave field names cannot have '-') -------------
labels  = {'S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R'};
function s = ord_dict(labels, vals)
  parts = {};
  for k = 1:length(labels)
    parts{end+1} = sprintf('"%s": %s', labels{k}, jsonnum(vals(k)));
  endfor
  s = ['{', strjoin(parts, ', '), '}'];
endfunction
function s = jsonnum(x)
  if isnan(x)
    s = 'null';
  elseif isinf(x)
    s = sprintf('%s', '"Infinity"');  % unlikely; let consumer decide
  else
    s = sprintf('%.10g', x);
  endif
endfunction

splits_str = sprintf(['{ "__source__": {"S": 1.0}, "S": {"E": 1.0}, "E": {"I-P": 1.0}, ' ...
  '"I-P": {"I-A": %.4f, "I-S": %.4f}, "I-A": {"R": 1.0}, ' ...
  '"I-S": {"R": %.4f, "I-H": %.4f}, "I-H": {"R": %.4f, "D": %.4f}, ' ...
  '"R": {"S": 1.0}, "D": {"main-sink": 1.0} }'], ...
  p_a, 1 - p_a, 1 - p_h, p_h, 1 - p_d, p_d);

main_block = sprintf(['{\n' ...
  '  "kernel": "octave",\n' ...
  '  "seed": 0,\n' ...
  '  "totals": {"created": %.4f, "absorbed": %.4f},\n' ...
  '  "finalPopulations": %s,\n' ...
  '  "transitionCounts": %s,\n' ...
  '  "splitProbs": %s,\n' ...
  '  "timeAvgPopulations": %s,\n' ...
  '  "peakPopulations": %s,\n' ...
  '  "elapsedMs": %.1f,\n' ...
  '  "_extras": {\n' ...
  '    "closedFormSteadyState": %s,\n' ...
  '    "differenceEquation": {\n' ...
  '       "dt": %g, "diverged": %s,\n' ...
  '       "finalPopulations": %s,\n' ...
  '       "timeAvgPopulations": %s,\n' ...
  '       "peakPopulations": %s,\n' ...
  '       "totals": {"created": %.4f, "absorbed": %.4f}\n' ...
  '    }\n' ...
  '  }\n' ...
  '}\n'], ...
  ode_C, ode_D, ...
  ord_dict(labels, ode_final), ...
  splits_str, splits_str, ...
  ord_dict(labels, ode_time_avg), ...
  ord_dict(labels, ode_peak), ...
  ode_elapsed_ms, ...
  ord_dict(labels, N_star), ...
  dt, ifelse(diverged, 'true', 'false'), ...
  ord_dict(labels, diff_final), ...
  ord_dict(labels, diff_time_avg), ...
  ord_dict(labels, diff_peak), ...
  diff_C, diff_deaths);

%% Make sure parent dir exists
[parent, ~, ~] = fileparts(out_path);
if !isempty(parent) && !exist(parent, 'dir')
  mkdir(parent);
endif

fid = fopen(out_path, 'w');
fprintf(fid, '%s', main_block);
fclose(fid);

printf('octave         -> %s  (%.1f ms LSODE)\n', out_path, ode_elapsed_ms);
if diverged
  printf('  diff-eq DIVERGED at dt=%g (expected if dt > 2*min(mu_c) = %g)\n', dt, 2 * min([mu_S mu_E mu_IP mu_IA mu_IS mu_IH mu_R mu_D]));
endif
