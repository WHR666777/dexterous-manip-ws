"""Compare a committed Nero IK baseline with the working tree on recorded goals.

Offline only: never constructs an SDK/hardware object. Example:
python scripts/benchmark_nero_ik.py --baseline-revision <commit>
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import ModuleType

import numpy as np
import pinocchio as pin
import scipy
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_control import nero_ik  # noqa: E402


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return dict(count=len(values), mean=float(np.mean(values)),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)), max=float(np.max(values)))


def load_baseline(revision):
    commit = subprocess.check_output(
        ['git', 'rev-parse', '--verify', f'{revision}^{{commit}}'], cwd=ROOT, text=True
    ).strip()
    source = subprocess.check_output(
        ['git', 'show', f'{commit}:robot_control/nero_ik.py'], cwd=ROOT
    )
    module = ModuleType('nero_ik_baseline')
    module.__file__ = str(ROOT / 'robot_control' / 'nero_ik.py')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module, commit, hashlib.sha256(source).hexdigest()


def load_cases(directory, gap_seconds):
    cases, files, total_rows = [], [], 0
    for path in sorted(directory.glob('*/trajectory.npz')):
        with np.load(path, allow_pickle=False) as data:
            targets = data['target_base_flange']
            feedback = data['arm_joint_position']
            recorded_actions = data['arm_action']
            timestamps = data['timestamp_monotonic']
        count = len(timestamps)
        if (targets.shape != (count, 4, 4) or feedback.shape != (count, 7)
                or recorded_actions.shape != (count, 7)
                or not np.isfinite(timestamps).all()
                or np.any(np.diff(timestamps) < 0)):
            raise ValueError(f'Invalid trajectory shapes/timestamps: {path}')
        total_rows += count
        files.append(dict(episode=path.parent.name, rows=count,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        # The recording lacks the seed read before the current command. Use the
        # preceding feedback for BOTH solvers; skip row 0 rather than invent it.
        for row in range(1, count):
            dt = float(timestamps[row] - timestamps[row - 1])
            cases.append(dict(episode=path.parent.name, row=row, dt_s=dt,
                              gap=dt > gap_seconds, target=targets[row].copy(),
                              seed=feedback[row - 1].copy(),
                              recorded_action=recorded_actions[row].copy()))
    if not cases:
        raise ValueError(f'No paired trajectory samples under {directory}')
    return cases, files, total_rows


def run_solver(solver, case, reference):
    started = time.perf_counter()
    q = solver.solve(case['target'], case['seed'])
    elapsed_ms = (time.perf_counter() - started) * 1000
    diag = solver.last_diagnostics
    result = dict(elapsed_ms=elapsed_ms, success=q is not None, code=diag['code'],
                  deadline_fallback=bool(diag.get('deadline_fallback', False)),
                  residual_evaluations=diag.get('residual_evaluations', 0),
                  jacobian_evaluations=diag.get('jacobian_evaluations'),
                  q_rad=None, seed_delta_max_deg=None, seed_delta_l2_deg=None,
                  recorded_action_delta_max_deg=None,
                  position_error_mm=None, rotation_error_deg=None)
    if q is not None:
        # Verify returned solutions using the OLD FK, outside the timed region.
        p, r = nero_ik.pose_error(reference.forward(q), case['target'])
        dq = np.rad2deg(q - case['seed'])
        if (not solver.in_limits(q) or not solver.continuous(q, case['seed'])
                or p >= nero_ik.MAX_FK_POSITION_ERROR_M
                or np.rad2deg(r) >= nero_ik.MAX_FK_ROTATION_ERROR_DEG):
            raise AssertionError(f'Returned solution failed independent acceptance: {case}')
        result.update(q_rad=q.tolist(), seed_delta_max_deg=float(np.max(np.abs(dq))),
                      seed_delta_l2_deg=float(np.linalg.norm(dq)),
                      recorded_action_delta_max_deg=float(np.max(np.abs(
                          np.rad2deg(q - case['recorded_action'])))),
                      position_error_mm=p * 1000, rotation_error_deg=float(np.rad2deg(r)))
    return result


def summarize(records):
    if not records:
        return None
    result = {'cases': len(records)}
    for label in ('old', 'new'):
        samples = [r[label] for r in records]
        accepted = [s for s in samples if s['success']]
        summary = dict(successes=len(accepted), success_rate=len(accepted) / len(samples),
                       codes=dict(Counter(s['code'] for s in samples)),
                       deadline_fallbacks=sum(s['deadline_fallback'] for s in samples))
        for metric in ('elapsed_ms', 'residual_evaluations', 'jacobian_evaluations',
                       'seed_delta_max_deg', 'seed_delta_l2_deg',
                       'position_error_mm', 'rotation_error_deg',
                       'recorded_action_delta_max_deg'):
            summary[metric] = distribution([s[metric] for s in samples if s[metric] is not None])
        steps, previous = [], {}
        for row in records:
            key = (row['repeat'], row['episode'])
            prev = previous.get(key)
            if (prev is not None and row['row'] == prev['row'] + 1
                    and not row['gap'] and row[label]['success'] and prev[label]['success']):
                steps.append(np.rad2deg(np.asarray(row[label]['q_rad'])
                                       - np.asarray(prev[label]['q_rad'])))
            previous[key] = row
        summary['adjacent_solution_step_max_deg'] = distribution(
            [float(np.max(np.abs(step))) for step in steps])
        summary['adjacent_solution_step_l2_deg'] = distribution(
            [float(np.linalg.norm(step)) for step in steps])
        summary['adjacent_solution_step_abs_deg_per_joint_p95'] = (
            np.percentile(np.abs(steps), 95, axis=0).tolist() if steps else None)
        result[label] = summary
    paired = [r for r in records if r['old']['success'] and r['new']['success']]
    differences = [np.rad2deg(np.asarray(r['new']['q_rad']) - np.asarray(r['old']['q_rad']))
                   for r in paired]
    result['paired_solution_delta_max_deg'] = distribution(
        [float(np.max(np.abs(dq))) for dq in differences])
    result['paired_solution_abs_delta_deg_per_joint_p95'] = (
        np.percentile(np.abs(differences), 95, axis=0).tolist() if differences else None)
    result['old_only_successes'] = sum(r['old']['success'] and not r['new']['success'] for r in records)
    result['new_only_successes'] = sum(r['new']['success'] and not r['old']['success'] for r in records)
    result['mean_speedup'] = result['old']['elapsed_ms']['mean'] / result['new']['elapsed_ms']['mean']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-revision', required=True)
    parser.add_argument('--episodes', type=Path, default=ROOT / 'data/episodes')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/quest3_nero_l20.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/benchmarks/nero_ik_pinocchio')
    parser.add_argument('--gap-seconds', type=float, default=.5)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.gap_seconds <= 0:
        parser.error('repeats and gap-seconds must be positive')
    baseline, commit, source_hash = load_baseline(args.baseline_revision)
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    joint_delta = float(config['arm']['max_joint_delta_rad'])
    old = baseline.NeroIK(nero_ik.URDF_PATH, max_joint_delta_rad=joint_delta)
    new = nero_ik.NeroIK(max_joint_delta_rad=joint_delta)
    # Fair comparison requires identical objectives, budgets and acceptance limits.
    constants = ('MAX_FK_POSITION_ERROR_M', 'MAX_FK_ROTATION_ERROR_DEG',
                 'REGULARIZATION_WEIGHT', 'MAX_NFEV', 'MAX_SOLVE_SECONDS')
    for name in constants:
        if getattr(baseline, name) != getattr(nero_ik, name):
            raise ValueError(f'Changed comparison constant: {name}')
    cases, files, total_rows = load_cases(args.episodes, args.gap_seconds)
    for case in cases[:3]:
        old.solve(case['target'], case['seed'])
        new.solve(case['target'], case['seed'])
    records = []
    total = len(cases) * args.repeats
    print(f'{len(files)} episodes, {total_rows} rows, {total} paired solves; '
          f'gap transitions={sum(c["gap"] for c in cases)}', flush=True)
    for repeat in range(args.repeats):
        for index, case in enumerate(cases):
            record = {k: case[k] for k in ('episode', 'row', 'dt_s', 'gap')}
            record['repeat'] = repeat
            # Alternate solver order to reduce order-dependent timing bias.
            order = [('old', old), ('new', new)]
            if (index + repeat) % 2:
                order.reverse()
            for label, solver in order:
                record[label] = run_solver(solver, case, old)
            records.append(record)
            if len(records) % 200 == 0:
                print(f'Completed {len(records)}/{total}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        baseline_commit=commit, baseline_source_sha256=source_hash,
        current_source_sha256=hashlib.sha256(Path(nero_ik.__file__).read_bytes()).hexdigest(),
        urdf_sha256=hashlib.sha256(nero_ik.URDF_PATH.read_bytes()).hexdigest(),
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                         numpy=np.__version__, scipy=scipy.__version__, pinocchio=pin.__version__),
        parameters=dict(max_joint_delta_rad=joint_delta, repeats=args.repeats,
                        gap_seconds=args.gap_seconds, seed='previous row arm_joint_position',
                        **{name: getattr(nero_ik, name) for name in constants}),
        episodes=files, total_rows=total_rows, skipped_first_rows=len(files),
        all_samples=summarize(records),
        continuous_samples=summarize([r for r in records if not r['gap']]),
        gap_samples=summarize([r for r in records if r['gap']]) if any(r['gap'] for r in records) else None,
    )
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    flat_records = []
    for record in records:
        flat = {k: v for k, v in record.items() if k not in ('old', 'new')}
        for label in ('old', 'new'):
            flat.update({f'{label}_{k}': json.dumps(v) if isinstance(v, list) else v
                         for k, v in record[label].items()})
        flat_records.append(flat)
    with (args.output / 'samples.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_records[0]))
        writer.writeheader()
        writer.writerows(flat_records)
    print(json.dumps(summary['all_samples'], indent=2), flush=True)
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
