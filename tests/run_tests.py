"""
Regression tests for scale_search.

Run from anywhere:

    python tests/run_tests.py            # check everything
    python tests/run_tests.py --update   # accept the current results as expected
    python tests/run_tests.py edo seeds  # only tests whose name contains a word

Three kinds of test:

  - Result tests run a short search or comparison with a fixed --random-seed
    and compare the JSON output with the stored copy in tests/expected/. A
    difference is not necessarily a bug: a change to the algorithm or a
    metric is expected to change results. It means "look at this", and once
    the new results are understood to be right, --update stores them.
  - Error tests check that invalid options stop with exit code 2 and a
    message, before any search starts.
  - Round-trip tests check that files the tool writes (--save-result,
    --export-seeds, --output-scl) can be read back and give the same scale
    and quality.

The expected results were produced with Python 3.13. Another Python version
may produce different but equally valid results from the same random seed;
in that case run --update once on a version of the code known to be good,
and check later changes against that.

Uses only the standard library. Takes about a minute.
"""

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, 'tests')
DATA = os.path.join(TESTS, 'data')
EXPECTED = os.path.join(TESTS, 'expected')
SEEDS = os.path.join('tests', 'data', 'seeds.txt')
ARCHIVE = os.path.join('tests', 'data', 'archive')

SEARCH = ['--random-seed', '7', '--json']

# name -> arguments. Kept small so the whole set runs in about a minute.
RESULT_CASES = {
    'flat_default': ['--notes', '7', '--generations', '30', '--population', '60'],
    'flat_range_multi': ['--min-notes', '5', '--max-notes', '8',
                         '--metric', 'tenney:1,entropy:0.5,steps:1',
                         '--generations', '20', '--population', '40'],
    'flat_negative_weight': ['--notes', '6', '--metric', 'barlow:1,propriety:-0.5',
                             '--generations', '20', '--population', '40'],
    'flat_include_exclude': ['--notes', '7', '--include', '7/4,9/4', '--exclude', '5/4',
                             '--generations', '20', '--population', '40'],
    'flat_edo': ['--notes', '7', '--metric', 'barlow,edo12',
                 '--generations', '20', '--population', '40'],
    'flat_worst_octaves0': ['--notes', '5', '--combine', 'worst', '--octaves', '0',
                            '--generations', '20', '--population', '40'],
    'flat_runs2': ['--notes', '6', '--runs', '2', '--generations', '15', '--population', '30'],
    'group_icosahedron': ['--groups', 'icosahedron', '--generations', '20', '--population', '40'],
    'group_cyclic': ['--groups', 'cyclic:12,7', '--combine', 'logloss',
                     '--generations', '15', '--population', '40'],
    'group_file_ordered': ['--groups', os.path.join('groups', '12tone-triads.txt'),
                           '--ordered-slots', '--generations', '15', '--population', '40'],
    'group_global_metric': ['--groups', '24cell', '--metric', 'soft_tenney',
                            '--global-metric', 'edo24:2,steps:1',
                            '--generations', '4', '--population', '20'],
    'seeds_file': ['--notes', '7', '--seeds', SEEDS, '--generations', '15', '--population', '40'],
    'seeds_archive_analyze': ['--notes', '7', '--seeds', ARCHIVE, '--seeds', SEEDS,
                              '--analyze', '--generations', '10', '--population', '40'],
    'compare_ratios': ['--compare', '9/8,5/4,4/3,3/2,5/3,15/8', '--metric', 'tenney,propriety'],
    'compare_cents': ['--compare', '200.0,400.0,500.0,700.0,900.0,1100.0'],
    'compare_sources': ['--compare', '--seeds', SEEDS, '--seeds', ARCHIVE,
                        '--metric', 'barlow:1,steps:0.5'],
}

# (name, arguments, text that must appear in the error message)
ERROR_CASES = [
    ('unknown_metric', ['--metric', 'foo'], 'Unknown metric'),
    ('combine_typo', ['--combine', 'geometic'], "Did you mean 'geometric'"),
    ('bad_weight', ['--metric', 'barlow:x'], 'weight'),
    ('cents_in_include', ['--include', '701.9'], 'cents'),
    ('exclude_tonic', ['--exclude', '1/1'], 'cannot exclude 1/1'),
    ('include_and_exclude', ['--include', '3/2', '--exclude', '3/2'], 'both required and excluded'),
    ('population_too_small', ['--population', '1'], 'population'),
    ('missing_seeds', ['--seeds', 'no/such/file.txt'], 'not found'),
    ('bad_edo_range', ['--edo-range', '12'], '--edo-range'),
    ('global_metric_without_groups', ['--global-metric', 'edo24'], 'only applies with --groups'),
    ('empty_global_metric', ['--groups', 'cube', '--global-metric', ''], 'no metrics given'),
    ('top_without_export', ['--top', '5'], 'only applies with --export-seeds'),
    ('export_without_seeds', ['--export-seeds', 'x.txt'], 'needs at least one --seeds'),
    ('compare_with_output', ['--compare', '--output-scl', 'x.scl'], '--compare does not search'),
    ('unknown_group', ['--groups', 'hypercube'], 'hypercube'),
]


def run(args, cwd=ROOT):
    """Run the tool; returns (exit code, stdout, stderr)."""
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, '-m', 'scale_search'] + args, cwd=cwd,
                       capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


def run_json(args):
    code, out, err = run(args)
    if code != 0:
        raise RuntimeError(f"exit code {code}: {err.strip() or out.strip()}")
    return json.loads(out)


def describe_difference(old, new):
    """A short account of how two JSON results differ."""
    if isinstance(old, dict) and isinstance(new, dict):
        if 'quality' in old and 'quality' in new and old.get('scale') != new.get('scale'):
            return (f"scale {' '.join(old['scale'])} (quality {old['quality']}) -> "
                    f"{' '.join(new['scale'])} (quality {new['quality']})")
        keys = [k for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
        return "differs in: " + ", ".join(keys)
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return f"{len(old)} entries -> {len(new)}"
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            if a != b:
                return f"entry {i + 1} {describe_difference(a, b)}"
    return "differs"


def result_tests(selected, update):
    failures = []
    for name, args in RESULT_CASES.items():
        if not selected(name):
            continue
        path = os.path.join(EXPECTED, name + '.json')
        try:
            result = run_json(args + SEARCH)
        except (RuntimeError, json.JSONDecodeError) as e:
            failures.append(name)
            print(f"FAIL  {name}: {e}")
            continue
        if update or not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=1, sort_keys=True)
                f.write('\n')
            print(f"saved {name}")
            continue
        with open(path, encoding='utf-8') as f:
            expected = json.load(f)
        if result == expected:
            print(f"ok    {name}")
        else:
            failures.append(name)
            print(f"FAIL  {name}: {describe_difference(expected, result)}")
    return failures


def error_tests(selected):
    failures = []
    for name, args, needle in ERROR_CASES:
        if not selected(name):
            continue
        # Short settings, so a missing check shows up as a fast wrong result
        # rather than a long search
        extra = [a for flag, value in (('--generations', '1'), ('--population', '10'))
                 if flag not in args for a in (flag, value)]
        code, out, err = run(args + extra)
        message = (err + out).strip()
        if code == 2 and needle.lower() in message.lower():
            print(f"ok    {name}")
        else:
            failures.append(name)
            print(f"FAIL  {name}: expected exit code 2 and '{needle}', "
                  f"got exit code {code}: {message.splitlines()[-1] if message else '(no output)'}")
    return failures


def check(name, condition, detail, failures):
    if condition:
        print(f"ok    {name}")
    else:
        failures.append(name)
        print(f"FAIL  {name}: {detail}")


def roundtrip_tests(selected):
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        if selected('roundtrip_save_flat'):
            saved = os.path.join(tmp, 'flat.txt')
            r = run_json(['--notes', '7', '--generations', '15', '--population', '40',
                          '--save-result', saved] + SEARCH)
            c = run_json(['--compare', '--seeds', saved, '--json'])['ranking']
            check('roundtrip_save_flat',
                  c[0]['scale'] == r['scale'] and c[0]['quality'] == r['quality'],
                  f"saved {r['scale']} {r['quality']}, read back {c[0]['scale']} "
                  f"{c[0]['quality']}", failures)

        if selected('roundtrip_save_grouped'):
            saved = os.path.join(tmp, 'grouped.txt')
            args = ['--groups', 'icosahedron', '--metric', 'soft_tenney',
                    '--global-metric', 'edo12', '--population', '40']
            r = run_json(args + ['--generations', '15', '--save-result', saved] + SEARCH)
            again = run_json(args + ['--generations', '0', '--seeds', saved] + SEARCH)
            check('roundtrip_save_grouped',
                  again['slots'] == r['slots'] and again['quality'] == r['quality'],
                  f"saved quality {r['quality']}, re-seeded {again['quality']}", failures)

        if selected('roundtrip_output_scl'):
            scl = os.path.join(tmp, 'result.scl')
            r = run_json(['--notes', '6', '--generations', '15', '--population', '40',
                          '--output-scl', scl] + SEARCH)
            c = run_json(['--compare', '--seeds', scl, '--json'])['ranking']
            check('roundtrip_output_scl',
                  c[0]['scale'] == r['scale'] and c[0]['quality'] == r['quality'],
                  f"wrote {r['scale']} {r['quality']}, read back {c[0]['scale']} "
                  f"{c[0]['quality']}", failures)

        if selected('roundtrip_export_seeds'):
            exported = os.path.join(tmp, 'exported.txt')
            code, out, err = run(['--seeds', SEEDS, '--seeds', ARCHIVE,
                                  '--export-seeds', exported])
            c = run_json(['--compare', '--seeds', exported, '--json'])['ranking'] if code == 0 else []
            sizes = sorted(len(e['scale']) for e in c)
            # seeds.txt holds a scale and one of its modes, and the archive holds
            # the same scale again in ratios: three entries, one kept. Bohlen-
            # Pierce is skipped as non-octave. Expected (scale lists include
            # 2/1, so the length is the note count): two pentatonic scales
            # (seeds.txt, slendro), three 7-note (Ptolemy, septimal, harmonics
            # 8-14), one 12-note.
            check('roundtrip_export_seeds',
                  code == 0 and sizes == [5, 5, 7, 7, 7, 12],
                  f"exit code {code}, note counts read back (with 2/1): {sizes}; "
                  f"{(err or out).strip()}", failures)

        if selected('roundtrip_export_top'):
            exported = os.path.join(tmp, 'top1.txt')
            code, out, err = run(['--seeds', SEEDS, '--seeds', ARCHIVE, '--top', '1',
                                  '--export-seeds', exported])
            c = run_json(['--compare', '--seeds', exported, '--json'])['ranking'] if code == 0 else []
            check('roundtrip_export_top',
                  code == 0 and sorted(len(e['scale']) for e in c) == [5, 7, 12],
                  f"exit code {code}, {len(c)} scales read back", failures)
    return failures


def main():
    words = [a for a in sys.argv[1:] if not a.startswith('--')]
    update = '--update' in sys.argv

    def selected(name):
        return not words or any(w in name for w in words)

    os.makedirs(EXPECTED, exist_ok=True)
    print("Result tests" + (" (updating expected results)" if update else ""))
    failures = result_tests(selected, update)
    print("\nError tests")
    failures += error_tests(selected)
    print("\nRound-trip tests")
    failures += roundtrip_tests(selected)

    print()
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        sys.exit(1)
    print("All tests passed.")


if __name__ == '__main__':
    main()
