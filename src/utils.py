"""Configuration, provenance and the single-command FlowCast workflow."""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import os
import random

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['Free-flow', 'Moderate', 'Heavy', 'Severe']
COLORS = ['#21918c', '#d5b63c', '#ee854a', '#b63758']


def config():
    """Read the single project configuration."""
    import yaml
    with (ROOT / 'config.yaml').open() as stream:
        return yaml.safe_load(stream)


def setup():
    """Seed random generators and create only the specified output folders."""
    import numpy as np
    cfg = config()
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    for folder in ['data/interim', 'data/processed', 'models', 'reports/figures']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(ROOT / 'reports/pipeline.log', mode='a'),
                                  logging.StreamHandler()], force=True)
    return cfg


def digest(path):
    """Return a source or artifact SHA-256 without altering its bytes."""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    """Write human-readable, strict JSON metadata."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    def finite(item):
        import math
        if isinstance(item, dict):
            return {str(k): finite(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    temporary.write_text(json.dumps(finite(value), indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)



def save_csv(frame, path, index=False):
    """Replace an artifact atomically so dashboard readers never see partial CSVs."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    compression = {'method': 'gzip', 'mtime': 0} if path.name.endswith('.gz') else None
    frame.to_csv(temporary, index=index, compression=compression)
    temporary.replace(path)


def save_model(bundle, path):
    """Publish a complete persisted estimator in a single filesystem replacement."""
    import joblib
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(path)


def markdown_table(frame, digits=4):
    """Render a small table without adding a tabulate dependency."""
    def cell(x):
        if isinstance(x, float):
            return f'{x:.{digits}f}'
        return str(x).replace('|', '/')
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, row)) + ' |'
                      for row in frame.itertuples(index=False, name=None)])



def verify_delivery():
    """Execute the notebooks and real-artifact tests, then record the evidence."""
    import re
    import sys
    import subprocess
    import tempfile
    import nbformat
    from nbclient import NotebookClient
    with tempfile.TemporaryDirectory(prefix='flowcast-notebooks-') as temporary:
        previous = {key: os.environ.get(key) for key in ['IPYTHONDIR', 'JUPYTER_RUNTIME_DIR']}
        os.environ['IPYTHONDIR'] = temporary
        os.environ['JUPYTER_RUNTIME_DIR'] = temporary
        try:
            for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
                notebook = nbformat.read(path, as_version=4)
                NotebookClient(notebook, timeout=180, kernel_name='python3',
                               resources={'metadata': {'path': str(ROOT)}}).execute()
                nbformat.write(notebook, path)
                logging.info('Executed notebook %s', path.name)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (ROOT / 'reports/test_results.txt').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError('Verification failed. See reports/test_results.txt.')
    count = re.search(r'Ran (\d+) tests', result.stdout)
    summary = f"{count.group(1) if count else 'All'} automated tests passed."
    report_path = ROOT / 'reports/final_report.md'
    with report_path.open('a') as stream:
        stream.write('\n## Verification\n\n' + summary +
                     ' All three notebooks executed successfully. The tests cover original CSV integrity, grid reconciliation, causal cleaning, lag timing, time splits, model reloads, matching deep/classical test keys, metric calculations, all nine dashboard views, prediction, upload and report exports. Training-interface tests check process launch and status with a test double, so they do not overwrite trained models. Full output is in `reports/test_results.txt`.\n')
    logging.info(summary)

def main():
    """Run all required stages, or one stage for a controlled rerun."""
    parser = argparse.ArgumentParser(description='Reproduce FlowCast from original CSV files.')
    parser.add_argument('--stage', choices=['all', 'data', 'ml', 'dl', 'report'], default='all')
    args = parser.parse_args()
    cfg = setup()
    from src.clean import prepare
    from src.features import load_dataset
    if args.stage in ['all', 'data']:
        prepare()
    if args.stage in ['all', 'ml']:
        from src.ml_models import train
        train(load_dataset(), cfg)
    if args.stage in ['all', 'dl']:
        from src.dl_model import train
        train(load_dataset(), cfg)
    if args.stage in ['all', 'report']:
        from src.evaluate import report
        report(load_dataset(), cfg)
        verify_delivery()
    logging.info('Requested workflow completed successfully.')


if __name__ == '__main__':
    # Set before NumPy/BLAS imports, including fresh-process dashboard training.
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '2'
    try:
        from filelock import FileLock
        with FileLock(str(ROOT / 'reports/pipeline.lock'), timeout=0):
            main()
    except Exception:
        logging.exception('FlowCast workflow failed')
        raise
