# Can Saliency Maps Flag Chest Radiograph Classifier Errors at a New Hospital? An External Validation Study
# Publication figures and supplementary analyses
# Aashish Chapagai (Last Edited: October 2026)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import logging
import warnings
import contextlib
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

@contextlib.contextmanager
def quiet_native_output():
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(2)
    sink = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 2)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(sink)
        os.close(saved)

with quiet_native_output():
    import tensorflow as tf
    import keras_hub  
    import shap
    from lime import lime_image
    tf.get_logger().setLevel('ERROR')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

def no_progress(iterable, *args, **kwargs):
    return iterable

lime_image.tqdm = no_progress

SEED = 42
N_BOOTSTRAP = 1000
MIN_EVENTS = 20
FOLDS = 5
COVERAGE = 0.5
LIME_SAMPLES = 1000
SHAP_BACKGROUND = 50
SHAP_SAMPLES = 100
MODELS = {'DenseNet-121': ('models/cnn_base.keras', ''), 'ViT-B/16': ('models/vit_base.keras', 'vit_')}
SITES = ['NIH', 'CheXpert', 'VinDr-CXR']
PARTITIONS = ['NIH evaluation', 'CheXpert', 'VinDr-CXR']
SITE_OF = {'NIH evaluation': 'NIH', 'CheXpert': 'CheXpert', 'VinDr-CXR': 'VinDr-CXR'}
SCOPES = ('Flagged findings', 'Unflagged findings')
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
            'Infiltration', 'Lesion', 'Pleural thickening', 'Pneumonia', 'Pneumothorax']
SCORES = ['Confidence margin', 'TTA agreement', 'Anatomical enrichment', 'Margin + TTA', 'Margin + TTA + anatomy']
TICKS = {'Confidence margin': 'Margin', 'TTA agreement': 'TTA', 'Anatomical enrichment': '$E$',
         'Margin + TTA': 'Margin\n+ TTA', 'Margin + TTA + anatomy': 'Margin\n+ TTA + $E$', 'All flags': 'All\nflags'}
COMBINERS = (('Margin + TTA', ['abs_margin', 'tta_gap']), ('Margin + TTA + anatomy', ['abs_margin', 'tta_gap', 'region_enrichment']))
COLOURS = {'NIH': '#2a78d6', 'CheXpert': '#eb6834', 'VinDr-CXR': '#1baf7a'}
INK, SECONDARY, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
REGION, LUNG, BOX = '#00c8e0', '#9be39b', '#ffd400'
MARKERS = {'DenseNet-121': 'o', 'ViT-B/16': 's'}
OFFSETS = {'NIH': -0.24, 'CheXpert': 0.0, 'VinDr-CXR': 0.24, 'NIH evaluation': -0.24}
AUC_F = 'AUROC$_\\mathrm{f}$'
OUT = os.path.join('figures', 'paper')
ED_NUMBERS = {'ED_Fig1': 'ED_Fig6', 'ED_Fig2': 'ED_Fig1', 'ED_Fig3': 'ED_Fig5', 'ED_Fig4': 'ED_Fig2', 'ED_Fig5': 'ED_Fig4',
              'ED_Fig6': 'ED_Fig3'}
MM = 1 / 25.4
WIDTH = 183.0
TICK = 6.5
LABEL = 7.0

def required_files():
    paths = ['models/test_patient_ids.npy', 'models/X_test.npy', 'models/anatomy_nih.npy', 'models/gt_boxes_nih.csv']
    for path, prefix in MODELS.values():
        paths += [path, f'models/{prefix}xai_results.json', f'models/{prefix}xai_decisions.csv',
                  f'models/{prefix}xai_cross_method.csv', f'models/{prefix}xai_outputs_nih.npz']
    return paths

def set_style():
    for path in ('/mnt/c/Windows/Fonts/arial.ttf', '/mnt/c/Windows/Fonts/arialbd.ttf', '/mnt/c/Windows/Fonts/ariali.ttf'):
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next(name for name in ('Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans') if name in available)
    plt.rcParams.update({
        'font.family': family, 'font.size': LABEL, 'axes.titlesize': LABEL, 'axes.labelsize': LABEL,
        'xtick.labelsize': TICK, 'ytick.labelsize': TICK, 'legend.fontsize': TICK, 'axes.linewidth': 0.6,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': SECONDARY,
        'ytick.color': SECONDARY, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.pad': 2.0,
        'ytick.major.pad': 2.0, 'axes.labelpad': 3.0, 'axes.titlepad': 4.0, 'axes.spines.top': False,
        'axes.spines.right': False, 'legend.frameon': False, 'legend.handletextpad': 0.3, 'legend.borderaxespad': 0.3,
        'legend.columnspacing': 1.4, 'legend.handlelength': 1.6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': 300, 'figure.dpi': 150, 'lines.linewidth': 1.0, 'mathtext.fontset': 'custom',
        'mathtext.rm': family, 'mathtext.it': f'{family}:italic', 'mathtext.bf': f'{family}:bold', 'mathtext.sf': family,
        'mathtext.cal': family, 'mathtext.tt': family, 'axes.unicode_minus': True,
    })
    return family

def patient_groups(patients):
    _, inverse = np.unique(np.asarray(patients).astype(str), return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    return np.split(order, np.cumsum(np.bincount(inverse))[:-1])

def bootstrap(statistics, groups, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in statistics}
    for _ in range(n_bootstrap):
        rows = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        for key, (statistic, arguments) in statistics.items():
            value = statistic(rows, **arguments)
            if value is not None and np.isfinite(value):
                samples[key].append(value)
    intervals = {}
    for key, values in samples.items():
        enough = len(values) >= n_bootstrap // 2
        intervals[key] = np.percentile(values, [2.5, 97.5]) if enough else np.array([np.nan, np.nan])
    return intervals

def safe_auc(truth, score):
    return roc_auc_score(truth, score) if 0 < truth.sum() < len(truth) else None

def aurc(confidence, correct):
    order = np.argsort(-confidence, kind='stable')
    risk = np.cumsum(1 - correct[order]) / np.arange(1, len(order) + 1)
    return float(risk.mean())

def risk_coverage(confidence, correct, points=200):
    order = np.argsort(-confidence, kind='stable')
    risk = np.cumsum(1 - correct[order]) / np.arange(1, len(order) + 1)
    keep = np.unique(np.linspace(0, len(order) - 1, min(points, len(order))).astype(int))
    return (keep + 1) / len(order), risk[keep]

def selective_risk(confidence, correct, coverage=COVERAGE):
    keep = int(np.ceil(coverage * len(correct)))
    order = np.argsort(-confidence, kind='stable')[:keep]
    return float(1 - correct[order].mean())

def auc_at(rows, correct, values):
    return safe_auc(correct[rows], values[rows])

def delta_auc_at(rows, correct, full, base):
    first, second = safe_auc(correct[rows], full[rows]), safe_auc(correct[rows], base[rows])
    return None if first is None or second is None else first - second

def delta_aurc_at(rows, correct, full, base):
    return aurc(full[rows], correct[rows]) - aurc(base[rows], correct[rows])

def risk_at(rows, correct, values):
    return selective_risk(values[rows], correct[rows])

def base_risk_at(rows, correct):
    return float(1 - correct[rows].mean())

def spearman_at(rows, first, second):
    if np.ptp(first[rows]) == 0 or np.ptp(second[rows]) == 0:
        return None
    return float(spearmanr(first[rows], second[rows]).statistic)

def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p, kind='stable')):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted

def fmt(value, low, high, digits=4, signed=False):
    sign = '+' if signed else ''
    return f"{value:{sign}.{digits}f} [{low:{sign}.{digits}f}, {high:{sign}.{digits}f}]"

def load():
    patients = np.load('models/test_patient_ids.npy', allow_pickle=True).astype(str)
    nih_patients = np.unique(patients)
    calibration_patients = set(np.random.default_rng(SEED).permutation(nih_patients)[:len(nih_patients) // 2])
    data = {'results': {}, 'decisions': {}, 'frames': {}, 'cross': {}, 'cams': {}, 'coefficients': {}, 'shift': {}}
    for model, (_, prefix) in MODELS.items():
        with open(f'models/{prefix}xai_results.json') as handle:
            results = json.load(handle)
        table = pd.read_csv(f'models/{prefix}xai_decisions.csv', dtype={'patient': str})
        has_enrichment = table['region_enrichment'].notna()
        frames = {}
        for scope, value in zip(SCOPES, (1, 0)):
            frame = table[(table['decision'] == value) & has_enrichment].copy()
            frame['abs_margin'] = frame['margin'].abs()
            nih_part = np.where(frame['patient'].isin(calibration_patients), 'NIH calibration', 'NIH evaluation')
            frame['partition'] = np.where(frame['dataset'] != 'NIH', frame['dataset'], nih_part)
            calibration = frame[frame['partition'] == 'NIH calibration']
            frame['Confidence margin'] = frame['abs_margin']
            frame['TTA agreement'] = -frame['tta_gap']
            frame['Anatomical enrichment'] = frame['region_enrichment']
            for label, features in COMBINERS:
                scaler = StandardScaler().fit(calibration[features])
                combiner = LogisticRegression(max_iter=1000).fit(scaler.transform(calibration[features]), calibration['correct'])
                frame[label] = combiner.predict_proba(scaler.transform(frame[features]))[:, 1]
            frames[scope] = frame
        shift = pd.DataFrame(results['enrichment_shift'])
        shift['holm'] = np.nan
        for site in ('CheXpert', 'VinDr-CXR'):
            rows = (shift['external'] == site) & (shift['finding'] != 'All findings')
            shift.loc[rows, 'holm'] = holm(shift.loc[rows, 'p_value'].clip(lower=1e-300))
        data['results'][model] = results
        data['decisions'][model] = table
        data['frames'][model] = frames
        data['cross'][model] = pd.read_csv(f'models/{prefix}xai_cross_method.csv', dtype={'patient': str})
        data['cams'][model] = np.load(f'models/{prefix}xai_outputs_nih.npz')['cams']
        data['coefficients'][model] = {entry['scope']: entry for entry in results['combiner_coefficients']
                                       if entry['model'] == 'Margin + TTA + anatomy'}
        data['shift'][model] = shift
        counts = table.groupby('dataset')['row'].nunique()
        print(f"Loaded {model}: {len(table):,} decisions on {int(counts.sum()):,} radiographs from {len(counts)} sites.")
    data['images'] = np.load('models/X_test.npy', mmap_mode='r')
    data['anatomy'] = np.load('models/anatomy_nih.npy', mmap_mode='r')
    data['boxes'] = pd.read_csv('models/gt_boxes_nih.csv')
    data['size'] = data['images'].shape[1]
    print("\nReproduction of the notebook failure-detection results")
    for model in MODELS:
        reported = pd.DataFrame(data['results'][model]['failure_detection'])
        reported = reported[(reported['metric'] == 'AUROC_f') & reported['score'].isin(SCORES)]
        gaps = []
        for row in reported.itertuples():
            part = data['frames'][model][row.scope]
            part = part[part['partition'] == row.partition]
            gaps.append(abs(roc_auc_score(part['correct'], part[row.score]) - row.value))
        print(f"  {model}: {len(gaps)} AUROC_f values recomputed, largest difference {max(gaps):.1e}")
    return data

def site_refit(data):
    print("\nSite-refitted combiners (grouped 5-fold cross-validation inside each site)")
    rows = []
    for model in MODELS:
        for scope in SCOPES:
            frame = data['frames'][model][scope]
            for partition in PARTITIONS:
                part = frame[frame['partition'] == partition].reset_index(drop=True)
                correct = part['correct'].to_numpy()
                if (1 - correct).sum() < MIN_EVENTS or correct.sum() < MIN_EVENTS:
                    print(f"  {model}, {scope.lower()}, {partition}: too few failures to evaluate.")
                    continue
                splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
                out_of_fold = {label: np.zeros(len(part)) for label, _ in COMBINERS}
                for train, test in splitter.split(part, correct, part['patient']):
                    for label, features in COMBINERS:
                        scaler = StandardScaler().fit(part.loc[train, features])
                        fitted = LogisticRegression(max_iter=1000).fit(scaler.transform(part.loc[train, features]), correct[train])
                        out_of_fold[label][test] = fitted.predict_proba(scaler.transform(part.loc[test, features]))[:, 1]
                features = COMBINERS[1][1]
                scaler = StandardScaler().fit(part[features])
                full = LogisticRegression(max_iter=1000).fit(scaler.transform(part[features]), correct)
                base, extended = out_of_fold['Margin + TTA'], out_of_fold['Margin + TTA + anatomy']
                nih_base, nih_extended = part['Margin + TTA'].to_numpy(), part['Margin + TTA + anatomy'].to_numpy()
                statistics = {
                    'auroc_base': (auc_at, {'correct': correct, 'values': base}),
                    'auroc_extended': (auc_at, {'correct': correct, 'values': extended}),
                    'delta_auroc': (delta_auc_at, {'correct': correct, 'full': extended, 'base': base}),
                    'delta_aurc': (delta_aurc_at, {'correct': correct, 'full': extended, 'base': base}),
                    'delta_auroc_nih_fitted': (delta_auc_at, {'correct': correct, 'full': nih_extended, 'base': nih_base}),
                }
                ci = bootstrap(statistics, patient_groups(part['patient']))
                everything = np.arange(len(part))
                entry = {'model': model, 'scope': scope, 'partition': partition, 'n': len(part), 'failures': int((1 - correct).sum())}
                for key, (statistic, arguments) in statistics.items():
                    entry[key] = float(statistic(everything, **arguments))
                    entry[f'{key}_low'], entry[f'{key}_high'] = float(ci[key][0]), float(ci[key][1])
                entry.update({'coef_margin': float(full.coef_[0][0]), 'coef_tta': float(full.coef_[0][1]),
                              'coef_enrichment': float(full.coef_[0][2])})
                rows.append(entry)
                print(f"  {model}, {scope.lower()}, {partition} (n={len(part):,}, failures={entry['failures']:,}): "
                      f"margin + TTA {entry['auroc_base']:.4f}, with anatomy {entry['auroc_extended']:.4f}; "
                      f"delta AUROC_f {fmt(entry['delta_auroc'], entry['delta_auroc_low'], entry['delta_auroc_high'], signed=True)}; "
                      f"delta AURC {fmt(entry['delta_aurc'], entry['delta_aurc_low'], entry['delta_aurc_high'], signed=True)}; "
                      f"enrichment coefficient {entry['coef_enrichment']:+.3f}")
    return pd.DataFrame(rows)

def per_finding_failure(data):
    print("\nFailure detection by finding (flagged findings, at least 20 correct and 20 wrong flags)")
    rows = []
    for model in MODELS:
        frame = data['frames'][model]['Flagged findings']
        for partition in PARTITIONS:
            for finding in FINDINGS:
                part = frame[(frame['partition'] == partition) & (frame['finding'] == finding)]
                correct = part['correct'].to_numpy()
                failures = int((1 - correct).sum())
                if failures < MIN_EVENTS or correct.sum() < MIN_EVENTS:
                    continue
                statistics = {key: (auc_at, {'correct': correct, 'values': part[score].to_numpy(dtype=float)})
                              for key, score in (('margin', 'Confidence margin'), ('enrichment', 'Anatomical enrichment'))}
                ci = bootstrap(statistics, patient_groups(part['patient']))
                everything = np.arange(len(part))
                entry = {'model': model, 'partition': partition, 'finding': finding, 'n': len(part), 'failures': failures}
                for key, (statistic, arguments) in statistics.items():
                    entry[key] = float(statistic(everything, **arguments))
                    entry[f'{key}_low'], entry[f'{key}_high'] = float(ci[key][0]), float(ci[key][1])
                rows.append(entry)
    table = pd.DataFrame(rows)
    for model in MODELS:
        part = table[table['model'] == model]
        above = part[part['enrichment_low'] > 0.5]
        below = part[part['enrichment_high'] < 0.5]
        better = part[part['enrichment'] > part['margin']]
        print(f"  {model}: {len(part)} finding-site cells. Enrichment above chance in {len(above)}, below chance in "
              f"{len(below)}, higher than the margin in {len(better)}.")
        for label, cells in (('Above chance', above), ('Below chance', below)):
            for row in cells.itertuples():
                print(f"    {label}: {row.finding}, {row.partition} (n={row.n}): enrichment "
                      f"{fmt(row.enrichment, row.enrichment_low, row.enrichment_high, 3)}, margin {row.margin:.3f}")
    return table

def redundancy(data):
    print("\nRedundancy of enrichment with the confidence margin (Spearman correlation of E and |m|)")
    rows = []
    for model in MODELS:
        for scope in SCOPES:
            frame = data['frames'][model][scope]
            for partition in PARTITIONS:
                part = frame[frame['partition'] == partition]
                first = part['region_enrichment'].to_numpy(dtype=float)
                second = part['abs_margin'].to_numpy(dtype=float)
                statistics = {'rho': (spearman_at, {'first': first, 'second': second})}
                ci = bootstrap(statistics, patient_groups(part['patient']))
                value = spearman_at(np.arange(len(part)), first, second)
                rows.append({'model': model, 'scope': scope, 'partition': partition, 'n': len(part),
                             'rho': float(value), 'low': float(ci['rho'][0]), 'high': float(ci['rho'][1])})
                print(f"  {model}, {scope.lower()}, {partition}: rho {fmt(value, ci['rho'][0], ci['rho'][1], 3, True)}")
    return pd.DataFrame(rows)

def flag_risk(data):
    print(f"\nFalse-flag rate before and after keeping the {int(COVERAGE * 100)}% highest-scoring flags")
    rows = []
    kept = ('Confidence margin', 'Margin + TTA', 'Margin + TTA + anatomy', 'Anatomical enrichment')
    for model in MODELS:
        frame = data['frames'][model]['Flagged findings']
        for partition in PARTITIONS:
            part = frame[frame['partition'] == partition]
            correct = part['correct'].to_numpy()
            statistics = {'All flags': (base_risk_at, {'correct': correct})}
            for score in kept:
                statistics[score] = (risk_at, {'correct': correct, 'values': part[score].to_numpy(dtype=float)})
            ci = bootstrap(statistics, patient_groups(part['patient']))
            everything = np.arange(len(part))
            values = {}
            for key, (statistic, arguments) in statistics.items():
                values[key] = float(statistic(everything, **arguments))
                rows.append({'model': model, 'partition': partition, 'score': key, 'n': len(part),
                             'risk': values[key], 'low': float(ci[key][0]), 'high': float(ci[key][1])})
            print(f"  {model}, {partition}: all flags {values['All flags']:.3f}; kept by margin {values['Confidence margin']:.3f}, "
                  f"margin + TTA {values['Margin + TTA']:.3f}, margin + TTA + anatomy {values['Margin + TTA + anatomy']:.3f}, "
                  f"enrichment alone {values['Anatomical enrichment']:.3f}")
    return pd.DataFrame(rows)

def supplementary(data):
    extra = {'refit': site_refit(data), 'by_finding': per_finding_failure(data), 'redundancy': redundancy(data),
             'risks': flag_risk(data)}
    extra['refit'].to_csv('models/paper_site_refit.csv', index=False)
    extra['by_finding'].to_csv('models/paper_per_finding_failure.csv', index=False)
    extra['redundancy'].to_csv('models/paper_redundancy.csv', index=False)
    extra['risks'].to_csv('models/paper_selective_risk.csv', index=False)
    holm_tables = {model: table[['external', 'finding', 'difference', 'low', 'high', 'p_value', 'holm']].to_dict('records')
                   for model, table in data['shift'].items()}
    with open('models/paper_extra_results.json', 'w') as handle:
        json.dump({'site_refit': extra['refit'].to_dict('records'), 'per_finding_failure': extra['by_finding'].to_dict('records'),
                   'redundancy': extra['redundancy'].to_dict('records'), 'selective_risk': extra['risks'].to_dict('records'),
                   'enrichment_shift_holm': holm_tables}, handle, indent=2, default=float)
    return extra

def upsample(cam, size):
    return cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)

def enrichment(attribution, layers):
    region = (layers & 1).astype(bool)
    positive = np.maximum(attribution, 0.0)
    if region.sum() == 0:
        return np.nan
    if positive.sum() <= 0:
        return 1.0
    return float(positive[region].sum() / positive.sum() / region.mean())

def region_share(attribution, layers):
    region = (layers & 1).astype(bool)
    positive = np.maximum(attribution, 0.0)
    return float(positive[region].sum() / max(positive.sum(), 1e-12)), float(region.mean())

def choose_examples(data):
    boxes = data['boxes']
    keyed = set(zip(boxes['row'], boxes['finding']))
    flags = data['frames']['DenseNet-121']['Flagged findings']
    flags = flags[(flags['dataset'] == 'NIH') & (flags['correct'] == 1)]
    pool = flags[[(r, f) in keyed for r, f in zip(flags['row'], flags['finding'])]]
    pool = pool if len(pool) else flags
    pick = pool.iloc[int((pool['region_enrichment'] - flags['region_enrichment'].median()).abs().to_numpy().argmin())]
    examples = [{'figure': 'Fig1', 'model': 'DenseNet-121', 'row': int(pick['row']), 'driver': int(pick['driver']),
                 'finding': pick['finding'], 'correct': 1, 'saved': {}}]
    for model in MODELS:
        part = data['cross'][model][data['cross'][model]['dataset'] == 'NIH']
        for outcome in (1, 0):
            group = part[part['correct'] == outcome]
            if len(group) == 0:
                continue
            chosen = group.iloc[int((group['Grad-CAM'] - group['Grad-CAM'].median()).abs().to_numpy().argmin())]
            examples.append({'figure': 'ED_Fig1', 'model': model, 'row': int(chosen['row']), 'driver': int(chosen['driver']),
                             'finding': chosen['finding'], 'correct': outcome,
                             'saved': {method: float(chosen[method]) for method in ('Grad-CAM', 'LIME', 'SHAP')}})
    return examples

def explain_examples(data, examples):
    images, size = data['images'], data['size']
    nih_count = len(images)
    background_rows = np.sort(np.random.default_rng(SEED + 1).choice(nih_count, min(SHAP_BACKGROUND, nih_count), replace=False))
    background = np.asarray(images[background_rows], dtype=np.float32)
    for model, (path, _) in MODELS.items():
        batch_size = 8 if model == 'ViT-B/16' else 16
        if model == 'ViT-B/16':
            tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        classifier = tf.keras.models.load_model(path, compile=False)

        def predict_rgb(batch, classifier=classifier, batch_size=batch_size):
            return classifier.predict(np.ascontiguousarray(batch[..., :1], dtype=np.float32), batch_size=batch_size, verbose=0)

        def select_class(z, c):
            return z[:, c:c + 1]

        for example in [e for e in examples if e['model'] == model]:
            row, driver = example['row'], example['driver']
            image = np.asarray(images[row], dtype=np.float32)
            layers = np.asarray(data['anatomy'][row])
            explainer = lime_image.LimeImageExplainer(random_state=SEED)
            explanation = explainer.explain_instance(np.repeat(image.astype(np.uint8), 3, axis=-1), predict_rgb, labels=[driver],
                                                     top_labels=None, hide_color=0, num_samples=LIME_SAMPLES, batch_size=64,
                                                     random_seed=SEED)
            weights = np.zeros(explanation.segments.max() + 1, dtype=np.float32)
            for segment, weight in explanation.local_exp.get(driver, []):
                weights[segment] = weight
            selector = tf.keras.layers.Lambda(select_class, arguments={'c': driver}, name=f'select_{driver}')
            head = tf.keras.Model([classifier.input], selector(classifier.output))
            values = shap.GradientExplainer(head, background, batch_size=batch_size).shap_values(
                image[None].astype(np.float32), nsamples=SHAP_SAMPLES, rseed=SEED)
            example['maps'] = {'Grad-CAM': upsample(data['cams'][model][row, :, :, driver], size),
                               'LIME': weights[explanation.segments],
                               'SHAP': np.asarray(values, dtype=np.float32).reshape(size, size)}
            example['enrichment'] = {method: enrichment(example['maps'][method], layers) for method in example['maps']}
            example['share'] = {method: region_share(example['maps'][method], layers) for method in example['maps']}
        tf.keras.backend.clear_session()
    return examples

def measure(strings, size=TICK, weight='normal'):
    probe = plt.figure(figsize=(3, 3))
    renderer = probe.canvas.get_renderer()
    width = 0.0
    for s in strings:
        text = probe.text(0, 0, s, fontsize=size, fontweight=weight)
        width = max(width, text.get_window_extent(renderer).width / probe.dpi * 25.4)
    plt.close(probe)
    return width

class Canvas:
    def __init__(self, height, width=WIDTH):
        self.width, self.height = width, height
        self.fig = plt.figure(figsize=(width * MM, height * MM))

    def axes(self, left, top, width, height, **kwargs):
        return self.fig.add_axes([left / self.width, 1 - (top + height) / self.height, width / self.width,
                                  height / self.height], **kwargs)

    def text(self, x, y, s, **kwargs):
        kwargs.setdefault('ha', 'left')
        kwargs.setdefault('va', 'baseline')
        return self.fig.text(x / self.width, 1 - y / self.height, s, **kwargs)

    def letter(self, x, y, letter):
        self.text(x, y, letter, fontsize=8, fontweight='bold', color=INK)

    def legend(self, handles, centre, top, ncol=None):
        self.fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(centre / self.width, 1 - top / self.height),
                        ncol=ncol or len(handles), frameon=False, borderaxespad=0, borderpad=0)

    def save(self, name):
        name = ED_NUMBERS.get(name, name)
        os.makedirs(OUT, exist_ok=True)
        self.fig.savefig(os.path.join(OUT, f'{name}.pdf'))
        self.fig.savefig(os.path.join(OUT, f'{name}.png'))
        if name.startswith('ED_'):
            tiff = os.path.join(OUT, f'{name}.tif')
            with Image.open(os.path.join(OUT, f'{name}.png')) as image:
                image.convert('RGB').save(tiff, compression='tiff_lzw', dpi=(300, 300))
            if os.path.getsize(tiff) > 10e6:
                with Image.open(os.path.join(OUT, f'{name}.png')) as image:
                    image.convert('RGB').save(os.path.join(OUT, f'{name}.jpg'), quality=95, dpi=(300, 300))
                print(f"{name}.tif is over 10 MB, so {name}.jpg was saved as well; upload the JPEG for this figure.")
        plt.close(self.fig)

def title(ax, text, bold=True):
    ax.set_title(text, fontsize=7.5 if bold else LABEL, fontweight='bold' if bold else 'normal', color=INK, pad=4.0)

def recessive_grid(ax, axis='x'):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)

def dot(ax, x, y, low, high, colour, marker='o', filled=True, size=3.2, horizontal=True):
    if np.isfinite(low) and np.isfinite(high):
        if horizontal:
            ax.errorbar(x, y, xerr=[[max(x - low, 0)], [max(high - x, 0)]], fmt='none', ecolor=colour, elinewidth=0.8)
        else:
            ax.errorbar(x, y, yerr=[[max(y - low, 0)], [max(high - y, 0)]], fmt='none', ecolor=colour, elinewidth=0.8)
    ax.plot(x, y, marker=marker, markersize=size, markerfacecolor=colour if filled else 'white', markeredgecolor=colour,
            markeredgewidth=0.8, linestyle='none', zorder=3)

def handle(label, colour=SECONDARY, marker='o', filled=True):
    return Line2D([], [], marker=marker, linestyle='none', markersize=4, markerfacecolor=colour if filled else 'white',
                  markeredgecolor=colour, markeredgewidth=0.8, label=label)

def line_handle(label, colour, style='-', width=1.1):
    return Line2D([], [], color=colour, linestyle=style, linewidth=width, label=label)

def site_handles(sites=SITES):
    return [handle(site, COLOURS[site]) for site in sites]

def model_handles():
    return [handle(model, SECONDARY, MARKERS[model]) for model in MODELS]

def rows_axis(ax, labels, separators=(), show=True):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels if show else [])
    ax.set_ylim(len(labels) - 0.5, -0.6)
    for position in separators:
        ax.axhline(position, color=AXIS, linewidth=0.5)
    ax.tick_params(axis='y', length=0)

def value_limits(values, floor, ceiling, pad=0.02):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return floor, ceiling
    return min(floor, values.min() - pad), max(ceiling, values.max() + pad)

def outline(ax, mask, colour, width=0.8, style='-'):
    contours, _ = cv2.findContours(np.ascontiguousarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        closed = np.vstack([contour[:, 0, :], contour[:1, 0, :]])
        ax.plot(closed[:, 0], closed[:, 1], color=colour, linewidth=width, linestyle=style)

def draw_boxes(ax, boxes, row, finding):
    found = boxes[(boxes['row'] == row) & (boxes['finding'] == finding)]
    for box in found.itertuples():
        ax.add_patch(Rectangle((box.x0, box.y0), box.x1 - box.x0, box.y1 - box.y0, fill=False, edgecolor=BOX, linewidth=0.9))
    return len(found)

def show_image(ax, image):
    ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='bilinear')
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.axis('off')

def overlay(ax, image, attribution, smooth=0.0):
    show_image(ax, image)
    positive = np.maximum(np.asarray(attribution, dtype=np.float32), 0.0)
    if smooth > 0:
        positive = gaussian_filter(positive, smooth)
    low, high = np.percentile(positive, [1, 99.5])
    if high > low:
        level = np.clip((positive - low) / (high - low), 0, 1)
        colours = plt.get_cmap('inferno')(level)
        colours[..., 3] = 0.8 * level
        ax.imshow(colours, interpolation='bilinear')

def result_frame(data, model, key):
    return pd.DataFrame(data['results'][model][key])

def hit_rate(source, group, finding):
    match = source[(source['group'] == group) & (source['finding'] == finding) & (source['metric'] == 'Hit rate')]
    return None if len(match) == 0 else match.iloc[0]

def score_limits(data, scope):
    bounds = []
    for model in MODELS:
        failure = result_frame(data, model, 'failure_detection')
        failure = failure[(failure['scope'] == scope) & (failure['metric'] == 'AUROC_f') & failure['score'].isin(SCORES)]
        bounds += failure['low'].tolist() + failure['high'].tolist()
    return value_limits(bounds, 0.45, 0.75)

def score_panel(ax, data, model, scope, show_labels=True):
    failure = result_frame(data, model, 'failure_detection')
    failure = failure[(failure['scope'] == scope) & (failure['metric'] == 'AUROC_f')]
    for position, score in enumerate(SCORES):
        for partition in PARTITIONS:
            match = failure[(failure['partition'] == partition) & (failure['score'] == score)]
            if len(match):
                entry = match.iloc[0]
                dot(ax, position + OFFSETS[partition], entry['value'], entry['low'], entry['high'], COLOURS[SITE_OF[partition]],
                    MARKERS[model], horizontal=False)
    ax.axhline(0.5, color=INK, linewidth=0.6, linestyle=':')
    ax.set_xticks(range(len(SCORES)))
    ax.set_xticklabels([TICKS[s] for s in SCORES])
    ax.set_xlim(-0.6, len(SCORES) - 0.4)
    ax.set_ylim(*score_limits(data, scope))
    if not show_labels:
        ax.tick_params(axis='y', labelleft=False)
    recessive_grid(ax, 'y')

def delta_panel(ax, extra, scope):
    labels, position = [], 0
    for model in MODELS:
        part = extra['refit'][(extra['refit']['model'] == model) & (extra['refit']['scope'] == scope)]
        for partition in PARTITIONS:
            entry = part[part['partition'] == partition]
            if len(entry) == 0:
                continue
            entry = entry.iloc[0]
            colour = COLOURS[SITE_OF[partition]]
            dot(ax, entry['delta_auroc_nih_fitted'], position - 0.16, entry['delta_auroc_nih_fitted_low'],
                entry['delta_auroc_nih_fitted_high'], colour, MARKERS[model], True)
            dot(ax, entry['delta_auroc'], position + 0.16, entry['delta_auroc_low'], entry['delta_auroc_high'], colour, MARKERS[model], False)
            labels.append(f"{SITE_OF[partition]}, {model}")
            position += 1
    ax.axvline(0, color=INK, linewidth=0.6)
    rows_axis(ax, labels, [2.5])
    recessive_grid(ax)
    return labels

def site_model_labels():
    return [f"{site}, {model}" for model in MODELS for site in SITES]

def legend_above(ax, handles, ncol=None, heading=None):
    ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=ncol or len(handles), borderaxespad=0.5,
              title=heading, title_fontsize=TICK)

def column_layout(label_widths, gap, right=4.0, total=WIDTH):
    columns = len(label_widths)
    width = (total - right - sum(label_widths) - gap * (columns - 1)) / columns
    lefts, x = [], 0.0
    for i, label_width in enumerate(label_widths):
        x += label_width
        lefts.append(x)
        x += width + gap
    return lefts, width

def figure_1(data, example):
    gap, margin = 8.0, 1.0
    column = (WIDTH - 2 * margin - 3 * gap) / 4
    xs = [margin + i * (column + gap) for i in range(4)]
    top, height = 5.0, 60.0
    image_top = top + height + 10.0
    canvas = Canvas(image_top + column + 12.5)
    ax = canvas.axes(0, top, WIDTH, height)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(height, 0)
    ax.axis('off')

    renderer = canvas.fig.canvas.get_renderer()

    def box(x, y, h, heading, lines, edge=AXIS, heading_colour=INK, formula=None):
        ax.add_patch(FancyBboxPatch((x, y), column, h, boxstyle='round,pad=0,rounding_size=1.8', linewidth=0.8,
                                    edgecolor=edge, facecolor='#fbfbfa'))
        texts = [ax.text(x + column / 2, y, heading, ha='center', va='top', fontsize=6.2, fontweight='bold', color=heading_colour)]
        for i, line in enumerate(lines):
            texts.append(ax.text(x + column / 2, y + 4.6 + 3.3 * i, line, ha='center', va='top', fontsize=5.6, color=SECONDARY))
        if formula:
            texts.append(ax.text(x + column / 2, y + 4.6 + 3.3 * len(lines) + 0.6, formula, ha='center', va='top', fontsize=5.9, color=INK))
        extents = [ax.transData.inverted().transform(t.get_window_extent(renderer).get_points())[:, 1] for t in texts]
        shift = y + h / 2 - (min(e.min() for e in extents) + max(e.max() for e in extents)) / 2
        for t in texts:
            t.set_y(t.get_position()[1] + shift)

    def arrow(start, end):
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='-|>', color=MUTED, linewidth=0.7,
                                                              shrinkA=0, shrinkB=0, mutation_scale=6))

    table = data['decisions']['DenseNet-121']
    counts = table.groupby('dataset')['row'].nunique()
    patients = table.groupby('dataset')['patient'].nunique()
    tall, short, tall_gap, short_gap = 27.0, 18.0, 6.0, 3.0
    box(xs[0], 0, tall, 'Development', ['NIH ChestX-ray14 training split', '90,286 radiographs, 24,644 patients',
                                         'Labels text-mined from reports', '15% of patients for validation'], COLOURS['NIH'])
    box(xs[0], tall + tall_gap, tall, 'Two classifiers', ['DenseNet-121 (CNN)', 'ViT-B/16 (transformer)',
                                                            'Weighted cross-entropy', 'Thresholds $t_c$ tuned on NIH'])
    sites = [('NIH internal test', COLOURS['NIH'], [f"{counts.get('NIH', 0):,} radiographs, {patients.get('NIH', 0):,} patients",
                                                    'Calibration and evaluation halves']),
             ('CheXpert (Stanford, USA)', COLOURS['CheXpert'],
              [f"{counts.get('CheXpert', 0):,} radiographs, {patients.get('CheXpert', 0):,} patients", 'Radiologist labels and outlines']),
             ('VinDr-CXR (Hanoi, Vietnam)', COLOURS['VinDr-CXR'], [f"{counts.get('VinDr-CXR', 0):,} radiographs",
                                                                  'Five-radiologist consensus labels'])]
    site_mid = []
    for i, (heading, colour, lines) in enumerate(sites):
        y = i * (short + short_gap)
        box(xs[1], y, short, heading, lines, colour, colour)
        site_mid.append(y + short / 2)
    box(xs[2], 0, tall, 'Signals for every decision', ['$m=\\mathrm{logit}\\,p-\\mathrm{logit}\\,t_c$', 'Flag the finding if $m>0$',
                                                        'TTA gap $g=|p_{\\mathrm{up}}-p_{\\mathrm{flip}}|$'])
    box(xs[2], tall + tall_gap, tall, 'Explanation plausibility', ['Grad-CAM on all decisions', 'LIME and SHAP on subsets',
                                                                     'Thoracic region $R$ (HybridGNet)'],
        formula='$E=\\dfrac{\\mathrm{share\\ of\\ attribution\\ in\\ }R}{\\mathrm{share\\ of\\ image\\ in\\ }R}$')
    analyses = [('Localization', ['Pointing game against', 'radiologist annotations']),
                ('Explanation shift (H1)', ['Change in $E$ between', 'hospitals, by finding']),
                ('Failure detection (H2, H3)', ['$E$ alone, and added to', '$|m|$ and $g$'])]
    analysis_mid = []
    for i, (heading, lines) in enumerate(analyses):
        y = i * (short + short_gap)
        box(xs[3], y, short, heading, lines)
        analysis_mid.append(y + short / 2)
    arrow((xs[0] + column / 2, tall), (xs[0] + column / 2, tall + tall_gap))
    classifier_mid = tall + tall_gap + tall / 2
    for y in site_mid:
        arrow((xs[0] + column, classifier_mid), (xs[1], y))
    bus = xs[1] + column + gap / 2
    ax.plot([bus, bus], [site_mid[0], site_mid[-1]], color=MUTED, linewidth=0.7)
    for y in site_mid:
        ax.plot([xs[1] + column, bus], [y, y], color=MUTED, linewidth=0.7)
    for y in (tall / 2, classifier_mid):
        arrow((bus, y), (xs[2], y))
    bus = xs[2] + column + gap / 2
    ax.plot([bus, bus], [min(tall / 2, analysis_mid[0]), max(classifier_mid, analysis_mid[-1])], color=MUTED, linewidth=0.7)
    for y in (tall / 2, classifier_mid):
        ax.plot([xs[2] + column, bus], [y, y], color=MUTED, linewidth=0.7)
    for y in analysis_mid:
        arrow((bus, y), (xs[3], y))
    canvas.letter(margin, top - 1.2, 'a')

    row = example['row']
    image = np.asarray(data['images'][row])[:, :, 0]
    layers = np.asarray(data['anatomy'][row])
    outcome = 'correct flag' if example['correct'] == 1 else 'wrong flag'
    area = example['share']['Grad-CAM'][1]
    panels = [('Radiograph', 'NIH test radiograph')] + [(method, f"DenseNet-121 {method}") for method in ('Grad-CAM', 'LIME', 'SHAP')]
    for i, (key, heading) in enumerate(panels):
        axi = canvas.axes(xs[i], image_top, column, column)
        if key == 'Radiograph':
            show_image(axi, image)
            outline(axi, layers & 1, REGION, 0.9)
            outline(axi, (layers >> 1) & 1, LUNG, 0.7, (0, (2, 1.2)))
            draw_boxes(axi, data['boxes'], row, example['finding'])
            captions = [f"{example['finding']}, {outcome}", f"Cyan: thoracic region $R$ ({area:.0%} of image)",
                        'Dashed green: lungs; yellow: radiologist box']
        else:
            overlay(axi, image, example['maps'][key], smooth=2.0 if key == 'SHAP' else 0.0)
            outline(axi, layers & 1, REGION, 0.8)
            share, area = example['share'][key]
            captions = [f"Attribution in $R$: {share:.0%}", f"$E$ = {share:.2f} / {area:.2f} = {example['enrichment'][key]:.2f}"]
        canvas.text(xs[i] + column / 2, image_top - 1.4, heading, ha='center', fontsize=6.2, color=INK)
        for k, line in enumerate(captions):
            canvas.text(xs[i] + column / 2, image_top + column + 3.6 + 3.2 * k, line, ha='center', fontsize=5.6,
                        color=SECONDARY if key == 'Radiograph' and k > 0 else INK)
        canvas.letter(xs[i], image_top - 1.4, 'bcde'[i])
    canvas.save('Fig1')

def figure_2(data):
    labels = {'AUROC': FINDINGS + ['Macro average, CheXpert findings', 'Macro average, VinDr-CXR findings'],
              'Sensitivity': FINDINGS}
    label_width = measure(labels['AUROC']) + 3.0
    (left_a, left_b), width = column_layout([label_width, 0.0], gap=7.0)
    row = 4.0
    tops = [14.0, 14.0 + 14 * row + 17.0]
    canvas = Canvas(tops[1] + 12 * row + 11.0)
    for column, (model, left) in enumerate(zip(MODELS, (left_a, left_b))):
        transfer = result_frame(data, model, 'classification_transfer')
        for r, metric in enumerate(('AUROC', 'Sensitivity')):
            ax = canvas.axes(left, tops[r], width, len(labels[metric]) * row)
            for position, label in enumerate(labels[metric]):
                for site in SITES:
                    if label.startswith('Macro'):
                        shared = 'CheXpert' if 'CheXpert' in label else 'VinDr-CXR'
                        if site == 'NIH':
                            key = ('NIH', f'Macro average, shared with {shared}')
                        elif site == shared:
                            key = (site, 'Macro average')
                        else:
                            continue
                    else:
                        key = (site, label)
                    match = transfer[(transfer['dataset'] == key[0]) & (transfer['finding'] == key[1]) & (transfer['metric'] == metric)]
                    if len(match):
                        entry = match.iloc[0]
                        dot(ax, entry['value'], position + OFFSETS[site], entry['low'], entry['high'], COLOURS[site])
            rows_axis(ax, labels[metric], [len(FINDINGS) - 0.5] if metric == 'AUROC' else [], show=column == 0)
            if metric == 'AUROC':
                ax.set_xlim(0.5, 1.0)
                ax.set_xlabel('AUROC (95% CI)')
                title(ax, model)
            else:
                ax.set_xlim(-0.02, 1.0)
                ax.set_xlabel('Sensitivity at the NIH-tuned threshold (95% CI)')
            recessive_grid(ax)
    for r, letter in enumerate('ab'):
        canvas.letter(1.0, tops[r] - 1.4, letter)
    canvas.legend(site_handles(), (left_a + left_b + width) / 2, 1.5)
    canvas.save('Fig2')

def figure_3(data):
    order = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Lesion', 'Pneumothorax', 'All findings']
    sources = {model: result_frame(data, model, 'localisation') for model in MODELS}
    labels_a = []
    for finding in order:
        human = hit_rate(sources['DenseNet-121'], 'CheXpert human benchmark', finding)
        labels_a.append(finding if human is None else f"{finding} ($n$ = {int(human['n'])})")
    labels_b = site_model_labels()
    (left_a, left_b), width = column_layout([measure(labels_a) + 3.0, measure(labels_b) + 3.0], gap=8.0)
    top, height = 6.0, 52.0
    canvas = Canvas(top + height + 17.0)
    ax = canvas.axes(left_a, top, width, height)
    for position, finding in enumerate(order):
        series = [(sources['DenseNet-121'], 'CheXpert Grad-CAM, benchmarked images', -0.2, COLOURS['CheXpert'], 'o', True),
                  (sources['ViT-B/16'], 'CheXpert Grad-CAM, benchmarked images', 0.0, COLOURS['CheXpert'], 's', False),
                  (sources['DenseNet-121'], 'CheXpert human benchmark', 0.2, INK, 'D', True)]
        for source, group, offset, colour, marker, filled in series:
            entry = hit_rate(source, group, finding)
            if entry is not None:
                dot(ax, entry['value'], position + offset, entry['low'], entry['high'], colour, marker, filled)
    rows_axis(ax, labels_a, [len(order) - 1.5])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel('Pointing-game hit rate, CheXpert benchmark set (95% CI)')
    recessive_grid(ax)
    canvas.legend([handle('DenseNet-121 Grad-CAM', COLOURS['CheXpert'], 'o', True),
                   handle('ViT-B/16 Grad-CAM', COLOURS['CheXpert'], 's', False), handle('Radiologists', INK, 'D', True)],
                  left_a + width / 2, top + height + 11.0)
    ax = canvas.axes(left_b, top, width, height)
    position = 0
    for model in MODELS:
        for site in SITES:
            for flagged, offset in ((False, -0.15), (True, 0.15)):
                entry = hit_rate(sources[model], f"{site} Grad-CAM" + (', flagged only' if flagged else ''), 'All findings')
                if entry is not None:
                    dot(ax, entry['value'], position + offset, entry['low'], entry['high'], COLOURS[site], MARKERS[model], not flagged)
            position += 1
    rows_axis(ax, labels_b, [2.5])
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel('Pointing-game hit rate, all findings (95% CI)')
    recessive_grid(ax)
    canvas.legend([handle('All annotated findings'), handle('Flagged findings only', filled=False)],
                  left_b + width / 2, top + height + 11.0)
    canvas.letter(1.0, top - 1.4, 'a')
    canvas.letter(left_a + width + 8.0, top - 1.4, 'b')
    canvas.save('Fig3')

def figure_4(data):
    findings = set()
    for model in MODELS:
        findings |= set(data['shift'][model]['finding'])
    labels = ['All findings'] + [f for f in FINDINGS if f in findings]
    label_width = max(measure(labels) + 3.0, 13.0)
    (left_a, left_b), width = column_layout([label_width, 0.0], gap=6.0)
    top_a, height_a = 14.0, 40.0
    top_b, row = top_a + height_a + 15.0, 4.4
    height_b = len(labels) * row
    canvas = Canvas(top_b + height_b + 17.0)
    top_axes, bottom_axes = [], []
    for column, (model, left) in enumerate(zip(MODELS, (left_a, left_b))):
        ax = canvas.axes(left, top_a, width, height_a)
        top_axes.append(ax)
        flags = data['decisions'][model]
        flags = flags[(flags['decision'] == 1) & flags['region_enrichment'].notna()]
        values, positions, faces, edges = [], [], [], []
        for i, site in enumerate(SITES):
            for j, outcome in enumerate((1, 0)):
                part = flags[(flags['dataset'] == site) & (flags['correct'] == outcome)]['region_enrichment'].to_numpy()
                if len(part):
                    values.append(part)
                    positions.append(3 * i + j)
                    faces.append(COLOURS[site] if outcome == 1 else 'white')
                    edges.append(COLOURS[site])
        boxes = ax.boxplot(values, positions=positions, widths=0.72, patch_artist=True, showfliers=False,
                           medianprops=dict(color=INK, linewidth=0.9), whiskerprops=dict(linewidth=0.6),
                           capprops=dict(linewidth=0.6), boxprops=dict(linewidth=0.7))
        for patch, face, edge in zip(boxes['boxes'], faces, edges):
            patch.set_facecolor(face)
            patch.set_edgecolor(edge)
        for part in ('whiskers', 'caps'):
            for k, line in enumerate(boxes[part]):
                line.set_color(edges[k // 2])
        ax.axhline(1.0, color=INK, linestyle=':', linewidth=0.7)
        ax.set_xticks([3 * i + 0.5 for i in range(len(SITES))])
        ax.set_xticklabels(SITES)
        ax.set_xlim(-0.8, 3 * len(SITES) - 1.2)
        if column == 0:
            ax.set_ylabel('Enrichment $E$ of flagged findings')
        title(ax, model)
        recessive_grid(ax, 'y')
        ax = canvas.axes(left, top_b, width, height_b)
        bottom_axes.append(ax)
        shift = data['shift'][model]
        for position, finding in enumerate(labels):
            for site, offset in (('CheXpert', -0.17), ('VinDr-CXR', 0.17)):
                match = shift[(shift['external'] == site) & (shift['finding'] == finding)]
                if len(match):
                    entry = match.iloc[0]
                    significant = entry['p_value'] < 0.05 if finding == 'All findings' else entry['holm'] < 0.05
                    dot(ax, entry['difference'], position + offset, entry['low'], entry['high'], COLOURS[site], 'o', bool(significant))
        ax.axvline(0, color=INK, linewidth=0.6)
        rows_axis(ax, labels, [0.5], show=column == 0)
        ax.set_xlabel('Change in median $E$ relative to NIH (95% CI)')
        recessive_grid(ax)
    low = min(ax.get_ylim()[0] for ax in top_axes)
    high = max(ax.get_ylim()[1] for ax in top_axes)
    for ax in top_axes:
        ax.set_ylim(low, high)
    top_axes[1].tick_params(axis='y', labelleft=False)
    low = min(ax.get_xlim()[0] for ax in bottom_axes)
    high = max(ax.get_xlim()[1] for ax in bottom_axes)
    for ax in bottom_axes:
        ax.set_xlim(low, high)
    centre = (left_a + left_b + width) / 2
    canvas.legend(site_handles() + [handle('Correct flags', SECONDARY, 's'), handle('Wrong flags', SECONDARY, 's', False)], centre, 1.5)
    canvas.legend([handle('Holm-adjusted $P$ < 0.05'), handle('Holm-adjusted $P$ ≥ 0.05', filled=False)], centre, top_b + height_b + 11.0)
    canvas.letter(1.0, top_a - 1.4, 'a')
    canvas.letter(1.0, top_b - 1.4, 'b')
    canvas.save('Fig4')

def figure_5(data, extra):
    forest_labels = measure(site_model_labels()) + 3.0
    numeric_labels = measure(['0.65']) + 7.5
    gap = 7.0
    (left_a, left_b), width = column_layout([forest_labels, numeric_labels], gap=gap)
    height = 40.0
    tops = [14.0, 14.0 + height + 19.0, 14.0 + 2 * (height + 19.0)]
    canvas = Canvas(tops[2] + height + 12.0)
    for column, (model, left) in enumerate(zip(MODELS, (left_a, left_b))):
        ax = canvas.axes(left, tops[0], width, height)
        score_panel(ax, data, model, 'Flagged findings')
        ax.set_ylabel(f'{AUC_F}, flagged findings')
        title(ax, model)
    ax = canvas.axes(left_a, tops[1], width, height)
    delta_panel(ax, extra, 'Flagged findings')
    ax.set_xlabel(f'Change in {AUC_F} from adding $E$ (95% CI)')
    legend_above(ax, [handle('Fitted on NIH'), handle('Refitted within the site', filled=False)])
    ax = canvas.axes(left_b, tops[1], width, height)
    methods = ['Grad-CAM', 'LIME', 'SHAP']
    bounds = []
    for model, nudge in (('DenseNet-121', -0.06), ('ViT-B/16', 0.06)):
        summary = result_frame(data, model, 'cross_method')
        for position, method in enumerate(methods):
            for site in SITES:
                match = summary[(summary['dataset'] == site) & (summary['method'] == method)]
                if len(match):
                    entry = match.iloc[0]
                    dot(ax, position + OFFSETS[site] + nudge, entry['auroc_f'], entry['low'], entry['high'], COLOURS[site],
                        MARKERS[model], horizontal=False)
                    bounds += [entry['low'], entry['high']]
    ax.axhline(0.5, color=INK, linewidth=0.6, linestyle=':')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([f'$E$ from {m}' for m in methods])
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.set_ylim(*value_limits(bounds, 0.4, 0.65))
    ax.set_ylabel(f'{AUC_F} of $E$ alone')
    recessive_grid(ax, 'y')
    ax = canvas.axes(left_a, tops[2], width, height)
    labels, position, values = [], 0, []
    for model in MODELS:
        part = extra['refit'][(extra['refit']['model'] == model) & (extra['refit']['scope'] == 'Flagged findings')]
        reference = data['coefficients'][model]['Flagged findings']['region_enrichment']
        for partition in PARTITIONS:
            entry = part[part['partition'] == partition]
            if len(entry) == 0:
                continue
            entry = entry.iloc[0]
            ax.plot([reference, entry['coef_enrichment']], [position, position], color=GRID, linewidth=1.0, zorder=1)
            ax.plot(reference, position, marker='|', markersize=7, markeredgewidth=1.0, color=INK, linestyle='none', zorder=2)
            ax.plot(entry['coef_enrichment'], position, marker=MARKERS[model], markersize=4, color=COLOURS[SITE_OF[partition]],
                    linestyle='none', zorder=3)
            values += [reference, entry['coef_enrichment']]
            labels.append(f"{SITE_OF[partition]}, {model}")
            position += 1
    ax.axvline(0, color=AXIS, linewidth=0.6)
    rows_axis(ax, labels, [2.5])
    span = max(abs(min(values)), abs(max(values))) * 1.15 if values else 0.3
    ax.set_xlim(-span, span)
    ax.set_xlabel('Standardized weight of $E$ in the combination')
    recessive_grid(ax)
    legend_above(ax, [Line2D([], [], marker='|', linestyle='none', markersize=7, markeredgewidth=1.0, color=INK, label='Fitted on NIH'),
                      handle('Refitted within the site')])
    ax = canvas.axes(left_b, tops[2], width, height)
    frame = data['frames']['DenseNet-121']['Flagged findings']
    part = frame[frame['partition'] == 'VinDr-CXR']
    correct = part['correct'].to_numpy()
    colour = COLOURS['VinDr-CXR']
    lines, curves, base = [], [], 1 - correct.mean()
    for score, style, line_colour, label in (('Margin + TTA', '-', colour, 'Margin + TTA'),
                                             ('Margin + TTA + anatomy', (0, (4, 1.5)), colour, 'Margin + TTA + $E$'),
                                             ('Anatomical enrichment', (0, (1, 1.2)), MUTED, '$E$ alone')):
        values = part[score].to_numpy(dtype=float)
        coverage, risk = risk_coverage(values, correct)
        ax.plot(coverage, risk, linestyle=style, color=line_colour, linewidth=1.1)
        lines.append(line_handle(f"{label} (AURC {aurc(values, correct):.3f})", line_colour, style))
        curves.append((coverage, risk))
    ax.axhline(base, color=AXIS, linewidth=0.6)
    lines.append(line_handle('All flags kept', AXIS, '-', 0.6))
    ax.set_xlabel('Coverage (share of flags kept)')
    ax.set_ylabel('False-flag rate among kept flags')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    heading = 'DenseNet-121 flags on VinDr-CXR'
    renderer = canvas.fig.canvas.get_renderer()
    placed = False
    for corner in ('upper right', 'lower right'):
        legend = ax.legend(handles=lines, loc=corner, fontsize=6.0, title=heading, title_fontsize=6.0)
        (x0, y0), (x1, y1) = np.sort(legend.get_window_extent(renderer).transformed(ax.transData.inverted()).get_points(), axis=0)
        clash = y0 <= base <= y1 or any(((c >= x0) & (c <= x1) & (r >= y0) & (r <= y1)).any() for c, r in curves)
        if not clash:
            placed = True
            break
        legend.remove()
    if not placed:
        legend_above(ax, lines, ncol=2, heading=heading)
    recessive_grid(ax, 'both')
    for letter, row in (('a', 0), ('b', 1), ('d', 2)):
        canvas.letter(1.0, tops[row] - 1.4, letter)
    for letter, row in (('c', 1), ('e', 2)):
        canvas.letter(left_a + width + gap, tops[row] - 1.4, letter)
    canvas.legend(site_handles() + model_handles(), (left_a + left_b + width) / 2, 1.5)
    canvas.save('Fig5')

def extended_figure_1(data, examples):
    examples = [e for e in examples if e['figure'] == 'ED_Fig1']
    if not examples:
        print("No NIH examples with LIME and SHAP were found.")
        return
    label_width, gap, side = 9.0, 2.5, 40.0
    left = (WIDTH - label_width - 4 * side - 3 * gap) / 2 + label_width
    header, row_gap, caption = 7.0, 3.0, 5.0
    canvas = Canvas(header + len(examples) * (side + caption + row_gap))
    for j, heading in enumerate(('Radiograph', 'Grad-CAM', 'LIME', 'SHAP')):
        canvas.text(left + j * (side + gap) + side / 2, header - 2.0, heading, ha='center', fontsize=7.0, fontweight='bold')
    for i, example in enumerate(examples):
        top = header + i * (side + caption + row_gap)
        image = np.asarray(data['images'][example['row']])[:, :, 0]
        layers = np.asarray(data['anatomy'][example['row']])
        outcome = 'correct flag' if example['correct'] == 1 else 'wrong flag'
        canvas.text(left - 3.0, top + side / 2, f"{example['model']}\n{example['finding']}, {outcome}", ha='center',
                    va='center', rotation=90, fontsize=6.3, linespacing=1.4)
        canvas.letter(1.0, top + 2.1, 'abcdefgh'[i])
        for j, key in enumerate(('Radiograph', 'Grad-CAM', 'LIME', 'SHAP')):
            ax = canvas.axes(left + j * (side + gap), top, side, side)
            if key == 'Radiograph':
                show_image(ax, image)
                outline(ax, layers & 1, REGION, 0.9)
                draw_boxes(ax, data['boxes'], example['row'], example['finding'])
            else:
                overlay(ax, image, example['maps'][key], smooth=2.0 if key == 'SHAP' else 0.0)
                outline(ax, layers & 1, REGION, 0.6)
                canvas.text(left + j * (side + gap) + side / 2, top + side + 3.6, f"$E$ = {example['enrichment'][key]:.2f}",
                            ha='center', fontsize=6.3)
    canvas.save('ED_Fig1')
    print("\nQualitative NIH examples (enrichment recomputed for the figure, saved notebook value in brackets)")
    for example in examples:
        outcome = 'correct' if example['correct'] == 1 else 'wrong'
        print(f"  {example['model']}, {example['finding']} ({outcome} flag): Grad-CAM {example['enrichment']['Grad-CAM']:.3f} "
              f"[{example['saved']['Grad-CAM']:.3f}], LIME {example['enrichment']['LIME']:.3f} [{example['saved']['LIME']:.3f}], "
              f"SHAP {example['enrichment']['SHAP']:.3f} [{example['saved']['SHAP']:.3f}]")

def extended_figure_2(data):
    labels = [f for f in FINDINGS if f != 'Emphysema'] + ['All findings']
    gap = 4.0
    (left_a, left_b), width = column_layout([measure(labels) + 3.0, 3.0], gap=gap)
    top, row = 14.0, 7.0
    height = len(labels) * row
    canvas = Canvas(top + height + 11.0)
    for column, (model, left) in enumerate(zip(MODELS, (left_a, left_b))):
        ax = canvas.axes(left, top, width, height)
        source = result_frame(data, model, 'localisation')
        present = set(source['finding'])
        for position, finding in enumerate(labels):
            if finding not in present:
                continue
            for site in SITES:
                for flagged, nudge in ((False, -0.07), (True, 0.07)):
                    entry = hit_rate(source, f"{site} Grad-CAM" + (', flagged only' if flagged else ''), finding)
                    if entry is not None:
                        dot(ax, entry['value'], position + OFFSETS[site] + nudge, entry['low'], entry['high'], COLOURS[site], 'o', not flagged, 2.8)
        rows_axis(ax, labels, [len(labels) - 1.5], show=column == 0)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel('Pointing-game hit rate (95% CI)')
        title(ax, f'{model} Grad-CAM')
        recessive_grid(ax)
    canvas.letter(1.0, top - 1.4, 'a')
    canvas.letter(left_a + width + gap, top - 1.4, 'b')
    canvas.legend(site_handles() + [handle('All annotated findings'), handle('Flagged findings only', filled=False)],
                  (left_a + left_b + width) / 2, 1.5)
    canvas.save('ED_Fig2')

def extended_figure_3(data, extra):
    label_width = measure(site_model_labels()) + 3.0
    gap, inner = 5.0, 3.0
    (left_a, left_b), width = column_layout([label_width, inner], gap=gap)
    height = 40.0
    tops = [14.0, 14.0 + height + 19.0]
    canvas = Canvas(tops[1] + height + 12.0)
    for column, (model, left) in enumerate(zip(MODELS, (left_a, left_b))):
        ax = canvas.axes(left, tops[0], width, height)
        score_panel(ax, data, model, 'Unflagged findings', show_labels=column == 0)
        if column == 0:
            ax.set_ylabel(f'{AUC_F}, unflagged findings')
        title(ax, model)
    ax = canvas.axes(left_a, tops[1], width, height)
    delta_panel(ax, extra, 'Unflagged findings')
    ax.set_xlabel(f'Change in {AUC_F} from adding $E$ (95% CI)')
    legend_above(ax, [handle('Fitted on NIH'), handle('Refitted within the site', filled=False)])
    ax = canvas.axes(left_b, tops[1], width, height)
    labels, position = [], 0
    for model in MODELS:
        part = extra['redundancy'][extra['redundancy']['model'] == model]
        for partition in PARTITIONS:
            for scope, nudge in (('Flagged findings', -0.15), ('Unflagged findings', 0.15)):
                entry = part[(part['partition'] == partition) & (part['scope'] == scope)]
                if len(entry):
                    entry = entry.iloc[0]
                    dot(ax, entry['rho'], position + nudge, entry['low'], entry['high'], COLOURS[SITE_OF[partition]],
                        MARKERS[model], scope == 'Flagged findings')
            labels.append(f"{SITE_OF[partition]}, {model}")
            position += 1
    ax.axvline(0, color=INK, linewidth=0.6)
    rows_axis(ax, labels, [2.5], show=False)
    ax.set_xlabel('Spearman correlation of $E$ with $|m|$ (95% CI)')
    recessive_grid(ax)
    legend_above(ax, [handle('Flagged findings'), handle('Unflagged findings', filled=False)])
    canvas.letter(1.0, tops[0] - 1.4, 'a')
    canvas.letter(1.0, tops[1] - 1.4, 'b')
    canvas.letter(left_a + width + gap, tops[1] - 1.4, 'c')
    canvas.legend(site_handles() + model_handles(), (left_a + left_b + width) / 2, 1.5)
    canvas.save('ED_Fig3')

def extended_figure_4(data):
    pairs = [('Grad-CAM', 'LIME'), ('Grad-CAM', 'SHAP'), ('LIME', 'SHAP')]
    label_a = measure(['−0.2', '1.0']) + 6.5
    label_b = measure(['−0.75', '1.00']) + 9.5
    gap = 8.0
    total = WIDTH - 4.0 - label_a - label_b - gap
    width_a, width_b = total * 0.62, total * 0.38
    left_a, left_b = label_a, label_a + width_a + gap + label_b
    top, height = 14.0, 50.0
    canvas = Canvas(top + height + 15.0)
    ax = canvas.axes(left_a, top, width_a, height)
    bounds = []
    for position, (first, second) in enumerate(pairs):
        for model, nudge in (('DenseNet-121', -0.42), ('ViT-B/16', 0.42)):
            cross = data['cross'][model]
            for site in SITES:
                part = cross[cross['dataset'] == site].dropna(subset=['Grad-CAM', 'LIME', 'SHAP'])
                if len(part) < MIN_EVENTS:
                    continue
                a, b = part[first].to_numpy(dtype=float), part[second].to_numpy(dtype=float)
                ci = bootstrap({'rho': (spearman_at, {'first': a, 'second': b})}, patient_groups(part['patient']))
                dot(ax, 2.2 * position + nudge + 0.7 * OFFSETS[site], spearman_at(np.arange(len(part)), a, b), ci['rho'][0],
                    ci['rho'][1], COLOURS[site], MARKERS[model], horizontal=False)
                bounds += [ci['rho'][0], ci['rho'][1]]
    ax.axhline(0, color=AXIS, linewidth=0.6)
    ax.set_xticks([2.2 * i for i in range(len(pairs))])
    ax.set_xticklabels([f'{a} and {b}' for a, b in pairs])
    for i in range(1, len(pairs)):
        ax.axvline(2.2 * i - 1.1, color=GRID, linewidth=0.6)
    ax.set_xlim(-1.1, 2.2 * len(pairs) - 1.1)
    ax.set_ylim(*value_limits(bounds, -0.3, 1.0, pad=0.04))
    ax.set_ylabel('Spearman correlation of $E$ (95% CI)')
    title(ax, 'Agreement between explainers', bold=False)
    recessive_grid(ax, 'y')
    ax = canvas.axes(left_b, top, width_b, height)
    bounds = []
    for model, nudge in (('DenseNet-121', -0.1), ('ViT-B/16', 0.1)):
        randomisation = result_frame(data, model, 'randomisation')
        for position, entry in enumerate(randomisation.itertuples()):
            dot(ax, position + nudge, entry.mean_spearman, entry.low, entry.high, COLOURS['NIH'], MARKERS[model], horizontal=False)
            bounds += [entry.low, entry.high, entry.mean_spearman]
    ax.axhline(0, color=AXIS, linewidth=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['None', 'Head', 'Head and\nlast block'])
    ax.set_xlabel('Layers re-initialized')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(*value_limits(bounds, -0.75, 1.08, pad=0.04))
    ax.set_ylabel('Spearman correlation with\nthe trained Grad-CAM map')
    title(ax, 'Model-randomization test (NIH)', bold=False)
    recessive_grid(ax, 'y')
    canvas.letter(1.0, top - 1.4, 'a')
    canvas.letter(left_a + width_a + gap, top - 1.4, 'b')
    canvas.legend(site_handles() + model_handles(), (left_a + left_b + width_b) / 2, 1.5)
    canvas.save('ED_Fig4')

def extended_figure_5(data):
    label_width = measure(['0.0', '1.0']) + 6.5
    gap = 7.0
    width = (WIDTH - 4.0 - label_width - 2 * gap) / 3
    height = 40.0
    tops = [14.0, 14.0 + height + 11.0]
    canvas = Canvas(tops[1] + height + 11.0)
    for i, model in enumerate(MODELS):
        frame = data['frames'][model]['Flagged findings']
        for j, partition in enumerate(PARTITIONS):
            ax = canvas.axes(label_width + j * (width + gap), tops[i], width, height)
            part = frame[frame['partition'] == partition]
            correct = part['correct'].to_numpy()
            colour = COLOURS[SITE_OF[partition]]
            for score, style, line_colour, line_width in (('Confidence margin', '-', INK, 0.8), ('Margin + TTA', '-', colour, 1.1),
                                                          ('Margin + TTA + anatomy', (0, (4, 1.5)), colour, 1.1),
                                                          ('Anatomical enrichment', (0, (1, 1.2)), MUTED, 1.1)):
                coverage, risk = risk_coverage(part[score].to_numpy(dtype=float), correct)
                ax.plot(coverage, risk, linestyle=style, color=line_colour, linewidth=line_width)
            ax.axhline(1 - correct.mean(), color=AXIS, linewidth=0.6)
            title(ax, f'{partition}, {model}', bold=False)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.02, 1.02)
            recessive_grid(ax, 'both')
            if j == 0:
                ax.set_ylabel('False-flag rate among kept flags')
            else:
                ax.tick_params(axis='y', labelleft=False)
            if i == 1:
                ax.set_xlabel('Coverage (share of flags kept)')
            else:
                ax.tick_params(axis='x', labelbottom=False)
        canvas.letter(1.0, tops[i] - 1.4, 'ab'[i])
    canvas.legend([line_handle('Margin', INK, '-', 0.8), line_handle('Margin + TTA (site color)', SECONDARY, '-'),
                   line_handle('Margin + TTA + $E$ (site color)', SECONDARY, (0, (4, 1.5))),
                   line_handle('$E$ alone', MUTED, (0, (1, 1.2))), line_handle('All flags kept', AXIS, '-', 0.6)],
                  label_width + (3 * width + 2 * gap) / 2, 1.5)
    canvas.save('ED_Fig5')

def extended_figure_6(data, extra):
    table = extra['by_finding']
    cells = [(partition, finding) for partition in PARTITIONS for finding in FINDINGS
             if ((table['partition'] == partition) & (table['finding'] == finding)).any()]
    labels = [f"{finding}, {SITE_OF[partition]}" for partition, finding in cells]
    separators = [i - 0.5 for i in range(1, len(cells)) if cells[i][0] != cells[i - 1][0]]
    gap = 7.0
    (left_a, left_b), width = column_layout([measure(labels) + 3.0, 0.0], gap=gap)
    top_a, row = 14.0, 4.2
    height_a = max(28.0, len(cells) * row)
    top_b, height_b = top_a + height_a + 21.0, 46.0
    canvas = Canvas(top_b + height_b + 12.0)
    forests, bounds = [], []
    for column, (model, left) in enumerate(zip(MODELS, (left_a, left_b))):
        ax = canvas.axes(left, top_a, width, height_a)
        forests.append(ax)
        part = table[table['model'] == model]
        for position, (partition, finding) in enumerate(cells):
            entry = part[(part['partition'] == partition) & (part['finding'] == finding)]
            if len(entry) == 0:
                continue
            entry = entry.iloc[0]
            colour = COLOURS[SITE_OF[partition]]
            dot(ax, entry['margin'], position - 0.16, entry['margin_low'], entry['margin_high'], colour, 'o', True, 2.8)
            dot(ax, entry['enrichment'], position + 0.16, entry['enrichment_low'], entry['enrichment_high'], colour, 'o', False, 2.8)
            bounds += [entry['margin_low'], entry['margin_high'], entry['enrichment_low'], entry['enrichment_high']]
        ax.axvline(0.5, color=INK, linewidth=0.6, linestyle=':')
        rows_axis(ax, labels, separators, show=column == 0)
        ax.set_xlabel(f'{AUC_F}, flagged findings (95% CI)')
        title(ax, model)
        recessive_grid(ax)
    low, high = value_limits(bounds, 0.4, 0.8, pad=0.03)
    for ax in forests:
        ax.set_xlim(low, high)
    canvas.legend([handle('Confidence margin'), handle('Enrichment $E$', filled=False)], (left_a + left_b + width) / 2,
                  top_a + height_a + 11.0)
    ax = canvas.axes(left_a, top_b, left_b + width - left_a, height_b)
    order = ['All flags', 'Confidence margin', 'Margin + TTA', 'Margin + TTA + anatomy', 'Anatomical enrichment']
    for model, nudge in (('DenseNet-121', -0.06), ('ViT-B/16', 0.06)):
        for position, score in enumerate(order):
            for partition in PARTITIONS:
                entry = extra['risks'][(extra['risks']['model'] == model) & (extra['risks']['partition'] == partition) &
                                       (extra['risks']['score'] == score)]
                if len(entry):
                    entry = entry.iloc[0]
                    dot(ax, position + OFFSETS[partition] + nudge, entry['risk'], entry['low'], entry['high'], COLOURS[SITE_OF[partition]],
                        MARKERS[model], True, 3.0, horizontal=False)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(['All flags'] + [f'{TICKS[s]} kept'.replace('\n', ' ') for s in order[1:]])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(f'False-flag rate with {int(COVERAGE * 100)}%\nof flags kept (95% CI)')
    recessive_grid(ax, 'y')
    legend_above(ax, model_handles())
    canvas.letter(1.0, top_a - 1.4, 'a')
    canvas.letter(1.0, top_b - 1.4, 'b')
    canvas.legend(site_handles(), (left_a + left_b + width) / 2, 1.5)
    canvas.save('ED_Fig6')

def main():
    missing = [path for path in required_files() if not os.path.exists(path)]
    if missing:
        print("These files are missing: " + ", ".join(missing))
        print("Run this script from the project folder after both XAI notebooks have finished.")
        return
    family = set_style()
    data = load()
    extra = supplementary(data)
    print("\nRecomputing LIME and SHAP for the example radiographs.")
    with quiet_native_output():
        examples = explain_examples(data, choose_examples(data))
    print(f"Drawing the figures in {family}.")
    figure_example = examples[0]
    share, area = figure_example['share']['Grad-CAM']
    print(f"Figure 1 example: NIH test row {figure_example['row']}, {figure_example['finding']}, {share:.1%} of Grad-CAM attribution "
          f"inside R, R covers {area:.1%} of the image, E = {figure_example['enrichment']['Grad-CAM']:.3f} "
          f"(LIME {figure_example['enrichment']['LIME']:.3f}, SHAP {figure_example['enrichment']['SHAP']:.3f})")
    figure_1(data, figure_example)
    figure_2(data)
    figure_3(data)
    figure_4(data)
    figure_5(data, extra)
    extended_figure_1(data, examples)
    extended_figure_2(data)
    extended_figure_3(data, extra)
    extended_figure_4(data)
    extended_figure_5(data)
    extended_figure_6(data, extra)
    written = sorted(name[:-4] for name in os.listdir(OUT) if name.endswith('.pdf'))
    print(f"\nSaved {len(written)} figures to {OUT} as PDF and PNG, the Extended Data figures also as TIFF: {', '.join(written)}.")
    print("Saved the supplementary analyses to models/paper_extra_results.json and models/paper_*.csv.")

if __name__ == '__main__':
    main()