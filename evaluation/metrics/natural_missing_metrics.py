import numpy as np
from collections import defaultdict

class NaturalMissingMetrics:

    def __init__(self):
        self.total_natural_missing = 0
        self.recovered = 0
        self.unrecovered = 0
        self.recovered_confidences = []
        self.unrecovered_confidences = []
        self.recovered_label_counts = defaultdict(int)
        self.dfg_valid_count = 0
        self.dfg_invalid_count = 0
        self.dfg_unknown_count = 0

    def evaluate(self, results, events, injected_indices, dfg):
        injected_set = set(injected_indices) if injected_indices else set()

        for idx, (result, event) in enumerate(zip(results, events)):
            if idx in injected_set:
                continue

            original_label = getattr(event, 'concept_name', None)
            is_natural_missing = (
                original_label is None
                or str(original_label).strip() == ''
                or str(original_label).lower() in ('nan', 'none', 'null')
            )
            if not is_natural_missing:
                continue

            self.total_natural_missing += 1
            flag = result.get('provenance')
            confidence = float(result.get('confidence', 0.0))
            recovered_label = result.get('recovered_label')

            if flag == 'RECOVERED_ML' and recovered_label:
                self.recovered += 1
                self.recovered_confidences.append(confidence)
                self.recovered_label_counts[recovered_label] += 1

                prev_activity = self._get_previous_activity(events, idx)
                if prev_activity and dfg is not None:
                    outgoing = dfg.get(prev_activity, {}) if prev_activity else {}
                    if outgoing:
                        if recovered_label in outgoing:
                            self.dfg_valid_count += 1
                        else:
                            self.dfg_invalid_count += 1
                    else:
                        self.dfg_unknown_count += 1
                else:
                    self.dfg_unknown_count += 1
            else:
                self.unrecovered += 1
                self.unrecovered_confidences.append(confidence)

    @staticmethod
    def _get_previous_activity(events, idx):
        if idx == 0:
            return None
        current = events[idx]
        case_id = getattr(current, 'case_id', None)
        for j in range(idx - 1, -1, -1):
            prev = events[j]
            if getattr(prev, 'case_id', None) == case_id:
                label = getattr(prev, 'concept_name', None)
                if label and str(label).strip():
                    return str(label)
                return None
        return None

    @property
    def recovery_rate(self):
        if self.total_natural_missing == 0:
            return 0.0
        return self.recovered / self.total_natural_missing

    def _quantiles(self, values):
        if not values:
            return {'mean': 0.0, 'median': 0.0, 'q1': 0.0, 'q3': 0.0, 'min': 0.0, 'max': 0.0}
        arr = np.array(values)
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'q1': float(np.quantile(arr, 0.25)),
            'q3': float(np.quantile(arr, 0.75)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
        }

    @property
    def dfg_plausibility(self):
        checked = self.dfg_valid_count + self.dfg_invalid_count
        if checked == 0:
            return 0.0
        return self.dfg_valid_count / checked

    def summary(self):
        return {
            'total_natural_missing': self.total_natural_missing,
            'recovered': self.recovered,
            'unrecovered': self.unrecovered,
            'recovery_rate': self.recovery_rate,
            'recovered_confidence_stats': self._quantiles(self.recovered_confidences),
            'unrecovered_confidence_stats': self._quantiles(self.unrecovered_confidences),
            'dfg_valid': self.dfg_valid_count,
            'dfg_invalid': self.dfg_invalid_count,
            'dfg_unknown_prev': self.dfg_unknown_count,
            'dfg_plausibility': self.dfg_plausibility,
            'top_recovered_labels': sorted(
                self.recovered_label_counts.items(),
                key=lambda x: -x[1]
            )[:10],
        }

    def print_summary(self, dataset_name=''):
        s = self.summary()
        print(f'\n=== Natural Missing Label Metrics: {dataset_name} ===')
        print(f'  Total natural missing: {s["total_natural_missing"]}')
        print(f'  Recovered: {s["recovered"]} ({s["recovery_rate"]:.1%})')
        print(f'  Unrecovered: {s["unrecovered"]}')
        if s['recovered'] > 0:
            rc = s['recovered_confidence_stats']
            print(f'  Recovered confidence:   mean={rc["mean"]:.3f}  '
                  f'median={rc["median"]:.3f}  IQR=[{rc["q1"]:.3f}, {rc["q3"]:.3f}]')
        if s['unrecovered'] > 0:
            uc = s['unrecovered_confidence_stats']
            print(f'  Unrecovered confidence: mean={uc["mean"]:.3f}  '
                  f'median={uc["median"]:.3f}  IQR=[{uc["q1"]:.3f}, {uc["q3"]:.3f}]')
        print(f'  DFG plausibility: {s["dfg_plausibility"]:.1%} '
              f'({s["dfg_valid"]} valid / {s["dfg_valid"] + s["dfg_invalid"]} checked, '
              f'{s["dfg_unknown_prev"]} unknown predecessor)')
        if s['top_recovered_labels']:
            print(f'  Top recovered labels: {s["top_recovered_labels"][:5]}')
