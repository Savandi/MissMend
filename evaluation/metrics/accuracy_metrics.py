from collections import defaultdict

class AccuracyMetrics:

    RECOVERY_FLAGS = ('RECOVERED_ML', 'RECOVERED_ML_SEQ')

    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.abstained = 0
        self.total_missing = 0
        self.per_activity = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        self.confidence_bins = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
        self.per_provenance = defaultdict(lambda: {'tp': 0, 'fp': 0, 'total': 0})

    def evaluate(self, results, ground_truth, injected_indices):
        for idx in injected_indices:
            if idx >= len(results):
                continue
            result = results[idx]
            true_label = ground_truth.get(idx)
            if not true_label:
                continue

            self.total_missing += 1
            recovered = result.get('recovered_label')
            flag = result.get('provenance')
            confidence = result.get('confidence', 0.0)

            if flag in self.RECOVERY_FLAGS and recovered:
                conf_bin = round(confidence, 1)
                self.per_provenance[flag]['total'] += 1
                if recovered == true_label:
                    self.true_positives += 1
                    self.per_activity[true_label]['tp'] += 1
                    self.confidence_bins[conf_bin]['correct'] += 1
                    self.per_provenance[flag]['tp'] += 1
                else:
                    self.false_positives += 1
                    self.false_negatives += 1
                    self.per_activity[true_label]['fn'] += 1
                    self.per_activity[recovered]['fp'] += 1
                    self.confidence_bins[conf_bin]['incorrect'] += 1
                    self.per_provenance[flag]['fp'] += 1
            else:
                self.abstained += 1
                self.false_negatives += 1
                self.per_activity[true_label]['fn'] += 1

    @property
    def precision(self):
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self):
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def recovery_rate(self):
        return (self.true_positives + self.false_positives) / self.total_missing if self.total_missing > 0 else 0.0

    @property
    def abstain_rate(self):
        return self.abstained / self.total_missing if self.total_missing > 0 else 0.0

    def summary(self):
        provenance_block = {}
        for flag, c in self.per_provenance.items():
            tp, fp, total = c['tp'], c['fp'], c['total']
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            provenance_block[flag] = {
                'tp': tp, 'fp': fp, 'total': total, 'precision': prec,
            }

        return {
            'total_injected': self.total_missing,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'abstained': self.abstained,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'recovery_rate': self.recovery_rate,
            'abstain_rate': self.abstain_rate,
            'per_provenance': provenance_block,
        }

    def per_activity_summary(self):
        results = {}
        for activity, counts in self.per_activity.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            results[activity] = {'precision': p, 'recall': r, 'f1': f, 'tp': tp, 'fp': fp, 'fn': fn}
        return results

    def calibration_summary(self):
        results = {}
        for conf_bin, counts in sorted(self.confidence_bins.items()):
            total = counts['correct'] + counts['incorrect']
            acc = counts['correct'] / total if total > 0 else 0.0
            results[conf_bin] = {'accuracy': acc, 'total': total, 'correct': counts['correct']}
        return results
