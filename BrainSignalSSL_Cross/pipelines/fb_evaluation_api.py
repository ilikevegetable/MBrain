from sklearn.metrics import roc_curve, auc


class EvaluationIndex:
    def __init__(self, acc=0., tp=0., fp=0., fn=0., tn=0., pre=0., rec=0., f_h=0., f1=0., f_d=0., auc_s=None, th=None):
        self.acc = acc
        self.micro_tp = tp
        self.micro_fp = fp
        self.micro_fn = fn
        self.micro_tn = tn
        self.micro_pre = pre
        self.micro_rec = rec
        self.micro_f_h = f_h
        self.micro_f1 = f1
        self.micro_f_d = f_d
        self.micro_auc = auc_s
        self.micro_th = th

        self.macro_tp = None
        self.macro_fp = None
        self.macro_fn = None
        self.macro_tn = None
        self.macro_pre = None
        self.macro_rec = None
        self.macro_f_h = None
        self.macro_f1 = None
        self.macro_f_d = None
        self.macro_auc = None
        self.macro_th = None

    def __str__(self):
        out = ''
        out += '-'*10 + 'Evaluation Result' + '-'*10 + '\n'
        out += 'Accuracy: ' + str(self.acc) + '\n'
        if self.macro_pre is None:
            out += 'TP: ' + str(self.micro_tp) + '\n'
            out += 'FP: ' + str(self.micro_fp) + '\n'
            out += 'FN: ' + str(self.micro_fn) + '\n'
            out += 'TN: ' + str(self.micro_tn) + '\n'
            out += 'Precision: ' + str(self.micro_pre) + '\n'
            out += 'Recall: ' + str(self.micro_rec) + '\n'
            out += 'F0.5: ' + str(self.micro_f_h) + '\n'
            out += 'F1: ' + str(self.micro_f1) + '\n'
            out += 'F2: ' + str(self.micro_f_d) + '\n'
            out += 'AUC: ' + str(self.micro_auc) + '\n'
            out += 'Threshold: ' + str(self.micro_th) + '\n'
        else:
            out += 'Micro TP: ' + str(self.micro_tp) + '\n'
            out += 'Micro FP: ' + str(self.micro_fp) + '\n'
            out += 'Micro FN: ' + str(self.micro_fn) + '\n'
            out += 'Micro TN: ' + str(self.micro_tn) + '\n'
            out += 'Micro Precision: ' + str(self.micro_pre) + '\n'
            out += 'Micro Recall: ' + str(self.micro_rec) + '\n'
            out += 'Micro F0.5: ' + str(self.micro_f_h) + '\n'
            out += 'Micro F1: ' + str(self.micro_f1) + '\n'
            out += 'Micro F2: ' + str(self.micro_f_d) + '\n'
            out += 'Micro AUC: ' + str(self.micro_auc) + '\n'
            out += 'Micro Threshold: ' + str(self.micro_th) + '\n'

            out += '-' * 10 + '\n'
            out += 'Macro TP: ' + str(self.macro_tp) + '\n'
            out += 'Macro FP: ' + str(self.macro_fp) + '\n'
            out += 'Macro FN: ' + str(self.macro_fn) + '\n'
            out += 'Macro TN: ' + str(self.macro_tn) + '\n'
            out += 'Macro Precision: ' + str(self.macro_pre) + '\n'
            out += 'Macro Recall: ' + str(self.macro_rec) + '\n'
            out += 'Macro F0.5: ' + str(self.macro_f_h) + '\n'
            out += 'Macro F1: ' + str(self.macro_f1) + '\n'
            out += 'Macro F2: ' + str(self.macro_f_d) + '\n'
            out += 'Macro AUC: ' + str(self.macro_auc) + '\n'
            out += 'Macro Threshold: ' + str(self.macro_th) + '\n'
        return out


class Evaluator:
    def __init__(self):
        pass

    def class_evaluation(self, y_true, y_pred, positive_pred=None, valid_flag=True):
        if positive_pred is not None:
            fpr, tpr, thresholds = roc_curve(y_true, positive_pred)
            auc_score = auc(fpr, tpr)
        else:
            auc_score = None

        # Compute the confusion matrix
        total_sample = y_pred.shape[0]
        tp = (y_true & y_pred).sum()
        fp = y_pred.sum() - tp
        fn = y_true.sum() - tp
        tn = total_sample - tp - fp - fn

        accuracy = (tp + tn) / total_sample
        # valid_flag means that if the true labels do not include '1's, other indexes will be 0 except for accuracy
        if not valid_flag:
            return EvaluationIndex(accuracy)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = self.f_beta_score(precision, recall, beta=1.0)
        f_half = self.f_beta_score(precision, recall, beta=0.5)
        f_doub = self.f_beta_score(precision, recall, beta=2.0)
        return EvaluationIndex(accuracy, tp, fp, fn, tn, precision, recall, f_half, f1, f_doub, auc_score)

    def prob_evaluation(self, y_true, y_prob, valid_flag=True, threshold=0.5):
        y_pred = (y_prob >= threshold)
        return self.class_evaluation(y_true, y_pred, y_prob, valid_flag)

    @staticmethod
    def f_beta_score(pre, rec, beta=1.0):
        if pre > 0 or rec > 0:
            return (1 + beta ** 2) * pre * rec / (beta ** 2 * pre + rec)
        else:
            return 0
