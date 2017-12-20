import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def evaluation(prediction, y):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    e = 0.000001
    threshhold = 0.5
    for i in range(len(y)):
        for j in range(len(y[i])):
            p = prediction[i]
            min_l = min(prediction[i])
            max_l = max(prediction[i])
            norm_score = ((prediction[i][j]-min_l)/(max_l-min_l))
            fout.write(str(y[i][j])+"\t"+str(prediction[i][j])+"\t"+str(norm_score)+"\n")
            if y[i][j] == 1 and norm_score >= threshhold:
               TP += 1
            if y[i][j] == 1 and norm_score < threshhold:
               FN += 1
            if y[i][j] == 0 and norm_score >= threshhold:
               FP += 1
            if y[i][j] == 0 and norm_score < threshhold:
               TN += 1
    accu = float(TP+TN)/(TP+TN+FP+FN+e)
    prec_r = float(TP)/(TP+FP+e) ## for rumor
    recall_r = float(TP)/(TP+FN+e)
    F_r = 2 * prec_r*recall_r / (prec_r + recall_r+e)
    prec_f = float(TN)/(TN+FN+e)  ## for fact
    recall_f = float(TN)/(TN+FP+e)
    F_f = 2 * prec_f*recall_f / (prec_f + recall_f+e)
    return [accu, prec_r, recall_r, F_r, prec_f, recall_f, F_f, TP, FN, FP, TN]