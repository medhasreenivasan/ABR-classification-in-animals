import numpy as np

class metrics:
    def __init__(self,y_true,y_pred):
        self.y_true = y_true
        self.y_pred = y_pred 

    def compute_confusion_matrix(self):
        '''
        t11 - first number represents the predicted class and the second number represents the true class for eg:
        f21 - true class: 1 , predicted class: 2 - false negative for class 1 
        '''
        t11,t22,t33,f12,f13,f21,f23,f31,f32 = 0,0,0,0,0,0,0,0,0
        predicted = self.y_pred
        actual = self.y_true
        for pred,act in zip(predicted,actual):
            if act == 0 and pred == 0:
                t11 += 1
            elif act == 0 and pred == 1:
                f12 += 1
            elif act == 0 and pred == 2:
                f13 += 1
            elif act == 1 and pred == 0:
                f21 += 1
            elif act == 1 and pred == 1:
                t22 += 1
            elif act == 1 and pred == 2:
                f23 += 1
            elif act == 2 and pred == 0:
                f31 += 1
            elif act == 2 and pred == 1:
                f32 += 1
            elif act == 2 and pred == 2:
                t33 += 1

        return t11,t22,t33,f12,f13,f21,f23,f31,f32
    
    def f1_score(self):
        t11,t22,t33,f12,f13,f21,f23,f31,f32 = self.compute_confusion_matrix()
        ## precision = TP/(TP+FP)  - zero division condiiton included
        pc1 = t11/(t11+f12+f13) if (t11+f12+f13)!=0 else 0
        pc2 = t22/(t22+f21+f23) if (t22+f21+f23)!=0  else 0
        pc3 = t33/(t33+f32+f31) if (t33+f32+f31)!=0  else 0
        precisions = [pc1,pc2,pc3]
    
        ## recall = TP/(TP+FN)
        rc1 = t11/(t11+f21+f31) if (t11+f21+f31)!=0 else 0
        rc2 = t22/(t22+f12+f32) if (t22+f12+f32)!=0 else 0
        rc3 = t33/(t33+f13+f23) if (t33+f13+f23)!=0 else 0
        recalls = [rc1,rc2,rc3]

        f1score_1 = 2*(pc1*rc1)/(pc1+rc1) if (pc1+rc1)!=0 else 0
        f1score_2 = 2*(pc2*rc2)/(pc2+rc2) if (pc2+rc2)!=0 else 0
        f1score_3 = 2*(pc3*rc3)/(pc3+rc3) if (pc3+rc3)!=0 else 0
        f1_scores = [f1score_1,f1score_2,f1score_3]
        macro_f1 = (f1score_1+f1score_2+f1score_3)/3
        weighted_f1 = (t11*f1score_1+t22*f1score_2+t33*f1score_3)/(len(self.y_pred))

        return precisions,recalls,f1_scores,macro_f1,weighted_f1
    
    def compute_accuracy(self):
        correct = sum([pred == act for pred,act in zip(self.y_pred,self.y_true)])
        acc = correct/len(self.y_true)
        return acc
        

