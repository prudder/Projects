#Importing the necessary packages for modeling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import ElasticNetCV, ElasticNet

class modeler():    
    def custom_cross_val(df,drop_feats,cat,fold,drop=True,model=LogisticRegression()):
        X_train = df[df[cat] != fold][[col for col in df.columns if 'is_right' not in col]]
        
        X_train = X_train.drop(columns=drop_feats)
        
        y_train = df[df[cat] != fold]['is_right']

        X_test = df[df[cat] == fold][[col for col in df.columns if 'is_right' not in col]]
        if drop==True:
            X_test = X_test.drop(columns=drop_feats)
        y_test = df[df[cat] == fold]['is_right']

        lr = model

        lr_mod = lr.fit(X_train,y_train)
        y_pred = lr_mod.predict(X_test)

        acc = lr_mod.score(X_test,y_test)
        rec = recall_score(y_test,y_pred)
        prec = precision_score(y_test,y_pred) 
        f1 = 2 * (prec * rec) / (prec + rec)
        
        return acc,rec,prec,f1

    def custom_scorer(cat,folds,d_f,dp,mod=LogisticRegression()):
        def custom_cross_val(cat,folds,df=d_f,drop_feats=dp,drop=True,model=mod):
            X_train = df[df[cat] != fold][[col for col in df.columns if 'is_right' not in col]]
            
            X_train = X_train.drop(columns=drop_feats)
            
            y_train = df[df[cat] != fold]['is_right']

            X_test = df[df[cat] == fold][[col for col in df.columns if 'is_right' not in col]]
            if drop==True:
                X_test = X_test.drop(columns=drop_feats)
            y_test = df[df[cat] == fold]['is_right']

            lr = model

            lr_mod = lr.fit(X_train,y_train)
            y_pred = lr_mod.predict(X_test)

            acc = lr_mod.score(X_test,y_test)
            rec = recall_score(y_test,y_pred)
            prec = precision_score(y_test,y_pred) 
            f1 = 2 * (prec * rec) / (prec + rec)
            
            return acc,rec,prec,f1
        
        new_dict = {}
        new_dict['accuracy'] = []
        new_dict['recall'] = []
        new_dict['precision'] = []
        new_dict['f1'] = []
        
        for fold in folds:
            acc,rec,prec,f1 = custom_cross_val(cat,fold,df=d_f,drop_feats=dp,model=mod)
            new_dict['accuracy'].append(acc)
            new_dict['recall'].append(rec)
            new_dict['precision'].append(prec)
            new_dict['f1'].append(f1)

        return pd.DataFrame(new_dict,index=folds)

    def confusion_matrix(y,predict):
        tp = np.sum((y == 1) & (predict == 1))
        fp = np.sum((y == 0) & (predict == 1))
        tn = np.sum((y == 0) & (predict == 0))
        fn = np.sum((y == 1) & (predict == 0))

        new_df = pd.DataFrame({'Predicted positive':[tp,fp],'Predicted negative':[fn,tn]})
        new_df.index = ['Actual Positive','Actual Negative']
        return new_df

    def probability_table(model,Xtest,ytest,ref_df):
        probas = model.predict_proba(Xtest)
        prob_df = pd.DataFrame(probas)
        prob_df['electorate'] = ref_df[ref_df['year'] == 2019].loc[:,'divisionnm']
        prob_df.rename(columns={0:'is_left_pct',1:'is_right_pct'},inplace=True)

        prob_df['is_left_pct'] = prob_df['is_left_pct'].apply(lambda x: '{0:.4f}'.format(x*100)).apply(float)
        prob_df['is_right_pct'] = prob_df['is_right_pct'].apply(lambda x: '{0:.4f}'.format(x*100)).apply(float)

        ypred = model.predict(Xtest)
        
        prob_df['predicted'] = ypred
        prob_df['predicted'] = prob_df['predicted'].apply(lambda x: 'right' if x==1 else 'left')
        
        prob_df['actual'] = ytest
        prob_df['actual'] = prob_df['actual'].apply(lambda x: 'right' if x==1 else 'left')
        
        return prob_df

    def roc_curve_grapher(test,predictions,title=''):
        # For class 1, find the area under the curve
        fpr, tpr, _ = roc_curve(test, predictions)
        roc_auc = auc(fpr, tpr)

        # Plot of a ROC curve for class 1 (has_cancer)
        plt.figure(figsize=[8,8])
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.title(title, fontsize=18)
        plt.legend(loc="lower right")
        plt.show()

    def finder(probability_table,df,coefs): 
        """
        Description:
        This function takes in a specified electorate and outputs the probability of the left or right leaningness of 
        that specific electorate. The probability is based off of the generated 'probability table' from another function.
        The coefficients are based off of the outputs from the specified model, while the reference data frame is used to
        generate the right leaning means and the means of the specified electorate.
        """
        while True:
            key = input('Enter your electorate name: ')
            record = probability_table[probability_table['electorate'].apply(lambda x:str(x).lower()) == key.lower()]
            left = coefs.sort_values(by='Value',ascending=False).tail(10).index
            right = coefs.sort_values(by='Value',ascending=False).head(10).index
            gr_df = df.groupby('divisionnm').mean()
            
            def percentiler(x,col,df=gr_df):
                ranked = df[col].sort_values().reset_index(drop = True)
                ranked_tab = pd.DataFrame({"data": ranked})
                return len(ranked_tab[ranked_tab['data']<= x])/len(df[col])*100
            
            def bar_labeller(ax,spacing=5,lsize=10):
                """Description:
                A function that passes in the matplotlib object and labels the bar. This only works for bar charts, 
                and is meant to generate neat labels at the top of each bar.

                """
                for rect in ax.patches:
                    y_value = rect.get_height()
                    x_value = rect.get_x() + rect.get_width() / 2
                    space = spacing
                    label = "{0:.2f}%".format(y_value)
                    va = 'bottom'
                    ax.annotate(
                        label,
                        (x_value,y_value),
                        xytext=(0,space),
                        textcoords = 'offset points',
                        ha='center',
                        va=va,
                        fontsize=lsize)

            try:
                if record.iloc[0,0] > record.iloc[0,1]:
                    print('The seat of', record.iloc[0,2], 'was predicted to have a',record.iloc[0,0], '% probability of being left leaning')
                    print('The 2019 election saw', record.iloc[0,2], 'as voting', record.iloc[0,4], 'leaning')
                    print('')
                    print('How your seat scored on top 10 left leaning predictors:')
                    print('------')
                    top_10L = df[df['divisionnm'] == record.iloc[0,2]][left].mean()
                    top_10L_ind = pd.DataFrame(top_10L).index
                    for x,y in zip(top_10L_ind,top_10L):
                        print(x + ' feature was in the ' + str(round(percentiler(y,x),2)) + 'th percentile.')
                        
                    left_avg = pd.DataFrame(df.groupby('is_right').mean()[left].loc[0])
                    left_avg.rename(columns={0.0:'Left leanng average'},inplace=True)
                    
                    rec_L = pd.DataFrame(df[df['divisionnm'] == record.iloc[0,2]][left].mean(),columns=[record.iloc[0,2]])
                    ax = pd.concat([left_avg,rec_L],axis=1).plot(kind='bar',figsize=(12,7),title='How your seat compares to the average')
                    bar_labeller(ax)
                    ax.set_xticklabels(pd.concat([left_avg,rec_L],axis=1).index,rotation=75,fontsize=10)
                    plt.show()
                    break
                elif record.iloc[0,0] < record.iloc[0,1]:
                    print('The seat of', record.iloc[0,2], 'was predicted to have a',record.iloc[0,1], '% probability of being right leaning')
                    print('The 2019 election saw',record.iloc[0,2],'as voting',record.iloc[0,4],'leaning')
                    print('')
                    print('How your seat scored on top 10 predictors right leaning predictors:')
                    print('------')
                    top_10R = df[df['divisionnm'] == record.iloc[0,2]][right].mean()
                    top_10R_ind = pd.DataFrame(top_10R).index
                    for x,y in zip(top_10R_ind,top_10R):
                        print(x + ' feature was in the ' + str(round(percentiler(y,x),2)) + 'th percentile.')
                    
                    right_avg = pd.DataFrame(df.groupby('is_right').mean()[right].loc[1])
                    right_avg.rename(columns={1.0:'Right leanng average'},inplace=True)
                    
                    rec_R = pd.DataFrame(df[df['divisionnm'] == record.iloc[0,2]][right].mean(),columns=[record.iloc[0,2]])
                    ax = pd.concat([right_avg,rec_R],axis=1).plot(kind='bar',figsize=(12,7),title='How your seat compares to the average')
                    bar_labeller(ax)
                    ax.set_xticklabels(pd.concat([right_avg,rec_R],axis=1).index,rotation=75,fontsize=10)
                    plt.show()
                    break
                else:
                    print('Seat is equally left and right')
                    break
            except IndexError:
                print('Electorate not found, please try again.')

    def logit_convert(x):
        odds = np.exp(x)
        p = odds / (1+ odds)
        return p

    def coef_grapher(df,size=(12,7),title=''):
        high = df.sort_values(by='Value',ascending=False).head(10)
        low = df.sort_values(by='Value',ascending=False).tail(10)

        barsh = high['Value'].index
        barsl = low['Value'].index
        bars = barsh.append(barsl)
        y_pos = np.arange(len(bars))
        height = list(high['Value'].values) + list(low['Value'].values)

        plt.figure(figsize=size)
        plt.barh(y_pos,height,color=['blue','blue','blue','blue','blue','blue','blue','blue','blue','blue',
                                'red','red','red','red','red','red','red','red','red','red'])
        plt.yticks(y_pos,bars)
        plt.title(title,fontsize=18)
        plt.show()

    def corr_finder(df,up_bound,low_bound):
        #Taking the correlation matrix and unstacking it, then sorting it from highest/lowest absolute values
        s = df.corr().unstack().sort_values(kind="quicksort",ascending=False)
        #Putting it into a dataframe
        corr_df = pd.DataFrame(s)
        #Renaming column to be clearer
        corr_df.rename(columns={0:'Correlation'},inplace=True)
        #Applying the row mask where correlation is less what we specify
        corr_id = corr_df[(corr_df['Correlation'] < up_bound) & (corr_df['Correlation'] >= low_bound)]
        return corr_id

    def var_checker(df,thresh):
        """
        Creating a for loop that loops through all the columns and return the column name and value count with the proportion
        of its largest category greater than 90%. This will help us in selecting columns that will fudge our model with
        low variance.
        """
        for col in df:
            #Checks if the values in the column are 'numeric', if it is numeric it is assumed to be a continuous variable
            #and therefore excluded, we only want categorical variables
            if all(df[col].apply(lambda x:str(x).isnumeric())) == False:
                if df[col].value_counts(normalize=True).max() > thresh:
                    print(col) 
                    print(dict(df[col].value_counts()))
                    print('-------------------')

    def col_merger(col1,col2,data,merged_col):
        data[merged_col] = data[col1] + data[col2]
        data.drop(columns=[col1,col2],inplace=True)

    
    