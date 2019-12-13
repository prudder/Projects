import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class graphy():
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
    
    def fig_grapher(df,group,feature,desc,sorter='',label=True,size=(12,7),rot=45):
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

        if len(feature) > 1:
            ax = df.groupby(group).mean()[feature].sort_values(by=sorter).plot(kind='bar',figsize=size)
            ax.set_xticklabels(df.groupby(group).mean()[feature].sort_values(by=sorter).index,rotation = rot,fontsize = 10)
        else:
            ax = df.groupby(group).mean()[feature[0]].sort_values().plot(kind='bar',figsize=size)
            ax.set_xticklabels(df.groupby(group).mean()[feature[0]].sort_values().index,rotation = rot,fontsize = 10)
        if label == True:
            bar_labeller(ax)
        plt.xlabel('')
        
        plt.title(desc,fontsize=15)
        plt.show()

    def two_line_grapher(a1,a2,t1='',t2='',title='',x='',size=(12,7)):
        fig, ax1 = plt.subplots(figsize=size)
        

        color = 'tab:red'
        ax1.set_xlabel(x,fontsize=14)
        ax1.set_ylabel(t1, color=color,fontsize=14)
        ax1.plot(a1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel(t2, color=color,fontsize=14)
        ax2.plot(a2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title(title,fontsize=18)
        plt.show()

    def party_profiler(df,feats,party,title='',size=(12,7),rot=45,label=True):
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

        p_ = df.groupby('partynm').mean()[feats].loc[party]
        u30 = pd.DataFrame(df.groupby('partynm').mean().loc[party][[col for col in df.columns if 'years' in col][:6]].sum(axis=1),columns=['under_30_pct'])
        o60 = pd.DataFrame(df.groupby('partynm').mean().loc[party][[col for col in df.columns if 'years' in col][12:]].sum(axis=1),columns=['over_60_pct'])
        inc = pd.DataFrame(df.groupby('partynm').mean().loc[party][[col for col in df.columns if 'income_' in col][8:10]].sum(axis=1),columns=['income_1500_plus_pct'])
        p_n = pd.concat([p_,u30,o60,inc],axis=1)
        ax = p_n.T.plot(kind='bar',figsize=size)
        ax.set_xticklabels(p_n.T.index,rotation = rot,fontsize = 10)
        plt.title(title,fontsize=18)
        plt.ylabel('Mean proportion',fontsize=12)
        if label == True:
            bar_labeller(ax)
        plt.show()

    