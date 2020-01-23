from statistics import mean

class toolbox():
    def axis_cleaner(x):
        """Description:
        This function takes the x-axis label of my graphs and removes all labels that are not March.
        
        """
        
        if not x[:3] == 'Mar':
            return ''
        else:
            return x
        
    def index_fixer(x):
        """Description:
        This is a simple function designed only to change the ABS' 7 into a 3 to make it more consistent 
        with the other data frames. Only created to make the dictionary for 'All sectors' in the wage dataframe
        to be equal to '3' instead of '7'. 
        
        """
        
        if x == 7:
            return int(3)
        else:
            return x
        
    def formatter(x):
        """Description:
        A simple function that rounds every item in a row to 2 decimal places, for easier reading.
        
        """
        
        return "{0:.2f}".format(x)
    
    def generate_growth_rate_list(*arg):
        """
        Description:
        This method returns a list of growth rates for any number of indices. This is useful for calculating a growth
        rate between two points or for any number of points with the use of *args. This method allows for flexibility incase
        I want to find a list of growth rates over a certain period or just the growth between two periods.
        """
        
        gen_growth_rate_list = []

        for x,y in zip(arg[::1], arg[1::1]):
            z = (y - x) / x
            gen_growth_rate_list.append(z)

        return gen_growth_rate_list
    
    def bar_labeller(ax,spacing=5):
        """Description:
        A function that passes in the matplotlib object and labels the bar. This only works for bar charts, 
        and is meant to generate neat labels at the top of each bar.
        
        """
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            space = spacing
            label = "{}%".format(y_value)
            va = 'bottom'
            ax.annotate(
                label,
                (x_value,y_value),
                xytext=(0,space),
                textcoords = 'offset points',
                ha='center',
                va=va,
                fontsize=12)
            
class growth_calculator():
    def __init__(self, column_list):
        """Instantiates the column list (i.e. the list of values that forms the value list"""
        self.column_list = column_list
    
    def get_pa_growth(self):
        """Description:
        A function that returns the per annum growth of a passed in values list as a starting point. 
        
        The function takes in the original index values list, the for loop then transforms this into a list of 
        percentage growth values between ((x1 - x0) / (x0)) for every value in the list. 
        
        Once the growth rate list is filled it then applies those rates to a baseline 100 index to generate a new
        index list. Once the index list is generated those values are plugged into a formula to calculate the per
        annum growth. This formula is based off of an industry standard formula that is used in the financial industry
        (mainly in managed funds).
        
        The purpose of this is to plot this new index list in a graph where every index starts at 100, 
        this is a more even comparison to show how wages/housing/cpi have grown differently over the time period.
        """
        
        #Values list is the original index from the ABS, the purpose of this is to set a new index starting at 100. 
        values_list = self.column_list.values
        growth_rate_list = []

        #The for loop iterates through a list that doesn't include the first value as the formula for % growth relies
        #on a previous entry. The end result is a list of growth rates between the values we had earlier. 
        for x, y in zip(values_list[::1], values_list[1::1]):
            z = (y - x) / x
            growth_rate_list.append(z)

        #For the per annum growth formula to work properly, we need our t=0 index to be baseline 100 and apply the growth
        #rates to each point in the index. The index list we will have will allow us to calculate pa growth properly. 
        index_list = [100]
        for num in range(len(growth_rate_list)):  
            i = index_list[num] + (index_list[num] * growth_rate_list[num])
            index_list.append(i)

        #Now that we have the index list we can use the pa growth formula.
        #The equation is based off of the financial formula for compounding interest.
        return (index_list[len(index_list)-1]/index_list[0])**(1/((len(index_list)-2)/4))-1
        
    def get_avg_growth(self):
        """Description:
        A function that returns the simple average growth of a passed in values list as a starting point. 
        
        The function takes in the original index values list, the for loop then transforms this into a list of 
        percentage growth values between ((x1 - x0) / (x0)) for every value in the list. 
        
        Once that is generated we just take a simple mean of the list (via the statistics module).
        
        """
        #Values list is the original index from the ABS, the purpose of this is to set a new index starting at 100. 
        values_list = self.column_list.values
        growth_rate_list = []

        #The for loop iterates through a list that doesn't include the first value as the formula for % growth relies
        #on a previous entry. The end result is a list of growth rates between the values we had earlier. 
        for x, y in zip(values_list[::1], values_list[1::1]):
            z = (y - x) / x
            growth_rate_list.append(z)

        #Average growth is just a simple mean of the growth rates we generated before. 
        return mean(growth_rate_list)
    
    def get_index_list(self):
        """Description:
        A function that returns a new rebalanced at 100 index list from a passed in original index list. 
        
        """
        #Values list is the original index from the ABS, the purpose of this is to set a new index starting at 100. 
        values_list = self.column_list.values
        growth_rate_list = []

        #The for loop iterates through a list that doesn't include the first value as the formula for % growth relies
        #on a previous entry. The end result is a list of growth rates between the values we had earlier. 
        for x, y in zip(values_list[::1], values_list[1::1]):
            z = (y - x) / x
            growth_rate_list.append(z)

        #For the per annum growth formula to work properly, we need our t=0 index to be baseline 100 and apply the growth
        #rates to each point in the index. The index list we will have will allow us to calculate pa growth properly. 
        index_list = [100]
        for num in range(len(growth_rate_list)):  
            i = index_list[num] + (index_list[num] * growth_rate_list[num])
            index_list.append(i)
            
        return index_list
        