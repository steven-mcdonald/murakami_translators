class Modelling:

    def __init__(self, df):
        self.df = df

    def feature_select(self, basic_counts=True, vader=False, pos_counts=False,
                   words=False, adv=False, adj=False):
        '''create column list depending on features to include in the modelling'''
        columns = []
        if basic_counts:
            columns += [i for i in self.df.columns if i.startswith('n_') & i.endswith('_norm')]
        if vader:
            columns += [i for i in self.df.columns if i.startswith('vader_')]
        if pos_counts:
            columns += [i for i in self.df.columns if i.endswith('_count_norm')]
        if words:
            columns += [i for i in self.df.columns if i.endswith('_w')]
        if adj:
            columns += [i for i in self.df.columns if i.endswith('_adj')]
        if adv:
            columns += [i for i in self.df.columns if i.endswith('_adv')]
        return columns
