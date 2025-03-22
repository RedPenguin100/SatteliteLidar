class MLSResult:
    def __init__(self):
        self.time = None
        self.error = None


class Experiment:
    def __init__(self):
        self.result_i = None
        self.result_plus = None
        self.result_union = None

        self.n = None
        self.m = None
        self.delta = None
        self.base_function = None


class AggregatedExperiment:
    def __init__(self):
        self.results_i = []
        self.results_plus = []
        self.results_union = []
        self.base_function = None
        self.m = None
        self.n = None
        self.delta = None

    def add(self, experiment: Experiment):
        assert len(self.results_i) == len(self.results_plus)
        assert len(self.results_i) == len(self.results_union)
        self.results_i.append(experiment.result_i)
        self.results_plus.append(experiment.result_plus)
        self.results_union.append(experiment.result_union)

        if self.base_function is None:
            self.base_function = experiment.base_function
        elif self.base_function != experiment.base_function:
            self.base_function = "INVALID"

        if self.m is None:
            self.m = experiment.m
        elif self.m != experiment.m:
            self.m = "INVALID"

        if self.n is None:
            self.n = experiment.n
        elif self.n != experiment.n:
            self.n = "INVALID"

        if self.delta is None:
            self.delta = experiment.delta
        elif self.delta != experiment.delta:
            self.delta = "INVALID"

    def size(self):
        return len(self.results_i)

    def get_times(self):
        mls_i_time = 0
        mls_plus_time = 0
        mls_union_time = 0
        for i in range(len(self.results_i)):
            mls_i_time += self.results_i[i].time
            mls_plus_time += self.results_plus[i].time
            mls_union_time += self.results_union[i].time

        return (mls_i_time, mls_plus_time, mls_union_time)

    def get_error_average(self, error='rmse'):
        mls_i_error = 0
        mls_plus_error = 0
        mls_union_error = 0

        for i in range(len(self.results_i)):
            if error == 'rmse':
                mls_i_error += self.results_i[i].error.root_err
                mls_plus_error += self.results_plus[i].error.root_err
                mls_union_error += self.results_union[i].error.root_err
            elif error == 'score':
                mls_i_error += self.results_i[i].error.score
                mls_plus_error += self.results_plus[i].error.score
                mls_union_error += self.results_union[i].error.score
            else:
                raise ValueError(f"Unknown error type: {error}")

        return mls_i_error, mls_plus_error, mls_union_error


class ExperimentStore:
    def __init__(self):
        self.experiments = dict()

    def add_experiment(self, experiment: AggregatedExperiment):
        if (experiment.n, experiment.m, experiment.delta, experiment.base_function) in self.experiments:
            self.experiments[(experiment.n, experiment.m, experiment.delta, experiment.base_function)].add(experiment)
        else:
            self.experiments[(experiment.n, experiment.m, experiment.delta, experiment.base_function)] = experiment

    def get_error_df(self, errors=['rmse', 'score']):
        import pandas as pd

        base_columns = ['n', 'm', 'delta', 'base_function']
        algo_columns = ['mls_i', 'mls_plus', 'mls_union']
        columns = base_columns
        for error in errors:
            for column in algo_columns:
                columns.append(f"{column}_{error}")

        df = pd.DataFrame(columns=columns)

        for setting, experiment in self.experiments.items():
            n, m, delta, base_function = setting
            base_values = [n, m, delta, base_function]
            for error in errors:
                error_averages = experiment.get_error_average(error)
                base_values.extend(error_averages)
            df.loc[len(df)] = base_values

        return df
