import copy

try:
    from ConfigSpace import ConfigurationSpace, Float, Integer
    from smac import MultiFidelityFacade, Scenario
except ImportError:
    raise ImportError("Please install graphbench with the [tuning] extra to use this feature.")

class Optimize():

    def __init__(self, args, train, cs=None) -> None:
        self.args = args
        self.train = train
        self.cs = cs
        #Given a training function compatible with the SMAC framework this method
        #allows for optimization using SMAC with multi-fidelity support
        #The training function must take args (from argparse) and a budget (either steps or epochs) as input and return an evaluation metric.
        #See: https://github.com/automl/SMAC3 for more details


    def optimize_model(self,):
        self._optimize(self.args)



    def _optimize(self):

        self._tune()

    def _get_configuration_space(self, cs):
        
        if self.cs is None:
            return self._get_default_configuration_space(self.args)
        else: 
            return cs


    def _get_default_configuration_space(self, args):
        cs = ConfigurationSpace(
            space={
                "learning_rate": Float("learning_rate", (1e-6, 1e-1), log=True),
                "weight_decay": Float("weight_decay", (1e-8, 1e-1), log=True),
                "warmup_iters": Integer("warmup_iters", (1000, args.num_steps // 5)),
                "dropout": Float("dropout", (0.0, 0.5)),
            }
        )
        return cs


    def _tune(self):
        #if not args.disable_task_defaults:
        #    args = load_task_defaults(args)

        #args.disable_task_defaults = True
        self.config = self._get_configuration_space(self.cs)
        for hp in self.cs:
            delattr(self.args, hp)

        scenario = Scenario(
            configspace=self.config,
            output_directory=self.args.path + "/smac",
            min_budget=self.args.min_fidelity,
            max_budget=self.args.num_steps,
            deterministic=True,
            n_trials=self.args.trials,
            seed=self.args.seed,
        )


        smac = MultiFidelityFacade(scenario=scenario, target_function=self._target_function)

        smac.optimize()

    def _target_function(self,config,seed, budget):
            cur_args = copy.deepcopy(self.args)
            for hp in self.cs:
                setattr(cur_args, hp, config[hp])

            return self.train(cur_args, int(budget))
    