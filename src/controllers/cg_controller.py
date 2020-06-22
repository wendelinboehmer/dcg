import torch as th
import contextlib
from .dcg_controller import DeepCoordinationGraphMAC
from modules.agents import REGISTRY as agent_REGISTRY


class SimpleCoordionationGraphMAC (DeepCoordinationGraphMAC):

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        # New RNN
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        self.agent = agent_REGISTRY[self.args.agent](input_shape * self.n_agents, self.args)
        # New utilities and payoffs
        self.utility_fun = super()._mlp(self.args.rnn_hidden_dim,
                                        args.cg_utilities_hidden_dim,
                                        self.n_actions * self.n_agents)
        self.payoff_fun = super()._mlp(self.args.rnn_hidden_dim,
                                       args.cg_payoffs_hidden_dim,
                                       (self.n_actions ** 2) * len(self.edges_from))

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().expand(batch_size, -1)  # bv

    def annotations(self, ep_batch, t, compute_grads=False, actions=None):
        """ Returns all outputs of the utility and payoff functions. Override to use a GNN. """
        with th.no_grad() if not compute_grads else contextlib.suppress():
            agent_inputs = self._build_inputs(ep_batch, t)
            self.hidden_states = self.agent(agent_inputs, self.hidden_states)[1].view(ep_batch.batch_size, -1)
            f_i = self.utilities(self.hidden_states).reshape(ep_batch.batch_size, self.n_agents, self.n_actions)
            f_ij = self.payoffs(self.hidden_states)
        return f_i, f_ij

    def payoffs(self, hidden_states):
        n = self.n_actions
        output = self.payoff_fun(hidden_states)
        return output.view(*output.shape[:-1], len(self.edges_from), n, n)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t].view(bs, -1))  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]).view(bs, -1))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1].view(bs, -1))
        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        return inputs
