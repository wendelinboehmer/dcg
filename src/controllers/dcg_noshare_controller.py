import torch as th
import contextlib
from .cg_controller import DeepCoordinationGraphMAC
from modules.agents import REGISTRY as agent_REGISTRY
import itertools
import numpy as np


class DCGnoshareMAC (DeepCoordinationGraphMAC):
    """ Implements DCG without any parameter sharing between agents (Boehmer et al., 2020). """

    # ================================ Constructors ===================================================================

    def __init__(self, scheme, groups, args):
        DeepCoordinationGraphMAC.__init__(self, scheme, groups, args)
        # New utilities and payoffs
        self.utility_fun = [DeepCoordinationGraphMAC._mlp(self.args.rnn_hidden_dim,
                                                          args.cg_utilities_hidden_dim,
                                                          self.n_actions)
                            for _ in range(self.n_agents)]
        payoff_out = 2 * self.payoff_rank * self.n_actions if self.payoff_decomposition else self.n_actions ** 2
        self.payoff_fun = [DeepCoordinationGraphMAC._mlp(2*self.args.rnn_hidden_dim, args.cg_payoffs_hidden_dim,
                                                         payoff_out) for _ in range(len(self.edges_from))]

    # ================== DCG Core Methods =============================================================================

    def annotations(self, ep_batch, t, compute_grads=False, actions=None):
        """ Returns all outputs of the utility and payoff functions. """
        with th.no_grad() if not compute_grads else contextlib.suppress():
            # Compute all hidden states
            agent_inputs = self._build_inputs(ep_batch, t).view(ep_batch.batch_size, self.n_agents, -1)
            for i, ag in enumerate(self.agent):
                self.hidden_states[i] = ag(agent_inputs[:, i, :], self.hidden_states[i])\
                                            [1].view(ep_batch.batch_size, -1)
            # Compute all utility functions
            f_i, f_ij = [], []
            for i, f in enumerate(self.utility_fun):
                f_i.append(f(self.hidden_states[i]).reshape(ep_batch.batch_size, -1))
            f_i = th.stack(f_i, dim=-2)
            # Compute all payoff functions
            if len(self.payoff_fun) > 0:
                for i, f in enumerate(self.payoff_fun):
                    f_ij.append(self.single_payoff(f, i, self.hidden_states))
                f_ij = th.stack(f_ij, dim=-3)
            else:
                f_ij = f_i.new_zeros(*f_i.shape[:-2], 0, self.n_actions, self.n_actions)
        return f_i, f_ij

    def single_payoff(self, payoff_fun, edge, hidden_states):
        """ Computes one payoff at a time, as each payoff function does have different parameters. """
        # Construct the inputs for all edges' payoff functions and their flipped counterparts
        n = self.n_actions
        inputs = th.stack([th.cat([hidden_states[self.edges_from[edge]], hidden_states[self.edges_to[edge]]], dim=-1),
                           th.cat([hidden_states[self.edges_to[edge]], hidden_states[self.edges_from[edge]]], dim=-1)],
                          dim=0)
        # Without action batching, all payoffs of a sample are computed at once
        output = payoff_fun(inputs)
        if self.payoff_decomposition:
            # If the payoff matrix is decomposed, we need to de-decompose it here: ...
            dim = list(output.shape[:-1])
            # ... reshape output into left and right bases of the matrix, ...
            output = output.view(*[np.prod(dim) * self.payoff_rank, 2, n])
            # ... outer product between left and right bases, ...
            output = th.bmm(output[:, 0, :].unsqueeze(dim=-1), output[:, 1, :].unsqueeze(dim=-2))
            # ... and finally sum over the above outer products of payoff_rank base-pairs.
            output = output.view(*(dim + [self.payoff_rank, n, n])).sum(dim=-3)
        else:
            # Without decomposition, the payoff_fun output must only be reshaped
            output = output.view(*output.shape[:-1], n, n)
        # The output of the backward messages must be transposed
        output[1] = output[1].transpose(dim0=-2, dim1=-1)
        # Compute the symmetric average of each edge with it's flipped counterpart
        return output.mean(dim=0)

    # ================== Override methods of DeepCoordinationGraphMAC =================================================

    def _build_agents(self, input_shape):
        """ Overloads method to build a list of input-encoders for the different agents. """
        self.agent = [agent_REGISTRY[self.args.agent](input_shape, self.args) for _ in range(self.n_agents)]

    def cuda(self):
        """ Overloads method to make sure all encoders, utilities and payoffs are on the GPU. """
        for ag in self.agent:
            ag.cuda()
        for f in self.utility_fun:
            f.cuda()
        for f in self.payoff_fun:
            f.cuda()
        if self.edges_from is not None:
            self.edges_from = self.edges_from.cuda()
            self.edges_to = self.edges_to.cuda()
            self.edges_n_in = self.edges_n_in.cuda()
        if self.duelling:
            self.state_value.cuda()

    def parameters(self):
        """ Overloads method to make sure the parameters of all encoders, utilities and payoffs are returned. """
        param = itertools.chain(*[ag.parameters() for ag in self.agent],
                                *[f.parameters() for f in self.utility_fun],
                                *[f.parameters() for f in self.payoff_fun])
        if self.duelling:
            param = itertools.chain(param, self.state_value.parameters())
        return param

    def load_state(self, other_mac):
        """ Overloads method to make sure the parameters of all encoders, utilities and payoffs are swapped. """
        for i in range(len(self.agent)):
            self.agent[i].load_state_dict(other_mac.agent[i].state_dict())
        for i in range(len(self.utility_fun)):
            self.utility_fun[i].load_state_dict(other_mac.utility_fun[i].state_dict())
        for i in range(len(self.payoff_fun)):
            self.payoff_fun[i].load_state_dict(other_mac.payoff_fun[i].state_dict())
        if self.duelling:
            self.state_value.load_state_dict(other_mac.state_value.state_dict())

    def init_hidden(self, batch_size):
        """ Overloads method to make sure the hidden states of all agents are intialized. """
        self.hidden_states = [ag.init_hidden().expand(batch_size, -1) for ag in self.agent]  # bv
