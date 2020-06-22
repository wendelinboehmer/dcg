import torch as th
import torch.nn as nn
import contextlib
import itertools
from .basic_controller import BasicMAC
from modules.agents import REGISTRY as agent_REGISTRY

class LowRankMAC (BasicMAC):
    """ Implements a low-rank Q-value approximation for MARL (Boehmer et al., 2020)."""

    # ================================ Constructors ===================================================================

    def __init__(self, scheme, groups, args):
        BasicMAC.__init__(self, scheme, groups, args)
        # If fully_observable, the agent requires a new RNN with fitting inputs
        if self.args.fully_observable:
            input_shape = scheme["obs"]["vshape"]
            if self.args.obs_last_action:
                input_shape += scheme["actions_onehot"]["vshape"][0]
            self.agent = agent_REGISTRY[self.args.agent](input_shape * self.n_agents, self.args)
        # Create function that computes for each agent and rank*actions outputs
        output_dim = args.n_actions * args.low_rank * (self.n_agents if self.args.fully_observable else 1)
        self.factor_fun = nn.Linear(args.rnn_hidden_dim, output_dim)
        # Add VDN utilities if specified
        if args.add_utilities:
            output_dim = args.n_actions * (self.n_agents if self.args.fully_observable else 1)
            self.utility_fun = nn.Linear(args.rnn_hidden_dim, output_dim)
        self.device = self.factor_fun.weight.device
        # Create indices for "all agents but a" for the argmax
        self.idx = th.LongTensor([i for i in range(1, args.n_agents)], device=self.device)
        self.idx = self.idx.unsqueeze(dim=0).repeat(args.n_agents, 1)
        for i in range(1, args.n_agents):
            self.idx[i, :i] = th.LongTensor([j for j in range(i)], device=self.device)

    # ================================ Overloaded forward function ====================================================

    def forward(self, ep_batch, t, actions=None, policy_mode=True, test_mode=False, compute_grads=False):
        """ Computes policy, greedy actions or Q-values in the same format as the CoordinationGraphMAC. """
        with th.no_grad() if not compute_grads or policy_mode else contextlib.suppress():
            agent_inputs = self._build_inputs(ep_batch, t)
            avail_actions = ep_batch["avail_actions"][:, t]
            _, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
            # There are 3 different modes to call this function
            if policy_mode:
                # Return a greedy "policy" tensor
                actions = self.greedy(avail_actions)
                policy = self.hidden_states.new_zeros(ep_batch.batch_size, self.n_agents, self.args.n_actions)
                policy.scatter_(dim=-1, index=actions, src=policy.new_ones(1, 1, 1).expand_as(actions))
                return policy
            if actions is None:
                # Return greedy "actions" tensor
                actions = self.greedy(avail_actions)
                return actions
            else:
                # Return the "values" for the given actions tensor
                values = self.q_values(actions)
                return values

    # ================================ Controller-specific methods ====================================================

    def q_factors(self, batch_size):
        factors = self.factor_fun(self.hidden_states)
        factors = factors.view(batch_size, self.n_agents, self.args.low_rank, self.args.n_actions)
        return factors

    def utilities(self, batch_size):
        utilities = self.utility_fun(self.hidden_states)
        utilities = utilities.view(batch_size, self.n_agents, self.args.n_actions)
        return utilities

    def q_values(self, actions):
        """ Computes Q-values for a given batch of actions and the current self.hidden_state. """
        factors = self.q_factors(actions.shape[0])
        factors = factors.gather(dim=-1, index=actions.expand(factors.shape[:-1]).unsqueeze(dim=-1)).squeeze(dim=-1)
        values = factors.prod(dim=-2).sum(dim=-1)
        if self.args.add_utilities:
            values = values + self.utilities(actions.shape[0]).gather(dim=-1, index=actions).squeeze(dim=-1).sum(dim=-1)
        return values

    def greedy(self, available_actions=None, policy_mode=False):
        dims = self.hidden_states.shape[:-1] if available_actions is None else available_actions.shape[:-1]
        unavailable_actions = None if available_actions is None else available_actions == 0
        # Initialize actions randomly and all max_values as -inf
        actions = th.randint(self.args.n_actions, (*dims, 1), device=self.device)
        max_actions = actions
        max_values = self.hidden_states.new_ones(*dims, 1) * (-float('inf'))
        # Compute factors/utilities for all actions
        factors = self.q_factors(dims[0])
        if self.args.add_utilities:
            utilities = self.utilities(dims[0])
        # Iteratively improve the selected actions
        for _ in range(self.args.max_iterations):
            # get the factors for all current actions
            values = factors.gather(dim=-1, index=actions.expand(factors.shape[:-1]).unsqueeze(dim=-1))
            # compute factors for each batch, agent and action
            idx = self.idx.unsqueeze(dim=0).unsqueeze(dim=3).expand(*dims, self.n_agents - 1, self.args.low_rank)
            values = values.squeeze(dim=3).unsqueeze(dim=1).expand(*dims, self.n_agents, self.args.low_rank)
            values = values.gather(dim=2, index=idx).prod(dim=2).unsqueeze(dim=-1)
            values = factors * values
            # get Q-values for each batch, agent and action by summing all factors
            values = values.sum(dim=-2)
            # add utilities if specified
            if self.args.add_utilities:
                values = values + utilities
            # mask out unavailable actions
            if unavailable_actions is not None:
                values.masked_fill_(unavailable_actions, -float('inf'))
            # select actions that maximize the Q-values for each batch and agent
            values, actions = values.max(dim=-1, keepdim=True)
            # anytime extension (Kok and Vlassis, 2006)
            select = values > max_values
            max_values[select] = values[select]
            max_actions[select] = actions[select]
            # early break if no improvement has been found
            if not select.any():
                break
        # Return the best actions found in the coordinate descend loop
        return max_actions

    def _build_inputs(self, batch, t):
        if not self.args.fully_observable:
            return BasicMAC._build_inputs(self, batch, t)
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

    # ================================ Overloaded background methods ==================================================

    def init_hidden(self, batch_size):
        if self.args.fully_observable:
            self.hidden_states = self.agent.init_hidden().expand(batch_size, -1)  # bv
        else:
            BasicMAC.init_hidden(self, batch_size)

    def cuda(self):
        self.agent.cuda()
        self.factor_fun.cuda()
        if self.args.add_utilities:
            self.utility_fun.cuda()
        self.idx = self.idx.cuda()
        self.device = self.factor_fun.weight.device

    def parameters(self):
        param = itertools.chain(BasicMAC.parameters(self), self.factor_fun.parameters())
        if self.args.add_utilities:
            param = itertools.chain(param, self.utility_fun.parameters())
        return param

    def load_state(self, other_mac):
        BasicMAC.load_state(self, other_mac)
        self.factor_fun.load_state_dict(other_mac.factor_fun.state_dict())
        if self.args.add_utilities:
            self.utility_fun.load_state_dict(other_mac.utility_fun.state_dict())

    def save_models(self, path):
        BasicMAC.save_models(self, path)
        th.save(self.factor_fun.state_dict(), "{}/factors.th".format(path))
        if self.args.add_utilities:
            th.save(self.utility_fun.state_dict(), "{}/utilities.th".format(path))

    def load_models(self, path):
        BasicMAC.load_models(self, path)
        self.factor_fun.load_state_dict(th.load("{}/factors.th".format(path),
                                                map_location=lambda storage, loc: storage))
        if self.args.add_utilities:
            self.utility_fun.load_state_dict(th.load("{}/utilities.th".format(path),
                                                     map_location=lambda storage, loc: storage))
