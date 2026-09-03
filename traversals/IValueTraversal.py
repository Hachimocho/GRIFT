"""The I-value traversal: one class, two walk strategies chosen by graph type.

This used to be four classes across two modules, and the split caused every bug found in
the traversal layer:

* ``IValueTraversalSubcluster`` and ``IValueTraversalClusterHopSubcluster`` selected an area
  by Louvain-community mean I-value. Both are **removed**. They never ran as designed: their
  outlier filter used ``v < mean + k*std`` and so excluded every node whenever the variance
  was zero, they were absent from ``CapabilityManager``'s allowlist and therefore ran on
  random I-values with no DQN at all, and they yielded 1 node per step against ~17 for the
  other two -- a 17x difference in training data that swamped any effect of the strategy
  itself. ``i-value-cluster-hop-subcluster`` additionally returned an empty batch on step
  one, which trained the model on nothing for three epochs in two consecutive sweeps while
  reporting an 82.5%-accurate result.
* The two survivors disagreed on details that had nothing to do with strategy: one
  pre-warmed ``i_values`` for every node at construction, the other left it empty; one fell
  back to a random node when selection failed, the other skipped the pointer.

So the strategy is now a branch rather than a subclass, and the shared machinery -- pointer
state, I-value caching, the fallbacks -- has exactly one implementation.

**Which walk runs is a property of the graph, not a separate traversal to pick.** A
clustered graph is built from disjoint race-gender groups, so a walk along edges cannot
leave the cluster it started in and needs to hop; an unclustered graph is connected, so
hopping would only discard the locality the I-value signal is meant to exploit. Passing
``cluster_hop`` overrides the detection.

Neither mode pre-warms I-values. The connected walk used to call
``trainer.get_i_value`` for every node in the graph, for every pointer, inside
``reset_pointers`` -- and again on every ``predictor_update_period``. That is one DQN forward
pass per node, so it was O(N x pointers) per refresh and simply not runnable on a large
graph. Both modes now fetch lazily through ``_get_i_value`` and cache what they touch.

The ``reset_pointers`` half of that was removed first and the periodic half was missed, so
the sweep survived on a timer: measured on the 562,214-node graph it cost ~6 min per
refresh and fired 6 times an epoch, which was 93% of the epoch. The timer now *clears* the
cache instead of refilling it, so staleness is still bounded but the cost is O(1).
"""

import random
from collections import defaultdict, deque

from nodes.atrnode import AttributeNode
from traversals.Traversal import Traversal

#: How `IValueTraversal` picks a candidate from the pool.
SELECTION_MODES = ("max", "band", "min", "midband")

#: Graph types whose clusters are disjoint, so a pointer cannot walk between them.
CLUSTERED_GRAPH_PREFIX = "clustered"


class IValueTraversal(Traversal):
    """Moves pointers toward information-rich nodes, hopping between clusters when the
    graph is built from disjoint ones.
    """

    tags = ["attributes", "i-value"]

    hyperparameters = {
        "parameters": {
            "steps": {"distribution": "int_uniform", "min": 100, "max": 500},
            "return_delay": {"distribution": "int_uniform", "min": 10, "max": 100},
            "warp_chance": {"distribution": "uniform", "min": 0.0, "max": 0.999},
            "predictor_update_period": {"distribution": "int_uniform", "min": 10, "max": 100},
            "bias_hop_period": {"distribution": "int_uniform", "min": 50, "max": 500},
        }
    }

    def __init__(self, graph, num_pointers, num_steps, trainer=None, return_delay=10,
                 warp_chance=0.005, predictor_update_period=50, bias_hop_period=100,
                 pessimistic_i_value=1.0, neutral_i_value=0.5, cluster_hop=None,
                 candidate_pool=0, selection_mode="max", selection_band=(0.4, 0.7),
                 group_targeting=None):
        """
        Args:
            cluster_hop: force the cluster-hopping walk on (True) or off (False). ``None``
                detects it from the graph, which is what the CLI does -- the two walks are
                not independently useful, they match two graph constructions.
            pessimistic_i_value: default for an unmeasured node in cluster-hop mode. High,
                so an unvisited node looks worth visiting.
            neutral_i_value: default for an unmeasured node in connected mode, preserving
                that walk's original 0.5.
        """
        super().__init__()
        self.graph = graph
        self.num_pointers = num_pointers
        self.num_steps = num_steps
        self.return_delay = return_delay
        self.t = 0
        self.warp_chance = warp_chance
        # Extra nodes drawn uniformly and considered alongside the current node's
        # neighbours. The argmax otherwise ranges over ~8 graph neighbours, which on a
        # measured average degree of 8.35 with `warp_chance=0.005` is a very weak selection
        # pressure -- and because neighbours are joined *by similarity* their I-values are
        # correlated, so the effective choice is narrower still. 0 keeps the old behaviour.
        self.candidate_pool = max(0, int(candidate_pool or 0))
        # How a candidate is picked from the pool.
        #
        # `max` is the historical behaviour and the premise of the whole method: take the
        # most informative candidate. `band` takes one from a quantile *range* instead --
        # the hypothesis being that the very hardest samples are outliers and label noise
        # on a 87.55%-imbalanced corpus, so the useful signal sits somewhere in the middle.
        # `min` is the deliberate opposite, kept as a control: if `min` also beats i.i.d.
        # then the ranking is not what is doing the work. `midband` is `band`'s smooth
        # analogue: instead of a hard cutoff at `selection_band`'s edges, every candidate is
        # eligible with a probability that rises from 0 at the low edge, plateaus over the
        # band's middle, and falls back to 0 at the high edge -- see
        # `trainers.capabilities.loss_weighting.trapezoid_desirability`, which this reuses
        # rather than reimplements, so the "avoid both extremes" shape cannot drift between
        # the selection and loss-weighting versions of the same idea.
        #
        # A quantile over the ~8 k-NN neighbours a bare walk sees is noise, so `band` and
        # `midband` are only meaningful with `candidate_pool` set -- Phase 0 measured those
        # neighbours at 2.3x less batch diversity than i.i.d., which is why the pool exists.
        if selection_mode not in SELECTION_MODES:
            raise ValueError(
                f"unknown selection_mode {selection_mode!r}; choose from "
                f"{', '.join(SELECTION_MODES)}"
            )
        self.selection_mode = selection_mode
        low, high = (float(selection_band[0]), float(selection_band[1]))
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                f"selection_band must be 0 <= low <= high <= 1, got {selection_band}"
            )
        self.selection_band = (low, high)
        # Restricts the drawn pool to demographic groups the model is weakest on,
        # while still drawing uniformly *within* them -- see group_targeting.py.
        self.group_targeting = group_targeting
        self.predictor_update_period = predictor_update_period
        self.trainer = trainer
        self.current_batch_nodes = []
        self.neutral_i_value = neutral_i_value

        # Cluster-hop state. Allocated in both modes so `get_hop_i_value_history` and the
        # visualization hooks do not have to branch on which walk is active.
        self.current_bias_hop_pointer_index = 0
        self.hop_i_value_history = []
        self.bias_hop_period = bias_hop_period
        self.pessimistic_i_value = pessimistic_i_value
        self.bias_attributes = ['Ground Truth Gender', 'Ground Truth Race', 'Ground Truth Age']

        self.cluster_hop = (
            self.detect_cluster_hop(graph) if cluster_hop is None else bool(cluster_hop)
        )
        self.reset_pointers()

    @staticmethod
    def detect_cluster_hop(graph):
        """Whether this graph needs hopping: True for a clustered construction.

        Reads `graph.graph_type`, which the dataloaders set. Defaults to False -- the
        connected walk -- because that is the safe answer for a graph whose construction is
        unknown: hopping a connected graph wastes the locality the I-value signal provides,
        while not hopping a clustered one merely confines each pointer to its own cluster.
        """
        graph_type = str(getattr(graph, 'graph_type', '') or '')
        return graph_type.startswith(CLUSTERED_GRAPH_PREFIX)

    def __len__(self):
        return self.num_steps

    def get_pointers(self):
        return self.pointers

    def get_current_batch_nodes(self):
        return [
            self.pointers[i]['current_node'] for i in range(self.num_pointers)
            if self.pointers[i]['current_node'] is not None
        ]

    def get_hop_i_value_history(self):
        """Average I-value per demographic subgroup at each hop. Empty in connected mode."""
        return self.hop_i_value_history

    def reset_pointers(self):
        """Place every pointer on a random node. I-values are fetched lazily.

        One shared implementation. The connected walk used to pre-warm `i_values` for the
        whole graph here -- O(N) DQN forward passes per pointer -- while the cluster-hop
        walk left the dict empty, and that difference is what made the removed subcluster
        variant see zero variance and select nothing.
        """
        self.t = 0
        self.pointers = []
        all_nodes = list(self.graph.get_nodes())
        if not all_nodes:
            print("Warning: No nodes found in graph during reset_pointers.")
            return

        for _ in range(self.num_pointers):
            self.pointers.append({
                'current_node': self.rng.choice(all_nodes),
                'last_visited': {},
                'i_values': {},
                'path': [],
                'last_node_id': None,
                'visited_nodes': set(),
                'steps': 0,
            })

    def _candidates(self, valid_neighbors, visited_this_batch):
        """The pool the I-value argmax ranges over.

        Draws from the traversal's own seeded `self.rng`, so widening the pool stays
        reproducible. Duplicates are filtered by node id rather than by object, because a
        drawn node may be the same graph node as a neighbour.
        """
        if not self.candidate_pool:
            return valid_neighbors

        seen = {n.node_id for n in valid_neighbors}
        extra = []
        for _ in range(self.candidate_pool):
            node = self.graph.get_random_node(rng=self.rng)
            if not isinstance(node, AttributeNode):
                continue
            if node.node_id in seen or node in visited_this_batch:
                continue
            targeting = self.group_targeting
            if targeting is not None and not targeting.is_targeted(node):
                continue
            seen.add(node.node_id)
            extra.append(node)
        return valid_neighbors + extra

    def _pick(self, candidates, i_values):
        """Choose one candidate according to `selection_mode`.

        `band` selects uniformly among the candidates whose I-value falls in the requested
        quantile range, so it is a *rank* criterion and unaffected by the I-value's scale --
        which matters because different estimators output wildly different ranges (the legacy
        family sits in a 0.02-wide band around 0.31; the fixed ones are unbounded).

        `midband` also ranks, then draws from *every* candidate with probability proportional
        to `trapezoid_desirability` -- 0 at and beyond `selection_band`'s edges, 1 on a
        plateau inside them. Unlike `band`, a candidate just outside the window is not
        impossible, only unlikely, and one just inside is not certain, only likely: the
        distribution is smooth rather than a hard yes/no.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        if self.selection_mode == "max":
            return candidates[i_values.index(max(i_values))]
        if self.selection_mode == "min":
            return candidates[i_values.index(min(i_values))]

        if self.selection_mode == "midband":
            from trainers.capabilities.loss_weighting import ivalue_weights

            # Reuses the loss-weighting module's `midband` shape rather than reimplementing
            # it, so this and `ivalue_loss_weight=midband` cannot silently diverge. `clip`
            # here only sets how sharply the plateau is preferred over the tails as a
            # *sampling* probability -- it has no relationship to any loss magnitude, so the
            # default is used rather than threading `--ivalue-weight-clip` through, which
            # belongs to the loss-weighting arm.
            weights = ivalue_weights(i_values, mode="midband", band=self.selection_band)
            if weights is None:
                return candidates[self.rng.randrange(len(candidates))]
            return candidates[self.rng.choices(
                range(len(candidates)), weights=weights.tolist(), k=1
            )[0]]

        # band: rank the candidates, keep the requested slice, draw from it.
        order = sorted(range(len(candidates)), key=lambda index: i_values[index])
        low, high = self.selection_band
        start = int(low * (len(order) - 1))
        end = int(high * (len(order) - 1))
        window = order[start:end + 1] or order
        return candidates[self.rng.choice(window)]

    def traverse(self, batch_size=32):
        """Collect a batch of nodes with the walk this graph calls for."""
        if self.cluster_hop:
            return self._traverse_cluster_hop(batch_size)
        return self._traverse_connected(batch_size)

    def _get_i_value(self, pointer_data, node, use_din=False, default=None,
                     fetch_on_miss=False):
            """Safely get the I-value for a node, from cache, the trainer, or a default.

            `default` lets each walk keep its own notion of an unmeasured node: the
            connected walk treats it as neutral (0.5), the cluster-hop walk as maximally
            worth visiting (`pessimistic_i_value`, 1.0).

            `fetch_on_miss` asks the trainer once for a node that is not cached yet, then
            caches it. This is how the connected walk stays as informed as it used to be
            without the cost: it previously pre-warmed every node in the graph for every
            pointer inside `reset_pointers` -- O(N x pointers) DQN forward passes, repeated
            on every refresh period, which is simply not runnable on a large graph. Fetching
            only the neighbors a decision actually examines is O(degree) and gives the same
            answer. The cluster-hop walk leaves it off on purpose: its pessimistic default
            for an unvisited node *is* its exploration bonus.
            """
            if default is None:
                default = self.pessimistic_i_value
            if (not use_din) and fetch_on_miss and self.trainer is not None \
                    and node not in pointer_data['i_values']:
                use_din = True

            # If using DQN, prioritize trainer call
            if use_din:
                if self.trainer:
                    try:
                        # Use the DQN-based method
                        i_val = self.trainer.get_i_value(node, 0)
                        if isinstance(i_val, (int, float)):
                            pointer_data['i_values'][node] = i_val # Update cache
                            #print(f"Using trainer I-value {i_val} for node {node.node_id}")
                            return i_val
                        else:
                            print(f"Trainer returned non-numeric I-value for {node.node_id}: {i_val}")
                    except Exception as e:
                        print(f"Trainer error getting I-value for {node.node_id}: {e}")
                        # Fall through to cache/pessimistic if trainer fails
                else:
                    print(f"No trainer available for DQN I-value prediction for {node.node_id}")

                # If trainer failed or wasn't available, try cache as fallback before pessimistic
                i_val_cached = pointer_data['i_values'].get(node, None)
                if i_val_cached is not None:
                    #print(f"Using cached I-value {i_val_cached} for node {node.node_id} after trainer failure/absence")
                    return i_val_cached
                else:
                    # If trainer failed AND not in cache, use pessimistic
                    pointer_data['i_values'][node] = default
                    #print(f"Using pessimistic I-value {self.pessimistic_i_value} for node {node.node_id} (trainer failed, not cached)")
                    return default

            # If not using DQN (use_din=False), check cache first
            else: 
                i_val_cached = pointer_data['i_values'].get(node, None)
                if i_val_cached is not None:
                    #print(f"Using cached I-value {i_val_cached} for node {node.node_id} (use_din=False)")
                    return i_val_cached
                else:
                    # Fallback to pessimistic value if not in cache and not using DQN
                    pointer_data['i_values'][node] = default
                    #print(f"Using pessimistic I-value {self.pessimistic_i_value} for node {node.node_id} (not cached, use_din=False)")
                    return default

    def _traverse_connected(self, batch_size=32):
            """Move pointers based on I-values and constraints."""
            if self.t >= self.num_steps:
                return []

            self.t += 1
            batch_nodes = []
            visited_this_batch = set()

            # Periodically drop cached I-values so later lookups re-ask the DQN, which has
            # trained since. This used to *repopulate* the cache for every node in the
            # graph, for every pointer -- one DQN forward pass each, so O(N x pointers) per
            # refresh. On the 562,214-node graph that was 6 min of wall clock per refresh
            # and, at 313 batches an epoch with period 50, it fired 6 times: ~36 min of a
            # 38.6 min epoch, with the GPU at 6%. Training ran in ~16 s bursts between
            # these sweeps. Clearing is O(1) and gets the same freshness, because
            # `_get_i_value(fetch_on_miss=True)` re-fetches whatever the walk actually
            # looks at -- which is a handful of neighbours, not half a million nodes.
            if self.trainer and self.t % self.predictor_update_period == 0:
                for pointer in self.pointers:
                    pointer['i_values'].clear()

            # Keep collecting nodes until we have enough or can't find more
            while len(batch_nodes) < batch_size:
                new_nodes = []
                for pointer in self.pointers:
                    try:
                        # Random warp with probability warp_chance
                        if self.rng.random() < self.warp_chance:
                            new_node = self.graph.get_random_node(rng=self.rng)
                            pointer['current_node'] = new_node
                            if new_node not in visited_this_batch:
                                new_nodes.append(new_node)
                                visited_this_batch.add(new_node)
                            continue

                        # Get neighboring nodes
                        neighbors = pointer['current_node'].get_adjacent_nodes()
                        if not neighbors:
                            new_node = self.graph.get_random_node(rng=self.rng)
                            pointer['current_node'] = new_node
                            if new_node not in visited_this_batch and isinstance(new_node, AttributeNode):
                                new_nodes.append(new_node)
                                visited_this_batch.add(new_node)
                            continue

                        # Filter out recently visited nodes
                        current_time = self.t
                        valid_neighbors = [
                            n for n in neighbors
                            if current_time - pointer['last_visited'].get(n, -self.return_delay) >= self.return_delay
                            and n not in visited_this_batch
                            and isinstance(n, AttributeNode)  # Only consider AttributeNodes
                        ]

                        if not valid_neighbors:
                            new_node = self.graph.get_random_node(rng=self.rng)
                            pointer['current_node'] = new_node
                            if new_node not in visited_this_batch and isinstance(new_node, AttributeNode):
                                new_nodes.append(new_node)
                                visited_this_batch.add(new_node)
                            continue

                        # Choose next node based on I-values
                        candidates = self._candidates(valid_neighbors, visited_this_batch)
                        i_values = [
                            self._get_i_value(
                                pointer, n, default=self.neutral_i_value,
                                fetch_on_miss=True,
                            )
                            for n in candidates
                        ]
                        next_node = self._pick(candidates, i_values)

                        # Update visited time and move pointer
                        pointer['last_visited'][next_node] = current_time
                        pointer['current_node'] = next_node

                        if next_node not in visited_this_batch:
                            new_nodes.append(next_node)
                            visited_this_batch.add(next_node)

                    except Exception as e:
                        print(f"Error in traverse: {str(e)}")
                        continue

                # If we couldn't find any new nodes, break
                if not new_nodes:
                    # If we haven't found enough nodes for a minimal batch, try random sampling
                    if len(batch_nodes) < 8:  # Minimum batch size threshold
                        # Sorted before sampling: this list comes from a set of Node objects, and
                        # Node.__hash__ hashes a string node_id, so its order is
                        # PYTHONHASHSEED-dependent and varies between processes.
                        remaining_nodes = sorted(
                            set(self.graph.get_nodes()) - visited_this_batch,
                            key=lambda node: str(node.node_id),
                        )
                        if remaining_nodes:
                            random_nodes = self.rng.sample(remaining_nodes, min(batch_size - len(batch_nodes), len(remaining_nodes)))
                            batch_nodes.extend([n for n in random_nodes if isinstance(n, AttributeNode)])
                            visited_this_batch.update(random_nodes)
                    break

                # Add new nodes to batch
                batch_nodes.extend(new_nodes)

                # If we've collected more than batch_size nodes, trim the excess
                if len(batch_nodes) > batch_size:
                    batch_nodes = batch_nodes[:batch_size]
                    break

            # Return whatever was found, however little. Discarding a short batch was not
            # merely wasteful: `[]` is how `BasicTrainingCapability` and
            # `DQNCapability.train_with_dqn` are told the traversal is *exhausted*, so they
            # stop collecting on the first one. On a graph small enough that a step yields
            # fewer than eight nodes, the very first call returned `[]` and the epoch trained
            # on nothing -- which is why `--traversal-type i-value` was unusable on a small
            # graph while working fine on the 5000-node splits, where a step always fills
            # the batch. A three-node batch is a valid training batch.
            self.current_batch_nodes = batch_nodes
            return batch_nodes

    def _traverse_cluster_hop(self, batch_size=32):
            """Move pointers based on I-values, constraints, and periodic bias hops."""
            if self.t >= self.num_steps:
                return []

            self.t += 1
            batch_nodes = []
            visited_this_batch = set()
            #print(f"\n--- Traversal Step {self.t} ---")
            # print(f"Bias hop period: {self.bias_hop_period}")
            # print(f"self.t % self.bias_hop_period: {self.t % self.bias_hop_period}")
            # --- Bias Hop Logic --- 
            if self.bias_hop_period > 0 and self.t > 0 and self.t % self.bias_hop_period == 0:
                pointer_to_hop_idx = self.current_bias_hop_pointer_index % self.num_pointers
                pointer_to_hop_data = self.pointers[pointer_to_hop_idx]
                #print(f"\n--- Bias Hop Check at t={self.t} for Pointer {pointer_to_hop_idx} ---")

                # Calculate average I-value for each subgroup defined by bias_attributes combination
                subgroup_i_values = defaultdict(lambda: {'sum': 0.0, 'count': 0})
                all_nodes_for_hop = list(self.graph.get_nodes()) # Consider all nodes for hop target pool

                for node in all_nodes_for_hop:
                    if not isinstance(node, AttributeNode) or not hasattr(node, 'attributes') or not node.attributes:
                        continue # Skip nodes without attributes

                    # Create subgroup tuple (handle missing attributes)
                    subgroup_key_list = []
                    skip_node = False
                    for attr_name in self.bias_attributes:
                        attr_value = node.attributes.get(attr_name, 'MISSING')
                        # Optional: Skip nodes missing any bias attribute for hop calculation
                        if attr_value == 'MISSING':
                            print(f"Skipping node {node.node_id} due to missing bias attribute {attr_name}")
                            skip_node = True
                            break
                        subgroup_key_list.append(f"{attr_name}_{attr_value}") # Create descriptive string keys

                    if skip_node:
                        continue

                    subgroup_key = tuple(sorted(subgroup_key_list)) # Use sorted tuple as dict key

                    # Get I-value safely (using the pointer's perspective for consistency)
                    # USE CACHED/DEFAULT FOR HOP CALCULATION
                    i_val = self._get_i_value(pointer_to_hop_data, node, use_din=False) 

                    subgroup_i_values[subgroup_key]['sum'] += i_val
                    subgroup_i_values[subgroup_key]['count'] += 1

                # Calculate averages and find the best subgroup
                avg_i_values = {}
                max_avg_i_value = -float('inf')
                best_subgroup_key = None

                # Remove noisy print statements
                for subgroup_key, data in subgroup_i_values.items():
                    if data['count'] > 0:
                        avg = data['sum'] / data['count']
                        avg_i_values[subgroup_key] = avg
                        if avg > max_avg_i_value:
                            max_avg_i_value = avg
                            best_subgroup_key = subgroup_key
                    # else:
                    #    print(f"  {subgroup_key}: N/A (Count: 0)")

                # Store the calculated averages for this hop instance
                if avg_i_values:
                    self.hop_i_value_history.append(avg_i_values)

                # Hop to a random node within the best subgroup
                if best_subgroup_key:
                    # print(f"Target Subgroup for Hop: {best_subgroup_key} (Max Avg I-Value: {max_avg_i_value:.4f})")
                    target_nodes_in_subgroup = []
                    for node in all_nodes_for_hop:
                         if not isinstance(node, AttributeNode) or not hasattr(node, 'attributes') or not node.attributes:
                             continue
                         # Recreate the key for comparison
                         current_node_key_list = []
                         valid_node = True
                         for attr_name in self.bias_attributes:
                              attr_value = node.attributes.get(attr_name, 'MISSING')
                              # if attr_value == 'MISSING': # Apply same skip logic as above if used
                              #    valid_node = False
                              #    break
                              current_node_key_list.append(f"{attr_name}_{attr_value}")

                         if not valid_node:
                             continue

                         current_node_key = tuple(sorted(current_node_key_list))
                         if current_node_key == best_subgroup_key:
                              target_nodes_in_subgroup.append(node)

                    if target_nodes_in_subgroup:
                        # --- Prevent hopping to the same node if it's the only one in the best subgroup ---
                        if len(target_nodes_in_subgroup) == 1 and target_nodes_in_subgroup[0] == pointer_to_hop_data['current_node']:
                            # print(f"Skipping hop for Pointer {pointer_to_hop_idx}: Target subgroup {best_subgroup_key} only contains the current node {pointer_to_hop_data['current_node'].node_id}.")
                            pass
                        else:
                            hop_node = self.rng.choice(target_nodes_in_subgroup)
                            # print(f"Hopping Pointer {pointer_to_hop_idx} to node {hop_node.node_id} in subgroup {best_subgroup_key}")
                            pointer_to_hop_data['current_node'] = hop_node
                            pointer_to_hop_data['last_visited'] = {}
                    else:
                        # print(f"Warning: No nodes found for the best subgroup {best_subgroup_key}. No hop performed.")
                        pass
                else:
                    # print("Warning: Could not determine best subgroup. No hop performed.")
                    pass

                self.current_bias_hop_pointer_index += 1 # Move to the next pointer for the next hop cycle
            # --- End Bias Hop Logic --- 

            # Periodically drop cached I-values so later lookups re-ask the DQN, which has
            # trained since. This used to *repopulate* the cache for every node in the
            # graph, for every pointer -- one DQN forward pass each, so O(N x pointers) per
            # refresh. On the 562,214-node graph that was 6 min of wall clock per refresh
            # and, at 313 batches an epoch with period 50, it fired 6 times: ~36 min of a
            # 38.6 min epoch, with the GPU at 6%. Training ran in ~16 s bursts between
            # these sweeps. Clearing is O(1) and gets the same freshness, because
            # `_get_i_value(fetch_on_miss=True)` re-fetches whatever the walk actually
            # looks at -- which is a handful of neighbours, not half a million nodes.
            if self.trainer and self.t % self.predictor_update_period == 0:
                for pointer in self.pointers:
                    pointer['i_values'].clear()

            # Keep collecting nodes until we have enough or can't find more
            while len(batch_nodes) < batch_size:
                new_nodes = []
                for pointer in self.pointers:
                    try:
                        # Random warp with probability warp_chance
                        if self.rng.random() < self.warp_chance:
                            new_node = self.graph.get_random_node(rng=self.rng)
                            pointer['current_node'] = new_node
                            if new_node not in visited_this_batch:
                                new_nodes.append(new_node)
                                visited_this_batch.add(new_node)
                            continue

                        # Get neighboring nodes
                        neighbors = pointer['current_node'].get_adjacent_nodes()
                        if not neighbors:
                            new_node = self.graph.get_random_node(rng=self.rng)
                            pointer['current_node'] = new_node
                            if new_node not in visited_this_batch and isinstance(new_node, AttributeNode):
                                new_nodes.append(new_node)
                                visited_this_batch.add(new_node)
                            continue

                        # Filter out recently visited nodes
                        current_time = self.t
                        valid_neighbors = [
                            n for n in neighbors
                            if current_time - pointer['last_visited'].get(n, -self.return_delay) >= self.return_delay
                            and n not in visited_this_batch
                            and isinstance(n, AttributeNode)  # Only consider AttributeNodes
                        ]

                        if not valid_neighbors:
                            new_node = self.graph.get_random_node(rng=self.rng)
                            pointer['current_node'] = new_node
                            if new_node not in visited_this_batch and isinstance(new_node, AttributeNode):
                                new_nodes.append(new_node)
                                visited_this_batch.add(new_node)
                            continue

                        # Choose next node based on I-values
                        # Use the safe getter for I-values
                        #print("Updating I-values for valid neighbors...")
                        candidates = self._candidates(valid_neighbors, visited_this_batch)
                        i_values = [self._get_i_value(pointer, n, use_din=True) for n in candidates]
                        # Old: i_values = [pointer['i_values'].get(n, self.pessimistic_i_value) for n in valid_neighbors] # Use pessimistic if not found

                        if not i_values: # Should not happen if valid_neighbors is not empty, but check
                            print(f"Warning: No I-values for valid neighbors of node {pointer['current_node'].id}")
                            continue

                        next_node = self._pick(candidates, i_values)

                        # Update visited time and move pointer
                        pointer['last_visited'][next_node] = current_time
                        pointer['current_node'] = next_node

                        if next_node not in visited_this_batch:
                            new_nodes.append(next_node)
                            visited_this_batch.add(next_node)

                    except Exception as e:
                        print(f"Error in traverse: {str(e)}")
                        continue

                # If we couldn't find any new nodes, break
                if not new_nodes:
                    # If we haven't found enough nodes for a minimal batch, try random sampling
                    if len(batch_nodes) < 8:  # Minimum batch size threshold
                        # Sorted before sampling: this list comes from a set of Node objects, and
                        # Node.__hash__ hashes a string node_id, so its order is
                        # PYTHONHASHSEED-dependent and varies between processes.
                        remaining_nodes = sorted(
                            set(self.graph.get_nodes()) - visited_this_batch,
                            key=lambda node: str(node.node_id),
                        )
                        if remaining_nodes:
                            random_nodes = self.rng.sample(remaining_nodes, min(batch_size - len(batch_nodes), len(remaining_nodes)))
                            batch_nodes.extend([n for n in random_nodes if isinstance(n, AttributeNode)])
                            visited_this_batch.update(random_nodes)
                    break

                # Add new nodes to batch
                batch_nodes.extend(new_nodes)

                # If we've collected more than batch_size nodes, trim the excess
                if len(batch_nodes) > batch_size:
                    batch_nodes = batch_nodes[:batch_size]
                    break

            # If we still don't have enough nodes for a minimal batch, skip this traversal
            if len(batch_nodes) < 8:  # Minimum batch size threshold
                return []

            self.current_batch_nodes = batch_nodes # Store the collected nodes
            return batch_nodes
