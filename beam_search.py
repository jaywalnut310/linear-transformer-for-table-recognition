import torch
import math
import tqdm

# Assuming EOS_ID is 2
EOS_ID = 2
# Default value for INF
INF = 1. * 1e7


def _merge_beam_dim(tensor):
    """Reshapes first two dimensions in to single dimension.
    Args:
        tensor: Tensor to reshape of shape [A, B, ...]
    Returns:
        Reshaped tensor of shape [A*B, ...]
    """
    shape = list(tensor.shape)
    shape[0] *= shape[1]  # batch -> batch * beam_size
    shape.pop(1)  # Remove beam dim
    return tensor.reshape(shape)


def _unmerge_beam_dim(tensor, batch_size, beam_size):
    """Reshapes first dimension back to [batch_size, beam_size].
    Args:
        tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
        batch_size: Tensor, original batch size.
        beam_size: int, original beam size.
    Returns:
        Reshaped tensor of shape [batch_size, beam_size, ...]
    """
    shape = list(tensor.shape)
    new_shape = [batch_size] + [beam_size] + shape[1:]
    return tensor.reshape(new_shape)


def _expand_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size.
    Args:
        tensor: tensor to tile [batch_size, ...]
        beam_size: How much to tile the tensor by.
    Returns:
        Tiled tensor [batch_size, beam_size, ...]
    """
    tensor = tensor.unsqueeze(1)
    tile_dims = [1] * len(tensor.shape)
    tile_dims[1] = beam_size
    return tensor.repeat(tile_dims)


def _gather_coordinates(tensor, coordinates):
    batch_size, *_ = tensor.shape
    beam_size = coordinates.size(0) // batch_size
    tensor_flat = _merge_beam_dim(tensor)
    tensor_gather = torch.index_select(tensor_flat, 0, coordinates)
    tensor = _unmerge_beam_dim(tensor_gather, batch_size, beam_size)
    return tensor


def compute_batch_indices(batch_size, beam_size):
    """Computes the i'th coordinate that contains the batch index for gathers.
    Batch pos is a tensor like [[0,0,0,0],[1,1,1,1],..]. It says which
    batch the beam item is in. This will create the i of the i,j coordinate
    needed for the gather.
    Args:
        batch_size: Batch size
        beam_size: Size of the beam.
    Returns:
        batch_pos: [batch_size, beam_size] tensor of ids
    """
    batch_pos = torch.arange(batch_size * beam_size) // beam_size
    batch_pos = batch_pos.reshape([batch_size, beam_size])
    return batch_pos


def compute_topk_scores_and_seq(sequences, scores, scores_to_gather, flags,
                                beam_size, batch_size,
                                states_to_gather=None):
    """Given sequences and scores, will gather the top k=beam size sequences.
    This function is used to grow alive, and finished. It takes sequences,
    scores, and flags, and returns the top k from sequences, scores_to_gather,
    and flags based on the values in scores.
    This method permits easy introspection using tfdbg.  It adds three named ops
    that are prefixed by `prefix`:
        - _topk_seq: the tensor for topk_seq returned by this method.
        - _topk_flags: the tensor for topk_finished_flags returned by this method.
        - _topk_scores: the tensor for tokp_gathered_scores returned by this method.
    Args:
        sequences: Tensor of sequences that we need to gather from.
            [batch_size, beam_size, seq_length]
        scores: Tensor of scores for each sequence in sequences.
            [batch_size, beam_size]. We will use these to compute the topk.
        scores_to_gather: Tensor of scores for each sequence in sequences.
            [batch_size, beam_size]. We will return the gathered scores from here.
            Scores to gather is different from scores because for grow_alive, we will
            need to return log_probs, while for grow_finished, we will need to return
            the length penalized scores.
        flags: Tensor of bools for sequences that say whether a sequence has reached
            EOS or not
        beam_size: int
        prefix: string that will prefix unique names for the ops run.
        states_to_gather: dict (possibly nested) of decoding states.
    Returns:
        Tuple of
        (topk_seq [batch_size, beam_size, decode_length],
         topk_gathered_scores [batch_size, beam_size],
         topk_finished_flags[batch_size, beam_size])
    """
    # sort scores
    _, topk_indexes = torch.topk(scores, k=beam_size)
    # The next three steps are to create coordinates for the gather to pull
    # out the topk sequences from sequences based on scores.
    # batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    # batch the beam item is in. This will create the i of the i,j coordinate
    # needed for the gather
    batch_pos = compute_batch_indices(batch_size, beam_size).to(device=scores.device)

    # top coordinates will give us the actual coordinates to do the gather.
    # top coordinates is a sequence of dimension batch * beam, each of which
    # contains the gathering coordinate.
    top_coordinates = (batch_pos * scores.size(1) + topk_indexes).view(-1)

    # Gather up the highest scoring sequences.  For each operation added, give it
    # a concrete name to simplify observing these operations with tfdbg.  Clients
    # can capture these tensors by watching these node names.
    topk_seq = _gather_coordinates(sequences, top_coordinates)
    topk_flags = _gather_coordinates(flags, top_coordinates)
    topk_gathered_scores = _gather_coordinates(scores_to_gather, top_coordinates)
    if states_to_gather:
        for state in states_to_gather:
            for k, v in state.items():
                state[k] = _gather_coordinates(v, top_coordinates)
    topk_gathered_states = states_to_gather
    return topk_seq, topk_gathered_scores, topk_flags, topk_gathered_states


def beam_search(symbols_to_logits_fn,
                initial_ids,
                beam_size,
                decode_length,
                vocab_size,
                alpha,
                states=None,
                eos_id=EOS_ID,
                stop_early=True):
    """Beam search with length penalties.
    Requires a function that can take the currently decoded symbols and return
    the logits for the next symbol. The implementation is inspired by
    https://arxiv.org/abs/1609.08144.
    When running, the beam search steps can be visualized by using tfdbg to watch
    the operations generating the output ids for each beam step.  These operations
    have the pattern:
    (alive|finished)_topk_(seq,scores)
    Operations marked `alive` represent the new beam sequences that will be
    processed in the next step.  Operations marked `finished` represent the
    completed beam sequences, which may be padded with 0s if no beams finished.
    Operations marked `seq` store the full beam sequence for the time step.
    Operations marked `scores` store the sequence's final log scores.
    The beam search steps will be processed sequentially in order, so when
    capturing observed from these operations, tensors, clients can make
    assumptions about which step is being recorded.
    WARNING: Assumes 2nd dimension of tensors in `states` and not invariant, this
    means that the shape of the 2nd dimension of these tensors will not be
    available (i.e. set to None) inside symbols_to_logits_fn.
    Args:
        symbols_to_logits_fn: Interface to the model, to provide logits.
            Shoud take [batch_size, decoded_ids] and return [batch_size, vocab_size]
        initial_ids: Ids to start off the decoding, this will be the first thing
            handed to symbols_to_logits_fn (after expanding to beam size)
            [batch_size]
        beam_size: Size of the beam.
        decode_length: Number of steps to decode for.
        vocab_size: Size of the vocab, must equal the size of the logits returned by
            symbols_to_logits_fn
        alpha: alpha for length penalty.
        states: dict (possibly nested) of decoding states.
        eos_id: ID for end of sentence.
        stop_early: a boolean - stop once best sequence is provably determined.
    Returns:
        Tuple of
        (decoded beams [batch_size, beam_size, decode_length]
         decoding probabilities [batch_size, beam_size])
    """
    batch_size = initial_ids.shape[0]

    # Assume initial_ids are prob 1.0
    initial_log_probs = torch.Tensor(
        [[0.] + [-float("inf")] * (beam_size - 1)]
    ).to(device=initial_ids.device)
    # Expand to beam_size (batch_size, beam_size)
    alive_log_probs = initial_log_probs.repeat([batch_size, 1])

    # Expand each batch and state to beam_size
    alive_seq = _expand_to_beam_size(initial_ids, beam_size)
    alive_seq = alive_seq.unsqueeze(2)  # (batch_size, beam_size, 1)
    if states:
        for state in states:
            for k, v in state.items():
                state[k] = _expand_to_beam_size(v, beam_size)
    else:
        states = None

    # Finished will keep track of all the sequences that have finished so far
    # Finished log probs will be negative infinity in the beginning
    # finished_flags will keep track of booleans
    finished_seq = torch.zeros_like(alive_seq)
    # Setting the scores of the initial to negative infinity.
    finished_scores = torch.ones_like(alive_log_probs) * -INF
    finished_flags = torch.zeros_like(alive_log_probs, dtype=torch.bool)

    def grow_finished(finished_seq, finished_scores, finished_flags, curr_seq,
                    curr_scores, curr_finished):
        """Given sequences and scores, will gather the top k=beam size sequences.
        Args:
            finished_seq: Current finished sequences.
                [batch_size, beam_size, current_decoded_length]
            finished_scores: scores for each of these sequences.
                [batch_size, beam_size]
            finished_flags: finished bools for each of these sequences.
                [batch_size, beam_size]
            curr_seq: current topk sequence that has been grown by one position.
                [batch_size, beam_size, current_decoded_length]
            curr_scores: scores for each of these sequences. [batch_size, beam_size]
            curr_finished: Finished flags for each of these sequences.
                [batch_size, beam_size]
        Returns:
            Tuple of
                (Topk sequences based on scores,
                 log probs of these sequences,
                 Finished flags of these sequences)
        """
        # First append a column of 0'ids to finished to make the same length with
        # finished scores
        finished_seq = torch.cat(
            [finished_seq,
             torch.zeros([batch_size, beam_size, 1], 
                         dtype=finished_seq.dtype, device=finished_seq.device)
            ], 2)

        # Set the scores of the unfinished seq in curr_seq to large negative
        # values
        curr_scores = curr_scores + (1. - curr_finished.to(dtype=curr_scores.dtype)) * -INF
        # concatenating the sequences and scores along beam axis
        curr_finished_seq = torch.cat([finished_seq, curr_seq], 1)
        curr_finished_scores = torch.cat([finished_scores, curr_scores], 1)
        curr_finished_flags = torch.cat([finished_flags, curr_finished], 1)
        return compute_topk_scores_and_seq(
            curr_finished_seq, curr_finished_scores, curr_finished_scores,
            curr_finished_flags, beam_size, batch_size)

    def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished, states):
        """Given sequences and scores, will gather the top k=beam size sequences.
        Args:
            curr_seq: current topk sequence that has been grown by one position.
                [batch_size, beam_size, i+1]
            curr_scores: scores for each of these sequences. [batch_size, beam_size]
            curr_log_probs: log probs for each of these sequences.
                [batch_size, beam_size]
            curr_finished: Finished flags for each of these sequences.
                [batch_size, beam_size]
            states: dict (possibly nested) of decoding states.
        Returns:
            Tuple of
                (Topk sequences based on scores,
                 log probs of these sequences,
                 Finished flags of these sequences)
        """
        # Set the scores of the finished seq in curr_seq to large negative
        # values
        curr_scores = curr_scores + curr_finished.to(dtype=curr_scores.dtype) * -INF
        return compute_topk_scores_and_seq(curr_seq, curr_scores, curr_log_probs,
                                           curr_finished, beam_size, batch_size,
                                           states)

    def grow_topk(i, alive_seq, alive_log_probs, states):
        r"""Inner beam search loop.
        This function takes the current alive sequences, and grows them to topk
        sequences where k = 2*beam. We use 2*beam because, we could have beam_size
        number of sequences that might hit <EOS> and there will be no alive
        sequences to continue. With 2*beam_size, this will not happen. This relies
        on the assumption the vocab size is > beam size. If this is true, we'll
        have at least beam_size non <EOS> extensions if we extract the next top
        2*beam words.
        Length penalty is given by = (5+len(decode)/6) ^ -\alpha. Pls refer to
        https://arxiv.org/abs/1609.08144.
        Args:
            i: loop index
            alive_seq: Topk sequences decoded so far [batch_size, beam_size, i+1]
            alive_log_probs: probabilities of these sequences. [batch_size, beam_size]
            states: dict (possibly nested) of decoding states.
        Returns:
            Tuple of
                (Topk sequences extended by the next word,
                 The log probs of these sequences,
                 The scores with length penalty of these sequences,
                 Flags indicating which of these sequences have finished decoding,
                 dict of transformed decoding states)
        """
        # Get the logits for all the possible next symbols
        flat_ids = alive_seq.reshape([batch_size * beam_size, -1])

        # (batch_size * beam_size, decoded_length)
        if states:
            for state in states:
                for k, v in state.items():
                    state[k] = _merge_beam_dim(v)
            flat_states = states
            flat_logits, flat_states = symbols_to_logits_fn(flat_ids, i, flat_states)
            for state in flat_states:
                for k, v in state.items():
                    state[k] = _unmerge_beam_dim(v, batch_size, beam_size)
            states = flat_states
        else:
            flat_logits = symbols_to_logits_fn(flat_ids)

        logits = flat_logits.reshape([batch_size, beam_size, -1])

        # Convert logits to normalized log probs
        candidate_log_probs = torch.log_softmax(logits, -1)

        # Multiply the probabilities by the current probabilities of the beam.
        # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
        log_probs = candidate_log_probs + alive_log_probs.unsqueeze(2)

        length_penalty = math.pow(((5. + i + 1) / 6.), alpha)

        curr_scores = log_probs / length_penalty
        # Flatten out (beam_size, vocab_size) probs in to a list of possibilities
        flat_curr_scores = curr_scores.reshape([-1, beam_size * vocab_size])

        topk_scores, topk_ids = torch.topk(flat_curr_scores, k=beam_size * 2)

        # Recovering the log probs because we will need to send them back
        topk_log_probs = topk_scores * length_penalty

        # Work out what beam the top probs are in.
        topk_beam_index = topk_ids // vocab_size
        topk_ids %= vocab_size  # Unflatten the ids

        # The next three steps are to create coordinates for the gather to pull
        # out the correct sequences from id's that we need to grow.
        # We will also use the coordinates to gather the booleans of the beam items
        # that survived.
        batch_pos = compute_batch_indices(batch_size, beam_size * 2).to(device=alive_seq.device)
        
        # top coordinates will give us the actual coordinates to do the gather.
        # top coordinates is a sequence of dimension batch * beam, each of which
        # contains the gathering coordinate.
        top_coordinates = (batch_pos * beam_size + topk_beam_index).view(-1)

        # Gather up the most probable 2*beams both for the ids and finished_in_alive
        # bools
        topk_seq = _gather_coordinates(alive_seq, top_coordinates)
        if states:
            for state in states:
                for k, v in state.items():
                    state[k] = _gather_coordinates(v, top_coordinates)

        # Append the most probable alive
        topk_seq = torch.cat([topk_seq, topk_ids.unsqueeze(2)], 2)

        topk_finished = torch.eq(topk_ids, eos_id)

        return topk_seq, topk_log_probs, topk_scores, topk_finished, states

    def inner_loop(i, alive_seq, alive_log_probs, finished_seq, finished_scores,
                    finished_flags, states):
        """Inner beam search loop.
        There are three groups of tensors, alive, finished, and topk.
        The alive group contains information about the current alive sequences
        The topk group contains information about alive + topk current decoded words
        the finished group contains information about finished sentences, that is,
        the ones that have decoded to <EOS>. These are what we return.
        The general beam search algorithm is as follows:
        While we haven't terminated (pls look at termination condition)
            1. Grow the current alive to get beam*2 topk sequences
            2. Among the topk, keep the top beam_size ones that haven't reached EOS
            into alive
            3. Among the topk, keep the top beam_size ones have reached EOS into
            finished
        Repeat
        To make things simple with using fixed size tensors, we will end
        up inserting unfinished sequences into finished in the beginning. To stop
        that we add -ve INF to the score of the unfinished sequence so that when a
        true finished sequence does appear, it will have a higher score than all the
        unfinished ones.
        Args:
            i: loop index
            alive_seq: Topk sequences decoded so far [batch_size, beam_size, i+1]
            alive_log_probs: probabilities of the beams. [batch_size, beam_size]
            finished_seq: Current finished sequences.
                [batch_size, beam_size, i+1]
            finished_scores: scores for each of these sequences.
                [batch_size, beam_size]
            finished_flags: finished bools for each of these sequences.
                [batch_size, beam_size]
            states: dict (possibly nested) of decoding states.
        Returns:
            Tuple of
                (Incremented loop index
                 New alive sequences,
                 Log probs of the alive sequences,
                 New finished sequences,
                 Scores of the new finished sequences,
                 Flags indicating which sequence in finished as reached EOS,
                 dict of final decoding states)
        """

        # Each inner loop, we carry out three steps:
        # 1. Get the current topk items.
        # 2. Extract the ones that have finished and haven't finished
        # 3. Recompute the contents of finished based on scores.
        topk_seq, topk_log_probs, topk_scores, topk_finished, states = grow_topk(
            i, alive_seq, alive_log_probs, states)
        alive_seq, alive_log_probs, _, states = grow_alive(
            topk_seq, topk_scores, topk_log_probs, topk_finished, states)
        finished_seq, finished_scores, finished_flags, _ = grow_finished(
            finished_seq, finished_scores, finished_flags, topk_seq, topk_scores,
            topk_finished)

        return (alive_seq, alive_log_probs, finished_seq, finished_scores,
                finished_flags, states)

    def _is_finished(alive_log_probs, finished_scores, finished_in_finished):
        """Checking termination condition.
        We terminate when we decoded up to decode_length or the lowest scoring item
        in finished has a greater score that the highest prob item in alive divided
        by the max length penalty
        Args:
            alive_log_probs: probabilities of the beams. [batch_size, beam_size]
            finished_scores: scores for each of these sequences.
                [batch_size, beam_size]
            finished_in_finished: finished bools for each of these sequences.
                [batch_size, beam_size]
        Returns:
            Bool.
        """
        max_length_penalty = math.pow(((5. + decode_length) / 6.), alpha)
        # The best possible score of the most likely alive sequence.
        upper_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty

        # Now to compute the lowest score of a finished sequence in finished
        # If the sequence isn't finished, we multiply it's score by 0. since
        # scores are all -ve, taking the min will give us the score of the lowest
        # finished item.
        lowest_score_of_finished_in_finished = torch.min(
            finished_scores * finished_in_finished.to(dtype=finished_scores.dtype), 1)[0]
        # If none of the sequences have finished, then the min will be 0 and
        # we have to replace it by -ve INF if it is. The score of any seq in alive
        # will be much higher than -ve INF and the termination condition will not
        # be met.
        lowest_score_of_finished_in_finished += (
            (1. - torch.any(
                finished_in_finished, 
                1).to(dtype=lowest_score_of_finished_in_finished.dtype)) * -INF)

        bound_is_met = torch.all(
            lowest_score_of_finished_in_finished > upper_bound_alive_scores)

        return bound_is_met

    for i in tqdm.tqdm(range(decode_length)):
        (alive_seq, alive_log_probs, finished_seq, finished_scores,
        finished_flags, states) = inner_loop(i, alive_seq, alive_log_probs, 
            finished_seq, finished_scores, finished_flags, states)
        if stop_early and _is_finished(alive_log_probs, finished_scores, finished_flags):
            break

    # Accounting for corner case: It's possible that no sequence in alive for a
    # particular batch item ever reached EOS. In that case, we should just copy
    # the contents of alive for that batch item. torch.any(finished_flags, 1)
    # if 0, means that no sequence for that batch index had reached EOS. We need
    # to do the same for the scores as well.
    finished_seq = torch.where(
        torch.any(finished_flags, 1).view(batch_size, *([1] * (finished_seq.dim() - 1))), 
        finished_seq, 
        alive_seq)
    finished_scores = torch.where(
        torch.any(finished_flags, 1).view(batch_size, *([1] * (finished_seq.dim() - 1))),
        finished_scores, 
        alive_log_probs)
    return finished_seq, finished_scores