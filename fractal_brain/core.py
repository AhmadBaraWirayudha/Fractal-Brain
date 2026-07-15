"""
fractal_brain/core.py
FractalBrain – all-in-one hybrid architecture, fused with:
- Signal processing (delay line + convolution)
- Wormholes (shortcut connections)
- JEPA (joint embedding predictive loss) -- now actually trained, not just computed
- Distillation (teacher-student loss)
- RAG fusion, now a genuine (trained) contribution to expert selection, not a computed-
  and-discarded side effect
- Real gradient descent on: each expert's output projection, the entire gate (raw
  projection, lasso tentacles, wormhole, RAG fusion), the PID gains, and JEPA's own
  encoder/predictor -- each via a real optimizer (SGD w/ momentum, Adam, gradient
  clipping, LR schedules -- see optimizer.py) rather than fixed-rate updates computed
  and applied in the same breath. See CHANGELOG.md for what this replaced and why.
"""
import math
from .math_utils import Vector, Matrix, softmax, kl_divergence
from .pid import PIDController
from .markov import build_fractal_chain, FractalMarkovNode
from .tentacles import LassoTentacles
from .moe import GatedMoE
from .rag import VectorStore, StateRAGFusion
from .synaptic import BCMPlasticity
from .wormhole import Wormhole
from .jepa import JEPA
from .distillation import distillation_loss
from .signal import DelayLine, convolve1d
from .optimizer import SGD, clip_grad_norm_matrix

class FractalBrain:
    def __init__(self, vocab_size=1000, d_model=256, num_experts=8,
                 num_heads=4, d_ff=512, num_layers=2,
                 num_markov_nodes=7, markov_states=3, max_level=3,
                 use_jepa=True, use_wormhole=True, teacher_model=None,
                 pid_kp=0.8, pid_ki=0.15, pid_kd=0.05, pid_temp_scale=5.0,
                 output_lr=0.05, pid_meta_lr=0.02, gate_lr=0.05, jepa_lr=0.05,
                 output_optimizer=None, pid_optimizer=None, gate_optimizer=None,
                 jepa_optimizer=None,
                 output_lr_scheduler=None, grad_clip_norm=None):
        """
        output_optimizer / pid_optimizer / gate_optimizer / jepa_optimizer: an
            optimizer.SGD / optimizer.Adam instance (or anything exposing the same
            step_matrix(key, weight, grad) / step_scalar(key, value, grad) /
            step_vector(key, vec, grad) interface). Each defaults to a plain SGD at a
            sensible fixed rate (output_lr, pid_meta_lr, gate_lr, jepa_lr respectively)
            -- pass your own (e.g. optimizer.Adam(lr=1e-3)) for momentum/adaptive
            rates/decay. Kept as four separate slots rather than one shared optimizer
            because they're different parameter families with different natural scales
            (a cross-entropy-driven expert readout vs. a gating decision vs. a PID gain
            vs. an L2-driven auxiliary encoder), not because they need to move in sync.
        output_lr_scheduler: optional optimizer.ConstantLR/StepLR/CosineAnnealingLR/
            LinearWarmupLR (or anything with .get_lr(step)); if set, overrides
            output_optimizer.lr at the start of every step()/train_batch() call based
            on self.step_count. Defaults to None, i.e. output_optimizer.lr never
            changes on its own.
        grad_clip_norm: optional float; if set, gradients for the expert output
            projections and the gate weights are rescaled (if needed) to have L2 norm
            at most this, before being handed to their optimizer. Defaults to None (no
            clipping).
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_markov_nodes = num_markov_nodes
        self.output_lr = output_lr
        self.pid_temp_scale = pid_temp_scale
        self.output_optimizer = output_optimizer if output_optimizer is not None else SGD(lr=output_lr)
        self.pid_optimizer = pid_optimizer if pid_optimizer is not None else SGD(lr=pid_meta_lr)
        self.gate_optimizer = gate_optimizer if gate_optimizer is not None else SGD(lr=gate_lr)
        self.jepa_optimizer = jepa_optimizer if jepa_optimizer is not None else SGD(lr=jepa_lr)
        self.output_lr_scheduler = output_lr_scheduler
        self.grad_clip_norm = grad_clip_norm

        # --- PID controller (drives gate *temperature*, see forward()) ---
        self.pid = PIDController(Kp=pid_kp, Ki=pid_ki, Kd=pid_kd)

        # --- Fractal Markov ensemble ---
        self.markov_nodes = [build_fractal_chain(max_level, num_states=markov_states)
                             for _ in range(num_markov_nodes)]
        self.current_markov_states = [0] * num_markov_nodes

        # --- LassoTentacles (input from concatenated leaf one-hots) ---
        input_dim = num_markov_nodes * markov_states
        self.tentacles = LassoTentacles(input_dim, num_experts, l1_lambda=0.01)

        # --- MoE (Transformer experts) ---
        self.moe = GatedMoE(num_experts, vocab_size, d_model, num_heads, d_ff, num_layers)

        # --- RAG ---
        self.rag_index = VectorStore(d_model)
        for i in range(100):
            doc_vec = Vector([(math.sin(i*0.1 + j*0.3)+1)/2 for j in range(d_model)])
            self.rag_index.add(doc_vec, i)
        self.rag_fusion = StateRAGFusion(d_model)
        self.W_state_proj = Matrix.he_init(num_markov_nodes * markov_states, d_model)
        # projects the RAG-fused state into gate space, same pattern as W_gate/wormhole --
        # this is what makes _retrieve_and_fuse's output actually participate in (and be
        # trained through) expert selection, rather than being computed and discarded
        self.W_rag_gate = Matrix.he_init(d_model, num_experts)

        # --- Wormhole (injects fractal state directly into gate) ---
        self.wormhole = Wormhole(num_markov_nodes * markov_states, num_experts) if use_wormhole else None

        # --- JEPA (predictive auxiliary loss) ---
        self.jepa = JEPA(input_dim=d_model, embed_dim=d_model // 2) if use_jepa else None

        # --- Delay line for convolution over token embeddings ---
        self.emb_delay = DelayLine(max_delay=5)   # retain last 5 token embeddings

        # --- Synaptic plasticity (BCM) ---
        self.bcm = BCMPlasticity()

        # --- Distillation teacher ---
        self.teacher = teacher_model   # any object with a .forward method returning logits

        # counters / caches
        self.step_count = 0
        self.last_state_vec = Vector.zeros(input_dim)   # for plasticity
        self._last_jepa_loss = 0.0
        self._last_jepa_context = None                  # for jepa.train_step(), set in forward()
        self._last_jepa_target = None
        self._last_gated_logits = None                  # pre-temperature gate logits, for PID meta-update
        self._last_gate_query = None                     # for _compute_gate_gradients
        self._last_state_vec_for_gate = None
        self._last_fused_state = None
        self._last_gate_temperature = None

    def _get_fractal_state(self):
        """Step all fractal nodes, return concatenated one-hot vector."""
        full = []
        for i, node in enumerate(self.markov_nodes):
            next_idx, emb = node.forward(self.current_markov_states[i])
            self.current_markov_states[i] = next_idx
            full.extend(emb)   # emb is one-hot (length markov_states)
        return Vector(full)

    def _retrieve_and_fuse(self, token_ids, state_vec):
        """Query RAG with mean embedding of input, fuse with state via cross-attention."""
        if token_ids:
            emb_rows = [self.moe.experts[0].embedding.data[idx] for idx in token_ids]
            query = Vector([sum(col)/len(emb_rows) for col in zip(*emb_rows)])
        else:
            query = Vector.zeros(self.d_model)
        doc_ids, _ = self.rag_index.search(query, k=5)
        doc_embs = [self.rag_index.get_vector(did).to_list() for did in doc_ids]
        state_proj = self.W_state_proj.linear(state_vec)
        fused = self.rag_fusion.forward(state_proj, doc_embs)
        return fused   # list of length d_model

    def _gate_temperature(self, pid_correction):
        """
        Map an unbounded PID correction to a gate softmax temperature in (min_temp, max_temp)
        via a logistic squash, not a hard clamp. A hard clamp has *exactly* zero gradient
        once saturated -- and a KL-divergence error of 15-25 (routine for an early-training
        model against a one-hot target) times Kp~0.8 alone reaches that saturation almost
        immediately, which silently zeroed out the PID meta-gradient every single step (see
        CHANGELOG.md). The logistic keeps a small but genuinely nonzero gradient everywhere.
        """
        min_temp, max_temp = 0.2, 5.0
        sig = 1.0 / (1.0 + math.exp(-pid_correction / self.pid_temp_scale))
        return min_temp + (max_temp - min_temp) * sig

    def forward(self, token_ids, pid_correction=0.0):
        """
        Full forward pass. Returns (logits Matrix, expert weights list, expert outputs list).
        Also updates internal state (fractal, RAG, delay line).
        """
        if not token_ids:
            raise ValueError("token_ids must be a non-empty sequence")
        seq_len = len(token_ids)

        # 1. Fractal state (before external context, deterministic)
        state_vec = self._get_fractal_state()
        self.last_state_vec = state_vec

        # 2. RAG-augmented fused state (d_model)
        fused_state = self._retrieve_and_fuse(token_ids, state_vec)
        fused_state_vec = Vector(fused_state)

        # 3. Token embeddings via first expert (common embedding)
        emb_rows = [self.moe.experts[0].embedding.data[idx] for idx in token_ids]
        if emb_rows:
            current_mean = Vector([sum(col)/seq_len for col in zip(*emb_rows)])
        else:
            current_mean = Vector.zeros(self.d_model)
        # push into delay line and get the delayed (previous step's) version;
        # None means "the buffer hasn't filled this slot yet" (see signal.DelayLine)
        delayed_mean = self.emb_delay.push(current_mean)
        gate_query = current_mean   # could blend delayed values via convolve1d

        # 4. Gate logits: raw projection + lasso tentacles + wormhole shortcut + RAG fusion.
        # None of these four depend on pid_correction, so they're cached pre-temperature
        # below for the cheap PID meta-gradient probe in _meta_update_pid_gains, and the
        # per-source inputs (gate_query, state_vec, fused_state) are cached for
        # _compute_gate_gradients so each contribution can be trained (see below --
        # this is what makes the RAG contribution a genuine, learned part of expert
        # selection rather than an untrained random projection sitting next to trained
        # ones, which would arguably be worse than not wiring it in at all).
        raw_gate = self.moe.W_gate.linear(gate_query)
        lasso_gate = Vector(self.tentacles.forward(state_vec))
        worm_gate = self.wormhole.transform(state_vec) if self.wormhole else Vector.zeros(self.num_experts)
        rag_gate = self.W_rag_gate.linear(fused_state_vec)
        base_gate = [raw_gate[i] + lasso_gate[i] + worm_gate[i] + rag_gate[i] for i in range(self.num_experts)]

        # Hard-prune masked-out experts with a large negative logit so they truly drop out
        # of the softmax (previously the whole logit was *multiplied* by 0/1, which only
        # ever neutralized a pruned expert's score to exactly 0 -- if a surviving expert's
        # raw score was negative, the "pruned" one could still out-compete it).
        NEG_INF = -1e9
        gated_logits = [base_gate[i] if self.tentacles.mask[i] > 0.5 else NEG_INF
                        for i in range(self.num_experts)]
        self._last_gated_logits = gated_logits
        self._last_gate_query = gate_query
        self._last_state_vec_for_gate = state_vec
        self._last_fused_state = fused_state_vec

        # PID output modulates gate *temperature* rather than shifting every logit by the
        # same amount. A uniform additive shift cancels out exactly under softmax, which
        # previously made the PID controller a complete no-op on the model's behaviour --
        # see CHANGELOG.md. A positive correction (e.g. sustained divergence from target)
        # flattens the distribution (more exploration across experts); a near-zero
        # correction keeps it sharp.
        gate_temperature = self._gate_temperature(pid_correction)
        self._last_gate_temperature = gate_temperature
        expert_weights = softmax(Vector([g / gate_temperature for g in gated_logits]))

        # 5. Compute expert outputs
        expert_outputs = [expert.forward(token_ids) for expert in self.moe.experts]

        # 6. Weighted sum
        combined_logits = Matrix.zeros(seq_len, self.vocab_size)
        for t in range(seq_len):
            for v in range(self.vocab_size):
                s = sum(expert_weights[i] * expert_outputs[i].data[t][v] for i in range(self.num_experts))
                combined_logits.data[t][v] = s

        # 7. JEPA auxiliary loss (if available) -- will be added to total loss later
        self._last_jepa_loss = 0.0
        self._last_jepa_context = None
        self._last_jepa_target = None
        if self.jepa and delayed_mean is not None:
            context = delayed_mean   # Vector, previous step's mean embedding
            target = current_mean
            loss_val, _, _ = self.jepa.loss(context, target)
            self._last_jepa_loss = loss_val
            self._last_jepa_context = context
            self._last_jepa_target = target

        return combined_logits, expert_weights.to_list(), expert_outputs

    def _compute_output_gradients(self, expert_outputs, expert_weights, target_distribution, probs, seq_len):
        """
        Pure/read-only: the exact softmax-cross-entropy gradient dL/dW_out for each
        expert whose gate weight isn't negligible, at the final token position:
        dL/dz = probs - target, scaled by that expert's current gate weight (so an
        expert the gate barely uses gets barely any gradient -- standard, correct
        behaviour for a mixture of experts, and it composes with the hard-pruning mask
        in forward(): a pruned expert has weight ~0 and is skipped here too).

        Returns {expert_index: Matrix} of raw gradients, not yet scaled by any learning
        rate or run through an optimizer -- see _apply_output_gradients(). Splitting
        "compute the gradient" from "apply an update" like this is what lets a caller
        average several examples' gradients before taking one step (see train_batch())
        instead of only ever being able to react to one example at a time.

        Everything upstream of W_out (attention, feed-forward, embeddings) stays a
        fixed random projection for now -- this is the "random features" / extreme-
        learning-machine end of the spectrum, not full backprop. See To-Do.md.
        """
        dz = [probs[v] - target_distribution[v] for v in range(self.vocab_size)]
        last_idx = seq_len - 1
        grads = {}
        for i, expert in enumerate(self.moe.experts):
            w_i = expert_weights[i]
            if w_i < 1e-8 or expert._last_hidden is None:
                continue
            x_last = expert._last_hidden.data[last_idx]   # pre-projection hidden state, length d_model
            grads[i] = Matrix([[w_i * x_last[d] * dz[v] for v in range(self.vocab_size)]
                               for d in range(self.d_model)])
        return grads

    def _apply_output_gradients(self, grads):
        """Apply gradients (as returned by _compute_output_gradients, optionally summed/
        averaged across a batch first) to each expert's W_out via self.output_optimizer,
        with optional gradient-norm clipping (self.grad_clip_norm) first."""
        for i, grad_matrix in grads.items():
            if self.grad_clip_norm is not None:
                grad_matrix = clip_grad_norm_matrix(grad_matrix, self.grad_clip_norm)
            self.output_optimizer.step_matrix(f"expert_{i}.W_out", self.moe.experts[i].W_out, grad_matrix)

    def _compute_gate_gradients(self, expert_outputs, expert_weights, target_distribution, probs, seq_len):
        """
        Pure/read-only: the analytic gradient of the cross-entropy loss with respect to
        each of the gate's own linear contributions (moe.W_gate, tentacles.W,
        wormhole.W/b, and W_rag_gate -- whichever are present), via the standard
        softmax-mixture-of-experts gating gradient:

            d(CE)/d(gated_logits[i]) = (expert_weights[i] / temperature) * (s_i - s_bar)

        where s_i = sum_v dz[v] * expert_outputs[i][v] (how much increasing expert i's
        share of the mixture would increase the loss, to first order) and s_bar is its
        expert_weights-weighted average. This is the same gating gradient used for
        soft/sparse mixture-of-experts routing generally, derived here for this
        architecture's specific combination of four additive gate sources.

        Masked-out (pruned) experts are explicitly skipped -- their own gate weights
        get no gradient, since forward() gives them a constant -1e9 logit regardless of
        what those weights are, so the true gradient there is exactly zero (not just
        numerically negligible via a near-zero expert_weights[i], though that would
        also come out close enough not to matter in practice).

        Returns {param_key: Matrix} of raw gradients (not yet scaled by any learning
        rate), keyed by 'moe.W_gate', 'tentacles.W', 'wormhole.W', 'wormhole.b' (a
        Vector, not Matrix), and 'W_rag_gate'. See _apply_gate_gradients().
        """
        gate_query = self._last_gate_query
        state_vec = self._last_state_vec_for_gate
        fused_state = self._last_fused_state
        if gate_query is None or state_vec is None or fused_state is None:
            return {}

        dz = [probs[v] - target_distribution[v] for v in range(self.vocab_size)]
        last_idx = seq_len - 1
        s = [sum(dz[v] * expert_outputs[i].data[last_idx][v] for v in range(self.vocab_size))
             for i in range(self.num_experts)]
        s_bar = sum(s[i] * expert_weights[i] for i in range(self.num_experts))
        temperature = self._last_gate_temperature
        d_gated = [0.0] * self.num_experts
        for i in range(self.num_experts):
            if self.tentacles.mask[i] <= 0.5:
                continue
            d_gated[i] = (expert_weights[i] / temperature) * (s[i] - s_bar)

        grads = {}
        grads['moe.W_gate'] = Matrix([[gate_query[d] * d_gated[i] for i in range(self.num_experts)]
                                      for d in range(self.d_model)])
        n_state = len(state_vec)
        grads['tentacles.W'] = Matrix([[state_vec[d] * d_gated[i] for i in range(self.num_experts)]
                                       for d in range(n_state)])
        if self.wormhole is not None:
            grads['wormhole.W'] = Matrix([[state_vec[d] * d_gated[i] for i in range(self.num_experts)]
                                          for d in range(n_state)])
            grads['wormhole.b'] = Vector([d_gated[i] for i in range(self.num_experts)])
        grads['W_rag_gate'] = Matrix([[fused_state[d] * d_gated[i] for i in range(self.num_experts)]
                                      for d in range(self.d_model)])
        return grads

    def _apply_gate_gradients(self, grads):
        """Apply gate gradients (as returned by _compute_gate_gradients, optionally
        averaged across a batch first) via self.gate_optimizer, with optional
        gradient-norm clipping (self.grad_clip_norm) on the matrix-shaped ones."""
        def _clip(g):
            return clip_grad_norm_matrix(g, self.grad_clip_norm) if self.grad_clip_norm is not None else g

        if 'moe.W_gate' in grads:
            self.gate_optimizer.step_matrix('moe.W_gate', self.moe.W_gate, _clip(grads['moe.W_gate']))
        if 'tentacles.W' in grads:
            self.gate_optimizer.step_matrix('tentacles.W', self.tentacles.W, _clip(grads['tentacles.W']))
        if 'wormhole.W' in grads:
            self.gate_optimizer.step_matrix('wormhole.W', self.wormhole.W, _clip(grads['wormhole.W']))
            self.gate_optimizer.step_vector('wormhole.b', self.wormhole.b, grads['wormhole.b'])
        if 'W_rag_gate' in grads:
            self.gate_optimizer.step_matrix('W_rag_gate', self.W_rag_gate, _clip(grads['W_rag_gate']))

    def _compute_pid_gradients(self, expert_outputs, seq_len, target_distribution, delta=0.05):
        """
        Pure/read-only: central-difference estimate of d(loss)/d(Kp,Ki,Kd). Cheap
        because the expensive part (each expert's transformer forward pass) is reused
        as-is from `expert_outputs`; only the gate temperature and the weighted
        recombination + softmax + cross-entropy at the final position are recomputed
        per probe -- a handful of length-vocab_size loops, not a full forward pass.

        Returns {"Kp": grad, "Ki": grad, "Kd": grad} (empty dict if there's no target
        to differentiate against). See _apply_pid_gradients() for turning this into an
        actual update, and train_batch() for averaging it across several examples first.
        """
        if not target_distribution or self._last_gated_logits is None:
            return {}
        gated_logits = self._last_gated_logits
        last_idx = seq_len - 1

        def loss_for_correction(correction):
            temp = self._gate_temperature(correction)
            weights = softmax(Vector([g / temp for g in gated_logits])).to_list()
            combined_last = [sum(weights[i] * expert_outputs[i].data[last_idx][v] for i in range(self.num_experts))
                              for v in range(self.vocab_size)]
            probs = softmax(Vector(combined_last))
            return -sum(target_distribution[i] * math.log(max(probs[i], 1e-12))
                        for i in range(self.vocab_size) if target_distribution[i] > 0)

        grads = {}
        for gain in ("Kp", "Ki", "Kd"):
            base_val = getattr(self.pid, gain)
            corr_plus = self.pid.compute_output(**{gain: base_val + delta})
            corr_minus = self.pid.compute_output(**{gain: base_val - delta})
            grads[gain] = (loss_for_correction(corr_plus) - loss_for_correction(corr_minus)) / (2 * delta)
        return grads

    def _apply_pid_gradients(self, grads):
        """Apply gradients (as returned by _compute_pid_gradients, optionally averaged
        across a batch first) to the PID gains via self.pid_optimizer, keeping them
        non-negative (a domain constraint on PID gains, not a general optimizer concern,
        so it's enforced here rather than inside the optimizer)."""
        for gain, grad in grads.items():
            base_val = getattr(self.pid, gain)
            new_val = self.pid_optimizer.step_scalar(f"pid.{gain}", base_val, grad)
            setattr(self.pid, gain, max(0.0, new_val))

    def _compute_losses(self, logits, token_ids, target_distribution):
        """
        Pure/read-only: computes every loss term from an already-computed forward pass.
        Shared by step() (which additionally applies learning updates) and evaluate()
        (which doesn't), so the two can never drift apart on what "the loss" means.
        Returns (has_target, ce_loss, kl, probs, l1, jepa_loss, distill_loss, total_loss).
        `probs` is None when there's no target to compare against.
        """
        has_target = bool(token_ids) and bool(target_distribution)
        ce_loss = 0.0
        kl = 0.0
        probs = None
        if has_target:
            last_logit = Vector(logits.data[-1])
            probs = softmax(last_logit)
            for i, t in enumerate(target_distribution):
                if t > 0:
                    ce_loss += -t * math.log(max(probs[i], 1e-12))
            kl = kl_divergence(Vector([math.log(max(p, 1e-12)) for p in probs.data]),
                                Vector(target_distribution))

        l1 = self.tentacles.l1_loss()
        jepa_loss = self._last_jepa_loss

        distill_loss = 0.0
        if self.teacher and token_ids:
            teacher_logits = self.teacher.forward(token_ids)   # assume Matrix (seq_len, vocab)
            if isinstance(teacher_logits, Matrix):
                t_logits_last = Vector(teacher_logits.data[-1])
                s_logits_last = Vector(logits.data[-1])
                distill_loss = distillation_loss(s_logits_last, t_logits_last, temperature=2.0, alpha=0.5,
                                                 true_labels=Vector(target_distribution) if target_distribution else None)

        total_loss = ce_loss + 0.01 * l1 + 0.1 * jepa_loss + 0.1 * distill_loss
        return has_target, ce_loss, kl, probs, l1, jepa_loss, distill_loss, total_loss

    def step(self, token_ids, target_distribution):
        """
        One training/inference step:
        - Forward pass with current PID correction
        - Compute loss (CE + KL + lasso + jepa + distillation)
        - Train each expert's output projection, the gate itself (raw projection,
          lasso tentacles, wormhole, RAG fusion), meta-update PID gains, train JEPA's
          own encoder/predictor, apply BCM plasticity, and periodically prune tentacles
        Returns logits, total_loss.

        For a read-only pass that computes the same loss without updating any weights
        (e.g. for validation/test metrics), use evaluate() instead. For training on
        several examples at once (gradients averaged before one optimizer step, rather
        than one step per example), use train_batch() instead.
        """
        if self.output_lr_scheduler is not None:
            self.output_optimizer.lr = self.output_lr_scheduler.get_lr(self.step_count)

        if not hasattr(self, '_pid_error'):
            self._pid_error = 0.0
        pid_correction = self.pid.step(self._pid_error)

        logits, expert_weights, expert_outputs = self.forward(token_ids, pid_correction)
        seq_len = len(token_ids)

        has_target, ce_loss, kl, probs, l1, jepa_loss, distill_loss, total_loss = \
            self._compute_losses(logits, token_ids, target_distribution)
        # store error for next PID step
        self._pid_error = kl

        # ----- Train each expert's output projection (real gradient descent) -----
        if has_target:
            grads = self._compute_output_gradients(expert_outputs, expert_weights, target_distribution, probs, seq_len)
            self._apply_output_gradients(grads)

            # ----- Train the gate itself (moe.W_gate, tentacles.W, wormhole, W_rag_gate) -----
            gate_grads = self._compute_gate_gradients(expert_outputs, expert_weights, target_distribution, probs, seq_len)
            self._apply_gate_gradients(gate_grads)

        # ----- Meta-update PID gains (cheap finite-difference gradient descent) -----
        pid_grads = self._compute_pid_gradients(expert_outputs, seq_len, target_distribution if has_target else None)
        self._apply_pid_gradients(pid_grads)

        # ----- Train JEPA's own encoder/predictor (real gradient descent + EMA target) -----
        if self.jepa and self._last_jepa_context is not None:
            self.jepa.train_step(self._last_jepa_context, self._last_jepa_target, self.jepa_optimizer)

        # ----- Synaptic plasticity (BCM) on tentacles and gate weights -----
        if token_ids:
            emb_rows = [self.moe.experts[0].embedding.data[idx] for idx in token_ids]
            pre_emb = Vector([sum(col)/len(emb_rows) for col in zip(*emb_rows)]) if emb_rows else Vector.zeros(self.d_model)
        else:
            pre_emb = Vector.zeros(self.d_model)
        post_weights = Vector(expert_weights)
        self.bcm.update(self.tentacles.W, self.last_state_vec, post_weights)
        self.bcm.update(self.moe.W_gate, pre_emb, post_weights)

        # Periodic pruning
        self.step_count += 1
        if self.step_count % 50 == 0:
            self.tentacles.prune()

        return logits, total_loss

    def train_batch(self, batch):
        """
        Train on several (token_ids, target_distribution) examples at once, averaging
        their output-projection, gate, and PID gradients before taking *one* optimizer
        step each, rather than reacting to every single example individually. This is
        the "Batching" item from To-Do.md.

        batch: list of (token_ids, target_distribution) pairs -- e.g. a chunk from
            dataset.TextDataset.batches(batch_size), or any list you build yourself.

        Sequences may be different lengths -- there's deliberately no padding. Unlike a
        numpy/GPU implementation, nothing here needs a single stacked tensor: each
        example still gets its own independent forward pass (advancing the fractal
        Markov chain / delay line / RAG index each time, exactly as step() would, once
        per example). What batching buys you here is the same thing mini-batch SGD
        always buys you -- averaging the gradient over several examples before moving,
        rather than reacting to each one's individual noise -- not a vectorized speedup
        (pure Python doesn't have one to offer here regardless of batch size).

        BCM plasticity and JEPA's own training are still applied per-example (BCM is a
        local Hebbian rule reacting to that example's own activity, not a loss gradient,
        so "batching" it doesn't have the same meaning; JEPA's train_step is a complete,
        self-contained update rather than a compute/apply-split gradient, so there's
        nothing to accumulate across examples the way there is for the output/gate/PID
        gradients), and step_count/pruning still advance once per example, exactly as
        repeated step() calls would.

        Returns (list_of_logits, average_total_loss).
        """
        if not batch:
            raise ValueError("batch must be non-empty")
        if self.output_lr_scheduler is not None:
            self.output_optimizer.lr = self.output_lr_scheduler.get_lr(self.step_count)

        all_logits = []
        loss_sum = 0.0
        grad_sums = {}     # expert_index -> Matrix (running sum)
        grad_counts = {}   # expert_index -> number of examples that contributed
        gate_grad_sums = {}    # param_key -> Matrix or Vector (running sum)
        gate_grad_counts = {}  # param_key -> number of examples that contributed
        pid_grad_sums = {}     # gain -> running sum
        pid_grad_count = 0

        for token_ids, target_distribution in batch:
            if not hasattr(self, '_pid_error'):
                self._pid_error = 0.0
            pid_correction = self.pid.step(self._pid_error)

            logits, expert_weights, expert_outputs = self.forward(token_ids, pid_correction)
            seq_len = len(token_ids)
            has_target, ce_loss, kl, probs, l1, jepa_loss, distill_loss, total_loss = \
                self._compute_losses(logits, token_ids, target_distribution)
            self._pid_error = kl
            all_logits.append(logits)
            loss_sum += total_loss

            if has_target:
                grads = self._compute_output_gradients(expert_outputs, expert_weights, target_distribution, probs, seq_len)
                for i, g in grads.items():
                    if i in grad_sums:
                        prev = grad_sums[i]
                        grad_sums[i] = Matrix([[prev.data[r][c] + g.data[r][c] for c in range(g.cols)]
                                               for r in range(g.rows)])
                        grad_counts[i] += 1
                    else:
                        grad_sums[i] = g
                        grad_counts[i] = 1

                gate_grads = self._compute_gate_gradients(expert_outputs, expert_weights, target_distribution, probs, seq_len)
                for key, g in gate_grads.items():
                    if key not in gate_grad_sums:
                        gate_grad_sums[key] = g
                        gate_grad_counts[key] = 1
                    elif isinstance(g, Matrix):
                        prev = gate_grad_sums[key]
                        gate_grad_sums[key] = Matrix([[prev.data[r][c] + g.data[r][c] for c in range(g.cols)]
                                                      for r in range(g.rows)])
                        gate_grad_counts[key] += 1
                    else:   # Vector (wormhole.b)
                        prev = gate_grad_sums[key]
                        gate_grad_sums[key] = Vector([prev[k] + g[k] for k in range(len(g))])
                        gate_grad_counts[key] += 1

            pid_grads = self._compute_pid_gradients(expert_outputs, seq_len, target_distribution if has_target else None)
            if pid_grads:
                for gain, g in pid_grads.items():
                    pid_grad_sums[gain] = pid_grad_sums.get(gain, 0.0) + g
                pid_grad_count += 1

            # JEPA trains per-example, like BCM below -- see train_batch()'s docstring
            if self.jepa and self._last_jepa_context is not None:
                self.jepa.train_step(self._last_jepa_context, self._last_jepa_target, self.jepa_optimizer)

            if token_ids:
                emb_rows = [self.moe.experts[0].embedding.data[idx] for idx in token_ids]
                pre_emb = Vector([sum(col)/len(emb_rows) for col in zip(*emb_rows)]) if emb_rows else Vector.zeros(self.d_model)
            else:
                pre_emb = Vector.zeros(self.d_model)
            post_weights = Vector(expert_weights)
            self.bcm.update(self.tentacles.W, self.last_state_vec, post_weights)
            self.bcm.update(self.moe.W_gate, pre_emb, post_weights)

            self.step_count += 1
            if self.step_count % 50 == 0:
                self.tentacles.prune()

        averaged_grads = {i: Matrix([[val / grad_counts[i] for val in row] for row in g.data])
                          for i, g in grad_sums.items()}
        self._apply_output_gradients(averaged_grads)

        averaged_gate_grads = {}
        for key, g in gate_grad_sums.items():
            n = gate_grad_counts[key]
            if isinstance(g, Matrix):
                averaged_gate_grads[key] = Matrix([[val / n for val in row] for row in g.data])
            else:
                averaged_gate_grads[key] = Vector([g[k] / n for k in range(len(g))])
        self._apply_gate_gradients(averaged_gate_grads)

        if pid_grad_count:
            averaged_pid_grads = {gain: total / pid_grad_count for gain, total in pid_grad_sums.items()}
            self._apply_pid_gradients(averaged_pid_grads)

        return all_logits, loss_sum / len(batch)

    def evaluate(self, token_ids, target_distribution=None):
        """
        Read-only counterpart to step(): the same forward pass and total_loss, but never
        updates any weights (expert output projections, the gate's four sources,
        PID gains, JEPA's encoder/predictor) and never advances step_count/pruning. Use
        this for validation/test metrics -- calling step() on held-out data would
        silently train on it, since step() has no "no-op" mode of its own.

        Two things this does *not* freeze, deliberately: it still advances the fractal
        Markov chain's internal state and the embedding delay line, exactly like step()
        does. Those are the model's ongoing internal dynamics (closer to an RNN's hidden
        state than to a trainable parameter) rather than something being *fit* to
        whatever you evaluate on, so letting them continue seemed more correct than
        freezing them -- but it does mean calling evaluate() interleaved with step() can
        affect subsequent calls either way, the same as calling step() itself repeatedly
        would. It also does not mutate the PID controller's integral/prev_error: it
        *peeks* at the correction the controller's current gains would produce via
        compute_output(), rather than calling step() on it.
        """
        pid_correction = self.pid.compute_output()
        logits, expert_weights, expert_outputs = self.forward(token_ids, pid_correction)
        _, _, _, _, _, _, _, total_loss = self._compute_losses(logits, token_ids, target_distribution)
        return logits, total_loss

    def sample(self, token_ids, temperature=1.0):
        """Generate next token probability distribution."""
        logits, _, _ = self.forward(token_ids, pid_correction=0.0)   # inference, no PID memory update
        last_logit = Vector(logits.data[-1])
        if temperature != 1.0:
            last_logit = Vector([l / temperature for l in last_logit.data])
        probs = softmax(last_logit)
        return probs.to_list()
