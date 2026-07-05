"""
fractal_brain/core.py
FractalBrain – all-in-one hybrid architecture, fused with:
- Signal processing (delay line + convolution)
- Wormholes (shortcut connections)
- JEPA (joint embedding predictive loss)
- Distillation (teacher-student loss)
- A real (if partial) learning signal: analytic gradient descent on each expert's
  output projection, plus a cheap finite-difference meta-gradient on the PID gains.
  See CHANGELOG.md for what this replaced and why.
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

class FractalBrain:
    def __init__(self, vocab_size=1000, d_model=256, num_experts=8,
                 num_heads=4, d_ff=512, num_layers=2,
                 num_markov_nodes=7, markov_states=3, max_level=3,
                 use_jepa=True, use_wormhole=True, teacher_model=None,
                 pid_kp=0.8, pid_ki=0.15, pid_kd=0.05, pid_temp_scale=5.0, output_lr=0.05):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_markov_nodes = num_markov_nodes
        self.output_lr = output_lr
        self.pid_temp_scale = pid_temp_scale

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
        self._last_gated_logits = None                  # pre-temperature gate logits, for PID meta-update

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

        # 2. RAG-augmented fused state (d_model) -- currently computed for its side effects
        # (exercising the RAG/cross-attention path) but not yet wired into the gate/logits;
        # see To-Do.md.
        fused_state = self._retrieve_and_fuse(token_ids, state_vec)

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

        # 4. Gate logits: raw projection + lasso tentacles + wormhole shortcut.
        # None of these three depend on pid_correction, so they're cached pre-temperature
        # below for the cheap PID meta-gradient probe in _meta_update_pid_gains.
        raw_gate = self.moe.W_gate.linear(gate_query)
        lasso_gate = Vector(self.tentacles.forward(state_vec))
        worm_gate = self.wormhole.transform(state_vec) if self.wormhole else Vector.zeros(self.num_experts)
        base_gate = [raw_gate[i] + lasso_gate[i] + worm_gate[i] for i in range(self.num_experts)]

        # Hard-prune masked-out experts with a large negative logit so they truly drop out
        # of the softmax (previously the whole logit was *multiplied* by 0/1, which only
        # ever neutralized a pruned expert's score to exactly 0 -- if a surviving expert's
        # raw score was negative, the "pruned" one could still out-compete it).
        NEG_INF = -1e9
        gated_logits = [base_gate[i] if self.tentacles.mask[i] > 0.5 else NEG_INF
                        for i in range(self.num_experts)]
        self._last_gated_logits = gated_logits

        # PID output modulates gate *temperature* rather than shifting every logit by the
        # same amount. A uniform additive shift cancels out exactly under softmax, which
        # previously made the PID controller a complete no-op on the model's behaviour --
        # see CHANGELOG.md. A positive correction (e.g. sustained divergence from target)
        # flattens the distribution (more exploration across experts); a near-zero
        # correction keeps it sharp.
        gate_temperature = self._gate_temperature(pid_correction)
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
        if self.jepa and delayed_mean is not None:
            context = delayed_mean   # Vector, previous step's mean embedding
            target = current_mean
            loss_val, _, _ = self.jepa.loss(context, target)
            self._last_jepa_loss = loss_val

        return combined_logits, expert_weights.to_list(), expert_outputs

    def _update_expert_output_layers(self, expert_outputs, expert_weights, target_distribution, probs, seq_len):
        """
        Real gradient-descent update for each expert's output projection (W_out), using
        the exact softmax-cross-entropy gradient at the final token position:
        dL/dz = probs - target. Each expert's share of that gradient is scaled by its
        current gate weight, so an expert the gate barely uses gets barely any update --
        standard, correct behaviour for a mixture of experts (and it composes fine with
        the hard-pruning mask above: a pruned expert has weight ~0 and is skipped).

        Everything upstream of W_out (attention, feed-forward, embeddings) stays a fixed
        random projection for now -- this is the "random features" / extreme-learning-
        machine end of the spectrum, not full backprop. See To-Do.md.
        """
        dz = [probs[v] - target_distribution[v] for v in range(self.vocab_size)]
        last_idx = seq_len - 1
        lr = self.output_lr
        for i, expert in enumerate(self.moe.experts):
            w_i = expert_weights[i]
            if w_i < 1e-8 or expert._last_hidden is None:
                continue
            x_last = expert._last_hidden.data[last_idx]   # pre-projection hidden state, length d_model
            row_scale = lr * w_i
            W = expert.W_out.data   # (d_model, vocab_size)
            for d in range(self.d_model):
                xd = x_last[d]
                if xd == 0.0:
                    continue
                coeff = row_scale * xd
                row = W[d]
                for v in range(self.vocab_size):
                    row[v] -= coeff * dz[v]

    def _meta_update_pid_gains(self, expert_outputs, seq_len, target_distribution, delta=0.05, meta_lr=0.02):
        """
        Central-difference estimate of d(loss)/d(Kp,Ki,Kd), then one small gradient step
        per gain. Cheap because the expensive part (each expert's transformer forward
        pass) is reused as-is from `expert_outputs`; only the gate temperature and the
        weighted recombination + softmax + cross-entropy at the final position are
        recomputed per probe -- a handful of length-vocab_size loops, not a full forward
        pass. This is what actually makes the PID gains adapt; previously this block
        perturbed a gain and immediately restored it without recomputing anything at all
        (see CHANGELOG.md) -- and even if it had recomputed something, the additive gate
        bug above meant the gain had zero effect on the loss anyway.
        """
        if not target_distribution or self._last_gated_logits is None:
            return
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

        for gain in ("Kp", "Ki", "Kd"):
            base_val = getattr(self.pid, gain)
            corr_plus = self.pid.compute_output(**{gain: base_val + delta})
            corr_minus = self.pid.compute_output(**{gain: base_val - delta})
            grad = (loss_for_correction(corr_plus) - loss_for_correction(corr_minus)) / (2 * delta)
            setattr(self.pid, gain, max(0.0, base_val - meta_lr * grad))

    def step(self, token_ids, target_distribution):
        """
        One training/inference step:
        - Forward pass with current PID correction
        - Compute loss (CE + KL + lasso + jepa + distillation)
        - Train each expert's output projection, meta-update PID gains, apply BCM
          plasticity, and periodically prune tentacles
        Returns logits, total_loss.
        """
        if not hasattr(self, '_pid_error'):
            self._pid_error = 0.0
        pid_correction = self.pid.step(self._pid_error)

        logits, expert_weights, expert_outputs = self.forward(token_ids, pid_correction)
        seq_len = len(token_ids)
        has_target = bool(token_ids) and bool(target_distribution)

        # ----- Losses -----
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
        # store error for next PID step
        self._pid_error = kl

        # Lasso penalty
        l1 = self.tentacles.l1_loss()

        # JEPA auxiliary loss
        jepa_loss = self._last_jepa_loss

        # Distillation loss (if teacher is given)
        distill_loss = 0.0
        if self.teacher and token_ids:
            teacher_logits = self.teacher.forward(token_ids)   # assume Matrix (seq_len, vocab)
            if isinstance(teacher_logits, Matrix):
                t_logits_last = Vector(teacher_logits.data[-1])
                s_logits_last = Vector(logits.data[-1])
                distill_loss = distillation_loss(s_logits_last, t_logits_last, temperature=2.0, alpha=0.5,
                                                 true_labels=Vector(target_distribution) if target_distribution else None)

        total_loss = ce_loss + 0.01 * l1 + 0.1 * jepa_loss + 0.1 * distill_loss

        # ----- Train each expert's output projection (real gradient descent) -----
        if has_target:
            self._update_expert_output_layers(expert_outputs, expert_weights, target_distribution, probs, seq_len)

        # ----- Meta-update PID gains (cheap finite-difference gradient descent) -----
        self._meta_update_pid_gains(expert_outputs, seq_len, target_distribution if has_target else None)

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

    def sample(self, token_ids, temperature=1.0):
        """Generate next token probability distribution."""
        logits, _, _ = self.forward(token_ids, pid_correction=0.0)   # inference, no PID memory update
        last_logit = Vector(logits.data[-1])
        if temperature != 1.0:
            last_logit = Vector([l / temperature for l in last_logit.data])
        probs = softmax(last_logit)
        return probs.to_list()
