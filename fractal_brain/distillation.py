"""
fractal_brain/distillation.py
Knowledge distillation: train a student model to match a teacher's output distribution.
Uses KL divergence (with temperature) and optional hard‑label cross‑entropy.
"""
import math
from .math_utils import Vector, softmax, kl_divergence

def distillation_loss(student_logits, teacher_logits, temperature=1.0, alpha=0.5, true_labels=None):
    """
    Combined loss: α * KL(teacher||student) + (1-α) * cross‑entropy(student, true_labels).
    - teacher_logits: Vector of raw logits from teacher.
    - student_logits: Vector from student.
    - temperature: softmax temperature.
    - alpha: blending factor.
    - true_labels: Vector of target probabilities (or one‑hot list).
    Returns total scalar loss.
    """
    # Soften distributions
    def softmax_temp(logits, T):
        scaled = Vector([l / T for l in logits.data])
        return softmax(scaled)
    teacher_probs = softmax_temp(teacher_logits, temperature)
    student_probs_temp = softmax_temp(student_logits, temperature)

    # KL divergence (teacher as true distribution, student as approximation)
    kl = kl_divergence(Vector([math.log(max(p, 1e-12)) for p in student_probs_temp.data]), teacher_probs)

    # Cross‑entropy with true labels
    ce = 0.0
    if true_labels is not None:
        # softmax of raw student logits (no temperature)
        student_probs = softmax(student_logits)
        for i in range(len(true_labels)):
            if true_labels[i] > 0:
                ce += -true_labels[i] * math.log(max(student_probs[i], 1e-12))

    total = alpha * kl + (1 - alpha) * ce
    return total