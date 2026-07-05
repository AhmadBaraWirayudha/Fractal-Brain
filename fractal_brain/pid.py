"""
fractal_brain/pid.py
Discrete PID controller.
No external dependencies.
"""

class PIDController:
    """A simple discrete PID controller with integral anti‑windup via clamping."""
    def __init__(self, Kp: float, Ki: float, Kd: float, setpoint: float = 0.0,
                 integral_min: float = -10.0, integral_max: float = 10.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral_min = integral_min
        self.integral_max = integral_max
        self.prev_error = 0.0
        self.integral = 0.0
        # breakdown of the most recent step(), cached for compute_output() below
        self.last_error = 0.0
        self.last_integral = 0.0
        self.last_derivative = 0.0

    def step(self, error: float, dt: float = 1.0) -> float:
        """
        Compute PID output given the current error signal.

        `error` should be the raw deviation you want driven to zero (e.g.
        measurement - target). It is compared against `self.setpoint` internally
        (e = error - setpoint); if you've already computed a deviation yourself,
        just leave setpoint at its default of 0 and pass that deviation straight in.
        """
        e = error - self.setpoint
        self.integral += e * dt
        # Clamp integral to avoid windup
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < self.integral_min:
            self.integral = self.integral_min

        derivative = (e - self.prev_error) / dt if dt else 0.0
        output = self.Kp * e + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = e
        self.last_error, self.last_integral, self.last_derivative = e, self.integral, derivative
        return output

    def compute_output(self, Kp=None, Ki=None, Kd=None) -> float:
        """
        Recompute the PID formula from the *cached* (error, integral, derivative) of the
        most recent step(), optionally substituting different gains -- without touching
        any internal state (integral, prev_error). Useful for probing "what would the
        output have been with a slightly different gain", e.g. for finite-difference
        gradient estimation, without corrupting the controller's real trajectory.
        """
        Kp = self.Kp if Kp is None else Kp
        Ki = self.Ki if Ki is None else Ki
        Kd = self.Kd if Kd is None else Kd
        return Kp * self.last_error + Ki * self.last_integral + Kd * self.last_derivative

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_error = 0.0
        self.last_integral = 0.0
        self.last_derivative = 0.0