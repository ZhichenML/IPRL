import functools


def clip_to_range(value, lw=-1, up=1):
    if value > up:
        return up
    if value < lw:
        return lw
    return value


def create_interval(value, delta):
    interval = (value - delta, value + delta)
    return interval

def fold(fun, obs, init):
    return functools.reduce(fun, obs, init)


class Controller():
    def __init__(self, pid_constants=(0, 0, 0), pid_target=0.0, pid_sensor=0, pid_sub_sensor=0, pid_increment=0.0, para_condition=0.0, condition='False'):
        self.pid_constants = pid_constants
        self.pid_target = pid_target
        self.pid_sensor = pid_sensor
        self.pid_sub_sensor = pid_sub_sensor
        self.pid_increment = pid_increment
        self.para_condition = para_condition
        self.condition = condition
        self.final_target = pid_target

    def fold_pid(self, acc, lobs):
        return acc + (self.final_target - lobs[self.pid_sensor][self.pid_sub_sensor])

    def pid_execute(self, obs):
        if eval(self.condition):
            self.final_target = self.pid_target + self.pid_increment
        else:
            self.final_target = self.pid_target
        act = self.pid_constants[0] * (self.final_target - obs[-1][self.pid_sensor][self.pid_sub_sensor]) + \
              self.pid_constants[1] * fold(self.fold_pid, obs, 0) + self.pid_constants[2] * \
              (obs[-2][self.pid_sensor][self.pid_sub_sensor] - obs[-1][self.pid_sensor][self.pid_sub_sensor])
        return act

    def update_parameters(self, pid_constants=(0, 0, 0), pid_target=0.0, pid_increment=0.0, para_condition=0.0):
        self.pid_constants = pid_constants
        self.pid_target = pid_target
        self.pid_increment = pid_increment
        self.para_condition = para_condition

    def pid_info(self):
        return [self.pid_constants, self.pid_target, self.pid_increment, self.para_condition]
