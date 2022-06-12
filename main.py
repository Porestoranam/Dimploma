import random
import pandas as pd
import numpy as np
import pulp as plp
from collections import deque

class SchedulerComparator():
    def __init__(self, job_stats):
        '''
          input: job_stats - DataFrame w columns r,p,w
          r - release date
          p - processing time
          w - job weight
        '''
        self.schedule_task_df = job_stats
        self.n_jobs = job_stats.shape[0]
        self.plan_horizon = np.sum(job_stats['p'])
        self.p_max = np.max(self.schedule_task_df["p"])
        self.max_number = 1e9

        self.jobs_parts = {i: int(job_stats.iloc[i]['p']) for i in range(self.n_jobs)}

        self.exact_value = None

    def construct_AP_matrix(self, bound_type='lower'):
        '''
            Constructs weight matrix for Lower(Upper) bound of Scheduling Problem
        '''
        M = np.array([0] * self.plan_horizon * self.plan_horizon).reshape(self.plan_horizon, self.plan_horizon)
        W = [[[0 for col in range(self.plan_horizon)] for col in range(self.p_max)] for row in range(self.n_jobs)]

        cur_idx = 0
        for i in range(self.n_jobs):
            cur_r = int(self.schedule_task_df.iloc[i]["r"])
            cur_w = self.schedule_task_df.iloc[i]["w"]
            cur_p = int(self.schedule_task_df.iloc[i]["p"])

            for j in range(cur_p):
                # processing part j of job i
                inf_values_idx_before = cur_r + j - 1
                inf_values_idx_after = self.plan_horizon - cur_p + j

                # insert into matrix M
                for k in range(inf_values_idx_before):
                    M[cur_idx][k] = self.max_number
                    W[i][j][k] = self.max_number
                for k in range(inf_values_idx_after + 1, self.plan_horizon):
                    M[cur_idx][k] = self.max_number
                    W[i][j][k] = self.max_number

                if j == (cur_p - 1):
                    for k in range(inf_values_idx_before, inf_values_idx_after + 1):
                        M[cur_idx][k] = (k + 1) * cur_w
                        W[i][j][k] = (k + 1) * cur_w

                if j < (cur_p - 1) and bound_type == 'upper':
                    for k in range(inf_values_idx_before, inf_values_idx_after + 1):
                        M[cur_idx][k] = (k + 1) * cur_w
                        W[i][j][k] = (k + 1) * cur_w
                cur_idx += 1

        if bound_type == 'lower':
            self.AP_lower_matrix = W
            self.AP_lower_matrix_indexed_job_parts = M

        if bound_type == 'upper':
            self.AP_upper_matrix = W
            self.AP_upper_matrix_indexed_job_parts = M

    def solve_AP_problem(self, matrix_type='lower'):
        '''
            Solves AP using algorithm of finding perfect match
            complexity - O(n^3)
        '''

        if matrix_type == 'lower':
            matrix = np.array(self.AP_lower_matrix_indexed_job_parts)
        else:
            matrix = np.array(self.AP_upper_matrix_indexed_job_parts)

        matrix = np.insert(matrix, 0, np.array([0] * self.plan_horizon), axis=0)
        matrix = np.insert(matrix, 0, np.array([0] * (self.plan_horizon + 1)), axis=1)

        u = [0] * (self.plan_horizon + 1)
        v = [0] * (self.plan_horizon + 1)
        parents = [0] * (self.plan_horizon + 1)
        way = [0] * (self.plan_horizon + 1)

        for i in range(1, self.plan_horizon + 1):
            parents[0] = i
            j0 = 0
            minv = [np.inf] * (self.plan_horizon + 1)
            used = [False] * (self.plan_horizon + 1)
            while True:
                used[j0] = True
                i0 = parents[j0]
                delta = np.inf
                j1 = 0
                for j in range(1, self.plan_horizon + 1):
                    if not used[j]:
                        cur = matrix[i0][j] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j

                for j in range(self.plan_horizon + 1):
                    if used[j]:
                        u[parents[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta

                j0 = j1

                if parents[j0] == 0:
                    break

            while True:
                j1 = way[j0]
                parents[j0] = parents[j1]
                j0 = j1
                if j0 == 0:
                    break

        ap_solution = [0] * (self.plan_horizon + 1)
        objective_function = 0

        for j in range(1, self.plan_horizon + 1):
            ap_solution[parents[j]] = j

        if matrix_type == 'lower':
            last_part_idx = 0
            for j in range(self.n_jobs):
                last_part_idx += self.jobs_parts[j]
                objective_function += matrix[last_part_idx][ap_solution[last_part_idx]]

            self.ap_lower_solution = ap_solution
            self.objective_lower_function = objective_function

        elif matrix_type == 'upper':
            cur_idx = 1

            for j in range(self.n_jobs):
                max_value_job_part = 0
                job_part_w_max_value = 0
                for k in range(self.jobs_parts[j]):
                    if max_value_job_part < matrix[cur_idx][ap_solution[cur_idx]]:
                        max_value_job_part = matrix[cur_idx][ap_solution[cur_idx]]
                        job_part_w_max_value = k
                    cur_idx += 1

                objective_function += max_value_job_part

            self.ap_upper_solution = ap_solution
            self.objective_upper_function = objective_function

    def construct_BLP(self):
        '''
          Constructs BLP model for finding exact solution and saves it
        '''
        opt_model = plp.LpProblem(name="MIP Model")

        # define continuous variables
        x_vars = {(j, k, t):
                      plp.LpVariable(cat=plp.LpBinary,
                                     name="x_{0}_{1}_{2}".format(j, k, t))
                  for j in range(self.n_jobs)
                  for k in range(self.jobs_parts[j])
                  for t in range(self.plan_horizon)}

        # Add constraints
        # each job part must be executed at only one moment
        for j in range(self.n_jobs):
            for k in range(self.jobs_parts[j]):
                opt_model.addConstraint(plp.LpConstraint(
                    e=plp.lpSum(x_vars[j, k, t] for t in range(self.plan_horizon)),
                    sense=plp.LpConstraintEQ,
                    rhs=1,
                    name="constraint_only_one_time_assigned_{0}_{1}".format(j, k)))

        # each time only one job part is assgned
        for t in range(self.plan_horizon):
            opt_model.addConstraint(plp.LpConstraint(
                e=plp.lpSum(x_vars[j, k, t] for j in range(self.n_jobs)
                            for k in range(self.jobs_parts[j])),
                sense=plp.LpConstraintEQ,
                rhs=1,
                name="constaraint_each_time_only_one_job_assigned_{0}".format(t)
            ))

        # the last part of each job must be assigned after all previous jobs
        for j in range(self.n_jobs):
            for t in range(self.plan_horizon - 1):
                opt_model.addConstraint(
                    plp.LpConstraint(
                        e=plp.lpSum(plp.lpSum(x_vars[j, k, i] for k in range(self.jobs_parts[j] - 1) for i in
                                              range(t, self.plan_horizon)) - self.jobs_parts[j] * (
                                                1 - x_vars[j, self.jobs_parts[j] - 1, t])),
                        sense=plp.LpConstraintLE,
                        rhs=0,
                        name="constraint_the_part_of_{0}_job_at_{1}_in_right_order".format(j, t)
                    ))

        # define objective function
        objective = plp.lpSum(self.AP_lower_matrix[j][k][t] * x_vars[j, k, t]
                              for j in range(self.n_jobs)
                              for k in range(self.jobs_parts[j])
                              for t in range(self.plan_horizon))

        # for minimization
        opt_model.sense = plp.LpMinimize
        opt_model.setObjective(objective)

        opt_model.solve()

        self.opt_model = opt_model
        self.x_vars = x_vars

    def get_wsrpt_solution(self):
        '''
          task_df - dataframe of scheduling task
          returns wsrpt solution
        '''

        weighted_deque = deque()
        jobs_deque = deque()

        plan_horizon = np.sum(self.schedule_task_df["p"])
        r_x_job = [[] for i in range(0, plan_horizon + 1)]
        job_x_w = self.schedule_task_df['w']
        job_x_p = self.schedule_task_df['p']
        r_dates = sorted(self.schedule_task_df['r'])

        for idx, row in self.schedule_task_df.iterrows():
            r_x_job[row['r']].append(idx)

        scheduling_order = [0]
        twcp_value = 0

        for cur_step in range(1, plan_horizon + 1):
            for job in r_x_job[cur_step]:
                cur_weighted_value = job_x_w[job] / job_x_p[job]

                if len(weighted_deque) == 0:
                    weighted_deque.insert(0, (job_x_w[job], job_x_p[job]))
                    jobs_deque.insert(0, job)
                    continue

                for idx in range(len(weighted_deque)):
                    if weighted_deque[idx][0] / weighted_deque[idx][1] > cur_weighted_value:
                        idx += 1

                        if idx == len(weighted_deque):
                            weighted_deque.insert(idx, (job_x_w[job], job_x_p[job]))
                            jobs_deque.insert(idx, job)
                            break

                    else:
                        weighted_deque.insert(idx, (job_x_w[job], job_x_p[job]))
                        jobs_deque.insert(idx, job)
                        break

            scheduling_order.append(jobs_deque[0])
            weighted_deque[0] = (weighted_deque[0][0], weighted_deque[0][1] - 1)

            if weighted_deque[0][1] == 0:
                twcp_value += cur_step * job_x_w[jobs_deque[0]]
                weighted_deque.popleft()
                jobs_deque.popleft()

        self.wsrpt_scheduling_order = scheduling_order
        self.wsrpt_value = twcp_value

    def get_optimal_schedule(self):
        '''
            Solves BLP model and saves optimal answer and optimal schedule
        '''
        opt_df = pd.DataFrame.from_dict(self.x_vars, orient="index",
                                        columns=["variable_object"])
        opt_df.index = pd.MultiIndex.from_tuples(opt_df.index,
                                                 names=["column_j", "column_k", "column_t"])
        opt_df.reset_index(inplace=True)
        opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.varValue)
        opt_df.drop(columns=["variable_object"], inplace=True)

        self.opt_df = opt_df
        job_parts_x_time = np.array([0] * self.n_jobs * self.p_max).reshape(self.n_jobs, self.p_max)
        time_x_job_part = [(0, 0)] * self.plan_horizon

        for _, row in opt_df.iterrows():
            if row['solution_value'] == 1:
                job_parts_x_time[int(row['column_j'])][int(row['column_k'])] = row['column_t']
                time_x_job_part[int(row['column_t'])] = (int(row['column_j']), int(row['column_k']))

        self.job_parts_x_time = job_parts_x_time
        self.time_x_job_part = time_x_job_part

        exact_value = 0
        for j in range(self.n_jobs):
            k = self.jobs_parts[j]
            exact_value += self.AP_lower_matrix[j][k - 1][job_parts_x_time[j][k - 1]]

        self.exact_value = exact_value

    def get_answers(self, exact_solution_flg=False):
        '''
            Calls all function above to solve each problem (BLP/AP/WSRPT)
        '''

        self.construct_AP_matrix('lower')
        self.construct_AP_matrix('upper')

        self.solve_AP_problem('lower')
        self.solve_AP_problem('upper')
        self.get_wsrpt_solution()
        if exact_solution_flg:
            self.construct_BLP()
            self.get_optimal_schedule()
            if self.objective_lower_function <= self.exact_value <= self.objective_upper_function and self.wsrpt_value >= self.exact_value:
                print(self.objective_lower_function,  self.objective_upper_function, self.wsrpt_value, self.exact_value)

        if self.objective_lower_function <= self.objective_upper_function and self.wsrpt_value >= self.objective_lower_function:
            print(self.objective_lower_function, self.objective_upper_function, self.wsrpt_value, self.exact_value)

        # assert (self.objective_lower_function <= self.objective_upper_function and self.wsrpt_value >= self.objective_lower_function)
        return self.objective_lower_function, self.objective_upper_function, self.wsrpt_value, self.exact_value


max_sump = 1000
opt_percent_lower = {i:{'instances': 0, 'min_ratio': 1, 'sum_ratio': 0} for i in range(2, max_sump + 1)}
opt_tasks_lower = {i:pd.DataFrame() for i in range(2, max_sump + 1)}

opt_percent_upper = {i:{'instances': 0, 'max_ratio': 0, 'sum_ratio': 0} for i in range(2, max_sump + 1)}
opt_tasks_upper = {i:pd.DataFrame() for i in range(2, max_sump + 1)}

all_tasks = []


def check_no_idle(r, p):
    sorted_pairs = sorted([(x, y) for x, y in zip(r, p)])

    for i in range(len(sorted_pairs)):
        if i == 0 and sorted_pairs[i][0] != 1:
            return False
        elif i != 0:
            if sorted_pairs[i][0] > np.sum(x[1] for x in sorted_pairs[:i]) + 1:
              return False
    return True

# save the worst case
idle_cases = 0
for i in range(10000):
    num_jobs = random.randint(2, 20)
    p = 2 * (np.random.randint(1, 20, size=num_jobs))
    if np.sum(p) > max_sump + 1:
        continue

    r = np.random.randint(1, max(np.sum(p) // 10, 2), size=num_jobs)
    if not check_no_idle(r, p):
        idle_cases += 1
        continue

    w = np.random.randint(1, 10, size=num_jobs)

    cur_task = pd.DataFrame({'r': r, 'p': p, 'w': w})

    #print(cur_task)
    cur_sc = SchedulerComparator(cur_task)
    lower, upper, wsrpt_value, exact = cur_sc.get_answers(False)

    if (np.sum(p), np.sum(w), np.sum(r), wsrpt_value) in all_tasks:
        continue
    all_tasks.append((np.sum(p), np.sum(w), np.sum(r), wsrpt_value))

    opt_percent_lower[np.sum(p)]['instances'] += 1
    opt_percent_lower[np.sum(p)]['sum_ratio'] += lower / wsrpt_value

    opt_percent_upper[np.sum(p)]['instances'] += 1
    opt_percent_upper[np.sum(p)]['sum_ratio'] += upper / wsrpt_value

    if opt_percent_lower[np.sum(p)]['min_ratio'] > lower / wsrpt_value:
        opt_percent_lower[np.sum(p)]['min_ratio'] = lower / wsrpt_value
        opt_tasks_lower[np.sum(p)] = cur_task

    if opt_percent_upper[np.sum(p)]['max_ratio'] < upper / wsrpt_value:
        opt_percent_upper[np.sum(p)]['max_ratio'] = upper / wsrpt_value
        opt_tasks_upper[np.sum(p)] = cur_task

    print("ITERATION {0}".format(i))

print(opt_percent_lower)
print(opt_tasks_lower)
print(np.sum(opt_percent_lower[i]['instances'] for i in range(2, max_sump + 1)))

print(opt_percent_upper)
print(opt_tasks_upper)

print(idle_cases)

# test_df = pd.DataFrame({'r': [4,3,3,1], 'p': [8,2,10,4], 'w': [7,5,1,6]})
#
# sc = SchedulerComparator(test_df)
# sc.get_answers()