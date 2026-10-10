import random, time
import robby
import numpy as np
import matplotlib.pyplot as plt



# simulates one episode of Rosie performing actions in her world

ACTION_NAMES = ['MoveNorth', 'MoveSouth', 'MoveEast', 'MoveWest', 'PickUpCan']
ACTION_CODES = [0, 1, 2,3,4]
rw = robby.World(10,10)

# new version that updates Q values (in learn mode)
def run_episode(Q=None, epsilon=0.5, discount_factor=0.8, learning_rate=1, mode='run'):
    assert mode in ['run', 'step', 'quiet', 'learn'], "invalid mode"
    if mode in ['quiet', 'learn']:
        rw.graphics_off()
        rw.goto(random.randrange(10), random.randrange(10))
        rw.distribute_cans(density= 0.50)
    else: 
        rw.goto(random.randrange(10), random.randrange(10))
        rw.distribute_cans()
        rw.graphics_on()
        time.sleep(1)
        
    score = 0
    for i in range(200):
        # observe current state
        state = rw.get_percept_code() 

        
        if Q == None:
            # choose an action randomly
            action = random.choice(ACTION_CODES)
        elif random.uniform(0, 1) < epsilon:
            # with probability epsilon choose an action randomly
            action = random.choice(ACTION_CODES)
        else:
            # with probability (1 - epsilon) choose "best" action
            action = argmax(Q[state])

        # perform action and observe reward
        action_name = ACTION_NAMES[action]
        reward = rw.perform_action(action_name)
        score += reward 

        # observe new state and update Q table
        if mode == 'learn':
            new_state = rw.get_percept_code()
            Q[state][action] += learning_rate * (reward + discount_factor * max(Q[new_state])
                                                 - Q[state][action])

    return score

#---------------------------------------------------------------------------

# updates Q table over multiple training episodes
def train(Q, num_episodes, epsilon=0.5, discount_factor=0.8, learning_rate=1):
    for episode_num in range(1, num_episodes+1):
        score = run_episode(Q, epsilon, discount_factor, learning_rate, mode='learn')
        print(f"Episode #{episode_num}: epsilon {epsilon:.4f}, {score}")

# collects statistics over multiple episodes
def stats(Q=None, epsilon=0.5, trials=100):
    total = 0
    for i in range(trials):
        total += run_episode(Q, epsilon, mode='quiet')
    avg = total / trials
    print(f"Over {trials} episodes, Robby's average score was {avg:.1f}")

# creates an empty Q table
def make_table(num_states=243): 
    table = []
    for i in range(num_states):
        new_row = [0.0] * len(ACTION_CODES)
        table.append(new_row)
    return table

# creates a Q table with random values, for demo purposes only
def make_random_table(num_states=243):
    table = []
    for i in range(num_states):
        new_row = [random.uniform(0, 10) for j in range(5)]
        table.append(new_row)
    return table

# prints out a Q table nicely
def print_table(Q):
    s = "State"
    for name in ACTION_NAMES: 
        s+= f"{name:>11}"
    print(s)
    for row in range(len(Q)): 
        s = f"{row:3}  "
        for q_value in Q[row]: 
            s+= f"{q_value:11.3}"
        print(s)

# returns the position/index number of the maximum value in a list
def argmax(values):
    m = max(values)
    # break ties randomly
    best_indices = [i for i in range(len(values)) if values[i] == m]
    return random.choice(best_indices)

def constant_schedule(num_episodes, epsilon):
    schedule = [epsilon] * num_episodes
    return schedule

def decreasing_schedule(num_episodes, start_epsilon=1.0, interval=50,
                        decrement=0.01, end_epsilon=0.01):
    schedule = []
    epsilon = start_epsilon
    for n in range(1, num_episodes+1):
        schedule.append(epsilon)
        if n % interval == 0:
            epsilon -= decrement
            epsilon = max(epsilon, end_epsilon)
    return schedule


def show_schedule(schedule):
    num_episodes = len(schedule)
    x_values = np.arange(1, num_episodes+1)
    plt.plot(x_values, schedule, 'r-')
    plt.title("Epsilon Schedule")
    plt.xlabel("episode number")
    plt.xlim(1, num_episodes)
    plt.ylabel("epsilon value")
    plt.ylim(0.0, 1.1)
    plt.show()

def train_by_schedule(Q, schedule, discount_factor=0.8, learning_rate=1):
    if type(schedule) is not list:
        print("Second argument must be an epsilon schedule")
        return
    episode_num = 1
    for epsilon in schedule:
        score = run_episode(Q, epsilon, discount_factor, learning_rate, mode='learn')
        print(f"Episode #{episode_num}: epsilon {epsilon:.4f}, scored {score}")
        episode_num += 1


def train_by_schedule_collect(Q, schedule, discount_factor = 0.9, learning_rate = 0.2): 
    scores = []
    episode_num = 1
    for epsilon in schedule: 
        score = run_episode(Q, epsilon, discount_factor, learning_rate, mode='learn')
        scores.append(score)
        if episode_num % 100 == 0:
            print(f"Episode {episode_num}: eps={epsilon:.3f}, score={score}")
        episode_num += 1 
    return scores

def tester(Q, num_episodes = 5000, epsilon = 0.1):
    scores = []
    for ep in range (1, num_episodes +1): 
        score = run_episode(Q, epsilon=epsilon, mode = 'quiet')
        scores.append(score)
        if ep % 100 == 0: 
            print(f"Test episode {ep}: score={score}")
    avg = sum(scores) / len(scores)
    return scores, avg  
    

def plot_100(scores, title): 
    x = list(range(100, len(scores) + 1, 100))    
    y = [scores[i-1] for i  in x]
    plt.figure()
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Episode Score")
    plt.show()

def learning_rate_trials(alphas, num_episodes = 5000): 
    schedule = decreasing_schedule(num_episodes=num_episodes, start_epsilon=1.0, interval= 50, decrement= 0.01, end_epsilon= 0.1)
    results = {} 
    for alpha in alphas: 
        print("\n" + "=" * 50)
        print(f"Experiment with alpha = {alpha}")
        print("=" * 50)
        Q = make_table()
        train_scores = train_by_schedule_collect(Q, schedule, discount_factor= 0.9, learning_rate= alpha)
        results[alpha] = train_scores 
        plot_100(train_scores, f"Training Reward (alpha = {alpha})")
    return results 

def constant_epsilon(epsilon_values, num_episodes = 5000): 
    results = {}
    for eps in epsilon_values:
        print("\n" + "=" * 50)
        print(f"Epsilon Experiment: epsilon = {eps}")
        print("=" * 50)
        Q = make_table()
        schedule = constant_schedule(num_episodes, eps)
        train_scores = train_by_schedule_collect(Q, schedule, discount_factor= 0.9, learning_rate= 0.2)
        results[eps] = train_scores
        plot_100(train_scores, f"Training Reward (constant epsilon = {eps})")
    return results 



def main(): 
    Q = make_table()
    schedule = decreasing_schedule(num_episodes=5000, start_epsilon=1.0, interval =50, decrement =0.01, end_epsilon=0.1)
    train_scores = train_by_schedule_collect(Q, schedule, discount_factor=0.9, learning_rate=0.2)
    plot_100(train_scores, "Training Reward")
    test_scores, avg = tester(Q) 
    plot_100(test_scores, "Test Reward")
    print("Average Test Score:", avg)

    alphas = [0.2, 0.4, 0.6, 0.8]
    learning_rate_trials(alphas)
    epsilon_values = [0.1, 0.3, 0.5, 0.7]
    constant_epsilon(epsilon_values)

if __name__ == "__main__":
    main()



            
        
