import numpy as np
import random
import time

# Q表初始化：每个玩家各自维护一个Q表
# 状态空间大小为 (22 x 10)，动作空间大小为 2
state_space_size = (22, 10)  # 玩家总点数(0-21) x 名牌(1-10)
action_space_size = 2  # 动作空间 {0: 要牌(y), 1: 停牌(n)}
#Q_table_player1 = np.zeros(state_space_size + (action_space_size,))
#Q_table_player2 = np.zeros(state_space_size + (action_space_size,))

# 训练参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索概率
episodes = 2000  # 训练轮数

win_point = 21 #游戏获胜最高点数

# 导入游戏类
from point2 import BlackjackGameTwoPlayers  # 替换为实际游戏文件名


class QTable:
    def __init__(self, state_size=22, action_size=2):
        """
        Q表类
        :param state_size: 状态空间大小
        :param action_size: 动作空间大小
        """
        self.q_table = np.zeros((state_size, action_size))

    def update(self, state, action, value):
        """
        更新 Q 值
        :param state: 当前状态
        :param action: 当前动作
        :param value: 新的 Q 值
        """
        self.q_table[state, action] = value

    def get_q_value(self, state, action):
        """
        获取 Q 值
        :param state: 当前状态
        :param action: 当前动作
        :return: Q 值
        """
        return self.q_table[state, action]

    def get_best_action(self, state):
        """
        获取最佳动作
        :param state: 当前状态
        :return: 最佳动作
        """
        return np.argmax(self.q_table[state])
    def import_experience(self,table):
        self.q_table = table

game = BlackjackGameTwoPlayers()  # 初始化新一局游戏



qtable = QTable()  #定义一个大家公用的表
class Agent1:
    def __init__(self, player_id,  alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        智能体类
        :param state_size: 状态空间大小
        :param action_size: 动作空间大小
        :param alpha: 学习率
        :param gamma: 折扣因子
        :param epsilon: 探索率
        """
        self.q_table = qtable
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.player_id = player_id
        self.hand = game.get_player_hand(self.player_id)  # 获取玩家手牌和当前得分

    def learn(self, state, action, new_state, reward):
        """
        学习并更新 Q 值
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        """
        player_sum, dealer_card = state
        player_sum_new, dealer_card = new_state
        current_q = self.q_table.q_table
        max_future_q = np.max(current_q[player_sum_new])  # 未来 Q 值

        new_q = current_q[player_sum,action] + self.alpha * (reward + self.gamma * max_future_q - current_q[player_sum,action])
        self.q_table.update(player_sum, action, new_q)


    def get_state(self, game):
        """
        获取当前玩家的状态，包括总点数和庄家名牌。
        """
        hand = game.get_player_hand(self.player_id)  # 获取玩家手牌和当前得分
        score = hand["score"] if hand["score"]<22 else 0 # 如果总点数超过21，则视为0  ？？这句咋回事？
        #print(score)
        dealer_card = hand["hand"][1]  # 获取庄家的明牌（假设为第一张牌）
        if dealer_card in ["J", "Q", "K"]:
            dealer_card = 10
        elif dealer_card in ["A"]:
            dealer_card = 1
        return score, dealer_card  # 返回状态元组 (玩家总点数, 庄家明牌)
    '''
    def choose_action(self, state):
        """
        选择动作（基于 ε-greedy 策略）
        :param state: 当前状态
        :return: 选择的动作
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.q_table.q_table.shape[1])  # 随机选择
        return self.q_table.get_best_action(state)  # 选择最佳动作
    '''


    def choose_action(self, state): #0是要牌，1是停牌
        """
        根据ε-greedy策略选择动作。
        """
        player_sum, dealer_card = state  # 提取状态信息,总分、庄家明牌
        if dealer_card in ["J", "Q", "K"]:
            dealer_card = 10
        elif dealer_card in ["A"]:
            dealer_card = 1

        if player_sum == 0:  #超过21点就别再要牌了。
            return 1
        else:
            if random.uniform(0, 1) < self.epsilon:  # ε概率探索
                return random.choice([0, 1])  # 随机选择动作
            else:  # (1-ε)概率利用
                return np.argmax(self.q_table.q_table[player_sum])  # 选择Q值最大的动作

# 训练程序调用
#if __name__ == "__main__":
start_time = time.time()
player1 = Agent1(1)
player2 = Agent1(2)

def train_Q(episodes):
    for episode in range(episodes):
        game = BlackjackGameTwoPlayers()  # 初始化新一局游戏


        done = False  # 游戏结束标志
        new_hand1, new_hand2 = game.dealcard()  # 每轮重新发一次牌


        # 获取玩家初始状态
        state1 = player1.get_state(game)  # 玩家1初始状态
        state2 = player2.get_state(game)  # 玩家2初始状态

        action1 = player1.choose_action(state1) #看玩家需不需要要牌。
        action2 = player2.choose_action(state2) #

        while not done:
            new_hand1 = game.player_action(1, 'y' if action1 == 0 else 'n')  # 执行动作,并返回当前手牌和分数。
            new_hand2 = game.player_action(2, 'y' if action2 == 0 else 'n')  # 执行动作
            # 判断游戏是否结束
            if action1 and action2:  # 如果两名玩家都停牌，游戏结束
                # if  action1 and  action2:  # 如果两名玩家都停牌，游戏结束
                winner = game.get_game_result()  # 获取游戏结果
                reward1, reward2 = (1, -1) if winner["winner"] == 1 else (-1, 1) if winner["winner"] == 2 else (
                0, 0)  # 分配奖励

                # 更新玩家1的Q值
                player_sum1, dealer_card1 = state1
                #dealer_card_index1 = dealer_card1 - 1  # 将庄家明牌转换为索引
                #print(player_sum1, dealer_card_index1, action1)
                player1.learn(state1, action1, state1, reward1)

                # 更新玩家2的Q值
                player_sum2, dealer_card2 = state2
                #dealer_card_index2 = dealer_card2 - 1  # 将庄家明牌转换为索引
                player2.learn(state2, action2, state2, reward2)

                done = True  # 标记游戏结束
            else:
                if not action1:#判断一下前一次是否是要牌，如果不要牌就不用抽牌了。
                    # Player 1 选择动作并执行
                    new_state1 = player1.get_state(game)  # 获取新的状态
                    player1.learn(state1, action1, new_state1, 0)
                    state1 = new_state1
                    action1 = player1.choose_action(state1)  # 玩家1选择动作


                if not action2:
                    # Player 2 选择动作并执行
                    new_state2 = player2.get_state(game)  # 获取新的状态
                    player2.learn(state2, action2, new_state2, 0)
                    state2 = new_state2
                    action2 = player2.choose_action(state2)  # 玩家2选择动作

        ###print("player2",new_hand2)
        #print("winner is", winner["winner"])
        #print(f"已完成第{episode + 1}轮训练")
    #print ("强化学习",player1.q_table.q_table)
    #print ("2",player2.q_table.q_table)
    # 保存训练后的Q表
    np.save('Q_table_player1.npy', player1.q_table.q_table)  # 保存玩家1的Q表
    np.save('Q_table_player2.npy', player2.q_table.q_table)  # 保存玩家2的Q表

    end_time = time.time()
    print("Q-learning训练完成，Q表已保存。")
    print("Q-learning训练时间为:", end_time - start_time, "秒")
