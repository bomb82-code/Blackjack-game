import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from point2 import BlackjackGameTwoPlayers  # 替换为实际游戏文件名

#from train_experience import Experiencechain
from train_enforce import Agent1,train_Q

from train_experience import Agent2,train_E
from A2C_train3 import Actor,Critic,Agent3,train_a2c
import time

num_episodes = 1000000
input_size = 2  # 简化的状态表示为玩家手牌值和庄家手牌值
output_size = 2  # 要牌或停牌两种动作


game = BlackjackGameTwoPlayers()  # 初始化新一局游戏

train_E(num_episodes)
train_Q(num_episodes)
train_a2c(num_episodes)




# 加载训练好的Q表
Q_table_player1 = np.load('Q_table_player1.npy')  # 玩家1的Q表
exprience2 = np.load('exprience1.npy')  # 玩家2的经验表

player_actor3 = Actor(input_size, output_size)
critic3 = Critic(input_size)
player_actor3.load_state_dict(torch.load('model_dict.pth'))# 玩家3的网络模型


class Player1(Agent1):  #强化学习
    def choose_action(self, state): #0是要牌，1是停牌
        """
        根据ε-greedy策略选择动作。
        """
        player_sum, dealer_card = state  # 提取状态信息,总分、庄家明牌
        if dealer_card in ["J", "Q", "K"]:
            dealer_card = 10
        elif dealer_card in ["A"]:
            dealer_card = 1
        if (player_sum==0):  #超过21点就别再要牌了。
            return 1
        else:
            if self.q_table.q_table[player_sum, 0] == self.q_table.q_table[player_sum,1]:
                return random.choice([0, 1])
            else:
                #print('max',np.argmax(self.experience.exp_unit[player_sum]))
                return np.argmax(self.q_table.q_table[player_sum])  # 选择经验最大的动作

class Player2(Agent2): #经验学习
    def choose_action(self, state): #0是要牌，1是停牌
        """
        根据ε-greedy策略选择动作。
        """
        player_sum, dealer_card = state  # 提取状态信息,总分、庄家明牌
        if dealer_card in ["J", "Q", "K"]:
            dealer_card = 10
        elif dealer_card in ["A"]:
            dealer_card = 1
        if (player_sum==0):  #超过21点就别再要牌了。
            return 1
        else:
            if self.experience.exp_unit[player_sum, 0] == self.experience.exp_unit[player_sum,1]:
                return random.choice([0, 1])
            else:
                #print('max',np.argmax(self.experience.exp_unit[player_sum]))
                return np.argmax(self.experience.exp_unit[player_sum])  # 选择经验最大的动作



# 测试轮数
test_episodes = 100000

player1 = Player1(1)  # 强化学习 Q-learning
player2 = Player2(2)  # 经验学习
player3 = Agent3(player_actor3,critic3, 1) # A2C学习
player1.q_table.import_experience(Q_table_player1)
player2.experience.import_experience(exprience2)

def test_Q_E(test_episodes=10000):
    player1_wins = 0
    player2_wins = 0
    draws = 0  # 统计平局数量
    for episode in range(test_episodes):
        game = BlackjackGameTwoPlayers()  # 初始化新一局游戏


        done = False  # 游戏结束标志
        game.dealcard()  # 每轮重新发一次牌
        # 获取玩家初始状态
        state1 = player1.get_state(game)  # 玩家1初始状态
        state2 = player2.get_state(game)  # 玩家2初始状态

        action1 = player1.choose_action(state1)
        action2 = player2.choose_action(state2)
        while not done:
            new_hand1 = game.player_action(1, 'y' if action1 == 0 else 'n')  # 执行动作,并返回当前手牌和分数。
            new_hand2 = game.player_action(2, 'y' if action2 == 0 else 'n')  # 执行动作
            # Player 1 选择动作并执行
            if not action1:

                new_state1 = player1.get_state(game)  # 获取新状态
                state1 = new_state1
                action1 = player1.choose_action(state1)  # 玩家1选择动作
            if not action2:
                # 玩家2根据Q表选择动作并执行
                new_state2 = player2.get_state(game)  # 获取新状态
                state2 = new_state2
                action2 = player2.choose_action(state2)  # 玩家2选择动作

            # 判断游戏是否结束
            if action1 and action2:  # 如果两名玩家都停牌，游戏结束
                winner = game.get_game_result()  # 获取游戏结果
                if winner["winner"] == 1:
                    player1_wins += 1
                elif winner["winner"] == 2:
                    player2_wins += 1
                else:
                    draws += 1
                done = True  # 标记游戏结束

        #print(f"玩家1手牌: {new_hand1 }, 动作: {action1}")
        #print(f"玩家2手牌: {new_hand2 }, 动作: {action2}")
        #print(f"游戏结果: 玩家{winner['winner']}获胜")

    # 打印测试结果
    print(f"第一组测试完成 {test_episodes} 局：")
    print(f"玩家1强化学习胜利次数：{player1_wins} ({player1_wins / test_episodes * 100:.2f}%)")
    print(f"玩家2经验学习胜利次数：{player2_wins} ({player2_wins / test_episodes * 100:.2f}%)")
    print(f"平局次数：{draws} ({draws / test_episodes * 100:.2f}%)")


def test_A2c_E(test_episodes=10000):
    player3_wins = 0
    player2_wins = 0
    draws = 0  # 统计平局数量
    for episode in range(test_episodes):
        game = BlackjackGameTwoPlayers()  # 初始化新一局游戏
        new_hand1, new_hand2 =game.dealcard()  # 每轮重新发一次牌
        # 获取玩家初始状态
        state1 = player3.get_state(game)  # 玩家3在这里坐在1号位置，初始状态
        state2 = player2.get_state(game)  # 玩家2初始状态

        #action2 = player2.choose_action(state2)

        done = False  # 游戏结束标志
        while not done:
            '''
            state1 = torch.FloatTensor(state1).unsqueeze(0)
            player1_action_probs = player_actor3(state1)
            player1_action_dist = torch.distributions.Categorical(player1_action_probs) #这个函数创建类别分布对象，各个类别的概率等于actor（state）输出的两个训练结果值。
            player1_action = player1_action_dist.sample() #从类别分布中随机采样。
            #print('player1_action',player1_action.item())
            action1 = player1_action.item()
            '''

            action1 = player3.choose_action(state1)  # 玩家2选择动作
            action2 = player2.choose_action(state2)  # 玩家2选择动作

            new_hand1 = game.player_action(1, 'y' if action1 == 0 else 'n')  # 执行动作,并返回当前手牌和分数。
            new_hand2 = game.player_action(2, 'y' if action2 == 0 else 'n')  # 执行动作

            if new_hand1["score"] > 21:
                action1 = 1
            if new_hand2["score"] > 21:
                action2 = 1

            if not action1:
                new_state1 = player3.get_state(game)  # 获取新状态
                state1 = new_state1

            if not action2:
                # 玩家2根据Q表选择动作并执行
                new_state2 = player2.get_state(game)  # 获取新状态
                state2 = new_state2


            # 判断游戏是否结束
            if action1 and action2:  # 如果两名玩家都停牌，游戏结束
                winner = game.get_game_result()  # 获取游戏结果
                if winner["winner"] == 1:
                    player3_wins += 1
                elif winner["winner"] == 2:
                    player2_wins += 1
                else:
                    draws += 1
                done = True  # 标记游戏结束

        # print(f"玩家1手牌: {new_hand1 }, 动作: {action1}")
        # print(f"玩家2手牌: {new_hand2 }, 动作: {action2}")
        # print(f"游戏结果: 玩家{winner['winner']}获胜")

    # 打印测试结果
    print(f"第二组测试完成 {test_episodes} 局：")
    print(f"玩家1强化学习胜利次数：{player3_wins} ({player3_wins / test_episodes * 100:.2f}%)")
    print(f"玩家2经验学习胜利次数：{player2_wins} ({player2_wins / test_episodes * 100:.2f}%)")
    print(f"平局次数：{draws} ({draws / test_episodes * 100:.2f}%)")


test_Q_E()


test_A2c_E()

